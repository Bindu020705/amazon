"""Text normalization, tokenization and blocking-key helpers for the ER pipeline.

Design notes
------------
* Records arrive in Latin script (S1 always) and in several Indic scripts (S2/S3 names).
  We transliterate Indic scripts to a coarse Latin form so that cross-script pairs
  (``अरिहंत मार्केटिंग`` <-> ``Arihant Marketing``) become comparable.
* Everything is derived from the training/test data itself: no external data sources,
  gazetteers or APIs. The stop-word / legal-suffix lists are generic English domain
  vocabulary, not looked-up business data.
* Blocking keys are 64-bit hashes (blake2b truncated) so that token sets can be stored
  in compact integer arrays.

This module is dependency-free (stdlib only) so it can be imported cheaply.
"""
from __future__ import annotations

import re
import unicodedata
import zlib

# --------------------------------------------------------------------------------------
# Transliteration (Indic -> coarse Latin)
# --------------------------------------------------------------------------------------
# Rather than hand-maintaining one table per script (the varnamala layout has gaps,
# e.g. Devanagari NNA between NA and PA), we derive the phonetic mapping from the
# official Unicode character names: 'DEVANAGARI LETTER KHA' -> 'kh',
# 'MALAYALAM VOWEL SIGN AA' -> 'aa'. That is stdlib-only and correct for every script.

_INDIC_RANGES = [
    (0x0900, 0x097F), (0x0980, 0x09FF), (0x0A00, 0x0A7F), (0x0A80, 0x0AFF),
    (0x0B00, 0x0B7F), (0x0B80, 0x0BFF), (0x0C00, 0x0C7F), (0x0C80, 0x0CFF),
    (0x0D00, 0x0D7F),
]
_INDIC_SCRIPT_PREFIXES = (
    "DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL",
    "TELUGU", "KANNADA", "MALAYALAM",
)

_CONSONANT_PHONEME = {
    "ka": "k", "kha": "kh", "ga": "g", "gha": "gh", "nga": "ng",
    "ca": "ch", "cha": "chh", "ja": "j", "jha": "jh", "nya": "n",
    "tta": "t", "ttha": "th", "dda": "d", "ddha": "dh", "nna": "n",
    "ta": "t", "tha": "th", "da": "d", "dha": "dh", "na": "n",
    "pa": "p", "pha": "ph", "ba": "b", "bha": "bh", "ma": "m",
    "ya": "y", "ra": "r", "la": "l", "va": "v", "wa": "v",
    "sha": "sh", "ssa": "sh", "sa": "s", "ha": "h", "lla": "l",
    "za": "j", "qa": "k", "khha": "kh", "ghha": "g", "rrha": "r",
    "rra": "d", "ddda": "d", "nnna": "n", "llla": "l", "rra": "d",
    "fa": "f", "gaa": "g", "khaa": "kh", "caa": "ch", "jaa": "j",
}
_VOWEL_PHONEME = {
    "a": "a", "aa": "aa", "i": "i", "ii": "ii", "u": "u", "uu": "uu",
    "vocalic r": "ri", "vocalic rr": "ri", "vocalic l": "lri", "vocalic ll": "lri",
    "e": "e", "ee": "ee", "ai": "ai", "o": "o", "oo": "oo", "au": "au",
    "short e": "e", "short o": "o", "candra e": "e", "candra o": "o",
}
_DIGIT_WORDS = "zero one two three four five six seven eight nine ten".split()
_INDIC_TABLE: dict[int, tuple[str, str]] = {}
# value: (kind, phoneme); kind in {vowel, matra, cons, virama, drop, digit, other}


def _build_indic_table() -> None:
    if _INDIC_TABLE:
        return
    for lo, hi in _INDIC_RANGES:
        for cp in range(lo, hi + 1):
            ch = chr(cp)
            try:
                nm = unicodedata.name(ch)
            except ValueError:
                continue
            script, _, rest = nm.partition(" ")
            if script not in _INDIC_SCRIPT_PREFIXES:
                continue
            low = rest.lower()
            if low.startswith("letter "):
                key = low[len("letter "):]
                if key in _CONSONANT_PHONEME:
                    _INDIC_TABLE[cp] = ("cons", _CONSONANT_PHONEME[key])
                elif key in _VOWEL_PHONEME:
                    _INDIC_TABLE[cp] = ("vowel", _VOWEL_PHONEME[key])
                elif key.startswith("vocalic"):
                    _INDIC_TABLE[cp] = ("vowel", "ri")
                elif key == "a" or key.startswith("a"):
                    _INDIC_TABLE[cp] = ("vowel", "a")
                else:
                    _INDIC_TABLE[cp] = ("cons", key[:2])
            elif low.startswith("vowel sign "):
                key = low[len("vowel sign "):]
                if key in _VOWEL_PHONEME:
                    _INDIC_TABLE[cp] = ("matra", _VOWEL_PHONEME[key])
                elif key in ("e", "short e", "candra e"):
                    _INDIC_TABLE[cp] = ("matra", "e")
                elif key in ("o", "short o", "candra o"):
                    _INDIC_TABLE[cp] = ("matra", "o")
                elif key.startswith("vocalic"):
                    _INDIC_TABLE[cp] = ("matra", "ri")
                else:
                    _INDIC_TABLE[cp] = ("matra", key[:2])
            elif "virama" in low or "halant" in low or "pulli" in low:
                _INDIC_TABLE[cp] = ("virama", "")
            elif low.startswith("digit "):
                digit = _DIGIT_WORDS.index(low[len("digit "):]) if low[len("digit "):] in _DIGIT_WORDS else -1
                _INDIC_TABLE[cp] = ("digit", str(digit) if digit >= 0 else "")
            else:
                # signs, nukta, danda, vowel signs without phoneme, ... -> drop
                _INDIC_TABLE[cp] = ("drop", "")


def _is_indic(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _INDIC_RANGES)


def transliterate(text: str) -> str:
    """Coarse Indic -> Latin transliteration (ITRANS-flavoured, no diacritics).

    Latin characters pass through unchanged. This is deliberately lossy: it exists so
    that a Devanagari name and its Latin counterpart land in the same coarse space.
    """
    _build_indic_table()
    out: list[str] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        cp = ord(ch)
        if cp < 128 or cp not in _INDIC_TABLE:
            out.append(ch)
            i += 1
            continue
        kind, phon = _INDIC_TABLE[cp]
        if kind == "vowel":
            out.append(phon)
            i += 1
        elif kind == "digit":
            out.append(phon)
            i += 1
        elif kind == "cons":
            out.append(phon)
            nxt = _INDIC_TABLE.get(ord(text[i + 1]), ("", ""))[0] if i + 1 < n else ""
            if nxt == "virama":
                i += 2
            elif nxt == "matra":
                out.append(_INDIC_TABLE[ord(text[i + 1])][1])
                i += 2
            else:
                out.append("a")
                i += 1
        elif kind == "matra":
            out.append(phon)
            i += 1
        elif kind == "virama":
            i += 1
        else:
            i += 1
    return "".join(out)


# --------------------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
# split "abc123" -> "abc", "123" keeping the joined form too
_ALNUM_SPLIT = re.compile(r"([a-z]+)([0-9]+)|([0-9]+)([a-z]+)")

PUNCT_MAP = {
    "&": " and ", "+": " and ", "@": " at ", "#": " ", "/": " ", "\\": " ",
    ".": " ", ",": " ", "-": " ", "_": " ", "'": "", "’": "", "`": "", '"': " ",
    ":": " ", ";": " ", "(": " ", ")": " ", "[": " ", "]": " ", "{": " ", "}": " ",
    "*": " ", "!": " ", "?": " ", "|": " ", "=": " ", "%": " ", "$": " ", "~": " ",
}

DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
}

# Generic legal / structural tokens: uninformative for identity, dropped from "core" forms.
LEGAL_TOKENS = {
    "ltd", "ltda", "limited", "llc", "llp", "lllp", "inc", "incorporated", "corp",
    "corporation", "co", "company", "companies", "pvt", "private", "pte", "plc",
    "gmbh", "sarl", "sas", "sa", "srl", "spa", "bv", "nv", "ab", "oy", "aps", "as",
    "ag", "kg", "sro", "spzoo", "ug", "sl", "srl", "sdn", "bhd", "pty", "kk", "gk",
    "yugen", "kaisha", "sac", "eirl", "surl", "cia", "lda", "unipessoal", "zrt",
    "kft", "bt", "nyrt", "dba", "trading", "holdings", "holding", "group", "the",
    "and", "of", "for", "at", "by", "to", "in", "on", "de", "la", "le", "les",
    "des", "du", "et", "societe", "société", "entreprise", "etablissement",
}

# Tokens that carry no locality/identity information inside addresses.
ADDR_STOP = {
    "no", "number", "plot", "flat", "shop", "unit", "suite", "floor", "flr", "fl",
    "building", "bldg", "block", "phase", "sector", "street", "st", "road", "rd",
    "avenue", "ave", "av", "drive", "dr", "lane", "ln", "highway", "hwy", "boulevard",
    "blvd", "way", "court", "ct", "circle", "cir", "place", "pl", "park", "square",
    "sq", "terrace", "ter", "trail", "trl", "pkwy", "parkway", "expressway", "route",
    "rue", "chemin", "ch", "impasse", "allee", "allée", "voie", "place", "quai",
    "near", "opp", "opposite", "beside", "behind", "front", "next", "adjacent",
    "nagar", "colony", "society", "complex", "market", "bazaar", "bazar", "chowk",
    "gali", "marg", "road", "cross", "main", "layout", "extension", "extn", "village",
    "po", "p o", "post", "dist", "district", "state", "taluk", "tehsil", "mandal",
    "india", "usa", "united", "states", "france", "french", "republic",
    "north", "south", "east", "west", "upper", "lower", "new", "old", "greater",
    "saint", "st", "sri", "shri", "shree", "mr", "mrs", "ms", "dr",
    "www", "http", "https", "com", "net", "org", "info", "biz", "in", "us", "fr",
}

# State / region names are *not* stop words (they are useful locality evidence) but
# short 2-letter codes are handled separately by the state-code extractor.
US_STATES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il",
    "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt",
    "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri",
    "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "dc", "pr",
}
US_STATE_NAMES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho", "illinois",
    "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland",
    "massachusetts", "michigan", "minnesota", "mississippi", "missouri", "montana",
    "nebraska", "nevada", "newhampshire", "newjersey", "newmexico", "newyork",
    "northcarolina", "northdakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhodeisland", "southcarolina", "southdakota", "tennessee", "texas", "utah",
    "vermont", "virginia", "washington", "westvirginia", "wisconsin", "wyoming",
}
IN_STATES = {
    "andhra", "arunachal", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
    "haryana", "himachal", "jharkhand", "karnataka", "kerala", "madhya", "maharashtra",
    "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "orissa", "punjab",
    "rajasthan", "sikkim", "tamilnadu", "telangana", "tripura", "uttarakhand",
    "uttarpradesh", "bengal", "delhi", "jammu", "kashmir", "ladakh", "puducherry",
    "chandigarh", "haryana", "uttaranchal", "pondicherry",
}
FR_REGIONS = {
    "iledefrance", "auvergne", "rhone", "provence", "occitanie", "normandie",
    "bretagne", "paysdelaloire", "centre", "bourgogne", "franche", "hautsdefrance",
    "grandest", "nouvelle", "aquitaine", "corse", "reunion", "guadeloupe",
}


def strip_diacritics(s: str) -> str:
    """Remove Latin diacritics (é -> e) without touching non-Latin scripts."""
    if s.isascii():
        return s
    out = []
    for ch in s:
        if ord(ch) < 128:
            out.append(ch)
            continue
        if _is_indic(ord(ch)):
            out.append(ch)
            continue
        d = unicodedata.normalize("NFD", ch)
        base = "".join(c for c in d if not unicodedata.combining(c))
        out.append(base if base else ch)
    return "".join(out)


def normalize_text(s: str | None) -> str:
    """Lower-case, de-punctuated, diacritic-free, transliterated text."""
    if not s:
        return ""
    s = s.replace("&", " and ")
    s = transliterate(s)
    s = strip_diacritics(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    out = []
    for ch in s:
        if ch.isalnum() and ord(ch) < 0x2500:
            out.append(ch)
        elif ch in PUNCT_MAP:
            out.append(PUNCT_MAP[ch])
        elif ord(ch) > 0x2000:
            out.append(" ")
        else:
            out.append(" ")
    return _WS.sub(" ", "".join(out)).strip()


def tokens(s_norm: str) -> list[str]:
    return _TOKEN_RE.findall(s_norm)


def token_variants(tok: str) -> list[str]:
    """Extra forms of an alphanumeric token: digit/letter split parts."""
    out = []
    for m in _ALNUM_SPLIT.finditer(tok):
        for g in m.groups():
            if g:
                out.append(g)
    return out


def core_tokens(name_tokens: list[str]) -> list[str]:
    """Name tokens with generic legal words removed (if anything remains)."""
    core = [t for t in name_tokens if t not in LEGAL_TOKENS]
    return core or name_tokens


def addr_tokens(address_tokens: list[str]) -> list[str]:
    core = [t for t in address_tokens if t not in ADDR_STOP]
    return core or address_tokens


def skeleton(tok: str) -> str:
    """Consonant skeleton: drop vowels, collapse repeats (``kylasam`` -> ``kylsm``)."""
    if not tok:
        return ""
    out = []
    prev = ""
    for ch in tok:
        if ch in "aeiou":
            continue
        if ch == prev:
            continue
        out.append(ch)
        prev = ch
    return "".join(out)


def state_code(toks: list[str], country: str | None) -> str:
    """Best-effort 2-letter state/region code from normalized tokens."""
    c = (country or "").lower()
    if c == "us":
        for t in reversed(toks):
            if t in US_STATES:
                return t
        joined = "".join(toks)
        for name in US_STATE_NAMES:
            if name in joined:
                return name
        return ""
    if c == "india":
        for t in reversed(toks):
            if t in IN_STATES:
                return t
        joined = "".join(toks)
        for name in IN_STATES:
            if name in joined:
                return name
        return ""
    if c == "france":
        joined = "".join(toks)
        for name in FR_REGIONS:
            if name in joined:
                return name
        return ""
    return ""


_NUM_RE = re.compile(r"\d+")


def numeric_codes(toks: list[str]) -> list[str]:
    """House/plot/street numbers as normalized digit strings (leading zeros stripped)."""
    out = []
    for t in toks:
        for d in _NUM_RE.findall(t):
            dd = d.lstrip("0")
            if dd and len(dd) <= 8:
                out.append(dd)
    return out


# --------------------------------------------------------------------------------------
# Hashing helpers (compact integer keys)
# --------------------------------------------------------------------------------------
_HASH_SEED = b"biz-er-2026"


def h64(*parts: str) -> int:
    """Stable 63-bit hash of the joined parts (fits in a signed int64).

    Two independently salted CRC32 passes give a cheap, deterministic 64-bit hash.
    (Deterministic across processes and platforms, unlike ``hash()``.)
    """
    s = "\x1f".join(parts).encode("utf-8", "ignore")
    lo = zlib.crc32(s, 1)
    hi = zlib.crc32(s, 0x9E3779B1)
    return ((hi << 31) ^ lo) & 0x7FFFFFFFFFFFFFFF


def compact(s: str) -> str:
    """Alphanumeric-only form (drops spaces) for domain / concatenation comparisons."""
    return "".join(ch for ch in s if ch.isalnum())
