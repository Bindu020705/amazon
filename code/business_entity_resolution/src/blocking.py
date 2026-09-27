"""Blocking / candidate generation.

Key idea
--------
A true match always shares a handful of *rare* keys with its Source-1 record: rare
tokens, rare token prefixes (suffix corruption), consonant skeletons (typos and
transliteration), rare-token/state composites, and compact name prefixes (the same name
written as one concatenation on one side). We index only keys whose document frequency
is below a threshold, then for every S1 record expand its keys *rarest first* until the
per-record candidate budget is used up. That keeps the candidate set small (good for the
ranking criterion) while the probe showed the recall ceiling stays above 0.99.

Compact on-disk/in-memory representation: one ``uint64`` per posting,
``(key40 << 24) | doc24`` - sorted ascending means (key, doc) order, so a key's posting
range is two binary searches. The index can be split into parts so that building it
never needs more RAM than the largest part.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import (  # noqa: E402
    FR_REGIONS, IN_STATES, US_STATE_NAMES, US_STATES, addr_tokens, compact, core_tokens,
    h64, numeric_codes, skeleton, tokens,
)

KEY_BITS = 40
DOC_BITS = 24
DOC_MASK = (1 << DOC_BITS) - 1

# tokens that never carry identity information (mirrors normalize.KEY_STOP semantics)
from normalize import ADDR_STOP, LEGAL_TOKENS  # noqa: E402

KEY_STOP = LEGAL_TOKENS | {
    "services", "service", "solutions", "systems", "technologies", "technology",
    "enterprises", "industries", "international", "global", "general", "national",
    "india", "indian", "usa", "american", "france", "french", "united", "states",
    "new", "old", "north", "south", "east", "west", "upper", "lower", "greater",
} | set(US_STATES) | set(IN_STATES)

# State/region names are hopelessly common blockers, but they are useful as the
# *scoping* half of a composite key, so they are stripped from token keys only.
_REGION_WORDS = US_STATE_NAMES | IN_STATES | FR_REGIONS
KEY_STOP |= _REGION_WORDS
KEY_STOP |= {w.replace(" ", "") for w in _REGION_WORDS}

MAX_KEYS = 20
RARE_DF = 400          # tokens this rare get unscoped prefix/skeleton keys
DF_THETA = 12_000      # keys above this document frequency are not indexed at all


# --------------------------------------------------------------------------------------
# token document frequencies
# --------------------------------------------------------------------------------------
def build_token_df(stores, out_path: str | None = None):
    """Exact document frequency of every core token over the corpus stores.

    Returns ``(sorted_hashes, counts)`` (uint64 / uint32). Also written to ``out_path``
    as two npy files when given.
    """
    parts = []
    for st in stores:
        parts.append(np.asarray(st.atoms_flat, dtype=np.uint64))
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint64)
    flat.sort()
    uniq, cnt = np.unique(flat, return_counts=True)
    uniq = uniq.astype(np.uint64)
    cnt = cnt.astype(np.uint32)
    if out_path:
        np.save(out_path + ".hash.npy", uniq)
        np.save(out_path + ".df.npy", cnt)
    return uniq, cnt


class TokenDF:
    """Token document frequencies, exposed as small membership sets for speed.

    Key generation only ever asks coarse questions - "is this token very common?" - so
    instead of a numpy lookup per token we keep two Python sets of the tokens that are
    above the thresholds. Both sets are small (tens of thousands of entries) even for a
    10M-record corpus, whereas the full hash table is ~8M entries.
    """

    def __init__(self, uniq: np.ndarray, cnt: np.ndarray, rare_df: int = RARE_DF,
                 theta: int = DF_THETA):
        self.uniq = uniq
        self.cnt = cnt
        self.rare_df = rare_df
        self.theta = theta
        self.common = set(uniq[cnt > rare_df].tolist())
        self.too_common = set(uniq[cnt > theta].tolist())

    @classmethod
    def load(cls, path: str, rare_df: int = RARE_DF, theta: int = DF_THETA):
        return cls(np.load(path + ".hash.npy"), np.load(path + ".df.npy"), rare_df, theta)

    def rank(self, h: int) -> int:
        if h in self.too_common:
            return 2
        if h in self.common:
            return 1
        return 0

    def df(self, h: int) -> int:
        p = int(np.searchsorted(self.uniq, np.uint64(h)))
        if p < self.uniq.size and self.uniq[p] == np.uint64(h):
            return int(self.cnt[p])
        return 0

    def dfs(self, hs) -> np.ndarray:
        hs = np.asarray(hs, dtype=np.uint64)
        if self.uniq.size == 0:
            return np.zeros(len(hs), dtype=np.int64)
        p = np.clip(np.searchsorted(self.uniq, hs), 0, self.uniq.size - 1)
        return np.where(self.uniq[p] == hs, self.cnt[p], 0).astype(np.int64)


# --------------------------------------------------------------------------------------
# key generation
# --------------------------------------------------------------------------------------
def record_keys(name_norm: str, addr_norm: str, state_h: int, tokdf: TokenDF,
                max_keys: int = MAX_KEYS, theta: int = DF_THETA) -> list[int]:
    """Blocking keys for one record (identical logic for index and query side)."""
    ncore = core_tokens(tokens(name_norm))
    atok = tokens(addr_norm)
    acore = addr_tokens(atok)

    # token -> hash, computed once; DF questions answered by set membership
    th: dict[str, int] = {}
    for t in ncore:
        if t not in th:
            th[t] = h64("t", t)
    for t in acore:
        if t not in th:
            th[t] = h64("t", t)
    order = list(th)
    rank = tokdf.rank
    too_common = tokdf.too_common

    def scoped(tag: str, tok: str, h: int) -> int:
        if h not in tokdf.common or state_h == 0:
            return h64(tag, tok)
        return h64(tag + "s", str(state_h), tok)

    keys: list[int] = []
    seen: set[int] = set()

    def add(k: int) -> bool:
        if k not in seen:
            seen.add(k)
            keys.append(k)
        return len(keys) < max_keys

    # --- per-token keys: exact token, prefix, consonant skeleton (rarest first)
    info = [(rank(th[t]), -len(t), t) for t in order if t not in KEY_STOP]
    info.sort()
    for _, _, t in info:
        if len(t) < 2 or t.isdigit():
            continue
        h = th[t]
        if h not in too_common:
            if not add(h):
                break
        if len(t) >= 6:
            if not add(scoped("p", t[:5], h)):
                break
        if len(t) >= 6 and not t.isdigit():
            sk = skeleton(t)
            if len(sk) >= 4:
                if not add(scoped("k", sk[:6], h)):
                    break

    # --- composites: always generated (their df is bounded by co-occurrence)
    n_rare = sorted([t for t in ncore if t not in KEY_STOP and len(t) >= 3],
                    key=lambda t: (rank(th[t]), -len(t)))
    a_rare = sorted([t for t in acore if t not in KEY_STOP and len(t) >= 3],
                    key=lambda t: (rank(th[t]), -len(t)))
    if len(n_rare) >= 2:
        add(h64("nn", str(state_h), n_rare[0], n_rare[1]))
    if n_rare and a_rare:
        add(h64("na", str(state_h), n_rare[0], a_rare[0]))
    if len(a_rare) >= 2:
        add(h64("aa", str(state_h), a_rare[0], a_rare[1]))
    if a_rare:
        add(h64("a1", str(state_h), a_rare[0]))
        ds = numeric_codes(atok)
        if ds:
            add(h64("ad", str(state_h), a_rare[0], ds[0]))
            # a house/plot number is a strong signal when paired with a locality half
            for d in ds[:3]:
                if rank(h64("t", d)) == 0 or rank(h64("t", d)) == 1:
                    add(h64("d", str(state_h), d))
                    break

    # --- compact name forms: S3 sometimes stores the whole name as one token / domain
    cname = compact("".join(ncore))
    if len(cname) >= 6:
        add(h64("c8", cname[:8]))
        add(h64("c7", cname[-7:]))
    if len(cname) >= 5:
        add(h64("c5", cname[:5]))
    return keys[:max_keys]


class _SetDF:
    """Minimal TokenDF replacement that only carries the two membership sets."""
    __slots__ = ("common", "too_common")

    def __init__(self, common, too_common):
        self.common = common
        self.too_common = too_common

    def rank(self, h: int) -> int:
        if h in self.too_common:
            return 2
        if h in self.common:
            return 1
        return 0


def _gen_keys_range(store, tokdf, lo: int, hi: int, max_keys: int, theta: int):
    """Fill a preallocated buffer with the keys of records [lo, hi).

    A preallocated numpy buffer avoids materialising tens of millions of Python ints.
    """
    n = hi - lo
    buf = np.empty(n * max_keys, dtype=np.uint64)
    off = np.empty(n + 1, dtype=np.int64)
    off[0] = 0
    pos = 0
    nb, no = store.name_b, store.name_o
    ab, ao = store.addr_b, store.addr_o
    for j, i in enumerate(range(lo, hi)):
        nm = bytes(nb[no[i]:no[i + 1]]).decode("utf-8")
        ad = bytes(ab[ao[i]:ao[i + 1]]).decode("utf-8")
        ks = record_keys(nm, ad, int(store.state[i]), tokdf, max_keys, theta)
        m = len(ks)
        buf[pos:pos + m] = ks
        pos += m
        off[j + 1] = pos
    return buf[:pos], off


def _key_worker(args):
    """Worker: generate keys for a range of one store (runs in a child process)."""
    store_path, common, too_common, lo, hi, max_keys, theta = args
    from norm_store import Store
    st = Store(store_path, with_ids=False)
    tokdf = _SetDF(common, too_common)
    keys, off = _gen_keys_range(st, tokdf, lo, hi, max_keys, theta)
    del st
    return keys, off


def _parallel_keys(store, tokdf: TokenDF, lo: int, hi: int, max_keys: int, theta: int,
                   workers: int):
    """Generate keys for records [lo, hi) using worker processes."""
    from concurrent.futures import ProcessPoolExecutor
    n = hi - lo
    chunk = max(100_000, n // (workers * 3))
    ranges = [(max(lo, a), min(hi, a + chunk)) for a in range(lo, hi, chunk)]
    args = [(store.path, tokdf.common, tokdf.too_common, a, b, max_keys, theta)
            for a, b in ranges]
    all_keys, all_off = [], []
    base = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, o in ex.map(_key_worker, args):
            all_off.append(o[1:] + base)
            base += int(o[-1])
            all_keys.append(k)
    flat = np.concatenate(all_keys) if all_keys else np.zeros(0, dtype=np.uint64)
    off = np.concatenate(all_off) if all_off else np.zeros(0, dtype=np.int64)
    off = np.concatenate([np.zeros(1, dtype=np.int64), off])
    return flat, off


def corpus_keys(store, tokdf, lo: int, hi: int, max_keys: int = MAX_KEYS,
                theta: int = DF_THETA):
    """Generate keys for records ``[lo, hi)`` of ``store`` (single process)."""
    return _gen_keys_range(store, tokdf, lo, hi, max_keys, theta)


# --------------------------------------------------------------------------------------
# inverted index
# --------------------------------------------------------------------------------------
class Index:
    """A key -> postings index split into independently sorted parts."""

    def __init__(self, parts: list[np.ndarray]):
        self.parts = parts

    def save(self, path: str) -> None:
        for i, p in enumerate(self.parts):
            np.save(f"{path}.part{i}.npy", p)

    @classmethod
    def load(cls, path: str, n_parts: int, mmap: bool = True):
        mode = "r" if mmap else None
        return cls([np.load(f"{path}.part{i}.npy", mmap_mode=mode)
                    for i in range(n_parts)])

    def total_postings(self) -> int:
        return int(sum(p.size for p in self.parts))

    def key_range(self, key: int):
        """Return (df, part_id, start, end) list across parts for one key."""
        k = (key & ((1 << KEY_BITS) - 1)) << DOC_BITS
        out = []
        for pi, part in enumerate(self.parts):
            a = int(np.searchsorted(part, np.uint64(k), side="left"))
            b = int(np.searchsorted(part, np.uint64(k + (1 << DOC_BITS)), side="left"))
            out.append((b - a, pi, a, b))
        return out


def build_index_parts(store, tokdf: TokenDF, part_records: int = 2_000_000,
                      max_keys: int = MAX_KEYS, theta: int = DF_THETA,
                      verbose: bool = True, workers: int = 8) -> Index:
    """Build the inverted index over a corpus store, one sorted part at a time.

    Key generation for the records of a part is farmed out to worker processes; the
    (key, doc) pairs are then packed and sorted in this process.
    """
    from concurrent.futures import ProcessPoolExecutor

    parts = []
    n = store.n
    t_start = __import__("time").time()
    for lo in range(0, n, part_records):
        hi = min(lo + part_records, n)
        if workers > 1 and (hi - lo) >= 100_000:
            keys, off = _parallel_keys(store, tokdf, lo, hi, max_keys, theta, workers)
        else:
            keys, off = corpus_keys(store, tokdf, lo, hi, max_keys, theta)
        if not off[-1]:
            continue
        flat = np.fromiter(keys, dtype=np.uint64, count=len(keys))
        docs = np.repeat(np.arange(lo, hi, dtype=np.uint64),
                         np.diff(np.asarray(off, dtype=np.int64)))
        packed = ((flat & ((1 << KEY_BITS) - 1)) << DOC_BITS) | (docs & DOC_MASK)
        packed.sort()
        parts.append(packed)
        if verbose:
            print(f"    index part {len(parts)}: records {lo}-{hi}, "
                  f"{packed.size:,} postings ({__import__('time').time()-t_start:.0f}s)",
                  flush=True)
    return Index(parts)


# --------------------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------------------
def query_keys(store, s1_idx, tokdf, max_keys: int = MAX_KEYS, theta: int = DF_THETA):
    """Blocking keys for a batch of S1 records (flattened arrays)."""
    nb, no = store.name_b, store.name_o
    ab, ao = store.addr_b, store.addr_o
    q_s1: list[int] = []
    q_key: list[int] = []
    for loc, i in enumerate(s1_idx):
        nm = bytes(nb[no[i]:no[i + 1]]).decode("utf-8")
        ad = bytes(ab[ao[i]:ao[i + 1]]).decode("utf-8")
        for k in record_keys(nm, ad, int(store.state[i]), tokdf, max_keys, theta):
            q_s1.append(loc)
            q_key.append(k)
    return (np.asarray(q_s1, dtype=np.int32), np.asarray(q_key, dtype=np.uint64))


def retrieve(index: Index, store, s1_idx: np.ndarray, tokdf: TokenDF,
             key_budget: int = 12, per_key_cap: int = 60, top_k: int = 150,
             min_keys: int = 0, max_keys: int = MAX_KEYS, theta: int = DF_THETA,
             verbose: bool = False):
    """Candidate retrieval for a batch of S1 records.

    For every S1 record we look up its ``key_budget`` rarest keys, expand up to
    ``per_key_cap`` postings of each, and score every candidate by the summed inverse
    document frequency of the keys it shares. The ``top_k`` best-scoring candidates per
    S1 record are returned as ``(s1_local, doc)`` int32 arrays.
    """
    empty = (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32),
             np.zeros(0, dtype=np.float32))
    q_s1, q_key = query_keys(store, s1_idx, tokdf, max_keys, theta)
    if q_s1.size == 0:
        return empty

    n_parts = len(index.parts)
    df = np.zeros(len(q_key), dtype=np.int64)
    starts = np.zeros((len(q_key), n_parts), dtype=np.int64)
    ends = np.zeros((len(q_key), n_parts), dtype=np.int64)
    for pi, part in enumerate(index.parts):
        k = ((q_key & ((1 << KEY_BITS) - 1)) << DOC_BITS)
        a = np.searchsorted(part, k, side="left")
        b = np.searchsorted(part, k + (1 << DOC_BITS), side="left")
        starts[:, pi] = a
        ends[:, pi] = b
        df += (b - a)

    # rarest key first within each S1 (np.lexsort: last key is the primary sort key)
    order = np.lexsort((df, q_s1))
    q_s1, q_key, df = q_s1[order], q_key[order], df[order]
    starts, ends = starts[order], ends[order]

    bounds = np.flatnonzero(np.r_[True, q_s1[1:] != q_s1[:-1]])
    group_start = np.repeat(bounds, np.diff(np.r_[bounds, len(q_s1)]))
    rank = np.arange(len(q_s1)) - group_start
    keep = (rank < key_budget) & (df > 0)
    # rare keys are always fully expanded; a hugely common key is not worth its cost
    allowed = np.minimum(df, per_key_cap)
    q_s1, df = q_s1[keep], df[keep]
    allowed = allowed[keep]
    starts, ends = starts[keep], ends[keep]
    if q_s1.size == 0:
        return empty

    weight = (1.0 / np.sqrt(df)).astype(np.float64)   # inverse-frequency weight

    pair_s1, pair_doc, pair_w = [], [], []
    for pi, part in enumerate(index.parts):
        a = starts[:, pi]
        cnt = np.minimum(ends[:, pi], a + allowed) - a
        tot = int(cnt.sum())
        if tot == 0:
            continue
        local_off = np.repeat(np.cumsum(cnt) - cnt, cnt)
        gather = np.repeat(a, cnt) - local_off + np.arange(tot)
        pair_doc.append((part[gather] & DOC_MASK).astype(np.int64))
        pair_s1.append(np.repeat(q_s1, cnt).astype(np.int64))
        pair_w.append(np.repeat(weight, cnt))
    if not pair_s1:
        return empty

    pair_s1 = np.concatenate(pair_s1)
    pair_doc = np.concatenate(pair_doc)
    pair_w = np.concatenate(pair_w)

    # aggregate the weight of every (S1, candidate) pair
    packed = (pair_s1 << 32) | pair_doc
    uniq, inv = np.unique(packed, return_inverse=True)
    score = np.bincount(inv, weights=pair_w, minlength=len(uniq))
    us1 = (uniq >> 32).astype(np.int64)
    udoc = (uniq & 0xFFFFFFFF).astype(np.int64)

    # top_k per S1 by score
    if top_k and len(uniq) > top_k:
        ordr = np.lexsort((-score, us1))
        us1, udoc, score = us1[ordr], udoc[ordr], score[ordr]
        b2 = np.flatnonzero(np.r_[True, us1[1:] != us1[:-1]])
        gs = np.repeat(b2, np.diff(np.r_[b2, len(us1)]))
        keep2 = (np.arange(len(us1)) - gs) < top_k
        us1, udoc, score = us1[keep2], udoc[keep2], score[keep2]

    return us1.astype(np.int32), udoc.astype(np.int32), score.astype(np.float32)


def dedupe_pairs(pair_s1: np.ndarray, pair_doc: np.ndarray, keep_max: int = 0):
    """Unique (s1, doc) pairs, optionally keeping at most ``keep_max`` per S1."""
    if pair_s1.size == 0:
        return pair_s1, pair_doc
    packed = (pair_s1.astype(np.int64) << 32) | pair_doc.astype(np.int64)
    packed = np.unique(packed)
    s1 = (packed >> 32).astype(np.int32)
    doc = (packed & 0xFFFFFFFF).astype(np.int32)
    if keep_max:
        order = np.arange(packed.size)
        bounds = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1]])
        gs = np.repeat(bounds, np.diff(np.r_[bounds, packed.size]))
        keep = (order - gs) < keep_max
        s1, doc = s1[keep], doc[keep]
    return s1, doc
