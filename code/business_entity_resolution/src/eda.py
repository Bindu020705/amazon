"""EDA: script mix, token DF stats, structural fields. Uses samples for speed."""
from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter

import polars as pl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(ROOT, "dataset")

SCRIPT_RANGES = [
    ("Devanagari", 0x0900, 0x097F),
    ("Bengali", 0x0980, 0x09FF),
    ("Gurmukhi", 0x0A00, 0x0A7F),
    ("Gujarati", 0x0A80, 0x0AFF),
    ("Odia", 0x0B00, 0x0B7F),
    ("Tamil", 0x0B80, 0x0BFF),
    ("Telugu", 0x0C00, 0x0C7F),
    ("Kannada", 0x0C80, 0x0CFF),
    ("Malayalam", 0x0D00, 0x0D7F),
    ("Sinhala", 0x0D80, 0x0DFF),
    ("Arabic", 0x0600, 0x06FF),
    ("Cyrillic", 0x0400, 0x04FF),
    ("Greek", 0x0370, 0x03FF),
    ("CJK", 0x4E00, 0x9FFF),
    ("Thai", 0x0E00, 0x0E7F),
]


def script_of(ch: str) -> str:
    cp = ord(ch)
    if cp < 128:
        return "Latin"
    for name, lo, hi in SCRIPT_RANGES:
        if lo <= cp <= hi:
            return name
    cat = unicodedata.category(ch)
    if cat.startswith("M"):
        return "combining"
    if 0x0080 <= cp <= 0x024F:
        return "Latin-ext"
    return "other"


def script_mix(series: pl.Series, label: str, sample: int = 400_000) -> None:
    if series.len() > sample:
        series = series.sample(sample, seed=1)
    c: Counter[str] = Counter()
    nonempty = 0
    for s in series.to_list():
        if not s:
            continue
        nonempty += 1
        c.update(script_of(ch) for ch in s if not ch.isspace())
    total = sum(c.values()) or 1
    top = ", ".join(f"{k}:{v/total:.3f}" for k, v in c.most_common(6))
    print(f"  {label}: nonempty={nonempty}/{len(series)} chars: {top}")


TOK = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(s: str):
    return TOK.findall(s or "")


def main() -> None:
    for split in ("train", "test"):
        print(f"===== {split} =====")
        for src in ("1", "2", "3"):
            path = os.path.join(DATA, split, f"{split}_source{src}.tsv")
            df = pl.read_csv(path, separator="\t", columns=["business_name", "business_address", "country"],
                             schema_overrides={c: pl.Utf8 for c in ("business_name", "business_address", "country")})
            print(f" source{src}: rows={df.height}")
            script_mix(df["business_name"], "name")
            script_mix(df["business_address"], "addr")
            print("   countries:", df.group_by("country").len().sort("len", descending=True).head(8).to_dicts())
            if src == "1":
                lens = df.select(
                    pl.col("business_name").str.len_chars().mean().alias("name_len"),
                    pl.col("business_address").str.len_chars().mean().alias("addr_len"),
                ).to_dicts()[0]
                print("   lens:", lens)
                print("   empty name:", df.filter(pl.col("business_name").is_null() | (pl.col("business_name") == "")).height,
                      " empty addr:", df.filter(pl.col("business_address").is_null() | (pl.col("business_address") == "")).height)
            del df

    # ---- token DF on a 600k sample of train S2+S3 ----
    print("\n===== token DF (sample of train S2+S3) =====")
    dfs = []
    for src in ("2", "3"):
        d = pl.read_csv(os.path.join(DATA, "train", f"train_source{src}.tsv"), separator="\t",
                        columns=["business_name", "business_address"],
                        schema_overrides={c: pl.Utf8 for c in ("business_name", "business_address")})
        dfs.append(d.sample(n=300_000, seed=7))
    d = pl.concat(dfs)
    print("sample rows:", d.height)
    toks = d.select(
        (pl.col("business_name").fill_null("") + " " + pl.col("business_address").fill_null(""))
        .str.to_lowercase()
        .str.extract_all(r"[^\W_]+")
        .alias("t")
    ).explode("t").drop_nulls()
    dfc = toks.group_by("t").len().sort("len", descending=True)
    n = d.height
    print("distinct tokens:", dfc.height, " tokens/record avg:", round(dfc["len"].sum() / n, 2))
    print("most common 60:", [t for t in dfc.head(60)["t"].to_list()])
    for thr in (2, 5, 20, 100, 1000, 5000):
        covered = dfc.filter(pl.col("len") >= thr)
        print(f"  tokens with df>={thr}: {covered.height} ; postings={covered['len'].sum()} "
              f"({covered['len'].sum()/dfc['len'].sum():.3f} of all)")
    print("df histogram:", dfc.select(pl.col("len").clip(1, 5).alias("k")).group_by("k").len().sort("k").to_dicts())
    # digit / alnum token shares
    dd = dfc.filter(pl.col("t").str.contains(r"^[0-9]+$"))
    print("pure-digit tokens:", dd.height, "postings:", dd["len"].sum())
    print("top digit tokens:", dd.head(10).to_dicts())


if __name__ == "__main__":
    main()
