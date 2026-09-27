"""Reconnaissance: understand ground-truth structure and the noise between matched records.

Run:  python code/business_entity_resolution/src/recon.py
Writes samples to work/recon/ for inspection.
"""
from __future__ import annotations

import os
import sys
import random
from collections import Counter

import polars as pl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(ROOT, "dataset")
WORK = os.path.join(ROOT, "work", "recon")
os.makedirs(WORK, exist_ok=True)


def load_gt(split: str) -> pl.DataFrame:
    return pl.read_csv(
        os.path.join(DATA, split, f"{split}_ground_truth.tsv"),
        separator="\t",
        schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
    )


def main() -> None:
    gt = load_gt("train")
    print("GT rows:", gt.height)

    gt = gt.with_columns(
        pl.col("matched_entity_ids").fill_null("").alias("m"),
    )
    gt = gt.with_columns(
        [
            pl.col("m").str.split(",").list.len().alias("n_match"),
            pl.col("m").str.contains("S2-").alias("has_s2"),
            pl.col("m").str.contains("S3-").alias("has_s3"),
        ]
    )
    n = gt.select(
        [
            pl.len().alias("total"),
            (pl.col("n_match") == 0).sum().alias("singletons"),
            (pl.col("n_match") > 0).sum().alias("with_matches"),
            pl.col("n_match").mean().alias("mean_match"),
            pl.col("n_match").max().alias("max_match"),
            pl.col("n_match").filter(pl.col("n_match") > 0).mean().alias("mean_match_nonzero"),
        ]
    )
    print(n.to_dicts()[0])

    print("\nmatch-count histogram (0..20 and >20):")
    hist = (
        gt.select(pl.col("n_match").clip(0, 20).alias("k"))
        .group_by("k")
        .len()
        .sort("k")
    )
    print(hist.to_dicts())

    combo = (
        gt.filter(pl.col("n_match") > 0)
        .select(
            pl.when(pl.col("has_s2") & pl.col("has_s3"))
            .then(pl.lit("both"))
            .when(pl.col("has_s2"))
            .then(pl.lit("s2_only"))
            .otherwise(pl.lit("s3_only"))
            .alias("combo")
        )
        .group_by("combo")
        .len()
        .sort("len", descending=True)
    )
    print("\nsource combo of matches:")
    print(combo.to_dicts())

    # ---- side by side examples: S1 vs its matched S2/S3 records ----
    pos = gt.filter(pl.col("n_match") > 0)
    sample = pos.sample(n=min(3000, pos.height), seed=17)

    s1 = pl.read_csv(
        os.path.join(DATA, "train", "train_source1.tsv"),
        separator="\t",
        schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8,
                          "business_address": pl.Utf8, "country": pl.Utf8},
    ).filter(pl.col("entity_id").is_in(sample["source1_entity_id"]))

    print("\nS1 sample rows:", s1.height)
    print("country mix S1 train:", s1.group_by("country").len().sort("len", descending=True).to_dicts())

    ids = []
    for row in sample.iter_rows(named=True):
        ids.extend([x for x in row["m"].split(",") if x])

    def fetch(src: str) -> pl.DataFrame:
        return (
            pl.scan_csv(
                os.path.join(DATA, "train", f"train_source{src}.tsv"),
                separator="\t",
                schema_overrides={c: pl.Utf8 for c in ("entity_id", "business_name",
                                                       "business_address", "country")},
            )
            .filter(pl.col("entity_id").is_in(ids))
            .collect()
        )

    s2 = fetch("2")
    s3 = fetch("3")
    print("matched S2 fetched:", s2.height, " matched S3 fetched:", s3.height)

    s2m = {r["entity_id"]: r for r in s2.iter_rows(named=True)}
    s3m = {r["entity_id"]: r for r in s3.iter_rows(named=True)}
    s1m = {r["entity_id"]: r for r in s1.iter_rows(named=True)}

    with open(os.path.join(WORK, "matched_examples.txt"), "w", encoding="utf-8") as f:
        for row in sample.head(300).iter_rows(named=True):
            s1r = s1m.get(row["source1_entity_id"])
            if not s1r:
                continue
            f.write("=" * 100 + "\n")
            f.write(f"S1 {s1r['entity_id']} | {s1r['business_name']} | {s1r['business_address']} | {s1r['country']}\n")
            for mid in row["m"].split(","):
                if not mid:
                    continue
                r = s2m.get(mid) or s3m.get(mid)
                if r:
                    f.write(f"   {mid} | {r['business_name']} | {r['business_address']} | {r['country']}\n")
                else:
                    f.write(f"   {mid} | <not found in train sources!>\n")

    # ---- also sample random S1 that are singletons, for contrast ----
    neg = gt.filter(pl.col("n_match") == 0).sample(n=50, seed=3)
    with open(os.path.join(WORK, "singleton_s1.txt"), "w", encoding="utf-8") as f:
        for row in neg.iter_rows(named=True):
            s1r = s1m.get(row["source1_entity_id"])
            if s1r:
                f.write(f"{s1r['entity_id']} | {s1r['business_name']} | {s1r['business_address']} | {s1r['country']}\n")

    # ---- test country mix ----
    t1 = pl.scan_csv(os.path.join(DATA, "test", "test_source1.tsv"), separator="\t",
                     schema_overrides={c: pl.Utf8 for c in ("entity_id", "business_name",
                                                            "business_address", "country")})
    print("\nTEST country mix S1:", t1.group_by("country").len().sort("len", descending=True).collect().to_dicts())
    t2 = pl.scan_csv(os.path.join(DATA, "test", "test_source2.tsv"), separator="\t",
                     schema_overrides={c: pl.Utf8 for c in ("entity_id", "business_name",
                                                            "business_address", "country")})
    print("TEST country mix S2:", t2.group_by("country").len().sort("len", descending=True).collect().to_dicts())


if __name__ == "__main__":
    main()
