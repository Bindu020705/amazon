"""Key-level blocking probe.

Builds the real key sets (as in the pipeline) over a large sample of train S2/S3,
measures key document frequencies, and reports:

  1. per-pair retrievability: the rarest *shared key* between an S1 record and each of
     its true matches (this is the recall ceiling of "expand keys with df <= cap"),
  2. per-entity F_0.5 ceiling (perfect precision assumed) for a range of df caps,
  3. index size / pool size statistics so the memory budget is known up front.
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from norm_store import encode_record  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(ROOT, "dataset")

N_CORPUS = 2_000_000
N_PROBE_S1 = 6000
FULL = 5_034_616 + 5_285_603


def chunk_encode(rows):
    keys_out = []
    for nm, ad, co in rows:
        e = encode_record(nm or "", ad or "", co or "")
        keys_out.append(e["keys"])
    return keys_out


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(1)
    frames = []
    for src in ("2", "3"):
        df = pl.read_csv(os.path.join(DATA, "train", f"train_source{src}.tsv"), separator="\t",
                         columns=["business_name", "business_address", "country"],
                         schema_overrides={c: pl.Utf8 for c in ("business_name", "business_address", "country")})
        idx = rng.choice(df.height, size=N_CORPUS // 2, replace=False)
        frames.append(df[idx.tolist()])
        del df
    corpus = pl.concat(frames)
    del frames
    rows = list(zip(corpus["business_name"].fill_null("").to_list(),
                    corpus["business_address"].fill_null("").to_list(),
                    corpus["country"].fill_null("").to_list()))
    del corpus
    print(f"corpus rows: {len(rows)}", flush=True)

    CH = 50_000
    all_keys: list[np.ndarray] = []
    counts_per_record: list[int] = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(chunk_encode, rows[i:i + CH]) for i in range(0, len(rows), CH)]
        for f in futs:
            for kl in f.result():
                counts_per_record.append(len(kl))
                if kl:
                    all_keys.append(np.fromiter(kl, dtype=np.uint64, count=len(kl)))
    del rows
    flat = np.concatenate(all_keys) if all_keys else np.zeros(0, dtype=np.uint64)
    del all_keys
    print(f"keys generated: {flat.size} ({flat.size/len(counts_per_record):.1f}/record) "
          f"in {time.time()-t0:.0f}s", flush=True)

    uniq, cnt = np.unique(flat, return_counts=True)
    del flat
    scale = FULL / N_CORPUS
    print(f"distinct keys: {uniq.size}   df scale: {scale:.2f}", flush=True)

    def df_of(h: int) -> int:
        p = np.searchsorted(uniq, np.uint64(h))
        if p >= uniq.size or uniq[p] != np.uint64(h):
            return 0
        return int(cnt[p])

    for capp in (3, 10, 100, 1000, 10000, 100000, 400000):
        sel = cnt <= capp
        print(f"  keys with sample df<={capp:>7}: {sel.sum():>10,} "
              f"postings={cnt[sel].sum():>12,} (full≈{cnt[sel].sum()*scale:>14,.0f})")

    # ---------------- probes ----------------
    gt = pl.read_csv(os.path.join(DATA, "train", "train_ground_truth.tsv"), separator="\t",
                     schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8})
    gt = gt.with_columns(pl.col("matched_entity_ids").fill_null(""))
    s1 = pl.read_csv(os.path.join(DATA, "train", "train_source1.tsv"), separator="\t",
                     schema_overrides={c: pl.Utf8 for c in ("entity_id", "business_name",
                                                            "business_address", "country")})
    take = gt.sample(n=N_PROBE_S1, seed=11)
    s1 = s1.join(take, left_on="entity_id", right_on="source1_entity_id", how="inner")
    print(f"probe S1: {s1.height}", flush=True)

    ids = []
    for m in s1["matched_entity_ids"].to_list():
        ids.extend([x for x in m.split(",") if x])
    idset = pl.Series("entity_id", ids).implode()
    tmap = {}
    for src in ("2", "3"):
        d = pl.read_csv(os.path.join(DATA, "train", f"train_source{src}.tsv"), separator="\t",
                        schema_overrides={c: pl.Utf8 for c in ("entity_id", "business_name",
                                                               "business_address", "country")})
        d = d.filter(pl.col("entity_id").is_in(idset))
        for r in d.iter_rows(named=True):
            tmap[r["entity_id"]] = r
        del d
    print(f"truth records: {len(tmap)}", flush=True)

    s1_rows = list(zip(s1["entity_id"].to_list(), s1["business_name"].to_list(),
                       s1["business_address"].to_list(), s1["country"].to_list(),
                       s1["matched_entity_ids"].to_list()))

    # per-entity list of pair-level (min shared key df)
    ent_pair: list[list[int]] = []
    key_kind = Counter()
    hard = []
    for eid, nm, ad, co, mid in s1_rows:
        e = encode_record(nm or "", ad or "", co or "")
        s1_keys = set(e["keys"])
        dfs = []
        for m in [x for x in mid.split(",") if x]:
            tr = tmap.get(m)
            if tr is None:
                continue
            te = encode_record(tr["business_name"] or "", tr["business_address"] or "", tr["country"] or "")
            shared = s1_keys & set(te["keys"])
            if not shared:
                dfs.append(10 ** 12)
                continue
            d = min(df_of(h) for h in shared)
            d = max(d, 1)  # absent from sample == very rare in full corpus
            dfs.append(d)
            if d > 3000:
                hard.append((d, nm, ad, tr["business_name"], tr["business_address"]))
        ent_pair.append(dfs)

    del tmap, s1, gt, take
    print(f"\n=== probe done ({time.time()-t0:.0f}s) ===")

    def ceiling(cap):
        tot = 0.0
        for dfs in ent_pair:
            if not dfs:
                continue
            r = sum(1 for d in dfs if d <= cap) / len(dfs)
            tot += 0.0 if r == 0 else (1.25 * r) / (0.25 + r)
        return tot / max(len(ent_pair), 1)

    print("\nper-entity F0.5 ceiling by key-df cap (perfect precision):")
    for cap in (10, 100, 1000, 3000, 10000, 35000, 100000, 350000, 10 ** 12):
        c = ceiling(cap)
        print(f"   cap sample-df<={cap:>8} (full≈{cap*scale:>12,.0f}): ceiling={c:.4f}")

    allpairs = np.array([d for dfs in ent_pair for d in dfs], dtype=np.int64)
    print(f"\npairs: {allpairs.size}  median rarest-shared-key df={np.median(allpairs):.0f}")
    for th in (1, 5, 20, 100, 500, 2000, 10000, 50000):
        print(f"   share of pairs with rarest shared key df<={th:>6}: {(allpairs<=th).mean()*100:6.2f}%")
    print(f"   pairs with NO shared key: {(allpairs>10**11).mean()*100:.3f}%")

    hard.sort(reverse=True)
    print("\nhardest pairs (rarest shared key df > 3000 in sample):")
    for d, n1, a1, n2, a2 in hard[:20]:
        tag = "NOLINK" if d > 10 ** 11 else f"df={d}"
        print(f"  {tag:>9} | {n1} | {a1}\n            | {n2} | {a2}")


if __name__ == "__main__":
    main()
