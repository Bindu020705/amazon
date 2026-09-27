"""Blocking feasibility probe.

Question 1: for a true (S1, S2/S3) pair, how "common" is the rarest shared token?
If a true pair always shares at least one rare token, rare-token blocking is enough.

Question 2: how big does the candidate pool get for different token-DF caps?

We answer both on a large sample of train S2/S3 (as the searchable corpus) with
document frequencies measured inside that sample (scaled = sample_df * 10M / N).
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import (  # noqa: E402
    addr_tokens, core_tokens, h64, normalize_text, skeleton, tokens, numeric_codes,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(ROOT, "dataset")

N_CORPUS = 1_500_000     # sampled S2/S3 rows used as the searchable corpus
N_PROBE_S1 = 4000        # S1 entities with ground truth probed
FULL_S2S3 = 5_034_616 + 5_285_603


def key_set(name: str, address: str, country: str) -> dict:
    """Token-level key sets for one record (used by the probe only)."""
    nn = normalize_text(name)
    an = normalize_text(address)
    nt = tokens(nn)
    at = tokens(an)
    ncore = core_tokens(nt)
    acore = addr_tokens(at)
    return {
        "name_core": ncore,
        "addr_core": acore,
        "all_core": list(dict.fromkeys(ncore + acore)),
        "nums": numeric_codes(at),
    }


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(0)

    # ---------------- corpus sample: S2 + S3 from train ----------------
    frames = []
    for src in ("2", "3"):
        df = pl.read_csv(os.path.join(DATA, "train", f"train_source{src}.tsv"), separator="\t",
                         columns=["business_name", "business_address"],
                         schema_overrides={c: pl.Utf8 for c in ("business_name", "business_address")})
        idx = rng.choice(df.height, size=N_CORPUS // 2, replace=False)
        frames.append(df[idx.tolist()])
        del df
    corpus = pl.concat(frames)
    del frames
    print(f"corpus sample: {corpus.height} rows  ({time.time()-t0:.1f}s)")

    names = corpus["business_name"].fill_null("").to_list()
    addrs = corpus["business_address"].fill_null("").to_list()
    del corpus

    tok_hashes: list[np.ndarray] = []
    rec_keys: list[tuple[list[str], list[str]]] = []
    for nm, ad in zip(names, addrs):
        nn = normalize_text(nm)
        an = normalize_text(ad)
        ncore = core_tokens(tokens(nn))
        acore = addr_tokens(tokens(an))
        allcore = list(dict.fromkeys(ncore + acore))
        rec_keys.append((ncore, acore))
        if allcore:
            tok_hashes.append(np.fromiter((h64("t", t) for t in allcore), dtype=np.uint64, count=len(allcore)))
    del names, addrs
    all_h = np.concatenate(tok_hashes)
    print(f"tokens in corpus: {all_h.size} distinct: ", end="", flush=True)
    uniq_h, counts = np.unique(all_h, return_counts=True)
    del all_h, tok_hashes
    print(f"{uniq_h.size}  ({time.time()-t0:.1f}s)")

    order = np.argsort(uniq_h)
    uniq_h = uniq_h[order]
    counts = counts[order]

    def df_of(h: int) -> int:
        pos = np.searchsorted(uniq_h, np.uint64(h))
        if pos >= uniq_h.size or uniq_h[pos] != np.uint64(h):
            return 0
        return int(counts[pos])

    scale = FULL_S2S3 / N_CORPUS
    print(f"DF scale factor (sample -> full corpus): {scale:.2f}")

    # ---------------- probe true pairs ----------------
    gt = pl.read_csv(os.path.join(DATA, "train", "train_ground_truth.tsv"), separator="\t",
                     schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8})
    gt = gt.with_columns(pl.col("matched_entity_ids").fill_null(""))
    gt = gt.filter(pl.col("matched_entity_ids") != "")
    s1 = pl.read_csv(os.path.join(DATA, "train", "train_source1.tsv"), separator="\t",
                     schema_overrides={c: pl.Utf8 for c in ("entity_id", "business_name",
                                                            "business_address", "country")})
    take = gt.sample(n=N_PROBE_S1, seed=5)
    s1 = s1.filter(pl.col("entity_id").is_in(take["source1_entity_id"].implode()))
    s1 = s1.join(take, left_on="entity_id", right_on="source1_entity_id", how="inner")
    print(f"probe S1 rows: {s1.height}")

    # fetch the matched S2/S3 records (full files, filtered)
    ids: list[str] = []
    for m in s1["matched_entity_ids"].to_list():
        ids.extend([x for x in m.split(",") if x])
    s2 = pl.read_csv(os.path.join(DATA, "train", "train_source2.tsv"), separator="\t",
                     schema_overrides={c: pl.Utf8 for c in ("entity_id", "business_name",
                                                            "business_address", "country")})
    s3 = pl.read_csv(os.path.join(DATA, "train", "train_source3.tsv"), separator="\t",
                     schema_overrides={c: pl.Utf8 for c in ("entity_id", "business_name",
                                                            "business_address", "country")})
    idset = pl.Series("entity_id", ids)
    truth = pl.concat([s2.filter(pl.col("entity_id").is_in(idset)),
                       s3.filter(pl.col("entity_id").is_in(idset))])
    del s2, s3
    print(f"truth records: {truth.height}")
    tmap = {r["entity_id"]: r for r in truth.iter_rows(named=True)}
    del truth

    rarest_shared: list[int] = []
    n_shared_tok: list[int] = []
    n_shared_by_type: Counter = Counter()
    worst: list[tuple[int, str, str, str, str]] = []
    pooled: Counter = Counter()   # tier -> total postings
    per_s1_pool: list[list[int]] = []
    tier_used: Counter = Counter()

    THETAS = [20, 100, 500, 5000, 50000, 10 ** 12]

    for row in s1.iter_rows(named=True):
        k = key_set(row["business_name"], row["business_address"], row["country"])
        s1_all = set(h64("t", t) for t in k["all_core"])
        s1_name = set(h64("t", t) for t in k["name_core"])
        s1_addr = set(h64("t", t) for t in k["addr_core"])
        s1_skel = set(h64("s", skeleton(t)) for t in k["all_core"] if len(skeleton(t)) >= 3)
        mids = [x for x in row["matched_entity_ids"].split(",") if x]

        # pool sizes: union of postings over own keys at each DF cap
        pool_sizes = []
        for th in THETAS:
            tot = 0
            for t in k["all_core"]:
                dh = df_of(h64("t", t))
                if dh <= th:
                    tot += dh
            pool_sizes.append(tot)
        per_s1_pool.append(pool_sizes)

        for mid in mids:
            tr = tmap.get(mid)
            if tr is None:
                continue
            tk = key_set(tr["business_name"], tr["business_address"], tr["country"])
            c_all = set(h64("t", t) for t in tk["all_core"])
            c_name = set(h64("t", t) for t in tk["name_core"])
            c_addr = set(h64("t", t) for t in tk["addr_core"])
            c_skel = set(h64("s", skeleton(t)) for t in tk["all_core"] if len(skeleton(t)) >= 3)

            shared = s1_all & c_all
            shared_skel = s1_skel & c_skel
            n_shared_tok.append(len(shared))
            if shared & s1_name and shared & c_name:
                n_shared_by_type["name"] += 1
            if shared & s1_addr and shared & c_addr:
                n_shared_by_type["addr"] += 1
            if shared_skel:
                n_shared_by_type["skeleton_only"] += 1

            if shared:
                rr = min(df_of(h) for h in shared)
            else:
                rr = 10 ** 12
            rarest_shared.append(rr)
            if rr > 5000:
                worst.append((rr, row["business_name"], row["business_address"],
                              tr["business_name"], tr["business_address"]))
                if len(worst) > 400:
                    worst.pop(0)

    del tmap, s1, gt, take

    ra = np.array(rarest_shared, dtype=np.int64)
    ns = np.array(n_shared_tok)
    print(f"\n=== true pairs probed: {ra.size} ===")
    print(f"shared core tokens: mean={ns.mean():.2f} median={np.median(ns):.0f} "
          f"zero={int((ns==0).sum())} ({(ns==0).mean()*100:.2f}%)")
    print("share of pairs whose rarest shared token has sample-DF <= X:")
    for th in (0, 1, 2, 5, 10, 20, 50, 100, 500, 1000, 5000, 20000, 100000):
        frac = (ra <= th).mean()
        print(f"   df<={th:>7}: {frac*100:6.2f}%   (full-corpus df<={th*scale:>9.0f})")
    print(f"   no shared token at all: {(ns==0).mean()*100:.2f}%")
    print("pair shares a name token:", n_shared_by_type["name"],
          " address token:", n_shared_by_type["addr"],
          " skeleton-only link:", n_shared_by_type["skeleton_only"])

    pools = np.array(per_s1_pool, dtype=np.int64)
    print("\n=== union-pool size over own keys (sample DF units) ===")
    for i, th in enumerate(THETAS):
        col = pools[:, i]
        print(f"  cap df<={th:>12}: mean={col.mean():9.1f} median={np.median(col):8.0f} "
              f"p90={np.percentile(col,90):9.0f} max={col.max():9.0f} "
              f"-> full-corpus mean≈{col.mean()*scale:,.0f}")

    print("\n=== hardest cases (rarest shared token > 5000 in sample) ===")
    worst.sort(reverse=True)
    seen = 0
    for rr, n1, a1, n2, a2 in worst:
        if rr > 10 ** 11:
            continue
        print(f"  df={rr:>7} | {n1} | {a1}\n            | {n2} | {a2}")
        seen += 1
        if seen >= 25:
            break
    nolink = [w for w in worst if w[0] > 10 ** 11]
    print(f"\n  total hard cases in buffer: {len(worst)}, of which no shared token: {len(nolink)}")
    for rr, n1, a1, n2, a2 in nolink[:15]:
        print(f"  NOLINK | {n1} | {a1}\n         | {n2} | {a2}")
    print(f"\n({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
