"""Predict matches for the test set and write the two submission TSVs.

Streams over test_source1.tsv in batches: build the test corpus index once, then for
every batch retrieve candidates, compute features, score with the trained LightGBM
model, and keep pairs above the tuned threshold. Writes:

* ``output/matching_results.tsv``  - one row per S1 entity, matches above threshold
* ``output/candidate_pairs.tsv``   - the full candidate list fed to the model
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from norm_store import Store  # noqa: E402
from pair_features import FeatureBuilder  # noqa: E402
from pipeline import (  # noqa: E402
    WORK, build_corpus_and_index, drop_store, open_corpus, process_batch,
)
from run_outputs import write_submission  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(ROOT, "dataset")
OUT = os.path.join(ROOT, "output")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(WORK, "model.txt"))
    ap.add_argument("--threshold", default=os.path.join(WORK, "threshold.npy"))
    ap.add_argument("--batch", type=int, default=100_000)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--key-budget", type=int, default=14)
    ap.add_argument("--per-key-cap", type=int, default=80)
    ap.add_argument("--tag", default="_full")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-corpus", type=int, default=0)
    ap.add_argument("--limit-s1", type=int, default=0)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--chunk-out", default=os.path.join(WORK, "cand", "test_chunks"))
    ap.add_argument("--save-chunks", action="store_true",
                    help="also save per-batch candidate npz (needs ~4 GB disk)")
    args = ap.parse_args()

    import lightgbm as lgb
    model = lgb.Booster(model_file=args.model)
    thr = float(np.load(args.threshold)[0])
    print(f"model loaded, threshold={thr:.3f}", flush=True)

    t0 = time.time()
    limit = args.limit_corpus if args.limit_corpus > 0 else None
    index, tokdf, s1_path, n_corpus = build_corpus_and_index(
        "test", args.tag, workers=args.workers, keep_corpus=True, limit=limit)
    from norm_store import Store
    corpus_path, corpus = open_corpus("test", args.tag)
    s1 = Store(s1_path)
    print(f"test: {s1.n:,} S1 entities, {n_corpus:,} corpus records "
          f"({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_chunks:
        os.makedirs(args.chunk_out, exist_ok=True)
    fb = None
    n_pairs = n_kept = 0
    match_fh = open(os.path.join(args.out_dir, "matching_results.tsv"), "w",
                    encoding="utf-8", newline="")
    cand_fh = open(os.path.join(args.out_dir, "candidate_pairs.tsv"), "w",
                   encoding="utf-8", newline="")
    match_fh.write("source1_entity_id\tmatched_entity_ids\n")
    cand_fh.write("source1_entity_id\tcandidate_entity_ids\n")

    if args.limit_s1 > 0:
        n_s1_use = min(args.limit_s1, s1.n)
    else:
        n_s1_use = s1.n

    for lo in range(0, n_s1_use, args.batch):
        hi = min(lo + args.batch, s1.n)
        ps, pd, sc, X, fb = process_batch(index, tokdf, corpus, s1, lo, hi,
                                          top_k=args.topk, key_budget=args.key_budget,
                                          per_key_cap=args.per_key_cap, fb=fb)
        pv = model.predict(X, num_iteration=getattr(model, "best_iteration", None) or None)
        keep = pv >= thr
        n_pairs += ps.size
        n_kept += int(keep.sum())

        ids = s1.ids[lo:hi]
        write_submission(match_fh, cand_fh, ids, ps, pd, pv, thr,
                         corpus_ids=corpus.ids, lo=lo)
        if args.save_chunks:
            np.savez(os.path.join(args.chunk_out, f"chunk_{lo:08d}.npz"),
                     ps=ps, pd=pd, pv=pv.astype(np.float32))
        print(f"  [{hi:,}/{n_s1_use:,}] pairs={n_pairs:,} kept={n_kept:,} "
              f"({time.time()-t0:.0f}s)", flush=True)

    match_fh.close()
    cand_fh.close()
    drop_store(corpus_path)
    print(f"\nDONE pairs={n_pairs:,} kept={n_kept:,} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
