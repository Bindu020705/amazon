"""Collect labeled training pairs: blocking + features + ground-truth labels, batch-wise.

The corpus index/tokdf are built once; then we stream over the first ``--n-s1`` train
Source-1 records, retrieving the top ``--topk`` candidates per record and computing pair
features. Each batch is written as one ``.npz`` chunk (features as float16 to keep the
disk footprint small). The merged corpus store is deleted at the end so the same index
can be reused later for test inference without holding two corpora on disk.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import h64  # noqa: E402
from pipeline import (  # noqa: E402
    S1_BATCH, WORK, build_corpus_and_index, drop_store, open_corpus, process_batch,
    save_chunk, store_path,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(ROOT, "dataset")


def load_gt_subset(path: str, s1_ids) -> dict[str, list[str]]:
    want = set(s1_ids)
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            if s1 in want:
                out[s1] = [x for x in rest.split(",") if x]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=200_000)
    ap.add_argument("--topk", type=int, default=60)
    ap.add_argument("--batch", type=int, default=S1_BATCH)
    ap.add_argument("--tag", default="_full")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    index, tokdf, s1_path, n_corpus = build_corpus_and_index(
        "train", args.tag, workers=args.workers, keep_corpus=True)

    chunks_dir = os.path.join(WORK, "cand", "train_chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    from norm_store import Store
    s1 = Store(s1_path)
    corpus_path, corpus = open_corpus("train", args.tag)
    n = min(args.n_s1, s1.n)
    print(f"collecting training pairs for {n:,} S1 entities (topk={args.topk})",
          flush=True)

    gt_cache: dict[str, list[str]] = {}
    n_pairs = 0
    n_pos = 0
    fb = None
    for lo in range(0, n, args.batch):
        hi = min(lo + args.batch, n)
        batch_ids = s1.ids[lo:hi]
        need = [x for x in batch_ids if x not in gt_cache]
        gt_cache.update(load_gt_subset(os.path.join(DATA, "train",
                                                    "train_ground_truth.tsv"), need))
        ps, pd, sc, X, fb = process_batch(index, tokdf, corpus, s1, lo, hi,
                                          top_k=args.topk, workers=args.workers,
                                          fb=fb)
        # labels: a pair is positive when the corpus doc's id is in the S1's truth list
        y = np.zeros(ps.size, dtype=np.int8)
        order = np.argsort(ps, kind="stable")
        ps_s, pd_s = ps[order], pd[order]
        bounds = np.flatnonzero(np.r_[True, ps_s[1:] != ps_s[:-1]])
        starts = np.r_[bounds, ps_s.size]
        for bi, b0 in enumerate(bounds):
            b1 = starts[bi + 1]
            truth = gt_cache.get(batch_ids[ps_s[b0] - lo], [])
            if not truth:
                continue
            ths = np.fromiter((h64("id", x) for x in truth), dtype=np.uint64,
                              count=len(truth))
            y[b0:b1] = np.isin(corpus.id_hash[pd_s[b0:b1]], ths)
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        y = y[inv]
        save_chunk((ps, pd, sc, X.astype(np.float16), y),
                   os.path.join(chunks_dir, f"chunk_{lo:08d}.npz"))
        n_pairs += ps.size
        n_pos += int(y.sum())
        print(f"  [{hi:,}/{n:,}] pairs={n_pairs:,} pos={n_pos:,} "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\nDONE pairs={n_pairs:,} positives={n_pos:,} "
          f"({n_pos/max(n_pairs,1)*100:.2f}%) in {time.time()-t0:.0f}s")
    # remove the corpus store; the index + tokdf stay for reuse
    drop_store(corpus_path)


if __name__ == "__main__":
    main()
