"""Evaluate blocking quality on the train split (recall ceiling, reduction ratio).

Ceiling = per-entity F_0.5 assuming a perfect classifier on the retrieved candidates:
an entity scores 1.0 when all of its true matches are in the candidate set, and
1.25r/(0.25+r) with r = found/true otherwise. Singletons score 1.0 (the model can always
predict an empty list, which is what a well-calibrated model does).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import h64  # noqa: E402
from norm_store import Store  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(ROOT, "dataset")
WORK = os.path.join(ROOT, "work")


def load_gt(path: str) -> dict[str, list[str]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            ids = [x for x in rest.split(",") if x]
            out[s1] = ids
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--tag", default="_full")
    ap.add_argument("--topk", type=int, nargs="*", default=[10, 20, 30, 60, 100, 200])
    args = ap.parse_args()

    tag = args.tag
    cand = np.load(os.path.join(WORK, "cand", f"{args.split}{tag}_candidates.npz"))
    pa, pb, sc = cand["s1"], cand["doc"], cand["score"]
    s1_store = Store(os.path.join(WORK, "store", f"{args.split}_1{tag}"))
    corpus = Store(os.path.join(WORK, "store", f"{args.split}_corpus{tag}"))
    print(f"pairs={pa.size:,}  s1_store={s1_store.n:,}  corpus={corpus.n:,}")

    # map ground-truth ids -> corpus doc index
    hs = np.asarray(corpus.id_hash)
    order = np.argsort(hs)
    hs_sorted = hs[order]

    gt = load_gt(os.path.join(DATA, "train", "train_ground_truth.tsv"))
    s1_range = np.arange(0, s1_store.n, dtype=np.int64)
    present = np.unique(pa)
    print(f"S1 with candidates: {present.size:,}")

    true_docs: dict[int, np.ndarray] = {}
    n_true_total = 0
    n_singleton = 0
    missing_truth = 0
    for i in present:
        ids = gt.get(s1_store.id(int(i)), [])
        if not ids:
            n_singleton += 1
            true_docs[int(i)] = np.zeros(0, dtype=np.int64)
            continue
        hh = np.fromiter((h64("id", x) for x in ids), dtype=np.uint64, count=len(ids))
        pos = np.searchsorted(hs_sorted, hh)
        pos = np.clip(pos, 0, max(len(hs_sorted) - 1, 0))
        ok = hs_sorted[pos] == hh
        docs = order[pos[ok]].astype(np.int64)
        missing_truth += int((~ok).sum())
        true_docs[int(i)] = docs
        n_true_total += docs.size
    print(f"true matches mapped: {n_true_total:,} (missing {missing_truth})  "
          f"singletons in sample: {n_singleton:,}")

    # order pairs by score descending within each S1 so we can truncate at any k
    ordr = np.lexsort((-sc, pa))
    pa, pb = pa[ordr], pb[ordr]
    sizes = np.bincount(pa, minlength=s1_store.n)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    print(f"mean candidates per S1 (full pool): {sizes[sizes > 0].mean():.1f}")

    for topk in args.topk:
        found = 0
        tot = 0
        ceiling = 0.0
        all_found = 0
        n_ent = 0
        cand_sum = 0
        for i in present:
            i = int(i)
            a, b = starts[i], starts[i + sizes[i]]
            take = min(topk, b - a)
            cand_docs = pb[a:a + take]
            tdocs = true_docs.get(i, np.zeros(0, dtype=np.int64))
            n_ent += 1
            cand_sum += take
            if tdocs.size == 0:
                ceiling += 1.0
                continue
            nf = int(np.isin(tdocs, cand_docs).sum())
            found += nf
            tot += tdocs.size
            r = nf / tdocs.size
            ceiling += 0.0 if r == 0 else (1.25 * r) / (0.25 + r)
            if nf == tdocs.size:
                all_found += 1
        macro = ceiling / max(n_ent, 1)
        print(f"\ntop_k={topk:>4}: pair_recall={found/max(tot,1):.4f} "
              f"entity_full_recall={all_found/max(n_ent-n_singleton,1):.4f} "
              f"mean_cands={cand_sum/max(n_ent,1):.1f} "
              f"reduction={1- (cand_sum/max(n_ent,1))/corpus.n:.6f}")
        print(f"            macro F0.5 ceiling = {macro:.4f}")


if __name__ == "__main__":
    main()
