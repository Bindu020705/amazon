"""Spot-check: print Source-1 records next to a sample of their blocked candidates."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from norm_store import Store  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WORK = os.path.join(ROOT, "work")


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    tag = sys.argv[2] if len(sys.argv) > 2 else "_lim300000"
    n_show = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    cand = np.load(os.path.join(WORK, "cand", f"{split}{tag}_candidates.npz"))
    s1 = Store(os.path.join(WORK, "store", f"{split}_1{tag}"))
    corpus = Store(os.path.join(WORK, "store", f"{split}_corpus{tag}"))
    s1i, doc = cand["s1"], cand["doc"]
    sizes = np.bincount(s1i, minlength=s1.n)
    order = np.argsort(-sizes)[:n_show]
    for i in order:
        print("=" * 110)
        print(f"S1[{i}] {s1.id(i)} | {s1.name_str(i)} | {s1.addr_str(i)} | cands={sizes[i]}")
        rows = doc[s1i == i][:8]
        for d in rows:
            print(f"    {corpus.id(d)} | {corpus.name_str(d)} | {corpus.addr_str(d)}")


if __name__ == "__main__":
    main()
