"""Driver: build stores for a split, build the inverted index, retrieve candidates.

Usage (from student_resource/)::

    python code/business_entity_resolution/src/run_blocking.py --split test
    python code/business_entity_resolution/src/run_blocking.py --split train --limit-s1 20000

Outputs ``work/cand/<split>_candidates.npz`` with ``s1`` (index into the S1 store) and
``doc`` (index into the materialised S2+S3 corpus store).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import (  # noqa: E402
    DF_THETA, Index, MAX_KEYS, TokenDF, build_index_parts, build_token_df, dedupe_pairs,
    retrieve,
)
from norm_store import Store, build_store, read_source_iter  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(ROOT, "dataset")
WORK = os.path.join(ROOT, "work")


def store_path(split: str, src: str, tag: str = "") -> str:
    return os.path.join(WORK, "store", f"{split}_{src}{tag}")


def ensure_store(split: str, src: str, limit: int | None, tag: str = "") -> Store:
    path = store_path(split, src, tag)
    if os.path.isfile(os.path.join(path, "ids.txt")):
        return Store(path)
    src_path = os.path.join(DATA, split, f"{split}_source{src}.tsv")
    print(f"  building store {split} source{src}", flush=True)
    n = build_store(read_source_iter(src_path, limit), path)
    print(f"    {n:,} rows", flush=True)
    return Store(path)


def materialize_concat(paths: list[str], out_path: str) -> Store:
    """Concatenate several stores into one (arrays copied one at a time to save RAM)."""
    if os.path.isfile(os.path.join(out_path, "ids.txt")):
        return Store(out_path)
    os.makedirs(out_path, exist_ok=True)
    print(f"  materialising corpus store {out_path}", flush=True)
    stores = [Store(p) for p in paths]
    for arr in ("name_b", "addr_b", "atoms_flat", "atoms_flag", "state", "country",
                "id_hash"):
        np.save(os.path.join(out_path, arr + ".npy"),
                np.concatenate([np.asarray(getattr(s, arr)) for s in stores]))
        print(f"    {arr}: {os.path.getsize(os.path.join(out_path, arr + '.npy'))/1e6:.0f} MB",
              flush=True)
    for off in ("name_o", "addr_o", "atoms_o"):
        arrs = [np.asarray(getattr(s, off)) for s in stores]
        base, parts = 0, []
        for j, a in enumerate(arrs):
            parts.append(a.copy() if j == 0 else a[1:] + base)
            base += int(a[-1])
        merged = np.concatenate(parts)
        np.save(os.path.join(out_path, off + ".npy"), merged)
        del arrs, merged, parts
    ids = []
    for s in stores:
        with open(os.path.join(s.path, "ids.txt"), encoding="utf-8") as f:
            ids.extend(ln.rstrip("\n") for ln in f)
    with open(os.path.join(out_path, "ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")
    del stores, ids
    return Store(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--limit-s1", type=int, default=None)
    ap.add_argument("--limit-corpus", type=int, default=None)
    ap.add_argument("--key-budget", type=int, default=12)
    ap.add_argument("--per-key-cap", type=int, default=60)
    ap.add_argument("--top-k", type=int, default=150)
    ap.add_argument("--keep-per-s1", type=int, default=0, help="0 = keep the whole pool")
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--part-records", type=int, default=1_500_000)
    ap.add_argument("--keep-parts", action="store_true",
                    help="keep the per-source stores instead of deleting them")
    ap.add_argument("--s1-start", type=int, default=0)
    ap.add_argument("--s1-end", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(os.path.join(WORK, "store"), exist_ok=True)
    os.makedirs(os.path.join(WORK, "cand"), exist_ok=True)
    t0 = time.time()
    split = args.split
    tag = args.tag or (f"_lim{args.limit_corpus}" if args.limit_corpus else "")

    a = ensure_store(split, "2", args.limit_corpus, tag)
    b = ensure_store(split, "3", args.limit_corpus, tag)
    corpus = materialize_concat([a.path, b.path], store_path(split, "corpus", tag))
    if not args.keep_parts:                     # free ~2 GB once merged
        for p in (a.path, b.path):
            if os.path.abspath(p) != os.path.abspath(corpus.path):
                shutil.rmtree(p, ignore_errors=True)
        del a, b
    print(f"corpus: {corpus.n:,} records ({time.time()-t0:.0f}s)", flush=True)

    df_path = os.path.join(WORK, "store", f"{split}{tag}_tokdf")
    if os.path.isfile(df_path + ".hash.npy"):
        tokdf = TokenDF.load(df_path)
    else:
        uniq, cnt = build_token_df([corpus], df_path)
        tokdf = TokenDF(uniq, cnt)
    print(f"token df table: {tokdf.uniq.size:,} distinct "
          f"(common={len(tokdf.common):,} too_common={len(tokdf.too_common):,}) "
          f"({time.time()-t0:.0f}s)", flush=True)

    idx_path = os.path.join(WORK, "store", f"{split}{tag}_index")
    n_parts = (corpus.n + args.part_records - 1) // args.part_records
    if os.path.isfile(idx_path + ".part0.npy"):
        index = Index.load(idx_path, n_parts)
    else:
        index = build_index_parts(corpus, tokdf, workers=args.workers,
                                  part_records=args.part_records)
        index.save(idx_path)
    print(f"index parts={len(index.parts)} postings={index.total_postings():,} "
          f"({time.time()-t0:.0f}s)", flush=True)

    s1 = ensure_store(split, "1", args.limit_s1, tag)
    print(f"S1 store: {s1.n:,} records", flush=True)

    out_s1, out_doc, out_sc = [], [], []
    s1_end = min(args.s1_end or s1.n, s1.n)
    for lo in range(args.s1_start, s1_end, args.batch):
        hi = min(lo + args.batch, s1_end)
        s1_idx = np.arange(lo, hi, dtype=np.int64)
        ps, pd, sc = retrieve(index, s1, s1_idx, tokdf, key_budget=args.key_budget,
                              per_key_cap=args.per_key_cap, top_k=args.top_k)
        if ps.size:
            out_s1.append((ps.astype(np.int64) + lo).astype(np.int32))
            out_doc.append(pd.astype(np.int32))
            out_sc.append(sc)
        print(f"  blocked {hi:,}/{s1.n:,} S1  pairs={sum(x.size for x in out_doc):,}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    s1_arr = np.concatenate(out_s1) if out_s1 else np.zeros(0, dtype=np.int32)
    doc_arr = np.concatenate(out_doc) if out_doc else np.zeros(0, dtype=np.int32)
    sc_arr = np.concatenate(out_sc) if out_sc else np.zeros(0, dtype=np.float32)
    np.savez(os.path.join(WORK, "cand", f"{split}{tag}_candidates.npz"),
             s1=s1_arr, doc=doc_arr, score=sc_arr, s1_start=args.s1_start,
             n_s1=s1.n, n_corpus=corpus.n)
    sizes = np.bincount(s1_arr, minlength=s1.n)
    print(f"\nDONE pairs={s1_arr.size:,} mean_candidates={sizes.mean():.1f} "
          f"median={np.median(sizes):.0f} p90={np.percentile(sizes, 90):.0f} "
          f"empty={(sizes == 0).sum():,} ({(sizes == 0).mean()*100:.1f}%) "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
