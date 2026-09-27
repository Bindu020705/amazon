"""Shared streaming pipeline: blocked retrieval + feature computation, batch by batch.

Both the training-data collector and the test predictor stream over Source-1 records in
file order, retrieving the top candidates for each batch and computing the pair features
on the fly, so peak memory stays around one batch regardless of dataset size.

Disk layout per split (``work/``):

* ``store/<split>_index.part*.npy``     - the corpus inverted index (kept, ~1.4 GB)
* ``store/<split>_tokdf.{hash,df}.npy`` - token document frequencies (kept, ~25 MB)
* ``cand/<split>_chunks/chunk_*.npz``   - per-batch candidates + features (kept)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import (  # noqa: E402
    Index, TokenDF, build_index_parts, build_token_df, retrieve,
)
from norm_store import (  # noqa: E402
    Store, build_store, read_source_iter,
)
from pair_features import FeatureBuilder  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(ROOT, "dataset")
WORK = os.path.join(ROOT, "work")

S1_BATCH = 30_000
CODE2COUNTRY = {1: "US", 2: "India", 3: "France"}


def store_path(split: str, src: str, tag: str = "") -> str:
    return os.path.join(WORK, "store", f"{split}_{src}{tag}")


def ensure_store(split: str, src: str, tag: str = "", limit: int | None = None) -> str:
    path = store_path(split, src, tag)
    if os.path.isfile(os.path.join(path, "ids.txt")):
        return path
    print(f"  building store {split} source{src}", flush=True)
    build_store(read_source_iter(os.path.join(DATA, split, f"{split}_source{src}.tsv"),
                                 limit), path)
    return path


def drop_store(path: str) -> None:
    import shutil
    shutil.rmtree(path, ignore_errors=True)


def _merge_concat_arrays(paths: list[str], out_path: str) -> int:
    """Merge several stores into one by concatenating their arrays one at a time."""
    import shutil
    if os.path.isfile(os.path.join(out_path, "ids.txt")):
        return Store(out_path, with_ids=False).n
    os.makedirs(out_path, exist_ok=True)
    stores = [Store(p, with_ids=False) for p in paths]
    arrays = ("name_b", "addr_b", "atoms_flat", "atoms_flag", "state", "country",
              "id_hash")
    for arr in arrays:
        np.save(os.path.join(out_path, arr + ".npy"),
                np.concatenate([np.asarray(getattr(s, arr)) for s in stores]))
        print(f"    merged {arr}", flush=True)
    for off in ("name_o", "addr_o", "atoms_o"):
        arrs = [np.asarray(getattr(s, off)) for s in stores]
        base, parts = 0, []
        for j, a in enumerate(arrs):
            parts.append(a.copy() if j == 0 else a[1:] + base)
            base += int(a[-1])
        np.save(os.path.join(out_path, off + ".npy"), np.concatenate(parts))
        del arrs, parts
        print(f"    merged {off}", flush=True)
    n_ids = 0
    with open(os.path.join(out_path, "ids.txt"), "w", encoding="utf-8") as out:
        for p in paths:
            with open(os.path.join(p, "ids.txt"), encoding="utf-8") as f:
                for ln in f:
                    s = ln.strip()
                    if s:
                        out.write(s + "\n")
                        n_ids += 1
    del stores
    return n_ids


def _iter_source_tsv(path: str, chunk: int = 25_000, limit: int | None = None):
    from norm_store import read_source_iter
    yield from read_source_iter(path, limit)


def build_corpus_direct(split: str, tag: str = "", limit: int | None = None) -> int:
    """Build the merged S2+S3 corpus store straight from the two TSVs.

    Chaining the two source iterators means no per-source intermediate store is ever
    written, which halves the peak disk footprint (the previous double-store merge
    needed ~5 GB; this needs only the final ~3.2 GB).
    """
    import itertools
    from norm_store import build_store

    corpus_path = store_path(split, "corpus", tag)
    if os.path.isfile(os.path.join(corpus_path, "ids.txt")):
        return Store(corpus_path, with_ids=False).n
    it = itertools.chain(
        _iter_source_tsv(os.path.join(DATA, split, f"{split}_source2.tsv"), limit=limit),
        _iter_source_tsv(os.path.join(DATA, split, f"{split}_source3.tsv"), limit=limit),
    )
    print("  building corpus store (S2+S3, streaming)", flush=True)
    n = build_store(it, corpus_path)
    print(f"  corpus: {n:,} records", flush=True)
    return n


def build_corpus_and_index(split: str, tag: str = "", workers: int = 8,
                           part_records: int = 1_500_000, keep_corpus: bool = False,
                           limit: int | None = None):
    """Build the merged corpus store + token DF + inverted index.

    The merged corpus store is the only large on-disk artifact (~3.2 GB for the 10.3M
    train corpus); the token-DF table and the inverted index are kept in memory and
    rebuilt on demand (they are deterministic, so rebuilding is safe).
    """
    df_path = os.path.join(WORK, "store", f"{split}{tag}_tokdf")
    corpus_path = store_path(split, "corpus", tag)

    n_corpus = build_corpus_direct(split, tag, limit=limit)

    if os.path.isfile(df_path + ".hash.npy"):
        tokdf = TokenDF.load(df_path)
    else:
        st = Store(corpus_path, with_ids=False)
        uniq, cnt = build_token_df([st], df_path)
        del st
        tokdf = TokenDF(uniq, cnt)
    print(f"  token df: {tokdf.uniq.size:,} distinct tokens", flush=True)

    st = Store(corpus_path, with_ids=False)
    index = build_index_parts(st, tokdf, workers=workers, part_records=part_records)
    print(f"  index: {len(index.parts)} parts, {index.total_postings():,} postings",
          flush=True)

    s1_path = ensure_store(split, "1", tag)
    if not keep_corpus:
        drop_store(corpus_path)          # frees ~3.2 GB; rebuilt on demand
    return index, tokdf, s1_path, n_corpus


def open_corpus(split: str, tag: str = ""):
    """Open (rebuilding if needed) the merged corpus store for a split, with ids."""
    corpus_path = store_path(split, "corpus", tag)
    if not os.path.isfile(os.path.join(corpus_path, "ids.txt")):
        _merge_concat_arrays([ensure_store(split, "2", tag),
                              ensure_store(split, "3", tag)], corpus_path)
    return corpus_path, Store(corpus_path, with_ids=True)


def process_batch(index, tokdf, corpus, s1, lo: int, hi: int, top_k: int = 150,
                  key_budget: int = 14, per_key_cap: int = 80, workers: int = 8,
                  fb=None):
    """Retrieve candidates for S1 rows [lo, hi) and compute their pair features.

    ``corpus`` / ``s1`` are open Store objects; only the FeatureBuilder is created here
    (its per-store caches make repeated batches cheap).
    """
    s1_idx = np.arange(lo, hi, dtype=np.int64)
    ps, pd, sc = retrieve(index, s1, s1_idx, tokdf, key_budget=key_budget,
                          per_key_cap=per_key_cap, top_k=top_k)
    if fb is None:
        fb = FeatureBuilder(corpus, s1, tokdf, workers=workers)
    ps_full = (ps.astype(np.int64) + lo).astype(np.int64)
    X = fb.features(ps_full, pd.astype(np.int64), sc)
    return (ps_full.astype(np.int32), pd.astype(np.int32), sc.astype(np.float32), X, fb)


def save_chunk(arrays, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, *arrays)
