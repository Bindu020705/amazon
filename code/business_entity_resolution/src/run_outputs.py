"""Streaming writer for the two submission TSVs.

Called once per Source-1 batch: writes one row per S1 entity in file order with its
matches (model probability >= threshold) and its full candidate list. IDs are resolved
through the corpus store's id list, which is guaranteed to contain only S2-/S3- ids of
the given split, and duplicates inside a list are impossible because candidates are
deduplicated per entity by construction.
"""
from __future__ import annotations

import numpy as np


def _ids_by_row(corpus_ids, rows: np.ndarray) -> list[str]:
    return [corpus_ids[int(d)] for d in rows]


def write_submission(match_fh, cand_fh, s1_ids, ps, pd, pv, thr,
                     corpus_ids, lo: int = 0) -> None:
    """Append the rows for S1 entities ``s1_ids`` (batch covers rows lo..lo+n)."""
    n = len(s1_ids)
    # group candidate pairs by S1 row
    order = np.argsort(ps, kind="stable")
    ps_s = ps[order]
    bounds = np.flatnonzero(np.r_[True, ps_s[1:] != ps_s[:-1]])
    starts = np.r_[bounds, ps_s.size]
    groups: dict[int, tuple] = {}
    for bi, b0 in enumerate(bounds):
        b1 = starts[bi + 1]
        row = int(ps_s[b0])
        idx = order[b0:b1]
        groups[row] = (pd[idx], pv[idx])

    for r in range(n):
        gid = lo + r
        pairs = groups.get(gid)
        if pairs is None:
            match_fh.write(s1_ids[r] + "\t\n")
            cand_fh.write(s1_ids[r] + "\t\n")
            continue
        docs, probs = pairs
        # sort by probability descending for readability
        od = np.argsort(-probs)
        docs_sorted = docs[od]
        probs_sorted = probs[od]
        cand_ids = _ids_by_row(corpus_ids, docs_sorted)
        cand_line = ",".join(cand_ids)
        keep_mask = probs_sorted >= thr
        match_line = ",".join([c for c, k in zip(cand_ids, keep_mask) if k])
        match_fh.write(f"{s1_ids[r]}\t{match_line}\n")
        cand_fh.write(f"{s1_ids[r]}\t{cand_line}\n")
