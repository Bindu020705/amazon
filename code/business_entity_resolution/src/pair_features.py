"""Feature engineering for (S1, candidate) pairs.

Two complementary families:

* **token features** - overlap of the normalized token sets (name-only, address-only,
  rarity buckets from the corpus document frequencies). Computed with a numba merge over
  the CSR token arrays, which keeps it fast for tens of millions of pairs.
* **string features** - rapidfuzz character-level similarities on the normalized name and
  address (ratio / partial / token-sort / token-set), plus the space-free compact forms
  that catch ``steelfarmer.com`` <-> ``Steel Farmer`` style records.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from rapidfuzz import fuzz, process
except ImportError:                                    # local vendored copy
    _vendor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor")
    sys.path.insert(0, os.path.abspath(_vendor))
    from rapidfuzz import fuzz, process

import numba  # noqa: E402
from numba import njit  # noqa: E402

FEATURE_NAMES = [
    "tok_shared", "tok_jaccard", "tok_shared_name", "tok_shared_addr",
    "tok_shared_r0", "tok_shared_r1", "tok_shared_r2",
    "tok_sizesum", "tok_sizeratio", "tok_rare_jaccard",
    "tok_name_jaccard", "tok_addr_jaccard",
    "state_match", "country_match", "block_score", "block_score_log",
    "name_ratio", "name_partial", "name_toksort", "name_tokset",
    "addr_ratio", "addr_partial", "addr_toksort", "addr_tokset",
    "cname_ratio", "cname_tokset", "caddr_ratio",
    "name_len_ratio", "addr_len_ratio", "cname_len_ratio",
]
N_FEATURES = len(FEATURE_NAMES)


@njit(cache=True, nogil=True, parallel=True)
def _intersect(flat_a, off_a, flag_a, rank_a, flat_b, off_b, flag_b, rank_b,
               pair_a, pair_b, out):
    """Per-pair overlap counts between two CSR token arrays."""
    for k in numba.prange(pair_a.shape[0]):
        ia = pair_a[k]
        ib = pair_b[k]
        a0, a1 = off_a[ia], off_a[ia + 1]
        b0, b1 = off_b[ib], off_b[ib + 1]
        shared = 0
        sh_name = 0
        sh_addr = 0
        r0 = r1 = r2 = 0
        for i in range(a0, a1):
            ha = flat_a[i]
            for j in range(b0, b1):
                if ha == flat_b[j]:
                    shared += 1
                    if flag_a[i] == 1 and flag_b[j] == 1:
                        sh_name += 1
                    if flag_a[i] == 2 and flag_b[j] == 2:
                        sh_addr += 1
                    r = rank_a[i] if rank_a[i] > rank_b[j] else rank_b[j]
                    if r == 0:
                        r0 += 1
                    elif r == 1:
                        r1 += 1
                    else:
                        r2 += 1
                    break
        out[k, 0] = shared
        out[k, 1] = sh_name
        out[k, 2] = sh_addr
        out[k, 3] = r0
        out[k, 4] = r1
        out[k, 5] = r2
        out[k, 6] = a1 - a0
        out[k, 7] = b1 - b0


def _decode(buf, off, i: int) -> str:
    return bytes(buf[off[i]:off[i + 1]]).decode("utf-8")


class FeatureBuilder:
    """Computes the feature matrix for (S1, corpus) candidate pairs."""

    def __init__(self, corpus, s1, tokdf, workers: int = 8, batch: int = 1_500_000):
        self.corpus = corpus
        self.s1 = s1
        self.tokdf = tokdf
        self.workers = workers
        self.batch = batch
        self._ranks = {}
        self._stats = {}

    # ---- rarity ranks per token atom (0 rare, 1 medium, 2 common) ----
    def ranks(self, store) -> np.ndarray:
        key = store.path
        if key in self._ranks:
            return self._ranks[key]
        flat = np.asarray(store.atoms_flat)
        uniq, cnt = self.tokdf.uniq, self.tokdf.cnt
        if uniq.size:
            p = np.clip(np.searchsorted(uniq, flat), 0, uniq.size - 1)
            df = np.where(uniq[p] == flat, cnt[p], 0).astype(np.int32)
        else:
            df = np.zeros(flat.size, dtype=np.int32)
        rank = np.zeros(flat.size, dtype=np.uint8)
        rank[df > self.tokdf.rare_df] = 1
        rank[df > self.tokdf.theta] = 2
        self._ranks[key] = rank
        return rank

    def _record_stats(self, store, rank):
        key = store.path
        if key in self._stats:
            return self._stats[key]
        off = np.asarray(store.atoms_o)
        n = off.size - 1
        counts = np.diff(off)
        rare = np.add.reduceat((rank == 0).astype(np.int32), off[:-1])
        rare[counts == 0] = 0
        flag = np.asarray(store.atoms_flag)
        name_cnt = np.add.reduceat((flag == 1).astype(np.int32), off[:-1])
        name_cnt[counts == 0] = 0
        addr_cnt = np.add.reduceat((flag == 2).astype(np.int32), off[:-1])
        addr_cnt[counts == 0] = 0
        out = counts.astype(np.int32), rare.astype(np.int32), \
            name_cnt.astype(np.int32), addr_cnt.astype(np.int32)
        self._stats[key] = out
        return out

    def features(self, pair_s1: np.ndarray, pair_doc: np.ndarray,
                 block_score: np.ndarray | None = None) -> np.ndarray:
        n = len(pair_s1)
        out = np.zeros((n, N_FEATURES), dtype=np.float32)
        if n == 0:
            return out
        corpus, s1 = self.corpus, self.s1
        rank_c = self.ranks(corpus)
        rank_s = self.ranks(s1)
        flat_c, off_c = np.asarray(corpus.atoms_flat), np.asarray(corpus.atoms_o)
        flag_c = np.asarray(corpus.atoms_flag)
        flat_s, off_s = np.asarray(s1.atoms_flat), np.asarray(s1.atoms_o)
        flag_s = np.asarray(s1.atoms_flag)

        for lo in range(0, n, self.batch):
            hi = min(lo + self.batch, n)
            pa = np.ascontiguousarray(pair_s1[lo:hi], dtype=np.int64)
            pb = np.ascontiguousarray(pair_doc[lo:hi], dtype=np.int64)
            m = hi - lo
            stats = np.zeros((m, 8), dtype=np.int32)
            _intersect(flat_s, off_s, flag_s, rank_s, flat_c, off_c, flag_c, rank_c,
                       pa, pb, stats)
            out[lo:hi] = self._assemble(pa, pb, stats, block_score[lo:hi]
                                        if block_score is not None else None)
        return out

    # ------------------------------------------------------------------
    def _assemble(self, pa, pb, stats, score) -> np.ndarray:
        corpus, s1 = self.corpus, self.s1
        cnt_s, rare_s, name_s, addr_s = self._record_stats(s1, self.ranks(s1))
        cnt_c, rare_c, name_c, addr_c = self._record_stats(corpus, self.ranks(corpus))
        state_a = np.asarray(s1.state)
        state_b = np.asarray(corpus.state)
        country_a = np.asarray(s1.country)
        country_b = np.asarray(corpus.country)

        shared = stats[:, 0].astype(np.float32)
        shared_name = stats[:, 1].astype(np.float32)
        shared_addr = stats[:, 2].astype(np.float32)
        r0, r1, r2 = stats[:, 3].astype(np.float32), stats[:, 4].astype(np.float32), \
            stats[:, 5].astype(np.float32)
        na = stats[:, 6].astype(np.float32)
        nb = stats[:, 7].astype(np.float32)

        union = na + nb - shared
        jac = np.divide(shared, np.maximum(union, 1.0))
        ra, rb = rare_s[pa].astype(np.float32), rare_c[pb].astype(np.float32)
        rare_jac = np.divide(r0, np.maximum(ra + rb - r0, 1.0))
        nam = name_s[pa].astype(np.float32)
        nac = name_c[pb].astype(np.float32)
        ada = addr_s[pa].astype(np.float32)
        adc = addr_c[pb].astype(np.float32)
        name_jac = np.divide(shared_name, np.maximum(nam + nac - shared_name, 1.0))
        addr_jac = np.divide(shared_addr, np.maximum(ada + adc - shared_addr, 1.0))

        # ---- strings ----
        u_s, inv_s = np.unique(pa, return_inverse=True)
        u_c, inv_c = np.unique(pb, return_inverse=True)
        a_name = [_decode(s1.name_b, s1.name_o, int(i)) for i in u_s]
        a_addr = [_decode(s1.addr_b, s1.addr_o, int(i)) for i in u_s]
        b_name = [_decode(corpus.name_b, corpus.name_o, int(i)) for i in u_c]
        b_addr = [_decode(corpus.addr_b, corpus.addr_o, int(i)) for i in u_c]
        la_n = [a_name[i] for i in inv_s]
        la_a = [a_addr[i] for i in inv_s]
        lb_n = [b_name[i] for i in inv_c]
        lb_a = [b_addr[i] for i in inv_c]
        la_cn = [x.replace(" ", "") for x in la_n]
        lb_cn = [x.replace(" ", "") for x in lb_n]
        la_ca = [x.replace(" ", "") for x in la_a]
        lb_ca = [x.replace(" ", "") for x in lb_a]

        w = self.workers
        cd = process.cpdist
        nm_ratio = cd(la_n, lb_n, scorer=fuzz.ratio, workers=w)
        nm_part = cd(la_n, lb_n, scorer=fuzz.partial_ratio, workers=w)
        nm_sort = cd(la_n, lb_n, scorer=fuzz.token_sort_ratio, workers=w)
        nm_set = cd(la_n, lb_n, scorer=fuzz.token_set_ratio, workers=w)
        ad_ratio = cd(la_a, lb_a, scorer=fuzz.ratio, workers=w)
        ad_part = cd(la_a, lb_a, scorer=fuzz.partial_ratio, workers=w)
        ad_sort = cd(la_a, lb_a, scorer=fuzz.token_sort_ratio, workers=w)
        ad_set = cd(la_a, lb_a, scorer=fuzz.token_set_ratio, workers=w)
        cn_ratio = cd(la_cn, lb_cn, scorer=fuzz.ratio, workers=w)
        cn_set = cd(la_cn, lb_cn, scorer=fuzz.token_set_ratio, workers=w)
        ca_ratio = cd(la_ca, lb_ca, scorer=fuzz.ratio, workers=w)

        state_match = (state_a[pa] == state_b[pb]) & (state_a[pa] != 0)
        country_match = country_a[pa] == country_b[pb]
        if score is None:
            score = np.zeros(len(pa), dtype=np.float32)

        a_len = np.array([len(x) for x in la_n], dtype=np.float32)
        b_len = np.array([len(x) for x in lb_n], dtype=np.float32)
        a_alen = np.array([len(x) for x in la_a], dtype=np.float32)
        b_alen = np.array([len(x) for x in lb_a], dtype=np.float32)
        a_clen = np.array([len(x) for x in la_cn], dtype=np.float32)
        b_clen = np.array([len(x) for x in lb_cn], dtype=np.float32)

        out = np.empty((len(pa), N_FEATURES), dtype=np.float32)
        cols = [
            shared, jac, shared_name, shared_addr, r0, r1, r2,
            na + nb, np.divide(np.minimum(na, nb), np.maximum(na, nb)), rare_jac,
            name_jac, addr_jac,
            state_match.astype(np.float32), country_match.astype(np.float32),
            score, np.log1p(np.maximum(score, 0)),
            nm_ratio, nm_part, nm_sort, nm_set,
            ad_ratio, ad_part, ad_sort, ad_set,
            cn_ratio, cn_set, ca_ratio,
            np.divide(np.minimum(a_len, b_len), np.maximum(a_len, b_len) + 1e-6),
            np.divide(np.minimum(a_alen, b_alen), np.maximum(a_alen, b_alen) + 1e-6),
            np.divide(np.minimum(a_clen, b_clen), np.maximum(a_clen, b_clen) + 1e-6),
        ]
        for j, c in enumerate(cols):
            out[:, j] = c
        return out
