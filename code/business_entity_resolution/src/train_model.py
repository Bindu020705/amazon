"""Train the pair classifier on blocked train candidates and tune the F_0.5 threshold.

Training pairs come from the *same* blocking configuration that runs on the test set,
so the negative distribution the model sees at inference time is the one it was trained
on. The decision threshold is tuned for the competition metric: macro F_0.5 over
Source-1 entities, singletons predicted empty.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import h64  # noqa: E402
from norm_store import Store  # noqa: E402
from pair_features import FEATURE_NAMES  # noqa: E402
from pipeline import WORK, store_path  # noqa: E402

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


def macro_f05(pred_keep: np.ndarray, is_pos: np.ndarray, ent: np.ndarray,
              n_true: np.ndarray, n_entities: int) -> float:
    tp = np.bincount(ent[pred_keep & is_pos], minlength=n_entities)
    n_pred = np.bincount(ent[pred_keep], minlength=n_entities)
    prec = np.divide(tp, np.maximum(n_pred, 1))
    rec = np.divide(tp, np.maximum(n_true, 1))
    f = np.divide(1.25 * prec * rec, 0.25 * prec + rec + 1e-12)
    singleton = n_true == 0
    f[singleton] = np.where(n_pred[singleton] == 0, 1.0, 0.0)
    return float(f.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=os.path.join(WORK, "cand", "train_chunks"))
    ap.add_argument("--holdout-frac", type=float, default=0.15,
                    help="fraction of S1 entities held out for threshold tuning")
    ap.add_argument("--max-neg-ratio", type=float, default=8.0)
    ap.add_argument("--rounds", type=int, default=800)
    ap.add_argument("--model-out", default=os.path.join(WORK, "model.txt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    files = sorted(glob.glob(os.path.join(args.chunks, "chunk_*.npz")))
    if not files:
        raise SystemExit("no chunks found - run collect_train.py first")
    print(f"{len(files)} chunks")

    # entity id hash -> local entity index (assigned as we stream)
    ent_index: dict[int, int] = {}
    X_list, y_list, ent_list, doc_list, sc_list = [], [], [], [], []
    n_pairs = n_pos = 0
    for fp in files:
        z = np.load(fp)
        ps, pd, sc, X, y = z["arr_0"], z["arr_1"], z["arr_2"], z["arr_3"], z["arr_4"]
        ps = ps.astype(np.int64)
        # entity keys are S1 store row indices; map them to compact ids
        eh = np.fromiter((h64("row", str(r)) for r in np.unique(ps)), dtype=np.uint64)
        uniq_rows = np.unique(ps)
        row_map = {int(r): i + len(ent_index) for i, r in enumerate(uniq_rows)}
        e = np.fromiter((row_map[int(r)] for r in ps), dtype=np.int64, count=ps.size)
        for r in uniq_rows:
            ent_index.setdefault(row_map[int(r)], len(ent_index))
        X_list.append(X.astype(np.float32))
        y_list.append(y.astype(np.int8))
        ent_list.append(e)
        doc_list.append(pd.astype(np.int64))
        sc_list.append(sc.astype(np.float32))
        n_pairs += ps.size
        n_pos += int(y.sum())
        del z, ps, pd, sc, X, y
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    ent = np.concatenate(ent_list)
    doc = np.concatenate(doc_list)
    sc = np.concatenate(sc_list)
    del X_list, y_list, ent_list, doc_list, sc_list
    n_ent = len(ent_index)
    print(f"pairs={n_pairs:,} positives={n_pos:,} ({n_pos/max(n_pairs,1)*100:.2f}%) "
          f"entities={n_ent:,}")

    # ---- ground truth n_true per entity (for the metric) ----
    s1_store = Store(store_path("train", "1", "_full"))
    n_true = np.zeros(n_ent, dtype=np.int64)
    ent_rows = np.zeros(n_ent, dtype=np.int64)
    for r, e in row_map.items():
        ent_rows[e] = r
    ids = [s1_store.ids[int(r)] for r in ent_rows]
    gt = load_gt_subset(os.path.join(DATA, "train", "train_ground_truth.tsv"), ids)
    for e, s1id in enumerate(ids):
        n_true[e] = len(gt.get(s1id, []))

    # ---- entity-level split ----
    uent = np.arange(n_ent)
    rng.shuffle(uent)
    n_hold = int(n_ent * args.holdout_frac)
    hold = np.zeros(n_ent, dtype=bool)
    hold[uent[:n_hold]] = True
    ent_hold = hold[ent]
    Xtr, ytr, etr = X[~ent_hold], y[~ent_hold], ent[~ent_hold]
    Xva, yva, eva = X[ent_hold], y[ent_hold], ent[ent_hold]
    # re-index validation entities compactly for the metric
    ue = np.unique(eva)
    remap = np.zeros(n_ent, dtype=np.int64)
    remap[ue] = np.arange(ue.size)
    eva_r = remap[eva]
    print(f"train: {Xtr.shape[0]:,} pairs / {int((~hold).sum()):,} entities; "
          f"valid: {Xva.shape[0]:,} pairs / {ue.size:,} entities")

    # ---- negative subsampling (keep all positives) ----
    pos_mask = ytr == 1
    max_neg = int(pos_mask.sum() * args.max_neg_ratio)
    neg_idx = np.flatnonzero(~pos_mask)
    if neg_idx.size > max_neg:
        neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)
    keep = np.concatenate([np.flatnonzero(pos_mask), neg_idx])
    keep.sort()
    Xtr, ytr, etr = Xtr[keep], ytr[keep], etr[keep]
    print(f"train rows after subsample: {Xtr.shape[0]:,}")

    import lightgbm as lgb
    params = {
        "objective": "binary",
        "learning_rate": 0.06,
        "num_leaves": 96,
        "min_data_in_leaf": 60,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "num_threads": os.cpu_count() or 8,
        "metric": "average_precision",
    }
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_NAMES)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr, feature_name=FEATURE_NAMES)
    model = lgb.train(params, dtr, num_boost_round=args.rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    imp = sorted(zip(FEATURE_NAMES, model.feature_importance("gain")), key=lambda x: -x[1])
    print("\ntop features:")
    for n_, g in imp[:20]:
        print(f"   {n_:<18} {g:,.0f}")

    # ---- threshold tuning ----
    pv = model.predict(Xva, num_iteration=model.best_iteration)
    n_true_v = n_true[ue]
    is_pos = yva.astype(bool)
    print("\nthreshold sweep (macro F0.5 on holdout entities):")
    best = (0.0, 0.5)
    for t in np.arange(0.05, 0.96, 0.05):
        f = macro_f05(pv >= t, is_pos, eva_r, n_true_v, ue.size)
        if f > best[0]:
            best = (f, float(t))
        print(f"   t={t:.2f}  F0.5={f:.4f}")
    for t in np.arange(max(0.01, best[1] - 0.05), best[1] + 0.05, 0.01):
        f = macro_f05(pv >= t, is_pos, eva_r, n_true_v, ue.size)
        if f > best[0]:
            best = (f, float(t))
    print(f"\nBEST threshold={best[1]:.3f}  holdout macro F0.5={best[0]:.4f}")

    model.save_model(args.model_out)
    np.save(os.path.join(WORK, "threshold.npy"), np.array([best[1]]))
    print(f"saved model -> {args.model_out}")


if __name__ == "__main__":
    main()
