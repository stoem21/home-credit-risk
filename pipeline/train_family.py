"""Train and cross-validate another model family on the same features, folds and rules as LightGBM
(several model families, same data and rules). Model code: homecredit/models/xgb.py (gradient-boosted trees from another library) and
homecredit/models/mlp.py (a plain neural network).

Same 5 seeds x 5 folds; same inner 10% hold-out for early stopping; settings fixed in advance (no tuning, like
every other model here). The final model for the test set is trained on all of train with the median number of
rounds / epochs chosen in CV x 1.1, as for LightGBM.

Usage:
    .venv/bin/python -m pipeline.train_family --model xgb|mlp --name NAME [--features v2 --add win,...]
                                              [--exclude stk_bb_] [--drop gender,family] [--seeds 0]
                                              [--jobs 5] [--threads 2] [--out-root DIR]
Outputs: processed/oof/<name>.parquet, processed/preds/<name>_test.parquet, analysis_output/<model>/<name>_folds.csv
"""

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from homecredit.data import N_FOLDS, N_SEEDS, drop_columns, exclude_prefixes, load_features, load_folds, scopes, select_groups
from homecredit.evaluate import auc, save_oof
from homecredit.models import mlp, xgb
from homecredit.paths import Paths, add_out_root

_cache = {}


def columns(P, features, add, drop, exclude=None):
    tr, te, cols = load_features(P, features, categories=True)
    cols = exclude_prefixes(drop_columns(select_groups(cols, add), drop), exclude)
    return tr, te, cols


def run_fold(model, out_root, features, add, drop, exclude, seed, k, threads):
    if "tr" not in _cache:
        P = Paths(out_root)
        tr, _, cols = columns(P, features, add, drop, exclude)
        _cache.update(tr=tr, cols=cols, fo=load_folds(P, tr.SK_ID_CURR))
    tr, cols, fo = _cache["tr"], _cache["cols"], _cache["fo"]
    f = fo[f"fold_s{seed}"].values
    trn, va = np.where(f != k)[0], np.where(f == k)[0]
    y = tr.TARGET.values.astype(int)
    t0 = time.time()
    with threadpool_limits(threads):
        if model == "xgb":
            bst, n = xgb.fit(tr[cols].iloc[trn], y[trn], seed, threads)
            p = xgb.predict(bst, tr[cols].iloc[va], n)
        else:
            prep, net, n = mlp.fit(tr[cols].iloc[trn], y[trn], seed)
            p = mlp.predict(prep, net, tr[cols].iloc[va])
    return seed, k, va, p, n, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["xgb", "mlp"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--features", default="v1")
    ap.add_argument("--add", default=None)
    ap.add_argument("--drop", default=None)
    ap.add_argument("--exclude", default=None, help="leave out columns starting with this prefix (e.g. stk_bb_)")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--folds-only", default="0,1,2,3,4", help="subset of folds, for timing runs (nothing written)")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    if P.oof(args.name).exists():
        ap.error(f"{P.oof(args.name)} exists: saved results are never overwritten; choose a new --name")
    seeds = [int(s) for s in args.seeds.split(",")]
    ks = [int(k) for k in args.folds_only.split(",")]
    t0 = time.time()

    tr, te, cols = columns(P, args.features, args.add, args.drop, args.exclude)
    y = tr.TARGET.values.astype(int)
    cash = scopes(tr.NAME_CONTRACT_TYPE)["cash"]
    print(f"{args.model}: {len(cols)} columns; {len(seeds) * len(ks)} fold fits", flush=True)
    res = Parallel(n_jobs=args.jobs)(
        delayed(run_fold)(args.model, args.out_root, args.features, args.add, args.drop, args.exclude, s, k,
                          args.threads)
        for s in seeds for k in ks)
    oof = pd.DataFrame({"SK_ID_CURR": tr.SK_ID_CURR})
    rows = []
    for s, k, va, p, n, secs in res:
        oof.loc[va, f"s{s}"] = p
        rows.append({"seed": s, "fold": k, "rounds_or_epochs": n, "seconds": round(secs, 1),
                     "auc_all": auc(y[va], p), "auc_cash": auc(y[va][cash[va]], p[cash[va]])})
    folds_df = pd.DataFrame(rows).sort_values(["seed", "fold"])
    print(folds_df.round(5).to_string(index=False), flush=True)
    print(f"mean AUC all {folds_df.auc_all.mean():.5f}, cash {folds_df.auc_cash.mean():.5f}; "
          f"elapsed {time.time() - t0:.0f} s", flush=True)
    if len(ks) != N_FOLDS:
        print("timing run only: no files written")
        return
    folds_df.to_csv(P.out(args.model) / f"{args.name}_folds.csv", index=False)
    save_oof(P, args.name, oof)
    if len(seeds) != N_SEEDS:
        print(f"exploratory run (seeds {seeds}): out-of-fold files written; no final model")
        return

    n_final = int(np.median(folds_df.rounds_or_epochs) * 1.1)
    threads = args.jobs * args.threads
    with threadpool_limits(threads):
        if args.model == "xgb":
            bst = xgb.fit_fixed(tr[cols], y, n_final, seed=0, threads=threads)
            p_te = xgb.predict(bst, te[cols], n_final)
        else:
            prep, net = mlp.fit_fixed(tr[cols], y, n_final, seed=0)
            p_te = mlp.predict(prep, net, te[cols])
    P.preds_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"SK_ID_CURR": te.SK_ID_CURR, "TARGET": p_te}).to_parquet(P.test_preds(args.name), index=False)
    print(f"final model: {n_final} rounds/epochs; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
