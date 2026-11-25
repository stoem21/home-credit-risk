"""Train and cross-validate LightGBM on feature table v1 (project step 4). Model code: homecredit/models/lgbm.py.

Same candidate columns as the `full` scorecard. Text columns are passed as pandas categories (LightGBM's own
handling); NaN is left as NaN. The LightGBM seed equals the fold seed, so the 5 seeds also measure training
randomness. The final model for the test set is trained on all of train with the median number of trees chosen
in CV x 1.1 (more data).

Usage:
    .venv/bin/python -m pipeline.train_lgbm [--seeds 0,1,2,3,4] [--jobs 5] [--threads 2] [--name lgbm_v1]
                                            [--features v2 --add win,stk,term,ext] [--drop gender,family,age]
                                            [--out-root DIR]
    --folds-only 0 (or any subset of folds) is a timing run: nothing is written.
    --seeds 0 (fewer than 5 seeds, all folds) is an exploratory run: out-of-fold files only, no final model;
    experiments/compare.py then compares the common seeds and marks the verdict exploratory.
Outputs: processed/oof/<name>.parquet (if all 5 seeds ran), processed/preds/<name>_test.parquet,
         processed/<name>_final.txt, analysis_output/lgbm/<name>_folds.csv, <name>_importance.csv
"""

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from homecredit.data import drop_columns, load_features, load_folds, scopes, select_groups
from homecredit.evaluate import auc, save_oof
from homecredit.models.lgbm import fit, fit_fixed, shap_select
from homecredit.paths import Paths, add_out_root

_cache = {}


def run_fold(out_root, seed, k, threads, drop=None, features="v1", add=None, exclude=None, lr=None, n_models=1,
             shap_top=None):
    if "tr" not in _cache:
        P = Paths(out_root)
        tr, _, cols = load_features(P, features, categories=True)
        cols = [c for c in drop_columns(select_groups(cols, add), drop) if not (exclude and c.startswith(exclude))]
        _cache.update(tr=tr, cols=cols, fo=load_folds(P, tr.SK_ID_CURR))
    tr, cols, fo = _cache["tr"], _cache["cols"], _cache["fo"]
    f = fo[f"fold_s{seed}"].values
    trn, va = np.where(f != k)[0], np.where(f == k)[0]
    y = tr.TARGET.values.astype(int)
    t0 = time.time()
    if shap_top:  # keep the shap_top columns with the largest mean |SHAP|, judged on the training part only
        cols = shap_select(tr[cols].iloc[trn], y[trn], shap_top, seed, threads)
    # n_models > 1: average of LightGBM fits with seeds seed, seed + 1000, ... (the first one is the usual fit)
    ps, trees, imp = [], [], 0
    for i in range(n_models):
        bst = fit(tr[cols].iloc[trn], y[trn], seed + 1000 * i, threads, lr)
        ps.append(bst.predict(tr[cols].iloc[va], num_iteration=bst.best_iteration))
        trees.append(bst.best_iteration)
        imp = imp + pd.Series(bst.feature_importance("gain"), index=cols)
    p = np.mean(ps, axis=0)
    n = trees[0] if n_models == 1 else float(np.mean(trees))
    return seed, k, va, p, n, time.time() - t0, imp / n_models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--folds-only", default="0,1,2,3,4", help="subset of folds, for timing runs")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--name", default="lgbm_v1")
    ap.add_argument("--features", default="v1", help="feature table version (processed/features_<v>.parquet)")
    ap.add_argument("--add", default=None, help="v2 feature groups to add to the v1 columns: win,stk,term,ext or all")
    ap.add_argument("--learning-rate", type=float, default=None, help="instead of the default 0.05")
    ap.add_argument("--n-models", type=int, default=1, help="average this many LightGBM fits per fold (seed averaging)")
    ap.add_argument("--shap-top", type=int, default=None, help="in each training fold, keep only the K columns "
                    "with the largest mean |SHAP| (homecredit.models.lgbm.shap_select)")
    ap.add_argument("--exclude", default=None, help="leave out columns starting with this prefix (e.g. stk_bb_)")
    ap.add_argument("--drop", default=None, help="leave out sensitive attributes, e.g. gender,family,age "
                    "(fairness step); use with a new --name")
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    if P.oof(args.name).exists():
        ap.error(f"{P.oof(args.name)} exists: saved results are never overwritten; choose a new --name")
    if (args.drop or args.add or args.features != "v1" or args.learning_rate or args.n_models > 1) \
            and args.name == "lgbm_v1":
        ap.error("--drop / --add / --features need a new --name, so the saved lgbm_v1 results are not overwritten")
    seeds = [int(s) for s in args.seeds.split(",")]
    ks = [int(k) for k in args.folds_only.split(",")]
    t0 = time.time()

    tr, te, cols = load_features(P, args.features, categories=True)
    cols = drop_columns(select_groups(cols, args.add), args.drop)
    cols = [c for c in cols if not (args.exclude and c.startswith(args.exclude))]
    y = tr.TARGET.values.astype(int)
    cash = scopes(tr.NAME_CONTRACT_TYPE)["cash"]
    print(f"{len(cols)} columns; {len(seeds) * len(ks)} fold fits", flush=True)

    res = Parallel(n_jobs=args.jobs)(
        delayed(run_fold)(args.out_root, s, k, args.threads, args.drop, args.features, args.add, args.exclude,
                          args.learning_rate, args.n_models, args.shap_top)
        for s in seeds for k in ks)
    oof = pd.DataFrame({"SK_ID_CURR": tr.SK_ID_CURR})
    rows, imps = [], []
    for s, k, va, p, n_trees, secs, imp in res:
        oof.loc[va, f"s{s}"] = p
        rows.append({"seed": s, "fold": k, "trees": n_trees, "seconds": round(secs, 1),
                     "auc_all": auc(y[va], p), "auc_cash": auc(y[va][cash[va]], p[cash[va]])})
        imps.append(imp)
    folds_df = pd.DataFrame(rows).sort_values(["seed", "fold"])
    print(folds_df.round(5).to_string(index=False), flush=True)
    print(f"mean AUC all {folds_df.auc_all.mean():.5f}, cash {folds_df.auc_cash.mean():.5f}; "
          f"elapsed {time.time() - t0:.0f} s", flush=True)

    if len(ks) != 5:
        print("timing run only: no files written")
        return
    out = P.out("lgbm")
    folds_df.to_csv(out / f"{args.name}_folds.csv", index=False)
    save_oof(P, args.name, oof)
    imp = pd.concat(imps, axis=1).mean(axis=1)  # with --shap-top: mean over the folds that kept the column
    imp_df = pd.DataFrame({"variable": imp.index, "mean_gain": imp.values,
                           "share_pct": (100 * imp / imp.sum()).round(2).values})
    if args.shap_top:
        kept = pd.Series([c for i in imps for c in i.index]).value_counts()
        imp_df["kept_in_folds"] = imp_df.variable.map(kept).values
        imp_df["folds"] = len(imps)
    imp_df.sort_values("mean_gain", ascending=False).to_csv(out / f"{args.name}_importance.csv", index=False)
    if len(seeds) != 5:
        print(f"exploratory run (seeds {seeds}): out-of-fold files written; no final model")
        return
    seed_means = folds_df.groupby("seed")[["auc_all", "auc_cash"]].mean()
    print("5-fold mean per seed:\n", seed_means.round(5).to_string())
    print("sd of seed means:", seed_means.std().round(5).to_dict())

    # Final model on all of train for test predictions: trees = median of the CV fold choices x 1.1 (more data).
    n_final = int(np.median(folds_df.trees) * 1.1)
    if args.shap_top:  # same selection rule on all of train
        cols = shap_select(tr[cols], y, args.shap_top, 0, args.jobs * args.threads)
    boosters = [fit_fixed(tr[cols], y, n_final, seed=1000 * i, threads=args.jobs * args.threads,
                          learning_rate=args.learning_rate) for i in range(args.n_models)]
    bst = boosters[0]
    P.preds_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"SK_ID_CURR": te.SK_ID_CURR, "TARGET": np.mean([b.predict(te[cols]) for b in boosters], axis=0)}).to_parquet(
        P.test_preds(args.name), index=False)
    bst.save_model(P.model_file(args.name))
    for i, b in enumerate(boosters[1:], start=1):  # the other fits of an average (--n-models > 1)
        b.save_model(P.model_file(f"{args.name}_{i}"))
    print(f"final model: {n_final} trees; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
