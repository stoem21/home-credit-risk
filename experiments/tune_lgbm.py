"""Bounded search of four LightGBM settings for the v3 model (EXPERIMENTS.md entry 32).

The plan was fixed before any result was seen:
  - settings searched: num_leaves 16 to 128, min_child_samples 20 to 500 (both whole numbers, log scale),
    feature_fraction 0.1 to 0.8, lambda_l2 0.01 to 10 (log scale); every other setting stays as in
    homecredit/models/lgbm.py;
  - 20 random combinations (numpy seed 0, so the list is fixed in advance), plus the current settings as trial 0,
    measured in the same way as a reference;
  - every trial: the 5 folds of seed 1, learning rate 0.05, one fit with early stopping on an inner 10%,
    on the columns the v3 model's SHAP rule (90%) keeps in each training fold. The SHAP selection does not depend
    on the searched settings (it uses the defaults), so it is computed once per fold and reused;
  - the best trial = highest mean AUC on all loans. It is then confirmed with the full setting (learning rate
    0.02, average of 3 fits, 5 seeds) and judged only on seeds 0, 2, 3 and 4, which were not used to choose
    anything: adopted only if it beats noise and wins at least 15 of 20 folds (the 18/25 share).
Random search instead of a library such as Optuna: no extra dependency, and with 20 trials the difference is small.

Usage:
    .venv/bin/python -m experiments.tune_lgbm --features v3b --add win,stk,term,agg,last,trend,load \
        --exclude <v3 list> --drop gender,family --shap-share 0.9 --name v3b_sh90 [--trials 20] [--jobs 5]
    --resume continues a search that was stopped: the trial list is fixed, so the finished trials (checked against
    the list) are kept and only the missing ones run; the saved SHAP columns are reused.
Outputs: analysis_output/tuning/<name>_trials.csv, <name>_best.json (read by train_lgbm --params),
         <name>_shap_columns.json (the columns kept in each fold)
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from homecredit.data import drop_columns, exclude_prefixes, load_features, load_folds, scopes, select_groups
from homecredit.evaluate import auc
from homecredit.models.lgbm import PARAMS, fit, shap_select
from homecredit.paths import Paths, add_out_root

SEED = 1
SEARCHED = ["num_leaves", "min_child_samples", "feature_fraction", "lambda_l2"]
_cache = {}


def trials(n: int) -> list[dict]:
    rng = np.random.default_rng(0)
    out = [{k: PARAMS[k] for k in SEARCHED}]  # trial 0: the current settings
    for _ in range(n):
        out.append({"num_leaves": int(round(np.exp(rng.uniform(np.log(16), np.log(128))))),
                    "min_child_samples": int(round(np.exp(rng.uniform(np.log(20), np.log(500))))),
                    "feature_fraction": round(float(rng.uniform(0.1, 0.8)), 3),
                    "lambda_l2": round(float(np.exp(rng.uniform(np.log(0.01), np.log(10)))), 4)})
    return out


def data(args):
    if "tr" not in _cache:
        P = Paths(args.out_root)
        tr, _, cols = load_features(P, args.features, categories=True)
        cols = exclude_prefixes(drop_columns(select_groups(cols, args.add), args.drop), args.exclude)
        f = load_folds(P, tr.SK_ID_CURR)[f"fold_s{SEED}"].values
        _cache.update(tr=tr, cols=cols, f=f, y=tr.TARGET.values.astype(int),
                      cash=scopes(tr.NAME_CONTRACT_TYPE)["cash"])
    return _cache


def select_fold(args, k):
    d = data(args)
    trn = np.where(d["f"] != k)[0]
    return k, shap_select(d["tr"][d["cols"]].iloc[trn], d["y"][trn], None, SEED, args.threads, share=args.shap_share)


def run_fold(args, k, cols, params):
    d = data(args)
    trn, va = np.where(d["f"] != k)[0], np.where(d["f"] == k)[0]
    t0 = time.time()
    bst = fit(d["tr"][cols].iloc[trn], d["y"][trn], SEED, args.threads, params=params)
    p = bst.predict(d["tr"][cols].iloc[va], num_iteration=bst.best_iteration)
    y, c = d["y"][va], d["cash"][va]
    return auc(y, p), auc(y[c], p[c]), bst.best_iteration, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in ["--features", "--add", "--exclude", "--drop", "--name"]:
        ap.add_argument(a, required=a in ("--features", "--name"), default=None)
    ap.add_argument("--shap-share", type=float, required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--resume", action="store_true", help="continue a stopped search (same fixed trial list)")
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    out = P.out("tuning")
    f_trials, f_sel = out / f"{args.name}_trials.csv", out / f"{args.name}_shap_columns.json"
    plan = trials(args.trials)
    t0 = time.time()
    rows = []
    if args.resume:
        done = pd.read_csv(f_trials)
        for r in done.itertuples():  # the finished trials must be the start of the fixed list
            assert all(abs(getattr(r, k) - plan[r.trial][k]) < 1e-9 for k in SEARCHED), r.trial
        rows = done.to_dict("records")
        sel = {int(k): v for k, v in json.loads(f_sel.read_text()).items()}
        print(f"resuming after {len(rows)} finished trials", flush=True)
    else:
        if f_trials.exists():
            ap.error(f"{f_trials} exists: saved results are never overwritten (use --resume to continue it)")
        sel = dict(Parallel(n_jobs=args.jobs)(delayed(select_fold)(args, k) for k in range(5)))
        f_sel.write_text(json.dumps({str(k): sel[k] for k in range(5)}, indent=1))
        print(f"SHAP selection, seed {SEED}: {[len(sel[k]) for k in range(5)]} columns per fold; "
              f"{time.time() - t0:.0f} s", flush=True)

    for i, params in enumerate(plan):
        if i < len(rows):
            continue
        t1 = time.time()
        res = Parallel(n_jobs=args.jobs)(delayed(run_fold)(args, k, sel[k], params) for k in range(5))
        a, c, n, _ = map(np.array, zip(*res))
        rows.append({"trial": i, **params, "auc_all": round(a.mean(), 5), "auc_cash": round(c.mean(), 5),
                     "auc_all_min_fold": round(a.min(), 5), "trees_median": int(np.median(n)),
                     "seconds": round(time.time() - t1)})
        pd.DataFrame(rows).to_csv(f_trials, index=False)
        print(pd.DataFrame(rows).tail(1).to_string(index=False, header=i == 0), flush=True)

    res = pd.DataFrame(rows)
    best = res.sort_values("auc_all", ascending=False).iloc[0]
    params = {k: (int(best[k]) if k in ("num_leaves", "min_child_samples") else float(best[k])) for k in SEARCHED}
    (out / f"{args.name}_best.json").write_text(json.dumps(
        {"trial": int(best.trial), "auc_all_seed1": float(best.auc_all),
         "auc_all_current_settings_seed1": float(res.auc_all.iloc[0]), "params": params}, indent=1))
    print(f"\nbest trial {int(best.trial)}: {params}, AUC {best.auc_all:.5f} vs current settings "
          f"{res.auc_all.iloc[0]:.5f}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
