"""Train and cross-validate the WOE scorecard (project step 4). Model code: homecredit/models/scorecard.py.

Cross-validation: 5 seeds x 5 folds; on each fold the bins, the selection and the logistic regression are fitted
on the training part only. Then the scorecard is refitted on all of train, turned into a points table, and used
to score the test set.

Variants: `app` uses application columns only; `full` adds the 22 history summaries of make_features.py.

Usage:
    .venv/bin/python -m pipeline.train_scorecard --variant app|full [--jobs 5] [--drop gender --name NAME]
                                                 [--out-root DIR]
Outputs: processed/oof/scorecard_<variant>.parquet (OOF, 5 seeds), processed/preds/scorecard_<variant>_test.parquet,
         analysis_output/scorecard/<variant>_*.csv
"""

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from homecredit.data import HIST_PREFIX, N_FOLDS, N_SEEDS, drop_columns, load_features, load_folds
from homecredit.evaluate import save_oof
from homecredit.models.scorecard import BASE_ODDS, BASE_SCORE, FACTOR, OFFSET, PDO, fit_scorecard, predict
from homecredit.paths import Paths, add_out_root

_cache = {}


def run_fold(out_root, cols, seed, k):
    if "X" not in _cache:
        P = Paths(out_root)
        tr, _, _ = load_features(P)
        _cache.update(X=tr, folds=load_folds(P, tr.SK_ID_CURR))
    X, fo = _cache["X"], _cache["folds"]
    f = fo[f"fold_s{seed}"].values
    trn, va = f != k, f == k
    y = X.TARGET.values.astype(float)
    specs, selected, lr, dropped = fit_scorecard(X[trn], y[trn], cols)
    return seed, k, np.where(va)[0], predict(X[va], specs, selected, lr), selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["app", "full"], required=True)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--drop", default=None, help="leave out sensitive attributes, e.g. gender (fairness step)")
    ap.add_argument("--name", default=None, help="output name (default scorecard_<variant>)")
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    out = P.out("scorecard")
    t0 = time.time()

    tr, te, cols = load_features(P)
    if args.variant == "app":
        cols = [c for c in cols if not c.startswith(HIST_PREFIX)]
    cols = drop_columns(cols, args.drop)
    name = args.name or f"scorecard_{args.variant}"
    if args.drop and name == f"scorecard_{args.variant}":
        ap.error("--drop needs a new --name, so the saved scorecard results are not overwritten")
    tag = name.removeprefix("scorecard_")  # file prefix in analysis_output/scorecard/
    print(f"variant {args.variant}: {len(cols)} candidate columns", flush=True)

    tasks = [(s, k) for s in range(N_SEEDS) for k in range(N_FOLDS)]
    results = Parallel(n_jobs=args.jobs)(delayed(run_fold)(args.out_root, cols, s, k) for s, k in tasks)
    oof = pd.DataFrame({"SK_ID_CURR": tr.SK_ID_CURR})
    sel_count = {}
    for s, k, idx, p, selected in results:
        col = f"s{s}"
        if col not in oof:
            oof[col] = np.nan
        oof.loc[idx, col] = p
        for c in selected:
            sel_count[c] = sel_count.get(c, 0) + 1
    assert not oof.isna().any().any()
    save_oof(P, name, oof)
    print(f"CV done in {time.time() - t0:.0f} s", flush=True)

    # Final scorecard on all of train
    y = tr.TARGET.values.astype(float)
    specs, selected, lr, dropped = fit_scorecard(tr, y, cols)
    base_points = OFFSET - FACTOR * lr.intercept_[0]
    rows = []
    for c, b in zip(selected, lr.coef_[0]):
        sp = specs[c]
        for lab, n, bad, w in zip(sp["labels"], sp["n"], sp["bad"], sp["woe"]):
            if n == 0:
                continue
            rows.append({"variable": c, "bin": lab, "applicants": int(n), "pct_of_train": round(100 * n / len(y), 2),
                         "default_rate_pct": round(100 * bad / n, 2), "woe": round(w, 4), "coef": round(b, 4),
                         "points": int(round(-FACTOR * b * w))})
    table = pd.DataFrame(rows)
    table.to_csv(out / f"{tag}_table.csv", index=False)
    var = pd.DataFrame({"variable": selected, "iv": [round(specs[c]["iv"], 4) for c in selected],
                        "coef": np.round(lr.coef_[0], 4),
                        "selected_in_folds_of_25": [sel_count.get(c, 0) for c in selected]})
    var.to_csv(out / f"{tag}_variables.csv", index=False)
    pd.DataFrame([{"base_points": round(base_points, 2), "factor": round(FACTOR, 4), "pdo": PDO,
                   "base_score": BASE_SCORE, "base_odds_good_to_bad": BASE_ODDS, "n_variables": len(selected),
                   "dropped_for_sign": ", ".join(dropped)}]).to_csv(out / f"{tag}_meta.csv", index=False)
    stab = pd.DataFrame(sorted(sel_count.items(), key=lambda kv: -kv[1]), columns=["variable", "selected_in_folds_of_25"])
    stab.to_csv(out / f"{tag}_selection_stability.csv", index=False)

    P.preds_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"SK_ID_CURR": te.SK_ID_CURR, "TARGET": predict(te, specs, selected, lr)}).to_parquet(
        P.test_preds(name), index=False)

    print(f"final scorecard: {len(selected)} variables, base points {base_points:.1f}; dropped for sign: {dropped}")
    print(var.to_string(index=False))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
