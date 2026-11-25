"""Calibration (project step 5): do predicted probabilities match observed default rates?

AUC only checks the ranking. A bank also needs the probability itself to be right: expected loss, pricing and
the approval cut-off (business step) all use it. Checked on out-of-fold predictions, all loans and cash loans.

Measures (per seed, then mean and sd over the 5 seeds; definitions in homecredit/evaluate.py):
  - mean predicted vs observed default rate ("calibration in the large");
  - calibration slope and intercept: logistic regression of the outcome on logit(prediction). Slope 1 and
    intercept 0 = perfect. Slope < 1 = predictions too extreme; slope > 1 = too timid;
  - ECE (expected calibration error): average gap |predicted - observed| over 20 equal-size bins, weighted;
  - Brier score and log loss (lower is better; both reward calibration and ranking).
  - reliability table for seed 0: 20 bins of predicted probability, mean predicted vs observed.

ECE by fold (file ece_by_fold<_tag>.csv): the pooled ECE of a cross-fitted map is biased low (found by an
external code review), so ECE is also measured inside each validation fold and averaged, next to the value a
perfectly calibrated model would show (outcomes simulated from the predictions themselves).

Recalibration, cross-fitted so no row is scored by a map fitted on its own label: for seed s and fold k the map
is fitted on the OOF predictions of the other four folds (same seed) and applied to fold k.
  - platt: logistic regression on logit(p) (2 parameters, keeps the ranking);
  - isotonic: a step function that only rises (flexible; may create ties, so AUC can move a little).

Usage:
    .venv/bin/python -m analysis.calibration MODEL [MODEL ...] [--tag v2] [--out-root DIR]   (names in processed/oof/)
Outputs analysis_output/calibration/summary<_tag>.csv, by_seed<_tag>.csv, ece_by_fold<_tag>.csv and
<MODEL>_reliability.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from homecredit.data import N_SEEDS, load_folds, load_target, scopes
from homecredit.evaluate import EPS, calibration_metrics, ece_by_fold, ece_floor, load_oof, logit, reliability
from homecredit.paths import Paths, add_out_root


def recalibrate(p, y, folds, method):
    out = np.empty_like(p)
    for k in np.unique(folds):
        fit, app = folds != k, folds == k
        if method == "platt":
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p[fit]).reshape(-1, 1), y[fit])
            out[app] = lr.predict_proba(logit(p[app]).reshape(-1, 1))[:, 1]
        else:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p[fit], y[fit])
            out[app] = iso.predict(p[app])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--tag", default="", help="suffix for summary / by_seed file names (e.g. v2), so earlier ones stay")
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    out = P.out("calibration")

    tr = load_target(P)
    fo = load_folds(P, tr.SK_ID_CURR)
    y = tr.TARGET.values
    masks = scopes(tr.NAME_CONTRACT_TYPE)

    rows, fold_rows = [], []
    for name in args.models:
        oof = load_oof(P, name, tr.SK_ID_CURR)
        rel = []
        for s in range(N_SEEDS):
            raw = oof[f"s{s}"].values
            versions = {"raw": raw,
                        "platt": recalibrate(raw, y, fo[f"fold_s{s}"].values, "platt"),
                        "isotonic": recalibrate(raw, y, fo[f"fold_s{s}"].values, "isotonic")}
            for ver, p in versions.items():
                for scope, m in masks.items():
                    rows.append({"model": name, "version": ver, "scope": scope, "seed": s,
                                 **calibration_metrics(y[m], p[m])})
                    f = fo[f"fold_s{s}"].values[m]
                    fold_rows.append({"model": name, "version": ver, "scope": scope, "seed": s,
                                      "ece_by_fold_pts": ece_by_fold(y[m], p[m], f),
                                      "floor_pts": ece_floor(p[m], f, seed=s)})
                    if s == 0:
                        t = reliability(y[m], p[m]).reset_index()
                        rel.append(t.assign(version=ver, scope=scope))
        pd.concat(rel).to_csv(out / f"{name}_reliability.csv", index=False)
        print(f"{name} done", flush=True)

    sfx = f"_{args.tag}" if args.tag else ""
    ef = pd.DataFrame(fold_rows)
    ef = ef.groupby(["model", "version", "scope"], sort=False)[["ece_by_fold_pts", "floor_pts"]].mean().round(4)
    ef.reset_index().to_csv(out / f"ece_by_fold{sfx}.csv", index=False)
    print(ef.to_string(), "\n")
    res = pd.DataFrame(rows)
    res.to_csv(out / f"by_seed{'_' + args.tag if args.tag else ''}.csv", index=False)
    keys = ["model", "version", "scope"]
    mean = res.groupby(keys, sort=False).mean(numeric_only=True).drop(columns="seed")
    sd = res.groupby(keys, sort=False)[["ece_pts", "log_loss", "slope"]].std().add_suffix("_sd")
    summ = mean.join(sd).reset_index()
    summ.to_csv(out / f"summary{'_' + args.tag if args.tag else ''}.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
