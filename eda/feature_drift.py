"""Feature table v1: does each feature behave the same in train and in test?

EDA step 5 compared the raw application columns. Here the check is repeated on the 149 columns the models actually
use, including the derived ratios and the 22 history summaries, and it adds two whole-table views.

  1. Per feature: share missing in train vs test; how well the feature alone separates test rows from train rows
     (numeric: AUC reported as max(AUC, 1 - AUC), 0.5 = same distribution, missing left out; text: total variation
     distance between category shares, 0 = same shares); and its AUC for TARGET in train, for scale.
  2. Adversarial validation: a LightGBM classifier (same settings as the default model) learns to tell test rows
     from train rows using all features, 5-fold cross-validated. AUC near 0.5 = test looks like train; the
     features it relies on most are the ones that differ.
  3. What the shift could mean for the score (an estimate, not a measurement): train rows are reweighted to look
     like test (weight = odds of "test" from the adversarial model, scaled to mean 1) and the saved out-of-fold
     predictions are scored with those weights. This assumes the shift is in the inputs only (same default
     mechanism given the inputs), which cannot be checked without test labels. The effective sample size of the
     weights says how much data the estimate really rests on.

No target information from test is used (there is none); the adversarial model only sees inputs. Nothing here is
used to choose a model setting.

Usage:
    .venv/bin/python -m eda.feature_drift [--features v1] [--jobs 5] [--threads 2] [--out-root DIR]
    With --features v2, every output name gets a "_v2" suffix (adversarial_v2.parquet) and nothing of v1 is touched.
Outputs: analysis_output/eda/feature_drift.csv, feature_drift_loan_terms.csv, feature_adversarial_importance.csv,
         feature_adversarial_summary.csv;
         processed/adversarial_v1.parquet (out-of-fold probability of "test" for every train and test row)
"""

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from homecredit.data import N_SEEDS, is_text, load_features, scopes
from homecredit.evaluate import auc, load_oof
from homecredit.models.lgbm import fit
from homecredit.paths import Paths, add_out_root

MODELS = ["logreg_baseline", "scorecard_full", "lgbm_v1"]


def adversarial_fold(X, is_test, trn, va, seed, threads):
    bst = fit(X.iloc[trn], is_test[trn], seed, threads)
    p = bst.predict(X.iloc[va], num_iteration=bst.best_iteration)
    return va, p, pd.Series(bst.feature_importance("gain"), index=X.columns), bst.best_iteration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="v1", help="feature table version")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    out = P.out("eda")

    ver = args.features
    sfx = "" if ver == "v1" else f"_{ver}"
    tr, te, cols = load_features(P, ver)
    y = tr.TARGET.values.astype(int)

    # ---- 1. per feature ----
    rows = []
    for c in cols:
        a, b = tr[c], te[c]
        r = {"feature": c, "kind": "text" if is_text(a) else "numeric",
             "pct_missing_train": round(100 * a.isna().mean(), 2), "pct_missing_test": round(100 * b.isna().mean(), 2)}
        if r["kind"] == "text":
            p = a.astype(object).fillna("(missing)").value_counts(normalize=True)
            q = b.astype(object).fillna("(missing)").value_counts(normalize=True)
            r["drift_tvd"] = round(0.5 * p.subtract(q, fill_value=0).abs().sum(), 4)
        else:
            v = np.r_[a.astype(float).values, b.astype(float).values]
            d = auc(np.r_[np.zeros(len(a)), np.ones(len(b))], v)
            t = auc(y, a.astype(float).values)
            r["drift_auc"] = round(max(d, 1 - d), 4)
            r["target_auc"] = round(max(t, 1 - t), 4)
        rows.append(r)
    fd = pd.DataFrame(rows)
    fd["missing_gap_pts"] = (fd.pct_missing_test - fd.pct_missing_train).round(2)
    fd = fd[["feature", "kind", "pct_missing_train", "pct_missing_test", "missing_gap_pts", "drift_auc", "drift_tvd",
             "target_auc"]]
    fd.to_csv(out / f"feature_drift{sfx}.csv", index=False)
    pd.set_option("display.width", 250)

    # Detail for the strongest shift (found in part 2): loan terms of cash loans. annuity / credit is roughly
    # 1 / number of payments (plus interest), so a higher value means a shorter or dearer loan.
    cash_tr, cash_te = tr[tr.NAME_CONTRACT_TYPE == "Cash loans"], te[te.NAME_CONTRACT_TYPE == "Cash loans"]
    det = []
    for c in ["annuity_to_credit", "credit_to_goods"]:
        for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
            det.append({"feature": c, "statistic": f"quantile {q}", "train_cash": round(cash_tr[c].quantile(q), 4),
                        "test_cash": round(cash_te[c].quantile(q), 4)})
    r_tr, r_te = cash_tr.annuity_to_credit.round(5), cash_te.annuity_to_credit.round(5)
    det.append({"feature": "annuity_to_credit", "statistic": "share of rows whose exact ratio (5 decimals) occurs "
                "in the other split", "train_cash": round(r_tr.isin(set(r_te)).mean(), 4),
                "test_cash": round(r_te.isin(set(r_tr)).mean(), 4)})
    pd.DataFrame(det).to_csv(out / f"feature_drift_loan_terms{sfx}.csv", index=False)
    print(pd.DataFrame(det).to_string(index=False), "\n")
    print("largest missing-share gaps (test - train, points):")
    print(fd.reindex(fd.missing_gap_pts.abs().sort_values(ascending=False).index).head(12).to_string(index=False))
    print("\nnumeric features that best separate test from train:")
    print(fd.sort_values("drift_auc", ascending=False).head(12).to_string(index=False))
    print("\ntext features with the largest share differences:")
    print(fd.sort_values("drift_tvd", ascending=False).head(5).to_string(index=False), "\n")

    # ---- 2. adversarial validation ----
    trc, tec, _ = load_features(P, ver, categories=True)
    X = pd.concat([trc[cols], tec[cols]], ignore_index=True)  # categories were set on train + test together
    is_test = np.r_[np.zeros(len(trc)), np.ones(len(tec))].astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    res = Parallel(n_jobs=args.jobs)(delayed(adversarial_fold)(X, is_test, trn, va, 0, args.threads)
                                     for trn, va in skf.split(X, is_test))
    p_test = np.empty(len(X))
    imps, fold_aucs, trees = [], [], []
    for va, p, imp, n in res:
        p_test[va] = p
        imps.append(imp)
        fold_aucs.append(auc(is_test[va], p))
        trees.append(n)
    adv = pd.DataFrame({"SK_ID_CURR": np.r_[trc.SK_ID_CURR.values, tec.SK_ID_CURR.values],
                        "split": np.where(is_test == 1, "test", "train"), "p_test": p_test})
    adv.to_parquet(P.processed / f"adversarial_{ver}.parquet", index=False)
    imp = pd.concat(imps, axis=1).mean(axis=1)
    imp_df = pd.DataFrame({"feature": imp.index, "mean_gain": imp.values,
                           "share_pct": (100 * imp / imp.sum()).round(2).values}).sort_values(
        "mean_gain", ascending=False)
    imp_df.to_csv(out / f"feature_adversarial_importance{sfx}.csv", index=False)
    print(f"adversarial AUC (test vs train), 5 folds: {np.round(fold_aucs, 4)}; trees {trees}")
    print(imp_df.head(15).to_string(index=False), "\n")

    # ---- 3. reweighted out-of-fold AUC (estimate) ----
    pt = np.clip(p_test[: len(trc)], 1e-3, 1 - 1e-3)
    w = pt / (1 - pt)
    w = w / w.mean()
    ess = w.sum() ** 2 / (w ** 2).sum()
    masks = scopes(trc.NAME_CONTRACT_TYPE)
    summ = [{"measure": "adversarial AUC, mean of 5 folds", "model": "", "scope": "", "value": np.mean(fold_aucs)},
            {"measure": "adversarial AUC, sd of 5 folds", "model": "", "scope": "", "value": np.std(fold_aucs, ddof=1)},
            {"measure": "effective sample size of weights (train rows)", "model": "", "scope": "", "value": ess},
            {"measure": "effective sample size, % of train", "model": "", "scope": "", "value": 100 * ess / len(w)},
            {"measure": "largest weight (mean = 1)", "model": "", "scope": "", "value": w.max()}]
    for name in MODELS:
        oof = load_oof(P, name, trc.SK_ID_CURR)
        for scope, m in masks.items():
            plain = [auc(y[m], oof[f"s{s}"].values[m]) for s in range(N_SEEDS)]
            wtd = [auc(y[m], oof[f"s{s}"].values[m], w[m]) for s in range(N_SEEDS)]
            summ += [{"measure": "OOF AUC, mean of 5 seeds", "model": name, "scope": scope, "value": np.mean(plain)},
                     {"measure": "OOF AUC reweighted to test mix, mean of 5 seeds (estimate)", "model": name,
                      "scope": scope, "value": np.mean(wtd)}]
    summ = pd.DataFrame(summ)
    summ["value"] = summ.value.round(5)
    summ.to_csv(out / f"feature_adversarial_summary{sfx}.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
