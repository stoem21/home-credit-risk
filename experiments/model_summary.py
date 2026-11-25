"""One table of every model trained so far: what it is, its cross-validated AUC, and whether it has a Kaggle file.

AUC = mean over the folds the model has (25 for 5-seed models; 5 for the exploratory seed-0 runs, marked), on all
loans and on cash loans, computed from the saved out-of-fold predictions with the shared evaluation code.
The description of each model is taken from the experiment log (EXPERIMENTS.md); nothing is retrained here.

Usage:
    .venv/bin/python -m experiments.model_summary [--out-root DIR]
Output: analysis_output/model_summary.csv
"""

import argparse

import pandas as pd

from experiments.noise_check import EXT
from homecredit.data import load_folds, load_target, read_raw, scopes
from homecredit.evaluate import fold_aucs, load_oof
from homecredit.paths import Paths, add_out_root

# name: (family, features, uses gender / family status, note)
MODELS = {
    "rule": ("rule, no training", "1 - mean of EXT_SOURCE_1/2/3", "no", "baseline"),
    "logreg_baseline": ("logistic regression", "15 hand-picked application columns", "gender", "baseline"),
    "scorecard_app": ("WOE scorecard", "application form only", "gender + family", ""),
    "scorecard_full": ("WOE scorecard", "v1: form + 22 history summaries", "gender", ""),
    "scorecard_full_no_gender": ("WOE scorecard", "v1", "no", "explainable comparison"),
    "lgbm_v1": ("LightGBM", "v1 (149 columns)", "gender + family", ""),
    "lgbm_v1_no_gender": ("LightGBM", "v1", "family", "fairness test"),
    "lgbm_v1_no_gender_family": ("LightGBM", "v1", "no", "fairness test; v2 reference"),
    "lgbm_v1_no_sensitive": ("LightGBM", "v1 without age too", "no (no age)", "fairness test"),
    "lgbm_v2_win": ("LightGBM", "v1 + recent-behaviour group", "no", "feature test"),
    "lgbm_v2_stk": ("LightGBM", "v1 + stacking group incl. bureau_balance", "no", "feature test"),
    "lgbm_v2_stk_nobb": ("LightGBM", "v1 + stacking group without bureau_balance", "no", "feature test"),
    "lgbm_v2_term": ("LightGBM", "v1 + loan-terms group", "no", "feature test"),
    "lgbm_v2_ext": ("LightGBM", "v1 + external-score combinations", "no", "feature test (no gain)"),
    "lgbm_v2_combo": ("LightGBM", "v2: v1 + win + stk (no bb) + term (186 columns)", "no", ""),
    "lgbm_v2_combo_with_gender": ("LightGBM", "v2 + gender and family", "gender + family", "fairness reference"),
    "lgbm_v2_combo_lr02": ("LightGBM", "v2, learning rate 0.02", "no", ""),
    "lgbm_v2_combo_lr02_x3": ("LightGBM", "v2, learning rate 0.02, average of 3 fits", "no", "CHOSEN MODEL"),
    "xgb_v2_combo_s0": ("XGBoost", "v2", "no", "seed 0 only (exploratory)"),
    "mlp_v2_combo_s0": ("neural net (MLP)", "v2", "no", "seed 0 only (exploratory)"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    tr = load_target(P)
    y = tr.TARGET.values
    folds = load_folds(P, tr.SK_ID_CURR)
    masks = scopes(tr.NAME_CONTRACT_TYPE)
    rows = []
    for name, (family, feats, sens, note) in MODELS.items():
        if name == "rule":  # same score as experiments/noise_check.py and pipeline/make_submissions.py
            app = read_raw(P, "application_train", EXT)
            score = 1 - app[EXT].mean(axis=1).fillna(app[EXT].mean(axis=1).mean()).values
            oof = pd.DataFrame({f"s{s}": score for s in range(5)})
        else:
            oof = load_oof(P, name, tr.SK_ID_CURR)
        seeds = [s for s in range(5) if f"s{s}" in oof.columns]
        fa = fold_aucs(y, oof, folds, masks, seeds)
        m = fa.groupby("scope").auc.mean()
        rows.append({"model": name, "family": family, "features": feats, "sensitive_columns_used": sens,
                     "folds": len(fa) // len(masks), "cv_auc_all": round(m["all"], 5),
                     "cv_auc_cash": round(m["cash"], 5),
                     "kaggle_file": f"submission/{name}.csv" if P.test_preds(name).exists() else "",
                     "note": note})
    res = pd.DataFrame(rows).sort_values("cv_auc_all").reset_index(drop=True)
    res.to_csv(P.results / "model_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
