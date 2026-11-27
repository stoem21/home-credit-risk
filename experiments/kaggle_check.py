"""Independent check: Kaggle late-submission scores next to our cross-validated AUC.

Input: analysis_output/kaggle/kaggle_scores.csv, typed by hand from the Kaggle submissions page (the source column
says so). Kaggle scores are a check only: no model or setting is chosen from them (project rule: no score-gaming).

  1. Per model: CV AUC on cash loans (the test set is 99% cash loans), Kaggle private and public AUC, and the gaps.
  2. Does Kaggle rank the models the way our validation does? Spearman rank correlation of CV vs private.
  3. The step-by-step changes we adopted on CV, as Kaggle sees them. Kaggle gives one number per file, so there is
     no noise estimate from Kaggle itself. A rough scale: the bootstrap SE of the same paired gain on our 307,511
     training applicants (experiments/compare.py), times sqrt(307,511 / 38,995), the approximate size of the
     private part (80% of 48,744). The CV gain and SE are those of cash loans, so the scale uses the 278,232
     train cash loans (an earlier version used all 307,511 rows, about 5% too large). This assumes test applicants vary like train applicants, which the drift check
     says is not quite true, so it is only a scale, not a test.

Usage:
    .venv/bin/python -m experiments.kaggle_check [--tag v2] [--out-root DIR]
    v3: --tag v3 --v3 (also reads the v3 scores, typed by hand into analysis_output/kaggle/kaggle_scores_v3.csv,
        and the v3 model table analysis_output/model_summary_v3.csv)
Output: analysis_output/kaggle/kaggle_vs_cv<_tag>.csv, kaggle_steps<_tag>.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from homecredit.paths import Paths, add_out_root

N_PRIVATE = round(0.8 * 48_744)  # the private part is about 80% of the test set (99% cash loans)
STEPS = [  # (reference, change), as adopted or tested on CV, with the comparison file that holds the CV SE
    ("lgbm_v1", "lgbm_v1_no_gender"),
    ("lgbm_v1", "lgbm_v1_no_gender_family"),
    ("lgbm_v1_no_gender_family", "lgbm_v2_win"),
    ("lgbm_v1_no_gender_family", "lgbm_v2_stk_nobb"),
    ("lgbm_v1_no_gender_family", "lgbm_v2_term"),
    ("lgbm_v2_stk_nobb", "lgbm_v2_stk"),
    ("lgbm_v1_no_gender_family", "lgbm_v2_combo"),
    ("lgbm_v2_combo_with_gender", "lgbm_v2_combo"),
    ("lgbm_v2_combo", "lgbm_v2_combo_lr02"),
    ("lgbm_v2_combo_lr02", "lgbm_v2_combo_lr02_x3"),
    ("lgbm_v2_combo_lr02_x3", "lgbm_v3b_sh90_lr02_x3"),
    ("lgbm_v3b_sh90_with_gender", "lgbm_v3b_sh90"),
    ("lgbm_v3b_sh90", "lgbm_v3b_sh90_lr02_x3"),
    ("lgbm_v3b_sh90_lr02_x3", "lgbm_v3b_sh90_lr02_x3_tuned"),
    ("lgbm_v3b_sh90_lr02_x3_tuned_with_gender", "lgbm_v3b_sh90_lr02_x3_tuned"),
    ("lgbm_v3b_sh90_lr02_x3_tuned", "lgbm_v3c_sh90_lr02_x3_tuned"),
    ("lgbm_v3c_sh90_lr02_x3_tuned", "lgbm_v3c_sh90_lr02_x3_tuned_mono"),
    ("lgbm_v3c_sh90_lr02_x3_tuned_mono_with_gender", "lgbm_v3c_sh90_lr02_x3_tuned_mono"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="suffix for the output files, so earlier ones stay")
    ap.add_argument("--v3", action="store_true", help="add the v3 scores and use the v3 model table")
    ap.add_argument("--v3t", action="store_true", help="add the v3 and v3t scores and use the v3t model table")
    ap.add_argument("--v3m", action="store_true", help="add every v3 score file and use the v3m model table")
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    out = P.out("kaggle")
    sfx = f"_{args.tag}" if args.tag else ""
    pd.set_option("display.width", 250)

    k = pd.read_csv(P.kaggle_scores)
    args.v3t = args.v3t or args.v3m
    extra = (["v3"] if args.v3 or args.v3t else []) + (["v3t"] if args.v3t else []) + (["v3c", "v3m"] if args.v3m else [])
    for v in [v for v in extra if P.kaggle_scores.with_name(f"kaggle_scores_{v}.csv").exists()]:
        k = pd.concat([k, pd.read_csv(P.kaggle_scores.with_name(f"kaggle_scores_{v}.csv"))], ignore_index=True)
    n_cash = int(pd.read_parquet(P.raw / "application_train.parquet", columns=["NAME_CONTRACT_TYPE"])
                 .NAME_CONTRACT_TYPE.eq("Cash loans").sum())
    s = pd.read_csv(P.results / ("model_summary_v3m.csv" if args.v3m else "model_summary_v3t.csv" if args.v3t else "model_summary_v3.csv" if args.v3
                                 else "model_summary.csv"))[
        ["model", "cv_auc_all", "cv_auc_cash"]]
    t = k.drop(columns="source").merge(s, on="model", how="left")
    t["private_minus_cv_cash"] = (t.private_auc - t.cv_auc_cash).round(5)
    t["public_minus_private"] = (t.public_auc - t.private_auc).round(5)
    t = t.sort_values("cv_auc_cash", ascending=False)
    both = t.dropna(subset=["cv_auc_cash", "private_auc"])  # a model with a Kaggle score but no CV row in this
    rho = spearmanr(both.cv_auc_cash, both.private_auc).statistic  # summary table is left out of the summary numbers
    t.to_csv(out / f"kaggle_vs_cv{sfx}.csv", index=False)
    print(t.to_string(index=False))
    print(f"\nmodels with both scores: {len(both)}; Spearman rank correlation CV (cash) vs private: {rho:.3f}; "
          f"mean private - CV: {t.private_minus_cv_cash.mean():+.4f}\n")

    sc = k.set_index("model")
    cv = s.set_index("model")
    rows = []
    for a, b in STEPS:
        if a not in sc.index or b not in sc.index:
            continue
        comp = P.results / "comparisons" / f"{a}_vs_{b}.csv"
        c = pd.read_csv(comp).set_index("scope").loc["cash"] if comp.exists() else None
        if c is None:  # some steps were compared inside the fairness step
            comp = P.results / "fairness" / "accuracy_cost.csv"
            f = pd.read_csv(comp)
            c = f[(f.reference == a) & (f.variant == b) & (f.scope == "cash")].iloc[0]
        se_private = float(c.boot_se_gain) * np.sqrt(n_cash / N_PRIVATE)
        gain_priv = sc.loc[b, "private_auc"] - sc.loc[a, "private_auc"]
        rows.append({"reference": a, "change": b, "cv_gain_cash": round(cv.loc[b, "cv_auc_cash"] - cv.loc[a, "cv_auc_cash"], 5),
                     "private_gain": round(gain_priv, 5),
                     "public_gain": round(sc.loc[b, "public_auc"] - sc.loc[a, "public_auc"], 5),
                     "rough_se_private": round(se_private, 5),
                     "private_gain_over_2se": bool(abs(gain_priv) > 2 * se_private),
                     "same_sign_as_cv": bool(np.sign(gain_priv) == np.sign(cv.loc[b, "cv_auc_cash"] - cv.loc[a, "cv_auc_cash"]))})
    steps = pd.DataFrame(rows)
    steps.to_csv(out / f"kaggle_steps{sfx}.csv", index=False)
    print(steps.to_string(index=False))


if __name__ == "__main__":
    main()
