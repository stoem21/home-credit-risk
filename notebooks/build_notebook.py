"""Build and run the report notebook (notebooks/home_credit_report.ipynb).

The notebook tells the README's story with the tables behind it. Cheap tables are computed live from the processed
data; results of expensive runs (cross-validation, calibration, reason codes, fairness, business) are read from the
saved files in analysis_output/. It shows aggregated numbers only, never an individual applicant's record.
Charts come from homecredit/figures.py, the same code that draws the README's PNGs.

Usage:
    .venv/bin/python -m notebooks.build_notebook       (from the repository root; writes and executes the notebook)
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

CELLS = [
    md("""# Home Credit default risk: report notebook

This notebook follows the README with the tables and charts behind each statement. Small tables are computed here
from the processed data; results of long runs are read from `analysis_output/` (each file's script is listed in
`OUTPUTS.md`). Only aggregated numbers are shown: the competition data may not be redistributed.

The final model is **LightGBM v3, tuned, with loan price, monotone** (`lgbm_v3c_sh90_lr02_x3_tuned_mono`): feature
table v3c (716 v3 candidate columns plus 5 estimated loan-price columns), in-fold SHAP selection at 90% (about 290
columns per fold), no gender or family status, 4 tuned settings, learning rate 0.02, average of 3 fits, monotone
constraints on a list of 28 columns (14 of them survive the SHAP cut and are used). Earlier versions (v1, v2) are shown where the README tells their part of the story."""),
    code("""import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))  # the repository root, so the project's own modules import
import pandas as pd
import io

import matplotlib.pyplot as plt
from IPython.display import Image, display

from homecredit import figures
from homecredit.data import read_raw
from homecredit.paths import Paths

P = Paths()
R = P.results
FINAL = "lgbm_v3c_sh90_lr02_x3_tuned_mono"
figures.use("v3m")
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)


def show(fig):
    \"\"\"Show a chart from homecredit/figures.py as a PNG image (the same rendering as the README files).\"\"\"
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return Image(buf.getvalue())


def cmp(a, b, suffix=""):
    \"\"\"One paired comparison file (experiments/compare.py).\"\"\"
    return pd.read_csv(R / "comparisons" / f"{a}_vs_{b}{suffix}.csv")"""),
    md("## 1. The data\nComputed live from the Parquet copies of the Kaggle files."),
    code("""tr = read_raw(P, "application_train", ["TARGET", "NAME_CONTRACT_TYPE"])
te = read_raw(P, "application_test", ["NAME_CONTRACT_TYPE"])
print(f"train applicants {len(tr):,}; test applicants {len(te):,}; "
      f"payment difficulties in train {100 * tr.TARGET.mean():.2f}%")
mix = pd.DataFrame({"train %": tr.NAME_CONTRACT_TYPE.value_counts(normalize=True) * 100,
                    "test %": te.NAME_CONTRACT_TYPE.value_counts(normalize=True) * 100}).round(2)
display(mix)"""),
    md("""## 2. First findings: the test set is a different mix
Single columns that differ between train and test (EDA step 5), the loan-term detail, how well a model can tell
test rows from train rows (adversarial validation; v1 and the v3b table), and the estimated term and rate of the
loan applied for (`pipeline/make_features_rate.py`): test loans are shorter and dearer."""),
    code("""display(pd.read_csv(R / "eda" / "drift_details.csv"))
display(pd.read_csv(R / "eda" / "feature_drift_loan_terms.csv"))
for f in ["feature_adversarial_summary.csv", "feature_adversarial_summary_v3b.csv"]:
    adv = pd.read_csv(R / "eda" / f)
    display(adv[adv.model.isna()])
display(pd.read_csv(R / "features" / "rate_term_model.csv"))
display(pd.read_csv(R / "features" / "rate_drift.csv"))"""),
    md("""## 3. Validation and noise
5 stratified folds x 5 seeds. `fold_sd` is how much one fold's AUC moves; the paired rule (gain over twice its
bootstrap SE and at least 18 of 25 wins) is applied in `experiments/compare.py`. Settings chosen on seed 1 are
judged on seeds 0, 2, 3 and 4 only (files ending in `_seeds0234`, at least 15 of 20 wins)."""),
    code("""display(pd.read_csv(R / "validation" / "noise_summary.csv"))"""),
    md("## 4. Results: every model"),
    code("""s = pd.read_csv(R / "model_summary_v3m.csv")
display(s[["model", "family", "features", "sensitive_columns_used", "folds", "cv_auc_all", "cv_auc_cash", "note"]])
show(figures.auc_by_model(P))"""),
    md("Each step of the main path, as paired comparisons (cash loans):"),
    code("""steps = [("logreg_baseline", "scorecard_app", ""), ("scorecard_app", "scorecard_full", ""),
         ("scorecard_full", "lgbm_v1", ""), ("lgbm_v1_no_gender_family", "lgbm_v2_combo", ""),
         ("lgbm_v2_combo", "lgbm_v2_combo_lr02", ""), ("lgbm_v2_combo_lr02", "lgbm_v2_combo_lr02_x3", ""),
         ("lgbm_v2_combo_lr02_x3", "lgbm_v3b_sh90_lr02_x3", ""),
         ("lgbm_v3b_sh90_lr02_x3", "lgbm_v3b_sh90_lr02_x3_tuned", "_seeds0234"),
         ("lgbm_v3b_sh90_lr02_x3_tuned", "lgbm_v3c_sh90_lr02_x3_tuned", "_seeds0234"),
         ("lgbm_v3c_sh90_lr02_x3_tuned", FINAL, "")]
display(pd.concat([cmp(a, b, x).query("scope == 'cash'") for a, b, x in steps], ignore_index=True))"""),
    md("""Choices inside v3, all on seed 1 (exploratory, 5 folds): the SHAP share (smallest share within noise of all
columns, rule fixed beforehand), the tuning search (20 random trials plus the current settings as trial 0), and the
model without the three external scores."""),
    code("""display(pd.concat([cmp("lgbm_v3b_all_s1", f"lgbm_v3b_sh{k}_s1") for k in (90, 95, 99)], ignore_index=True))
display(pd.read_csv(R / "tuning" / "v3b_sh90_trials.csv").sort_values("auc_all", ascending=False))
display(cmp("lgbm_v3b_sh90_s1", "lgbm_v3b_sh90_noext_s1"))"""),
    md("""Is the loan-price gain smaller on test-like applicants? (`experiments/test_like_gain.py`; out-of-fold, no
model trained.)"""),
    code("""display(pd.read_csv(R / "comparisons" /
                    "test_like_gain_lgbm_v3b_sh90_lr02_x3_tuned_vs_lgbm_v3c_sh90_lr02_x3_tuned.csv"))"""),
    md("""## 5. Monotone constraints
Share of 20,000 training applicants whose predicted risk goes DOWN after one small step the worse way on a column
with a business direction (`experiments/monotone_check.py`, final models trained on all of train). Only the 14
listed columns that the final model uses are checked; related columns that were not on the list stay
unconstrained."""),
    code("""v = pd.read_csv(R / "monotone_v3c" / "violations.csv")
display(v.groupby("model").pct_risk_down_after_worse_value.agg(["count", "mean", "median", "max"]).round(2))
display(v.pivot(index="column", columns="model", values="pct_risk_down_after_worse_value"))"""),
    md("""## 6. The independent check: Kaggle late submissions
Scores typed from the Kaggle submissions page (`analysis_output/kaggle/kaggle_scores*.csv`). Nothing was chosen or
changed because of them. `private_minus_cv_cash` moves from about +0.001 (v1) to -0.0011 (v3b tuned) and then to
-0.0029 when the loan price is added (final model -0.0030). The cause is unknown; optimism from many choices on the
same folds is one guess, but the loan price was one planned idea. The hidden-test score is the better estimate of
the level."""),
    code("""display(pd.read_csv(R / "kaggle" / "kaggle_vs_cv_v3m.csv"))
display(pd.read_csv(R / "kaggle" / "kaggle_steps_v3m.csv"))
show(figures.cv_vs_kaggle(P))"""),
    md("""## 7. Calibration
ECE = average gap between predicted and observed, in points. Measured inside each validation fold and compared with
the value a perfectly calibrated model would show (`floor_pts`); the pooled ECE in `summary_v3m.csv` flatters the
cross-fitted isotonic correction."""),
    code("""display(pd.read_csv(R / "calibration" / "ece_by_fold_v3m.csv"))
c = pd.read_csv(R / "calibration" / "summary_v3m.csv")
display(c[["model", "version", "scope", "auc", "mean_pred_pct", "observed_pct", "slope", "brier", "log_loss"]])
show(figures.calibration(P))"""),
    md("""## 8. What the model relies on (global SHAP)
Final model, 20,000 training applicants (`analysis/global_shap.py`). Share of total mean |SHAP| per reason group,
the largest columns with the direction of their effect (Spearman correlation of value and SHAP), and the mean SHAP
per decile of the largest numeric columns. The group share adds up each column's mean |SHAP|, which favours groups
with many related columns (see the `columns` count); read the shares as rough."""),
    code("""display(pd.read_csv(R / "global_shap_v3m" / "groups.csv"))
display(pd.read_csv(R / "global_shap_v3m" / "top_columns.csv"))
show(figures.shap_groups(P))"""),
    code("""show(figures.shap_dependence(P))"""),
    md("""## 9. Reason codes
At a cut-off that declines about 6%. Top-3 reason groups per declined applicant (SHAP for LightGBM; points below
the approved average for the scorecard), and the counterfactual test: replace one reason group with an approved
applicant's values and re-score. `top4` (the next reason) is a closer control than a random unused group."""),
    code("""rs = pd.read_csv(R / "reasons_v3m_cut06" / "top_reason_share.csv")
display(rs[rs.model == "lightgbm (SHAP)"].head(8))
display(pd.read_csv(R / "reasons_v3m_cut06" / "counterfactual_summary.csv"))
display(pd.read_csv(R / "reasons_v3m_cut06" / "sensitive_reasons.csv"))
show(figures.reason_counterfactual(P))"""),
    md("""## 10. Error analysis (final model)
AUC inside each group and its share of the wrongly ordered pairs; a `missed_ratio` above 1 means the group's
defaulters cost more than their share."""),
    code("""seg = pd.read_csv(R / "errors_v3m" / "segments.csv")
display(seg[seg.segment.isin(["age band", "income type", "loan term band (annuity / credit quintile)"])][
    ["segment", "group", "applicants", "default_rate_pct", "auc_within", "missed_ratio", "false_alarm_ratio"]])
display(pd.read_csv(R / "validation" / "duplicate_check.csv"))"""),
    md("""## 11. Fairness
Accuracy cost of leaving gender and family out at the final setting; approval gaps at a stricter rule (decline the
riskiest 15%); proxies (how well the final model's candidate columns reveal each attribute); and the direct effect
of age and gender at a 6% decline rule. The direct test was measured on the untuned v3 setting (fairness_v3), not
on the final model. These tables use the raw model scores. Approval by group at the break-even rule is in
section 12, where predicted risks are the calibrated (isotonic) probabilities used for money."""),
    code("""display(pd.read_csv(R / "fairness_v3m" / "accuracy_cost.csv").query("scope == 'cash'"))
display(pd.read_csv(R / "fairness_v3m" / "disparity_summary.csv"))
display(pd.read_csv(R / "fairness_v3m" / "proxy_check.csv"))
display(pd.read_csv(R / "fairness_v3" / "counterfactual_age_gender.csv"))"""),
    md("""## 12. Business: choosing the cut-off
Assumed margin 20% of the credit for a repaid loan, loss 60% for a loan with difficulties (not in the data).
Profit is a back-test on past (approved) applicants, in the data's (undisclosed) currency per applicant.
Probabilities are the cross-fitted isotonic ones. The tuning step's money difference (+30, interval -22 to 84) is in
`business_v3t/differences.csv`."""),
    code("""display(pd.read_csv(R / "business_v3m" / "policies.csv"))
display(pd.read_csv(R / "business_v3m" / "differences.csv"))
sens = pd.read_csv(R / "business_v3m" / "sensitivity.csv")
display(sens[sens.model == FINAL])
show(figures.profit_curve(P))"""),
    md("Score bands, KS, the swap-set between LightGBM and the scorecard, and population stability (PSI)."),
    code("""for f in ["score_bands", "ks", "swap_set", "psi"]:
    display(pd.read_csv(R / "credit_v3m" / f"{f}.csv"))"""),
    code("""display(pd.read_csv(R / "business_v3m" / "policy_groups.csv").query("model == @FINAL"))
show(figures.approval_by_group(P))"""),
    md("""## 13. Technical details
- **Code layout:** `homecredit/` holds the shared code (paths, data loading, the one evaluation module
  `evaluate.py`, model code in `models/`, charts in `figures.py`); `eda/`, `pipeline/` (data, feature tables,
  model training, submissions), `experiments/` (noise, paired comparisons, tuning, monotone check, Kaggle check,
  duplicate check, reproducibility) and `analysis/` (calibration, reason codes, global SHAP, error analysis,
  fairness, proxies, business, credit tables, charts). Scripts run as modules from the repository root, e.g.
  `.venv/bin/python -m pipeline.make_features`. `run_all.sh` runs v1 and v2; `run_v3.sh` runs v3 up to the final
  model and its analyses.
- **Final model:** LightGBM on feature table v3c (`pipeline/make_features_v3.py --fixed` plus
  `pipeline/make_features_rate.py`), candidate groups `win, stk, term, agg, last, trend, load, rate`, without gender,
  family status, the bureau's monthly history and 12 other columns that separate test from train (bureau instalment amounts
  and closing dates, and 2 debt-burden ratios built on them);
  in-fold SHAP selection keeping 90% of the total mean |SHAP|; settings from `analysis_output/tuning/v3b_sh90_best.json`
  (num_leaves 30, min_child_samples 174, feature_fraction 0.24, lambda_l2 6.7); monotone constraints from
  `pipeline/monotone_v3.json` (method "intermediate"); learning rate 0.02, early stopping on an inner 10% of each
  training fold, average of 3 fits.
- **Adoption rule:** paired comparison on the same folds (`experiments/compare.py`); gain more than 2 x the bootstrap
  standard error of the gain (one seed's out-of-fold predictions, 200 resamples) and at least 18 of 25 wins (15 of
  20 when seed 1 chose the setting). Exploratory one-seed runs compare 5 folds and need 4 of 5 wins before a
  5-seed confirmation.
- **Reason codes:** SHAP values of the 3 averaged models are averaged in log-odds, while the prediction averages
  probabilities (a small mismatch). The counterfactual swaps a whole reason group with a random approved applicant's
  values (5 donors), so it tests faithfulness to the model, not real-world causes. The 5 loan-price columns count
  as "Loan size and terms vs income".
- **For scale (Kaggle leaderboard, not a script):** the competition's winning team scored 0.806 on the private
  leaderboard with, by its public write-up, hundreds to thousands of features and about 90 stacked models.
- **Reproducibility:** `run_all.sh` and `run_v3.sh` rerun into a separate folder (about 11 hours) and compared
  file by file with `experiments/check_reproduced.py` (`analysis_output/reproducibility/v3_full_rerun.csv`). Of
  443 files in both runs, 400 match byte for byte and 34 apart from their run-time columns. The other 9: the v3b
  drift check (`adversarial_v3b.parquet` and its two summaries; adversarial AUC 0.98236 vs 0.98237, cause not
  checked, read by no later step) and 6 files that list hand-made runs (4 submission manifests, 2 model
  summaries). Files only in the saved results: the first v3 try (`FIRST_TRY=1`), the runs in `OUTPUTS.md` "Run by
  hand", the Kaggle scores typed by hand and superseded first-version files. The earlier v2-only rerun:
  `v2_full_rerun.csv`."""),
    md("""## 14. Limits
- The target's exact definition (X days late on one of the first Y instalments) is not disclosed.
- Validation uses random folds; there is no time order in the data.
- Only previously approved loans are in the data; the profit numbers are a back-test with assumed margins and losses.
- Only LightGBM was tuned; the other families used common settings.
- The direct age and gender test was measured on the untuned v3 setting, not on the final model.
- The loan-price gain on Kaggle's hidden test was smaller than in validation (section 6).
- If Home Credit priced loans by risk, the estimated rate partly carries the lender's earlier risk view; a new
  pricing policy would change what it means.
- The SHAP cut, tuning, loan price and monotone rule were all judged on the same 25 folds, so the final validation
  score may be a little optimistic; most of the gap to Kaggle appeared at the loan-price step (section 6).
- The business back-test uses the same 20% margin for every loan, although declined loans tend to be dearer; a
  flat margin likely overstates the gain from declining (not measured).
- The target is early payment difficulty with an undisclosed threshold, not a regulatory default definition.
- The test set is a different mix of loans; without dates I cannot tell whether it is also a later period.
- Possible repeat applicants: grouped folds did not lower the score (section 10)."""),
]

def main() -> None:
    nb = nbformat.v4.new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    out = HERE / "home_credit_report.ipynb"
    nbformat.write(nb, out)
    print(f"written and executed: {out}")


if __name__ == "__main__":
    main()
