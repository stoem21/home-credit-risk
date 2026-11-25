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

The chosen model is **LightGBM v2** (`lgbm_v2_combo_lr02_x3`): 186 columns, no gender or family status, learning
rate 0.02, average of 3 fits."""),
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
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)


def show(fig):
    \"\"\"Show a chart from homecredit/figures.py as a PNG image (the same rendering as the README files).\"\"\"
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return Image(buf.getvalue())"""),
    md("## 1. The data\nComputed live from the Parquet copies of the Kaggle files."),
    code("""tr = read_raw(P, "application_train", ["TARGET", "NAME_CONTRACT_TYPE"])
te = read_raw(P, "application_test", ["NAME_CONTRACT_TYPE"])
print(f"train applicants {len(tr):,}; test applicants {len(te):,}; "
      f"payment difficulties in train {100 * tr.TARGET.mean():.2f}%")
mix = pd.DataFrame({"train %": tr.NAME_CONTRACT_TYPE.value_counts(normalize=True) * 100,
                    "test %": te.NAME_CONTRACT_TYPE.value_counts(normalize=True) * 100}).round(2)
display(mix)"""),
    md("""## 2. First findings: the test set is a different mix
Single columns that differ between train and test (EDA step 5), the loan-term detail, and how well a model can
tell test rows from train rows (adversarial validation)."""),
    code("""display(pd.read_csv(R / "eda" / "drift_details.csv"))
display(pd.read_csv(R / "eda" / "feature_drift_loan_terms.csv"))
adv = pd.read_csv(R / "eda" / "feature_adversarial_summary.csv")
display(adv[adv.model.isna()])"""),
    md("""## 3. Validation and noise
5 stratified folds x 5 seeds. `fold_sd` is how much one fold's AUC moves; the paired rule (gain over twice its
bootstrap SE and at least 18 of 25 wins) is applied in `experiments/compare.py`."""),
    code("""display(pd.read_csv(R / "validation" / "noise_summary.csv"))"""),
    md("## 4. Results: every model"),
    code("""s = pd.read_csv(R / "model_summary.csv")
display(s[["model", "family", "features", "sensitive_columns_used", "folds", "cv_auc_all", "cv_auc_cash", "note"]])
show(figures.auc_by_model(P))"""),
    md("Each step of the main path, as paired comparisons on the same 25 folds (cash loans):"),
    code("""steps = [("logreg_baseline", "scorecard_app"), ("scorecard_app", "scorecard_full"),
         ("scorecard_full", "lgbm_v1"), ("lgbm_v1_no_gender_family", "lgbm_v2_combo"),
         ("lgbm_v2_combo", "lgbm_v2_combo_lr02"), ("lgbm_v2_combo_lr02", "lgbm_v2_combo_lr02_x3"),
         ("lgbm_v1", "lgbm_v2_combo_lr02_x3")]
rows = [pd.read_csv(R / "comparisons" / f"{a}_vs_{b}.csv").query("scope == 'cash'") for a, b in steps]
display(pd.concat(rows, ignore_index=True))"""),
    md("""## 5. The independent check: Kaggle late submissions
Scores typed from the Kaggle submissions page (`analysis_output/kaggle/kaggle_scores.csv`). Nothing was chosen or
changed because of them."""),
    code("""display(pd.read_csv(R / "kaggle" / "kaggle_vs_cv.csv"))
display(pd.read_csv(R / "kaggle" / "kaggle_steps_v2.csv"))
show(figures.cv_vs_kaggle(P))"""),
    md("""## 6. Calibration
ECE = average gap between predicted and observed, in points. Measured inside each validation fold and compared with
the value a perfectly calibrated model would show (`floor_pts`); the pooled ECE in `summary_v2.csv` flatters the
cross-fitted isotonic correction."""),
    code("""display(pd.read_csv(R / "calibration" / "ece_by_fold_v2.csv"))
c = pd.read_csv(R / "calibration" / "summary_v2.csv")
display(c[["model", "version", "scope", "auc", "mean_pred_pct", "observed_pct", "slope", "brier", "log_loss"]])
show(figures.calibration(P))"""),
    md("""## 7. Reason codes
At the proposed cut-off (decline about 6%). Top-3 reason groups per declined applicant (SHAP for LightGBM; points
below the approved average for the scorecard), and the counterfactual test: replace one reason group with an
approved applicant's values and re-score. `top4` (the next reason) is a closer control than a random unused group."""),
    code("""rs = pd.read_csv(R / "reasons_v2_cut06" / "top_reason_share.csv")
display(rs[rs.model == "lightgbm (SHAP)"].head(8))
display(pd.read_csv(R / "reasons_v2_cut06" / "counterfactual_summary.csv"))
display(pd.read_csv(R / "reasons_v2_cut06" / "sensitive_reasons.csv"))
show(figures.reason_counterfactual(P))"""),
    md("""## 8. Error analysis (chosen model)
AUC inside each group and its share of the wrongly ordered pairs; a `missed_ratio` above 1 means the group's
defaulters cost more than their share."""),
    code("""seg = pd.read_csv(R / "errors_v2" / "segments.csv")
display(seg[seg.segment.isin(["age band", "income type"])][
    ["segment", "group", "applicants", "default_rate_pct", "auc_within", "missed_ratio", "false_alarm_ratio"]])
display(pd.read_csv(R / "errors_v2" / "candidate_signals.csv"))
display(pd.read_csv(R / "validation" / "duplicate_check.csv"))"""),
    md("""## 9. Fairness
Accuracy cost of leaving columns out; approval gaps at a stricter rule (decline the riskiest 15%, files fairness*);
proxies; and the direct effect of age and gender at the proposed cut-off (decline about 6%). Approval by group at the
proposed cut-off is in section 10 (business_v2/policy_groups.csv)."""),
    code("""display(pd.read_csv(R / "fairness" / "accuracy_cost.csv").query("scope == 'cash'"))
display(pd.read_csv(R / "fairness_v2" / "accuracy_cost.csv").query("scope == 'cash'"))
display(pd.read_csv(R / "fairness_v2" / "disparity_summary.csv"))
display(pd.read_csv(R / "fairness" / "gender_counterfactual.csv"))
display(pd.read_csv(R / "fairness_v2" / "proxy_check.csv"))
display(pd.read_csv(R / "fairness_v2" / "counterfactual_age_gender.csv"))"""),
    md("""## 10. Business: choosing the cut-off
Assumed margin 20% of the credit for a repaid loan, loss 60% for a loan with difficulties (not in the data).
Profit is a back-test on past (approved) applicants, in the data's (undisclosed) currency per applicant."""),
    code("""display(pd.read_csv(R / "business_v2" / "policies.csv"))
display(pd.read_csv(R / "business_v2" / "differences.csv"))
sens = pd.read_csv(R / "business_v2" / "sensitivity.csv")
display(sens[sens.model == "lgbm_v2_combo_lr02_x3"])
show(figures.profit_curve(P))"""),
    md("Score bands, KS, the swap-set between LightGBM and the scorecard, and population stability (PSI)."),
    code("""for f in ["score_bands", "ks", "swap_set", "psi"]:
    display(pd.read_csv(R / "credit_v2" / f"{f}.csv"))"""),
    code("""display(pd.read_csv(R / "business_v2" / "policy_groups.csv").query("model == 'lgbm_v2_combo_lr02_x3'"))
show(figures.approval_by_group(P))"""),
    md("""## 11. Technical details
- **Code layout:** `homecredit/` holds the shared code (paths, data loading, the one evaluation module
  `evaluate.py`, model code in `models/`, charts in `figures.py`); `eda/`, `pipeline/` (data, feature tables,
  model training, submissions), `experiments/` (noise, paired comparisons, Kaggle check, duplicate check,
  reproducibility) and `analysis/` (calibration, reason codes, error analysis, fairness, proxies, business, credit
  tables, charts). Scripts run as modules from the repository root, e.g. `.venv/bin/python -m pipeline.make_features`.
  `run_all.sh` runs everything in order; `run_v3.sh` holds the v3 work in progress.
- **Chosen model:** LightGBM, 186 columns (v1 plus the `win_`, `stk_` and `term_` groups of
  `pipeline/make_features_v2.py`, without gender, family status and the bureau's monthly history), learning rate
  0.02, early stopping on an inner 10% of each training fold, average of 3 fits (`lgbm_v2_combo_lr02_x3`).
- **Adoption rule:** paired comparison on the same 25 folds (`experiments/compare.py`); gain more than 2 x the
  bootstrap standard error of the gain (seed-0 out-of-fold predictions, 200 resamples) and at least 18 of 25 wins.
  Exploratory one-seed runs compare 5 folds and need 4 of 5 wins; nothing was adopted from them.
- **Calibration:** measured inside each validation fold (section 6). The raw model's average gap is 0.48 points
  against 0.35 from chance alone; cross-fitted isotonic regression brings it to 0.36. A pooled measure across folds
  flatters cross-fitted corrections. Brier score and log loss barely change.
- **Reason codes:** SHAP values of the 3 averaged models are averaged in log-odds, while the prediction averages
  probabilities (a small mismatch). The counterfactual swaps a whole reason group with a random approved applicant's
  values (5 donors), so it tests faithfulness to the model, not real-world causes.
- **Credit tables:** score bands, KS (0.43 for the chosen model vs 0.38 for the scorecard), the swap-set and the
  population stability index are in section 10 (`analysis/credit_tables.py`).
- **For scale (Kaggle leaderboard, not a script):** the competition's winning team scored 0.806 on the private
  leaderboard with, by its public write-up, hundreds to thousands of features and about 90 stacked models."""),
    md("""## 12. Limits
- The target's exact definition (X days late on one of the first Y instalments) is not disclosed.
- Validation uses random folds; there is no time order in the data.
- Only previously approved loans are in the data; the profit numbers are a back-test with assumed margins and losses.
- The fairness cost measured on Kaggle's hidden test (0.003 AUC) is about twice the validation estimate (0.002).
- Possible repeat applicants: grouped folds did not lower the score (section 8).
- A third feature version (v3) with SHAP-based feature reduction is in progress and not part of these results."""),
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
