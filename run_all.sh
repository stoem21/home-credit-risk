#!/usr/bin/env bash
# Runs the whole project in order, from the Kaggle CSVs in data/raw/ to every result in analysis_output/.
#
# Usage (from anywhere):
#   ./run_all.sh               writes processed/ and analysis_output/ in the repository
#   ./run_all.sh /tmp/hc_run   writes them under /tmp/hc_run instead (smoke test, reproducibility check)
#
# The training scripts refuse to overwrite a model that already has results, so a second run in the repository
# stops at the first existing model; rerun into a new folder instead.
# Run times on a 10-core laptop are logged in EXPERIMENTS.md; the whole chain takes roughly 10 minutes.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
ROOT_ARGS=()
if [[ $# -gt 0 ]]; then ROOT_ARGS=(--out-root "$1"); fi
run() { echo "== $*"; "$PY" -m "$@" ${ROOT_ARGS[@]+"${ROOT_ARGS[@]}"}; }

# Data preparation
run pipeline.make_parquet

# EDA (project step 2)
run eda.keys
run eda.target_missing
run eda.quirks
run eda.drift_leakage
run eda.side_tables
run eda.sensitive

# Validation design and noise (step 3); also the logistic baseline (step 4)
run pipeline.make_folds
run experiments.noise_check

# Features and models (step 4)
run pipeline.make_features
run pipeline.train_scorecard --variant app
run pipeline.train_scorecard --variant full
run pipeline.train_lgbm
run experiments.compare logreg_baseline scorecard_app
run experiments.compare logreg_baseline scorecard_full
run experiments.compare scorecard_app scorecard_full
run experiments.compare scorecard_full lgbm_v1
run experiments.compare logreg_baseline lgbm_v1

# Calibration (step 5) and reason codes (step 6)
run analysis.calibration logreg_baseline scorecard_full lgbm_v1
run analysis.reason_codes

# Fairness (step 8): the same models without sensitive columns, paired comparisons, then the analysis
run pipeline.train_lgbm --drop gender --name lgbm_v1_no_gender
run pipeline.train_lgbm --drop gender,family --name lgbm_v1_no_gender_family
run pipeline.train_lgbm --drop gender,family,age --name lgbm_v1_no_sensitive
run pipeline.train_scorecard --variant full --drop gender --name scorecard_full_no_gender
run experiments.compare lgbm_v1 lgbm_v1_no_gender
run experiments.compare lgbm_v1 lgbm_v1_no_gender_family
run experiments.compare lgbm_v1 lgbm_v1_no_sensitive
run experiments.compare scorecard_full scorecard_full_no_gender
run analysis.fairness
run pipeline.make_submissions

# Phase 2 checks: features in train vs test (needs the saved OOF of the three models), then error analysis
run eda.feature_drift
run analysis.error_analysis
# eda.one_applicant prints one applicant's records to the terminal and saves nothing; run it by hand.

# LightGBM v2 (agreed plan: top-solution ideas judged on our CV). Reference: lgbm_v1_no_gender_family.
run pipeline.make_features_v2
run eda.feature_drift --features v2
for g in win stk term ext; do
  run pipeline.train_lgbm --features v2 --add "$g" --drop gender,family --name "lgbm_v2_$g"
  run experiments.compare lgbm_v1_no_gender_family "lgbm_v2_$g"
done
run pipeline.train_lgbm --features v2 --add stk --exclude stk_bb_ --drop gender,family --name lgbm_v2_stk_nobb
run experiments.compare lgbm_v1_no_gender_family lgbm_v2_stk_nobb
run experiments.compare lgbm_v2_stk_nobb lgbm_v2_stk
run pipeline.train_lgbm --features v2 --add win,stk,term --exclude stk_bb_ --drop gender,family --name lgbm_v2_combo
run experiments.compare lgbm_v1_no_gender_family lgbm_v2_combo
run experiments.compare lgbm_v2_stk lgbm_v2_combo
run pipeline.train_lgbm --features v2 --add win,stk,term --exclude stk_bb_ --drop gender,family --learning-rate 0.02 --name lgbm_v2_combo_lr02
run experiments.compare lgbm_v2_combo lgbm_v2_combo_lr02
run pipeline.train_lgbm --features v2 --add win,stk,term --exclude stk_bb_ --drop gender,family --learning-rate 0.02 --n-models 3 --name lgbm_v2_combo_lr02_x3
run experiments.compare lgbm_v2_combo_lr02 lgbm_v2_combo_lr02_x3
run experiments.compare lgbm_v1 lgbm_v2_combo_lr02_x3
run experiments.compare scorecard_full_no_gender lgbm_v2_combo_lr02_x3
# other families, 1 seed (exploratory)
run pipeline.train_family --model xgb --seeds 0 --features v2 --add win,stk,term --exclude stk_bb_ --drop gender,family --name xgb_v2_combo_s0
run experiments.compare lgbm_v2_combo xgb_v2_combo_s0
run pipeline.train_family --model mlp --seeds 0 --features v2 --add win,stk,term --exclude stk_bb_ --drop gender,family --name mlp_v2_combo_s0
run experiments.compare lgbm_v2_combo mlp_v2_combo_s0
# fairness reference with gender and family, calibration, fairness and reason codes for the v2 candidate
run pipeline.train_lgbm --features v2 --add win,stk,term --exclude stk_bb_ --name lgbm_v2_combo_with_gender
run experiments.compare lgbm_v2_combo_with_gender lgbm_v2_combo
run analysis.calibration lgbm_v1 lgbm_v2_combo_lr02_x3 --tag v2
run analysis.fairness --reference lgbm_v2_combo_with_gender --variants lgbm_v2_combo,lgbm_v2_combo_lr02_x3 --tag v2
run analysis.error_analysis --model lgbm_v2_combo_lr02_x3 --tag v2
run analysis.proxy_check
run experiments.duplicate_check
run analysis.reason_codes --features v2 --add win,stk,term --exclude stk_bb_ --drop gender,family --learning-rate 0.02 --n-models 3 --tag v2
run pipeline.make_submissions --tag v2
# reason codes again at the business cut-off (decline about 6%), with the 4th-ranked reason as a closer control
run analysis.reason_codes --features v2 --add win,stk,term --exclude stk_bb_ --drop gender,family --learning-rate 0.02 --n-models 3 --decline-rate 0.06 --with-fourth --tag v2_cut06

# Business cut-off (step 7)
run analysis.business
run experiments.model_summary
# Kaggle check (needs analysis_output/kaggle/kaggle_scores.csv, typed from Kaggle by hand)
run experiments.kaggle_check --tag v2

# Charts for the README and the report notebook
run analysis.figures
"$PY" -m notebooks.build_notebook   # reads the results in the repository, not an --out-root folder

# Credit tables and the direct effect of age and gender (added after the second review)
run analysis.credit_tables
run analysis.sensitive_counterfactual
