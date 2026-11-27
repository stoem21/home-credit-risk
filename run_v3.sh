#!/usr/bin/env bash
# LightGBM v3: broader features, two bureau data fixes, a drift-based exclusion list and in-fold SHAP feature
# reduction by a cumulative-share rule (EXPERIMENTS.md entries 26 and 31). Needs the outputs of run_all.sh first
# (feature table v2, folds, lgbm_v2_* models and their result files).
#
# Usage: ./run_v3.sh [OUT_ROOT]
#   FIRST_TRY=1 ./run_v3.sh ...  also reruns the first v3 attempt (before the bureau fixes, SHAP top K; entry 26),
#                                 whose results are kept for the log but not used in the README.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
ROOT_ARGS=()
if [[ $# -gt 0 ]]; then ROOT_ARGS=(--out-root "$1"); fi
run() { echo "== $*"; "$PY" -m "$@" ${ROOT_ARGS[@]+"${ROOT_ARGS[@]}"}; }

# ---- first attempt (superseded) ----
if [[ "${FIRST_TRY:-0}" == "1" ]]; then
  V3=(--features v3 --add win,stk,term,agg,last,trend,load --exclude stk_bb_ --drop gender,family)
  run pipeline.make_features_v3
  run pipeline.train_lgbm "${V3[@]}" --seeds 0 --name lgbm_v3_all_s0
  run experiments.compare lgbm_v2_combo lgbm_v3_all_s0
  for K in 100 200 350; do
    run pipeline.train_lgbm "${V3[@]}" --seeds 0 --shap-top "$K" --name "lgbm_v3_shap${K}_s0"
    run experiments.compare lgbm_v3_all_s0 "lgbm_v3_shap${K}_s0"
  done
  run pipeline.train_lgbm "${V3[@]}" --name lgbm_v3_all
  run pipeline.train_lgbm "${V3[@]}" --shap-top 350 --name lgbm_v3_shap350
  run experiments.compare lgbm_v2_combo lgbm_v3_all
  run experiments.compare lgbm_v2_combo lgbm_v3_shap350
  run experiments.compare lgbm_v3_all lgbm_v3_shap350
  run pipeline.train_lgbm "${V3[@]}" --shap-top 350 --learning-rate 0.02 --n-models 3 --name lgbm_v3_shap350_lr02_x3
  run experiments.compare lgbm_v3_shap350 lgbm_v3_shap350_lr02_x3
  run experiments.compare lgbm_v2_combo_lr02_x3 lgbm_v3_shap350_lr02_x3
fi

# ---- v3b: corrected bureau features, drift check ----
run pipeline.make_features_v3 --fixed
run eda.feature_drift --features v3b
# columns left out after the drift check (bureau_balance summaries and bureau instalment amounts)
EX=stk_bb_,agg_bur_AMT_ANNUITY,agg_bur_act_AMT_ANNUITY,agg_bur_act_DAYS_ENDDATE_FACT,load_total_annuity,load_existing_annuity
B=(--features v3b --add win,stk,term,agg,last,trend,load --exclude "$EX")

# ---- choosing the SHAP share on seed 1 (exploratory; rule fixed beforehand: smallest share within noise) ----
run pipeline.train_lgbm "${B[@]}" --drop gender,family --seeds 1 --name lgbm_v3b_all_s1
for S in 90 95 99; do
  run pipeline.train_lgbm "${B[@]}" --drop gender,family --seeds 1 --shap-share "0.$S" --name "lgbm_v3b_sh${S}_s1"
  run experiments.compare lgbm_v3b_all_s1 "lgbm_v3b_sh${S}_s1"
done
run experiments.compare lgbm_v3b_sh90_s1 lgbm_v3b_sh95_s1
run experiments.compare lgbm_v2_combo lgbm_v3b_all_s1

# ---- final setting on seed 1, then 5 seeds ----
FINAL=("${B[@]}" --drop gender,family --shap-share 0.90 --learning-rate 0.02 --n-models 3)
run pipeline.train_lgbm "${FINAL[@]}" --seeds 1 --name lgbm_v3b_sh90_lr02_x3_s1
run experiments.compare lgbm_v2_combo_lr02_x3 lgbm_v3b_sh90_lr02_x3_s1
run pipeline.train_lgbm "${B[@]}" --seeds 1 --shap-share 0.90 --name lgbm_v3b_sh90_with_gender_s1
run experiments.compare lgbm_v3b_sh90_with_gender_s1 lgbm_v3b_sh90_s1
# without the 3 external credit scores (entry 31; read by model_summary --v3t)
run pipeline.train_lgbm --features v3b --add win,stk,term,agg,last,trend,load --exclude "$EX,EXT_SOURCE_" \
  --drop gender,family --seeds 1 --shap-share 0.90 --name lgbm_v3b_sh90_noext_s1
run experiments.compare lgbm_v3b_sh90_s1 lgbm_v3b_sh90_noext_s1
run pipeline.train_lgbm "${FINAL[@]}" --name lgbm_v3b_sh90_lr02_x3
run experiments.compare lgbm_v2_combo_lr02_x3 lgbm_v3b_sh90_lr02_x3
run pipeline.train_lgbm "${B[@]}" --drop gender,family --shap-share 0.90 --name lgbm_v3b_sh90
run pipeline.train_lgbm "${B[@]}" --shap-share 0.90 --name lgbm_v3b_sh90_with_gender
run experiments.compare lgbm_v3b_sh90_with_gender lgbm_v3b_sh90
run experiments.compare lgbm_v3b_sh90 lgbm_v3b_sh90_lr02_x3

# ---- downstream analyses on the v3 model (same steps as v2, written to *_v3 folders) ----
run analysis.calibration lgbm_v2_combo_lr02_x3 lgbm_v3b_sh90_lr02_x3 --tag v3
run analysis.fairness --reference lgbm_v3b_sh90_with_gender --variants lgbm_v3b_sh90,lgbm_v3b_sh90_lr02_x3 --tag v3
run analysis.error_analysis --model lgbm_v3b_sh90_lr02_x3 --tag v3
run analysis.proxy_check "${B[@]}" --tag v3
run analysis.sensitive_counterfactual "${B[@]}" --shap-share 0.9 --age-model lgbm_v3b_sh90 --gender-model lgbm_v3b_sh90_with_gender --tag v3
run analysis.reason_codes "${FINAL[@]}" --decline-rate 0.06 --with-fourth --tag v3_cut06
run pipeline.make_submissions --tag v3
run analysis.business --main lgbm_v3b_sh90_lr02_x3 --also lgbm_v2_combo_lr02_x3 --tag v3
run analysis.credit_tables --main lgbm_v3b_sh90_lr02_x3 --features v3b --tag v3
run experiments.model_summary --v3
# Kaggle check (needs analysis_output/kaggle/kaggle_scores_v3.csv, typed from Kaggle by hand)
if [[ -f analysis_output/kaggle/kaggle_scores_v3.csv ]]; then run experiments.kaggle_check --tag v3 --v3; fi
run analysis.global_shap --model lgbm_v3b_sh90_lr02_x3 --features v3b --tag v3
run analysis.figures --version v3

# ---- bounded tuning of four LightGBM settings (EXPERIMENTS.md entry 32) ----
run experiments.tune_lgbm "${B[@]}" --drop gender,family --shap-share 0.9 --name v3b_sh90
run pipeline.train_lgbm "${FINAL[@]}" --params "${1:-.}/analysis_output/tuning/v3b_sh90_best.json" --name lgbm_v3b_sh90_lr02_x3_tuned
run experiments.compare lgbm_v3b_sh90_lr02_x3 lgbm_v3b_sh90_lr02_x3_tuned --seeds 0,2,3,4
run experiments.compare lgbm_v3b_sh90_lr02_x3 lgbm_v3b_sh90_lr02_x3_tuned  # all 25 folds, for information

# ---- follow-up analyses for the tuned model (adopted; written to *_v3t folders) ----
T=lgbm_v3b_sh90_lr02_x3_tuned
PRM="${1:-.}/analysis_output/tuning/v3b_sh90_best.json"
run pipeline.make_submissions --tag v3t
run analysis.global_shap --model "$T" --features v3b --tag v3t
run analysis.calibration "$T" --tag v3t
run analysis.business --main "$T" --also lgbm_v3b_sh90_lr02_x3,lgbm_v2_combo_lr02_x3 --tag v3t
run analysis.credit_tables --main "$T" --features v3b --tag v3t
run analysis.error_analysis --model "$T" --tag v3t
run pipeline.train_lgbm "${B[@]}" --shap-share 0.90 --learning-rate 0.02 --n-models 3 --params "$PRM" --name "${T}_with_gender"
run experiments.compare "${T}_with_gender" "$T"
run experiments.model_summary --v3t
run analysis.fairness --reference "${T}_with_gender" --variants "$T" --tag v3t
run analysis.reason_codes "${FINAL[@]}" --params "$PRM" --decline-rate 0.06 --with-fourth --tag v3t_cut06
run pipeline.make_submissions --tag v3t_2
# Kaggle check (needs analysis_output/kaggle/kaggle_scores_v3t.csv, typed from Kaggle by hand)
if [[ -f analysis_output/kaggle/kaggle_scores_v3t.csv ]]; then run experiments.kaggle_check --tag v3t --v3t; fi
run analysis.figures --version v3t

# ---- estimated term and rate (rate_, entry 33) and monotone constraints (entry 34); final model ----
run pipeline.make_features_rate
C=(--features v3c --add win,stk,term,agg,last,trend,load,rate --exclude "$EX")
run pipeline.train_lgbm "${B[@]}" --drop gender,family --seeds 1 --shap-share 0.90 --params "$PRM" --name lgbm_v3b_sh90_tuned_s1
run pipeline.train_lgbm "${C[@]}" --drop gender,family --seeds 1 --shap-share 0.90 --params "$PRM" --name lgbm_v3c_sh90_tuned_s1
run experiments.compare lgbm_v3b_sh90_tuned_s1 lgbm_v3c_sh90_tuned_s1
CF=("${C[@]}" --shap-share 0.90 --learning-rate 0.02 --n-models 3 --params "$PRM")
run pipeline.train_lgbm "${CF[@]}" --drop gender,family --name lgbm_v3c_sh90_lr02_x3_tuned
run experiments.compare lgbm_v3b_sh90_lr02_x3_tuned lgbm_v3c_sh90_lr02_x3_tuned --seeds 0,2,3,4
run experiments.compare lgbm_v3b_sh90_lr02_x3_tuned lgbm_v3c_sh90_lr02_x3_tuned  # all 25 folds, for information
run experiments.test_like_gain
run experiments.monotone_check --models lgbm_v3b_sh90_lr02_x3_tuned --features v3b --tag v3t
M=lgbm_v3c_sh90_lr02_x3_tuned_mono
run pipeline.train_lgbm "${CF[@]}" --drop gender,family --monotone pipeline/monotone_v3.json --name "$M"
run experiments.compare lgbm_v3c_sh90_lr02_x3_tuned "$M"
run experiments.monotone_check --models lgbm_v3c_sh90_lr02_x3_tuned,"$M" --features v3c --tag v3c
run pipeline.train_lgbm "${CF[@]}" --monotone pipeline/monotone_v3.json --name "${M}_with_gender"
run experiments.compare "${M}_with_gender" "$M"
run pipeline.make_submissions --tag v3m
run analysis.global_shap --model "$M" --features v3c --tag v3m
run analysis.calibration "$M" --tag v3m
run analysis.business --main "$M" --also lgbm_v3c_sh90_lr02_x3_tuned,lgbm_v3b_sh90_lr02_x3_tuned,lgbm_v2_combo_lr02_x3 --tag v3m
run analysis.credit_tables --main "$M" --features v3c --tag v3m
run analysis.error_analysis --model "$M" --tag v3m
run analysis.proxy_check "${C[@]}" --tag v3m
run experiments.model_summary --v3m
run analysis.fairness --reference "${M}_with_gender" --variants "$M" --tag v3m
run analysis.reason_codes "${CF[@]}" --drop gender,family --monotone pipeline/monotone_v3.json --decline-rate 0.06 --with-fourth --tag v3m_cut06
run experiments.kaggle_check --tag v3m --v3m
run analysis.figures --version v3m
