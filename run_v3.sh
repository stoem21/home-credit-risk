#!/usr/bin/env bash
# LightGBM v3 (in progress, not part of the README results yet): broader features and in-fold SHAP feature
# reduction. Needs the outputs of run_all.sh first (feature table v2, folds, lgbm_v2_* models).
# Usage: ./run_v3.sh [OUT_ROOT]
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
ROOT_ARGS=()
if [[ $# -gt 0 ]]; then ROOT_ARGS=(--out-root "$1"); fi
run() { echo "== $*"; "$PY" -m "$@" ${ROOT_ARGS[@]+"${ROOT_ARGS[@]}"}; }

V3="--features v3 --add win,stk,term,agg,last,trend,load --exclude stk_bb_ --drop gender,family"
run pipeline.make_features_v3
run pipeline.train_lgbm $V3 --seeds 0 --name lgbm_v3_all_s0
run experiments.compare lgbm_v2_combo lgbm_v3_all_s0
for K in 100 200 350; do
  run pipeline.train_lgbm $V3 --seeds 0 --shap-top "$K" --name "lgbm_v3_shap${K}_s0"
  run experiments.compare lgbm_v3_all_s0 "lgbm_v3_shap${K}_s0"
done
run pipeline.train_lgbm $V3 --name lgbm_v3_all
run pipeline.train_lgbm $V3 --shap-top 350 --name lgbm_v3_shap350
run experiments.compare lgbm_v2_combo lgbm_v3_all
run experiments.compare lgbm_v2_combo lgbm_v3_shap350
run experiments.compare lgbm_v3_all lgbm_v3_shap350
run pipeline.train_lgbm $V3 --shap-top 350 --learning-rate 0.02 --n-models 3 --name lgbm_v3_shap350_lr02_x3
run experiments.compare lgbm_v3_shap350 lgbm_v3_shap350_lr02_x3
run experiments.compare lgbm_v2_combo_lr02_x3 lgbm_v3_shap350_lr02_x3

