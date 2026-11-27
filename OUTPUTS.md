# Where each result file comes from

Every file below is written by one script. `run_all.sh` (v1, v2) and `run_v3.sh` (v3 up to the final model) run
them in order, except the few exploratory runs listed under "Run by hand" at the end. Files under `processed/` are
not in the repository (they are large or derived from the Kaggle data), so run the pipeline to recreate them.
Run a script from the repository root with `.venv/bin/python -m <module>`.

## processed/ (not committed)
| File | Written by |
|---|---|
| `raw/*.parquet`, `raw/manifest.csv` | `pipeline.make_parquet` |
| `folds.parquet` | `pipeline.make_folds` |
| `features_v1.parquet` | `pipeline.make_features` |
| `oof/logreg_baseline.parquet` | `experiments.noise_check` |
| `oof/scorecard_app.parquet`, `oof/scorecard_full.parquet`, `preds/scorecard_*_test.parquet` | `pipeline.train_scorecard --variant app` / `full` |
| `oof/lgbm_v1.parquet`, `preds/lgbm_v1_test.parquet`, `lgbm_v1_final.txt` | `pipeline.train_lgbm` |
| `adversarial_v1.parquet` | `eda.feature_drift` |
| `preds/rule_test.parquet`, `preds/logreg_baseline_test.parquet` | `pipeline.make_submissions` |
| `features_v2.parquet` | `pipeline.make_features_v2` |
| `adversarial_v2.parquet` | `eda.feature_drift --features v2` |
| `oof/lgbm_v2_*.parquet`, `preds/lgbm_v2_*_test.parquet`, `lgbm_v2_*_final.txt` | `pipeline.train_lgbm --features v2 ...` (see `run_all.sh`) |
| `features_v3.parquet` | `pipeline.make_features_v3` |
| `oof/lgbm_v3_*.parquet` (`*_s0`: seed 0 only), `preds/lgbm_v3_*_test.parquet` | `pipeline.train_lgbm --features v3 ...` (first v3 try; rerun with `FIRST_TRY=1 ./run_v3.sh`) |
| `features_v3b.parquet` (bureau fixes), `adversarial_v3b.parquet` | `pipeline.make_features_v3 --fixed`, `eda.feature_drift --features v3b` |
| `features_v3c.parquet` (v3b plus the `rate_` loan-price columns) | `pipeline.make_features_rate` |
| `oof/lgbm_v3b_*.parquet`, `oof/lgbm_v3c_*.parquet`, `preds/..._test.parquet`, `..._final.txt` (+ `_1`, `_2`) | `pipeline.train_lgbm` (see `run_v3.sh`; `*_s1`: seed 1 only) |
| `oof/xgb_v2_combo_s0.parquet`, `oof/mlp_v2_combo_s0.parquet` (seed 0 only) | `pipeline.train_family` |
| `oof/lgbm_v1_no_*.parquet`, `preds/lgbm_v1_no_*_test.parquet`, `lgbm_v1_no_*_final.txt` | `pipeline.train_lgbm --drop ... --name ...` |
| `oof/scorecard_full_no_gender.parquet`, `preds/scorecard_full_no_gender_test.parquet` | `pipeline.train_scorecard --variant full --drop gender --name scorecard_full_no_gender` |

## analysis_output/
| Folder / file | Written by |
|---|---|
| `eda/keys_uniqueness.csv`, `join_coverage.csv`, `rows_per_applicant.csv`, `bureau_balance_coverage.csv` | `eda.keys` |
| `eda/target_overall.csv`, `target_by_category.csv`, `missing_by_column.csv`, `missing_groups.csv`, `missing_vs_default.csv` | `eda.target_missing` |
| `eda/numeric_profile.csv`, `placeholder_values.csv`, `quirk_checks.csv` | `eda.quirks` |
| `eda/column_drift_and_target_auc.csv`, `drift_details.csv`, `id_order_check.csv`, `id_auc.csv` | `eda.drift_leakage` |
| `eda/side_time_spans.csv`, `side_status_shares.csv`, `installments_behaviour.csv`, `side_signals_vs_default.csv` | `eda.side_tables` |
| `eda/sensitive_inventory.csv` | `eda.sensitive` |
| `eda/feature_drift.csv`, `feature_drift_loan_terms.csv`, `feature_adversarial_importance.csv`, `feature_adversarial_summary.csv` | `eda.feature_drift` |
| `validation/noise_by_fold.csv`, `noise_summary.csv`, `noise_extra.csv` | `experiments.noise_check` |
| `scorecard/<variant>_table.csv`, `_variables.csv`, `_meta.csv`, `_selection_stability.csv` | `pipeline.train_scorecard` |
| `lgbm/lgbm_v1_folds.csv`, `lgbm_v1_importance.csv` | `pipeline.train_lgbm` |
| `comparisons/<A>_vs_<B>.csv` | `experiments.compare A B` |
| `calibration/summary.csv`, `by_seed.csv`, `<model>_reliability.csv` | `analysis.calibration` |
| `reasons/reason_groups.csv`, `top_reason_share.csv`, `sensitive_reasons.csv`, `model_agreement.csv`, `counterfactual_by_fold.csv`, `counterfactual_summary.csv`, `examples_no_ids.csv` | `analysis.reason_codes` |
| `scorecard/full_no_gender_*.csv`, `lgbm/lgbm_v1_no_*_folds.csv`, `lgbm/lgbm_v1_no_*_importance.csv` | the `--drop` runs above |
| `fairness/accuracy_cost.csv`, `group_decisions.csv`, `disparity_summary.csv`, `gender_counterfactual.csv` | `analysis.fairness` |
| `eda/feature_drift_v2.csv`, `feature_drift_loan_terms_v2.csv`, `feature_adversarial_importance_v2.csv`, `feature_adversarial_summary_v2.csv` | `eda.feature_drift --features v2` |
| `lgbm/lgbm_v2_*_folds.csv`, `lgbm/lgbm_v2_*_importance.csv` | `pipeline.train_lgbm` (v2 runs) |
| `lgbm/lgbm_v3_*_folds.csv`, `lgbm/lgbm_v3_*_importance.csv` (with `--shap-top`: `kept_in_folds`) | `pipeline.train_lgbm` (v3 runs) |
| `xgb/xgb_v2_combo_s0_folds.csv`, `mlp/mlp_v2_combo_s0_folds.csv` | `pipeline.train_family` |
| `calibration/summary_v2.csv`, `by_seed_v2.csv`, `lgbm_v2_combo_lr02_x3_reliability.csv` | `analysis.calibration ... --tag v2` |
| `fairness_v2/accuracy_cost.csv`, `group_decisions.csv`, `disparity_summary.csv` | `analysis.fairness ... --tag v2` |
| `reasons_v2/*.csv` | `analysis.reason_codes ... --tag v2` |
| `submissions/manifest_v2.csv` | `pipeline.make_submissions --tag v2` |
| `business/approval_curve.csv`, `policies.csv`, `sensitivity.csv`, `policy_groups.csv` | `analysis.business` |
| `errors/segments.csv`, `missed_defaulters.csv`, `candidate_signals.csv` (`mean_pred_by_group.csv` from entry 38 on) | `analysis.error_analysis` |
| `submissions/manifest.csv` (and `submission/<model>.csv` in the repository root, not committed) | `pipeline.make_submissions` |
| `model_summary.csv` | `experiments.model_summary` |
| `kaggle/kaggle_scores.csv` | typed by hand from the Kaggle submissions page (the only input not written by a script) |
| `kaggle/kaggle_vs_cv.csv`, `kaggle_steps.csv` | `experiments.kaggle_check` |
| `calibration/ece_by_fold_v2.csv` | `analysis.calibration ... --tag v2` (since the review) |
| `business_v2/approval_curve.csv`, `policies.csv`, `differences.csv`, `sensitivity.csv`, `policy_groups.csv` | `analysis.business` (version 2) |
| `errors_v2/*.csv` | `analysis.error_analysis --model lgbm_v2_combo_lr02_x3 --tag v2` |
| `fairness_v2/proxy_check.csv`, `proxy_top_columns.csv` | `analysis.proxy_check` |
| `validation/duplicate_check.csv` | `experiments.duplicate_check` |
| `reasons_v2_cut06/*.csv` | `analysis.reason_codes ... --decline-rate 0.06 --with-fourth --tag v2_cut06` |
| `kaggle/kaggle_steps_v2.csv`, `kaggle_vs_cv_v2.csv` | `experiments.kaggle_check --tag v2` |
| `credit_v2/score_bands.csv`, `ks.csv`, `swap_set.csv`, `psi.csv` | `analysis.credit_tables` |
| `fairness_v2/counterfactual_age_gender.csv` | `analysis.sensitive_counterfactual` |
| `eda/feature_drift_v3b.csv`, `feature_drift_loan_terms_v3b.csv`, `feature_adversarial_*_v3b.csv` | `eda.feature_drift --features v3b` |
| `lgbm/lgbm_v3b_*`, `lgbm/lgbm_v3c_*` (`_folds.csv`, `_importance.csv`, `kept_in_folds`) | `pipeline.train_lgbm` (v3 runs) |
| `tuning/v3b_sh90_trials.csv`, `v3b_sh90_best.json`, `v3b_sh90_shap_columns.json` | `experiments.tune_lgbm` |
| `features/rate_term_model.csv`, `rate_drift.csv` | `pipeline.make_features_rate` |
| `monotone_v3t/violations.csv`, `monotone_v3c/violations.csv` | `experiments.monotone_check` |
| `comparisons/<A>_vs_<B>_seeds0234.csv` | `experiments.compare A B --seeds 0,2,3,4` |
| `comparisons/test_like_gain_<A>_vs_<B>.csv` | `experiments.test_like_gain` |
| `calibration/*_v3.csv`, `*_v3t.csv`, `*_v3m.csv` | `analysis.calibration ... --tag v3 / v3t / v3m` |
| `business_v3/`, `business_v3t/`, `business_v3m/` | `analysis.business --main ...` |
| `credit_v3/`, `credit_v3t/`, `credit_v3m/` | `analysis.credit_tables --main ... --tag ...` |
| `errors_v3/`, `errors_v3t/`, `errors_v3m/` | `analysis.error_analysis --model ... --tag ...` |
| `fairness_v3/`, `fairness_v3t/`, `fairness_v3m/` | `analysis.fairness`, `analysis.proxy_check`, `analysis.sensitive_counterfactual` (v3 only) |
| `reasons_v3_cut06/`, `reasons_v3t_cut06/`, `reasons_v3m_cut06/` | `analysis.reason_codes ... --decline-rate 0.06 --tag ...` |
| `global_shap_v2/`, `global_shap_v3/`, `global_shap_v3t/`, `global_shap_v3m/` | `analysis.global_shap` |
| `submissions/manifest_v3*.csv` | `pipeline.make_submissions --tag ...` |
| `model_summary_v3.csv`, `model_summary_v3t.csv`, `model_summary_v3m.csv` | `experiments.model_summary --v3 / --v3t / --v3m` |
| `kaggle/kaggle_scores_v3.csv`, `kaggle_scores_v3t.csv`, `kaggle_scores_v3c.csv`, `kaggle_scores_v3m.csv` | typed by hand from the Kaggle submissions page |
| `kaggle/kaggle_vs_cv_v3m.csv`, `kaggle_steps_v3m.csv` (also `_v3`, `_v3t`) | `experiments.kaggle_check --tag v3m --v3m` |
| `reproducibility/<label>.csv` | `experiments.check_reproduced` |

`eda.one_applicant` prints one applicant's records to the terminal and writes no file (individual records of
the competition data must not be redistributed).

## Charts and notebook (committed)
| File | Written by |
|---|---|
| `figures/*.png` (README and DETAILS.md charts: `*_v3m.png` for the final model; no suffix = v2, `_v3`, `_v3t` = earlier versions) | `analysis.figures --version ...` (chart code in `homecredit/figures.py`) |
| `notebooks/home_credit_report.ipynb` (executed, with outputs) | `notebooks.build_notebook` |

## Superseded files (kept, not used by the README or DETAILS.md)
Results are never overwritten. These were written by earlier versions of a script (see git history and
EXPERIMENTS.md) and replaced by the files named in brackets:
`business/*` (first business version; `business_v2/`), `kaggle/kaggle_vs_cv.csv` and `kaggle/kaggle_steps.csv`
(`*_v2.csv`; the steps file used a 5% too large noise scale). `model_summary.csv` is not superseded: it still
holds the v1 and v2 models.

The v2, v3 and v3t result folders and charts belong to earlier main models. The README and DETAILS.md mostly use the
`*_v3m` files, but also cite a few earlier ones where the final model has no counterpart:
`business_v3t/differences.csv` (money from tuning), `fairness/group_decisions.csv` (LightGBM v1 without age),
`fairness_v3/` (direct age and gender test) and `reasons_v2_cut06/` (share of declines with the external scores
first, v2).

## Run by hand (not in the run scripts yet)
| File | Command |
|---|---|
| final models and test predictions of `lgbm_v3b_all_s1`, `lgbm_v3b_sh99_s1` | `pipeline.train_lgbm ... --final-only` with the settings and `--name` of the exploratory run |
| `submissions/manifest_v3_extra.csv`, `manifest_v3c.csv`, `manifest_v3c_mono.csv` | `pipeline.make_submissions --tag ...` for the Kaggle checks |
