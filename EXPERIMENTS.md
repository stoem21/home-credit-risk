# Experiment log

Every entry says what was run, what was measured, and what it means for the next step. Failed ideas stay in.
Interpretations that are not measured are marked as guesses.

## 1. EDA step 0, raw data to Parquet
- **Run:** `pipeline/make_parquet.py` (22 s). Output `processed/raw/` and `processed/raw/manifest.csv`.
- **Measured:** parsed rows equal CSV line counts for all 10 files; 346 columns in total (matches the Data page).
  Parquet equals the CSV value for value for application_train, bureau and previous_application (checked by
  hand with `pandas.testing.assert_frame_equal`; the other 7 tables only by row and column counts).

## 2. EDA step 1, structure and keys
- **Run:** `eda/keys.py` (about 6 s). Output in `analysis_output/eda/`.
- **Keys:** `SK_ID_CURR` is unique in train and test, `SK_ID_BUREAU` in bureau, `SK_ID_PREV` in
  previous_application. Train and test share no applicant. sample_submission lists exactly the test IDs.
- **"hashed" IDs:** not a problem for joins. 100% of `SK_ID_CURR` values in every side table are found in
  application train or test.
- **Orphan rows:** some child keys have no parent: 5.27% of `SK_ID_BUREAU` in bureau_balance are not in bureau;
  4.0% (POS), 10.9% (credit card) and 3.9% (installments) of `SK_ID_PREV` are not in previous_application.
  Where the parent exists, the owner (`SK_ID_CURR`) always agrees. So aggregate the monthly tables by
  `SK_ID_CURR`, not only through previous_application, or the orphan rows are lost.
- **Main finding: bureau_balance coverage differs between train and test.** 35.7% of train bureau credits have
  monthly history, against 99.9% in test. It is all or nothing per applicant: in train 55.7% of applicants have
  bureau credits but no monthly history, in test 0.02%. Default rate in train: 8.11% with history, 7.51% without,
  10.12% with no bureau rows at all.
  - **Why it matters:** a feature built from bureau_balance would be missing for most train rows but present for
    most test rows, so the model would learn "missing" from one population and be scored on another.
  - **Guess, not verified:** monthly history was supplied for a sample of train applicants only.
  - **To decide later (feature step):** use bureau_balance only if a validation fold that mimics test coverage
    shows it helps; otherwise leave it out and say so.
- **Smaller train/test gap:** applicants with no previous Home Credit application: 5.35% train, 1.94% test.
  Same direction for POS and installments. Cause not known; check in step 5 (time / how test was split).

## 3. EDA steps 2-3, target and missing values
- **Run:** `eda/target_missing.py` (about 5 s). Output in `analysis_output/eda/`.
- **Target:** 24,825 defaults in 307,511 train applicants: **8.07%**. Only values 0 and 1. Cash loans 8.35%
  (90.5% of train), revolving loans 5.48%. All text columns by category: `target_by_category.csv`; several
  categories have under 25 applicants, so their rates are noise (for example "Unemployed": 22 applicants).
- **Missing values (NaN only):** application has 64 of 120 columns with missing values, 41 of them over 50%
  missing. 39 of the 41 are building descriptions (`*_AVG`, `*_MODE`, `*_MEDI`), which go missing together in
  a few groups (`missing_groups.csv`). The other two are `OWN_CAR_AGE` (66%) and `EXT_SOURCE_1` (56%).
- **Train vs test gaps in missingness:** `bureau.AMT_ANNUITY` 77.1% missing in train vs 38.5% in test;
  `EXT_SOURCE_1` 56.4% vs 42.1%. Other gaps are under 8 points. With step 1 (bureau_balance), this is the
  third sign that test rows were not drawn the same way as train rows. Guess, not verified: test may come from
  a later period, when more fields were filled in. Step 5 should test this.
- **Does "missing" carry signal?** Among columns with at least 1,000 missing rows, the largest gaps in default
  rate are modest: social-circle counts missing 3.53% vs 8.09% (only 1,021 rows); credit-bureau enquiry counts
  missing 10.34% vs 7.72% (13.5% of rows); `OCCUPATION_TYPE` missing 6.51% vs 8.79%; `EXT_SOURCE_1` missing
  8.52% vs 7.50%; `EXT_SOURCE_3` missing 9.31% vs 7.77%. So keep missing as information (tree models handle
  NaN; the scorecard can give "missing" its own bin) instead of filling in values.
  - **Guess, not verified:** enquiry counts are missing for about the same share as applicants without bureau
    rows (13.5% vs 14.3%), so they may mean "no bureau file"; and missing occupation may be mostly pensioners
    (5.39% default rate). Both to check in step 4.
- **Script fix:** the first version ranked columns with a handful of missing rows at the top (12 missing rows,
  0 defaults). Ranking now uses only columns with at least 1,000 missing rows; the rest are flagged.

## 4. EDA step 4, odd values and placeholder codes
- **Run:** `eda/quirks.py` (about 5 s). Output: `numeric_profile.csv`, `placeholder_values.csv`,
  `quirk_checks.csv`.
- **365243 is a code, not a number of days** (it would be 1,000 years). In application `DAYS_EMPLOYED` it marks
  18.01% of train: 55,352 of the 55,362 pensioners plus the 22 unemployed. `ORGANIZATION_TYPE = "XNA"` has the
  same count (55,374), so it is the same people. In previous_application it fills five DAYS_* columns, up to
  55.95% of rows (`DAYS_FIRST_DRAWING`). Treat it as missing with its own flag; never as a real value.
- **XNA / XAP placeholders** in previous_application text columns: `NAME_CASH_LOAN_PURPOSE` 95.8%,
  `CODE_REJECT_REASON` 81.3% (XAP, presumably "not applicable" because the application was not refused: guess),
  `NAME_PRODUCT_TYPE` 63.7%. They are "not recorded / not applicable", not real categories.
- **Dates after the application:** in event columns none. Positive values appear only in planned dates
  (`bureau.DAYS_CREDIT_ENDDATE`: 602,603 rows; `previous_application.DAYS_LAST_DUE_1ST_VERSION`: 224,392 rows),
  which are known at application time, plus 17 rows of `bureau.DAYS_CREDIT_UPDATE` up to +372 days (oddity, too
  few to matter). Some bureau end dates go back about 115 years (`DAYS_CREDIT_ENDDATE` min -42,060): data error.
- **Other quirks:** 4 applicants with `CODE_GENDER = "XNA"`; one income of 117,000,000 (median 147,150; 99.9th
  percentile 900,000).
- **Step 3 guesses checked:** (1) missing credit-bureau enquiry counts = no bureau rows: confirmed, all 41,519
  have no bureau rows (another 2,501 without bureau rows do have counts). (2) missing occupation = mostly
  pensioners: partly, 55,357 of 96,391 (57%); 24,920 are "Working".

## 5. EDA steps 5-6, train vs test, time order, leakage
- **Run:** `eda/drift_leakage.py` (about 10 s). Output: `column_drift_and_target_auc.csv`,
  `drift_details.csv`, `id_order_check.csv`, `id_auc.csv`.
- **No time order is available.** There are no absolute dates, only days relative to each application.
  `SK_ID_CURR` is not a time proxy: test IDs are spread evenly over the ID range (12.8% to 14.7% of each decile)
  and the train default rate is flat across ID deciles (7.91% to 8.29%); AUC of the ID for TARGET is 0.4978.
  So the train/test split cannot be reproduced by time, and a time-based validation is not possible.
- **But test is a different mix (measured):**
  - **Product:** revolving loans are 9.52% of train but 0.90% of test.
  - **Recent bureau enquiries** (`AMT_REQ_CREDIT_BUREAU_QRT` >= 1): 16.5% of train, 39.3% of test
    (single-column AUC test vs train 0.62, the largest of all columns).
  - With step 1 (bureau_balance) and step 3 (EXT_SOURCE_1, bureau AMT_ANNUITY): five signs, all pointing the same
    way. **Guess, not verified:** test comes from a later period, after revolving loans were mostly dropped and
    when more data was recorded. After the two enquiry-count columns (0.62 and 0.58), every other numeric column
    has a test-vs-train AUC of 0.565 or less.
- **Consequence for validation (step 3 of the project, proposal):** stratified K-fold on train (no time order
  exists), plus a second score on cash loans only, because test is 99% cash loans. Expect the Kaggle score to
  differ from cross-validation because of this shift; report both.
- **Leakage check:** no single column is suspiciously strong. Best single-column AUCs for TARGET: EXT_SOURCE_3
  0.679, EXT_SOURCE_1 0.666, EXT_SOURCE_2 0.656 (external credit scores, plausible), then age 0.583 and
  employment length 0.582. Nothing near 0.9.

## 6. EDA step 7, side tables
- **Run:** `eda/side_tables.py` (about 3 s). Output: `side_time_spans.csv`, `side_status_shares.csv`,
  `installments_behaviour.csv`, `side_signals_vs_default.csv`.
- **History window:** all side tables reach back at most 96 months (2,922 days, 8 years): the data was cut to
  8 years before each application. Monthly tables end at month -1, events at day 0 or earlier.
- **Statuses:** bureau credits 62.9% closed, 36.7% active. Previous applications 62.1% approved, 18.9%
  cancelled, 17.4% refused. bureau_balance: 1.3% of months 1+ months overdue (STATUS 1 to 5), 21.3% unknown (X).
- **Installments:** 8.43% of installments paid after the due day, 9.52% paid less than due (row level; a
  partial payment can be split across rows, so "short" overstates it: to check when building features).
- **First signals (train default rate, one summary at a time):**
  - **Late installments:** 6.80% (none late) to 11.99% (more than 20% late).
  - **Refused previous applications:** 7.07% (none refused) to 15.93% (more than half refused).
  - **Any bureau credit overdue now:** 15.90% (yes, 1.1% of applicants) vs 7.62% (no).
  - **POS months past due (after tolerance):** 10.80% vs 7.74%.
  - **Credit-card months past due:** no clear signal (7.75% yes vs 8.86% no).
  - **No history at all** in Home Credit tables: about 6.0% default, lower than average (guess: new customers
    pass a stricter first check).

## 7. Validation design and noise (project step 3)
- **Decision:** stratified 5-fold CV on train, stratified on TARGET x contract type,
  repeated with 5 seeds (`pipeline/make_folds.py`, output `processed/folds.parquet`). Every model is scored on
  all loans and on cash loans only (test is 99% cash loans). Reason: no time order exists (EDA step 5).
  Check: every fold has a default rate of 8.07% and 9.52% revolving loans (seed 0).
- **Run:** `experiments/noise_check.py` (about 20 s). Output `analysis_output/validation/`.
- **Scorers:** a rule with no training (minus the mean of EXT_SOURCE_1/2/3) and a logistic regression on 15
  hand-picked application features (`build_features` in the script).
- **Results (mean of 25 fold AUCs):** logistic regression 0.7373 all loans / 0.7359 cash loans; rule
  0.7165 / 0.7164. These are the first baselines.
- **Noise, measured three ways:**
  - **One fold:** AUC moves by about 0.0024 to 0.0028 (sd across folds).
  - **Changing the seed:** almost nothing (sd of the 5-fold mean 0.00001 to 0.00004), because neither scorer has
    training randomness and the 5-fold mean always covers every row. To re-measure for LightGBM, which does.
  - **Another sample of applicants:** bootstrap SE of the out-of-fold AUC 0.0017 (logistic regression, seed 0;
    OOF AUC 0.7373). So the AUC is known to about +/- 0.003 (two SE).
  - **Two models compared on the same rows:** bootstrap SE of the difference 0.0009, half the single-model SE,
    because both models make similar errors. Logistic regression beats the rule by 0.021 (all) and 0.020 (cash)
    and wins 25 of 25 folds.
- **Rule for adopting a change (proposal, from the previous project):** compare on the same folds; adopt only
  if the mean gain is more than 2 x the bootstrap SE of the difference for that comparison (computed each time,
  not a fixed number) and the new model wins in most of the 25 folds (at least 18). Report both all-loan and
  cash-loan AUC.

## 8. Feature table v1
- **Run:** `pipeline/make_features.py` (about 5 s). Output `processed/features_v1.parquet`: 356,255 rows
  (train + test), 127 application candidates (raw columns, 365243 code turned into NaN + flag, 6 derived
  columns) and 22 history summaries (bureau, previous applications, installments, POS, credit card).
  bureau_balance left out (coverage gap, EDA step 1).
- **Checked:** after combining split payments per installment, only 0.1% of installments are underpaid on
  average, against 9.52% at row level. The row-level number was an artefact of split payments.
- **Bug found and fixed before any model used it:** an installment with no payment at all was counted as "not
  late". It now counts as late.

## 9. Scorecard (WOE + logistic regression), project step 4
- **Run:** `pipeline/train_scorecard.py --variant app|full` (about 25 s each, 5 parallel jobs) and
  `experiments/compare.py` for paired comparisons. Output `analysis_output/scorecard/` and
  `analysis_output/comparisons/`.
- **Method:** binning and WOE fitted inside each training fold; settings fixed in advance, not tuned (10 fine
  bins, monotonic merge, bins >= 1% of rows, IV >= 0.02, WOE correlation <= 0.7, at most 30 variables, all
  coefficients positive). Points: 600 = 50:1 good:bad odds, PDO 20.
- **Results (mean AUC over 25 folds, all loans / cash loans):**
  - logistic baseline (15 features): 0.7373 / 0.7359
  - scorecard, application only: 0.7433 / 0.7416 (+0.0060 vs baseline, SE 0.0007, wins 25 of 25)
  - scorecard, application + history: **0.7565 / 0.7554** (+0.0132 vs application only, SE 0.0006, wins
    25 of 25)
  All three steps beat noise by the adoption rule.
- **Final scorecard (all train):** 26 variables, scores from about 398 to 715 points. 22 of 26 variables were
  selected in all 25 folds; the least stable (employed_years 18, FLOORSMAX_AVG 19, REG_CITY_NOT_LIVE_CITY 16,
  OWN_CAR_AGE 16) are the ones to question first. History adds 7 variables, the strongest being days since the
  last bureau credit, share of refused applications, late installments in the last year and card utilisation.
- **Dropped for a wrong sign:** DAYS_BIRTH (age), AMT_GOODS_PRICE, REG_CITY_NOT_WORK_CITY,
  REGION_POPULATION_RELATIVE. Age on its own is the 4th strongest column (EDA step 6), but next to employment
  years, income type and the EXT_SOURCE scores its coefficient turned negative. Guess: those columns already
  carry the age effect (EXT_SOURCE scores may include age). Relevant for fairness: removing age would not remove
  its influence.
- **For the fairness step:** `CODE_GENDER` is in the scorecard (men -5 points, women +3). Many lenders may not
  use gender at all. To decide with evidence: AUC and approval rates with and without it.
- **Small quirk:** the 4 applicants with gender "XNA" fall in the pooled OTHER bin, whose WOE (+0.24) comes from
  smoothing on 4 rows with no default. Harmless at this size; a minimum bin size for OTHER would fix it.

## 10. LightGBM v1, project step 4
- **Run:** `pipeline/train_lgbm.py` (about 2 minutes for 25 fold fits, 5 parallel jobs x 2 threads; a timing run on
  seed 0 first). Output `analysis_output/lgbm/`, OOF `processed/oof/lgbm_v1.parquet`.
- **Same 149 candidate columns as the full scorecard.** Settings fixed in advance (learning rate 0.05, 31
  leaves, min 100 rows per leaf, 80% feature and row sampling, L2 1.0). Number of trees chosen by early stopping
  on an inner 10% of each training fold, never on the validation fold: 291 to 503 trees.
- **Result:** mean AUC **0.7792 all loans / 0.7788 cash loans**. Against the full scorecard: +0.0227 (SE 0.0008),
  wins 25 of 25. Against the logistic baseline: +0.0419.
- **Seed noise (now real, LightGBM samples rows and columns):** sd of the 5-fold mean over 5 seeds 0.00018
  (all) and 0.00019 (cash). Still small against the gains seen so far.
- **Importance (gain share):** EXT_SOURCE_3 14.9%, EXT_SOURCE_2 13.8%, ORGANIZATION_TYPE 7.9%, EXT_SOURCE_1
  5.1%, annuity_to_credit 3.8%; history columns from 2.3% down. ORGANIZATION_TYPE ranks 3rd in LightGBM but has IV
  0.058 in the scorecard (8th). Guess, not verified: LightGBM's own handling of a 58-category column can overfit
  and gain overstates it. A candidate check: the same model with ORGANIZATION_TYPE encoded differently or left
  out. Gain share is not impact on decisions either (lesson from the previous project).
- **Accuracy vs explainability, so far:** scorecard 26 variables, one table of points, 0.7565; LightGBM 149
  columns, about 450 trees, 0.7792. What 0.023 AUC is worth in approvals and losses is for the business step.
- **Final model:** refitted on all train with 446 trees (median fold choice x 1.1); test predictions saved, not
  submitted.

## 11. Calibration, project step 5
- **Run:** `analysis/calibration.py logreg_baseline scorecard_full lgbm_v1` (about 18 s). Output
  `analysis_output/calibration/` (summary, by seed, reliability tables for seed 0).
- **All three models are already close to calibrated on OOF predictions** (mean over 5 seeds, all loans):
  - mean predicted vs observed: baseline 8.08% vs 8.07%, scorecard 8.07% vs 8.07%, LightGBM 8.00% vs 8.07%;
  - calibration slope: 0.998, 0.994, 0.986 (1 = perfect);
  - ECE (average gap over 20 bins): 0.17, 0.28, 0.40 percentage points.
- **Where LightGBM is off (seed 0, 20 bins of about 15,400 applicants):** it under-predicts in the middle-high
  range, by 0.5 to 1.0 points in the bins with 5% to 17% predicted risk (for example 10.6% predicted vs 11.5%
  observed; the sampling SE of one bin there is about 0.25 points, so this is real), and over-predicts the
  riskiest 5% (38.2% vs 36.3%). Cash loans only: 8.26% predicted vs 8.35% observed.
- **Recalibration, cross-fitted on the other four folds:**
  - Platt (2 parameters): fixes the average (8.07%) but leaves ECE at 0.37 points.
  - Isotonic: ECE 0.40 -> 0.10 points (sd over seeds 0.03), cash 0.40 -> 0.12. Cost: AUC -0.0006 (ties),
    log loss +0.00004 (sd 0.00009, within noise), Brier unchanged.
- **Decision:** rank with the raw LightGBM score, use isotonic-calibrated probabilities
  where the probability value matters (expected loss, cut-off). The gain is about half a point of default rate in
  the 5-17% band, where cut-offs are likely to sit.
- **Limit (important):** this is calibration to the train population (8.07% defaults). The test population has
  a different mix (EDA step 5) and its default rate is unknown, so calibration on test cannot be checked. In use,
  probabilities would need monitoring against recent outcomes and recalibration when the mix changes.

## 12. Reason codes, project step 6
- **Run:** `analysis/reason_codes.py` (about 1 min 40 s). Output `analysis_output/reasons/`. Seed-0 folds: for
  each fold a LightGBM and a scorecard are fitted on the training part and explain the validation part.
  (The first run had only the "points below max" scorecard method; it was re-run the same day with the second
  method added, overwriting its own output before anything was reported. LightGBM numbers were identical.)
- **Provisional decline rule:** riskiest 15% (85th percentile of each model's training-part predictions). Not the
  final cut-off; that is the business step. Default rate among declined: LightGBM 25.25% vs 5.06% approved;
  scorecard 23.29% vs 5.39%.
- **Reasons are groups, not columns:** 149 columns mapped to 19 readable groups (`reason_groups.csv`), for
  example "External credit scores" (EXT_SOURCE_1/2/3), "Repayment of earlier Home Credit loans" (installments,
  POS), "Loan size and terms vs income".
- **Counterfactual check of the LightGBM (SHAP) reasons, the main test:** for each declined applicant, replace
  the columns of one reason group with those of a random approved applicant (5 donors). Mean fall in predicted
  risk and share that flips to approve:
  - top-1 reason: -14.0 points, 74.3% flip;
  - top-2: -7.4 points, 41.5%;
  - top-3: -4.8 points, 27.5%;
  - control (a random group not in the top 3): -0.2 points, 6.3%.
  The order holds in all 5 folds. So the stated reasons do drive the decision, in the stated order.
- **What comes first:** "External credit scores" is the top reason for 77.2% of LightGBM declines, then loan size
  vs income 8.4%, repayment of earlier loans 7.5%. In the scorecard it is first in 95% to 100%. This is honest
  but not very useful to an applicant: EXT_SOURCE are scores from outside sources whose content is not described,
  so the applicant cannot see what to improve. Worth stating as a limit.
- **Two scorecard methods compared:**
  - **Points below the best bin** (a common default): "Education" appears in the top 3 for 93.7% of declines,
    because almost everyone is below the best education bin. An artefact of the method, not a real driver.
  - **Points below the approved applicants' average:** Education drops to 1.0%; reasons agree more with LightGBM
    (on average 2.32 of 3 reasons shared, vs 1.75). Proposal: use this method.
- **Sensitive reasons (important for fairness):** "Gender" is among the top 3 reasons for 9.8% of LightGBM
  declines and 13.2% of scorecard declines (below-average method). A lender cannot tell an applicant "declined
  partly because you are a man". Age appears in 1.0% (LightGBM). Application day / hour and contact details:
  under 2%. Input for the fairness step: test the models without gender.

## 13. EDA step 8, sensitive attributes (inventory only)
- **Run:** `eda/sensitive.py` (about 2 s). Output: `sensitive_inventory.csv`.
- **Default rate by group (train):** men 10.14% vs women 7.00%; age falls from 12.29% (<25) to 3.66% (65+);
  education from 10.93% (lower secondary) to 5.36% (higher); region rating 4.82% (1) to 11.10% (3); family
  status 5.82% (widow) to 9.94% (civil marriage); children 7.71% (none) to 10.04% (3+). Train and test shares
  are within about 2.5 points for every group.
- **Not decided yet:** which of these count as protected attributes and whether the model may use them. That
  is the fairness step, with evidence (for example, how much accuracy each one adds and how approval rates change
  with and without it). Age is also tied to pension status, so excluding it may not remove its effect.

## 14. Repository restructure before the first commit (no result changes)
- **Why:** a planned layout, one shared evaluation module, every number traceable to a script, and smoke tests
  that never write next to real results. All scripts sat in `pipeline/`; the AUC, the cash-loans
  scope, the per-fold AUC loop, the bootstrap, fold loading and `is_text` were copied in 2 to 6 places (the EDA
  used its own rank-based AUC); lgbm and scorecard wrote to fixed paths whatever `--out` said.
- **What changed:**
  - shared module `homecredit/`: `paths.py` (every location, plus `--out-root`), `data.py` (loading, folds with
    ID check, scopes, `is_text`, constants), `evaluate.py` (the one AUC, per-fold AUC, bootstrap, paired adoption
    rule, calibration measures, OOF files), `models/scorecard.py`, `models/lgbm.py`;
  - scripts split into `eda/`, `pipeline/` (data, features, model training), `experiments/` (noise, paired
    comparisons, reproducibility check) and `analysis/` (calibration, reason codes); run as modules from the
    repository root (`.venv/bin/python -m ...`), no `sys.path` edits;
  - `run_all.sh [OUT_ROOT]` runs everything in order; `OUTPUTS.md` lists which script writes each file;
  - `experiments/noise_check.py` no longer refits the seed-0 logistic baseline a second time for the bootstrap; it
    reuses the seed-0 out-of-fold predictions it already has;
  - `analysis/reason_codes.py` now writes `examples_no_ids.csv` (no applicant ID, no outcome) instead of
    `examples.csv`, which showed individual records of the competition data. I deleted the old file after
    the check below, so it can never be published.
- **Check (measured):** `run_all.sh` into a separate folder (501 s wall clock), then
  `experiments/check_reproduced.py`, report `analysis_output/reproducibility/refactor_2026-11-25.csv`: 69 of 73
  files byte-identical, including every OOF and test prediction, the LightGBM model file and every EDA table (so
  the switch of the EDA AUC from ranks to scikit-learn changed no stored value). The only differences: the
  `seconds` timing columns of `processed/raw/manifest.csv` and `lgbm_v1_folds.csv`, and the examples file above
  (same five applicants and reasons, minus the two removed columns; checked by hand in Python, not saved). This
  also confirms the earlier category fix in the scorecard code did not change its results.

## 15. One applicant in detail
- **Run:** `eda/one_applicant.py` (about 4 s). Prints to the terminal only and saves nothing: it shows an
  individual record of the competition data, which must not be redistributed, so this entry describes what the
  record taught, not its values. Drawn at random (seed 0) among the 2,452 train defaulters with rows in every
  side table.
- **Check (measured):** 10 features recomputed by hand from the printed rows (counts, late share, max days late,
  POS months, card flag, credit / income) all match `features_v1.parquet`.
- **The record shows things our summary features do not see.** One record is an anecdote: each point below is a
  hypothesis to count on all applicants (error analysis, feature v2), not a finding.
  - **Burst of recent applications:** several Home Credit applications in the last year, most of them in the
    last 2 months, many cancelled. Our features only have totals and "days since last decision", no recent
    window and no cancelled count.
  - **Loan stacking:** an earlier Home Credit cash loan of similar size is still running, and a new credit with
    another lender opened weeks before this application. No feature measures debt still running at Home Credit
    or remaining instalments (`CNT_INSTALMENT_FUTURE`).
  - **Small delays vs real arrears:** all late instalments were 1-3 days late, yet they count as "late" in
    `inst_late_share`. A threshold on days late may separate habits from real trouble.
  - **bureau_balance shows a month of arrears** in two other-lender credits that `bureau` alone does not show
    (`CREDIT_DAY_OVERDUE` is 0 today). We left bureau_balance out because of its train/test coverage gap.
- **Data quirks seen, to handle before using these columns:** in `previous_application`, 365243 in
  `DAYS_LAST_DUE` / `DAYS_TERMINATION` means "not ended yet"; cancelled applications carry amounts of 0 and
  "XNA", not real values; the last monthly status in `POS_CASH_balance` can be "Active" years ago, because the
  monthly history stops, so it is not the current status; an other-lender credit opened weeks ago has no debt
  recorded (NaN).
- **Models on this record:** all three put this applicant in the riskier part of train (seed-0 out-of-fold
  risk higher than about 79% / 95% / 88% of applicants for logreg / scorecard / LightGBM); the applicant did
  default.

## 16. Feature table v1, train vs test
- **Run:** `eda/feature_drift.py` (about 1.5 min; 5 parallel LightGBM fits). Output `analysis_output/eda/feature_drift*.csv`,
  `feature_adversarial_*.csv` and `processed/adversarial_v1.parquet`. Ran twice (the second run added the
  loan-term table); the adversarial numbers were identical.
- **Single features (measured):** 7 of 133 numeric features separate test from train with AUC above 0.55. The
  largest: `annuity_to_credit` 0.669, `AMT_REQ_CREDIT_BUREAU_QRT` 0.623, `credit_to_income` 0.590. The 22 history
  summaries all stay below 0.545. 11 features differ by more than 2 points in share missing; the largest is
  `EXT_SOURCE_1` (56.4% missing in train, 42.1% in test), then the history summaries, which are missing more
  often in train (for example installments 5.2% vs 1.6%: more test applicants have a Home Credit history).
- **All features together (measured): test is easy to tell apart.** An adversarial LightGBM separates test rows
  from train rows with AUC 0.957 (5 folds, sd 0.0007). Its most used features are `annuity_to_credit` (23% of
  gain) and `credit_to_goods` (15%), then loan amounts and enquiry counts.
- **What differs is the loan term mix (measured):** for cash loans, the median annuity / credit is 0.0616 in
  test vs 0.0484 in train, so test loans are shorter (or dearer). 98% of test ratios also occur in train, but only
  58% of train ratios occur in test: same product grid, different mix. **Interpretation (a guess):** the test
  applications may come from a different period or channel with different loan products; the data cannot tell.
- **Estimate of the effect on AUC: not reliable.** Reweighting train to look like test gives lgbm_v1 0.7793 vs
  0.7792 unweighted, scorecard_full 0.7476 vs 0.7565, logreg 0.7252 vs 0.7373. But the shift is so strong that the
  weights rest on an effective 3.1% of train (largest weight 1,557 times the mean), so these numbers are too noisy
  to report as an expectation. The Kaggle late submission stays the only honest check.

## 17. Error analysis of LightGBM v1
- **Run:** `analysis/error_analysis.py` (about 7 s, saved out-of-fold predictions only). Output
  `analysis_output/errors/`. Checked: 1 - wrong pairs / all pairs reproduces the seed-0 AUC (0.77927).
- **Where the model ranks worse (AUC inside the group, mean of 5 seeds):** applicants under 25 (0.738), 55+
  (0.749 and 0.746), pensioners (0.747), the longest-term fifth of loans (0.741), and the least test-like tenth of
  train (0.729). The other groups of at least 1,000 applicants range from 0.755 to 0.811. Part of this is
  expected: inside a narrow group there is less spread to rank.
- **Who causes the lost AUC:** missed defaulters are concentrated in the safe-looking groups. Pensioners hold
  18.0% of train but 18.4% of the missed-defaulter pairs and only 12.7% of the false-alarm pairs; per defaulter,
  a defaulter aged 65+ costs 2.1 times the average, 55-64 1.5 times, region rating 1 1.5 times, revolving loans
  1.4 times. False alarms are concentrated in young, male and working applicants and region rating 3. The model
  is close to calibrated inside every group of at least 1,000 applicants: mean predicted within 0.7 points of
  the default rate (largest gaps: lower secondary education 10.93% vs 10.26%, test-likeness decile 5 9.04% vs
  8.39%).
- **Missed defaulters look like good customers on every feature we have.** Defaulters in the safest quarter
  (mean predicted 4.2% vs 7.1% for non-defaulters) have median external scores, card use, refusal share and late
  share close to non-defaulters, far from caught defaulters. Our features hold nothing that flags them.
- **Candidate signals from the one-applicant record, counted on all applicants:** observed default rate vs mean
  predicted per bucket (overall gap +0.07 points, SE 0.05). Most buckets match within 2 SE, so the model already
  captures them through existing features. Small delays are already told apart (at most 5 days late: 8.6%
  defaults; 6-30 days: 9.7%; over 30: 10.7%; never late: 6.8%), so that hypothesis is rejected. Gaps above 2.5 SE:
  2+ other-lender credits opened in the last 90 days (2.3% of applicants, +1.11 points, SE 0.44), 2+ Home Credit
  loans still running (4.6%, +0.65, SE 0.23), 2+ cancelled applications in the last year (17.4%, +0.38, SE 0.12).
  About 25 buckets were tested, so one gap near 2 SE could be chance. **Reading:** loan stacking is the one
  theme with some evidence; the gaps are small, so any AUC gain from it is likely small (a guess, to test in v2).

## 18. Tuning budget and model families
- **Tuning budget (stated, not a new run):** no model was tuned. Logistic regression uses scikit-learn defaults
  on 15 hand-picked features; the scorecard uses binning and selection rules fixed in advance (10 bins, 1%
  minimum, IV 0.02, correlation 0.7, 30 variables); LightGBM uses common starting values with fold-safe early
  stopping (the number of trees is chosen on an inner 10% hold-out, the same rule for every fold). So the
  comparison is fair in the sense that every model got zero tuning; it does not show how far each could go.
- **Model families tried:** three (logistic regression, WOE scorecard, gradient-boosted trees). Not tried: other
  boosting libraries (XGBoost, CatBoost; expected to be close to LightGBM, a guess) and neural networks (a
  genuinely different family). Both were tried later on the v2 features (see the LightGBM v2 entry).

## 19. README draft started
- `README.md` drafted as a story (summary first, technical details last), with the findings so far. Every number in it was
  copied from a file in `analysis_output/` (target, comparisons, noise, calibration, reasons, errors, feature
  drift, reproducibility) and checked against it. Business, fairness and the final write-up are marked as not
  done. `run_all.sh` now also runs `eda.feature_drift` and `analysis.error_analysis`.

## 20. Fairness, project step 8
- **Run:** `pipeline/train_lgbm.py --drop gender|gender,family|gender,family,age` and
  `pipeline/train_scorecard.py --variant full --drop gender` (new `--drop` option; about 1.5 min per LightGBM run,
  same folds and settings, new output names, the saved v1 files untouched), `experiments/compare.py` for each,
  then `analysis/fairness.py` (about 2.7 min). Output `analysis_output/fairness/`.
- **Decision rule for this step (provisional):** decline the riskiest 15% of each model's out-of-fold predictions
  (per seed), so every model approves 85% and groups can be compared across models. The business step will set
  the real cut-off.
- **Check (measured):** the full models refitted inside the fairness script reproduce the saved seed-0
  predictions exactly (max difference 0.0).
- **Accuracy cost of leaving columns out (measured, 25 paired folds):** LightGBM without gender -0.0015 AUC
  (SE 0.0004, wins 2 of 25); without gender and family -0.0019 (1 of 25); without gender, family and age -0.0033
  (0 of 25). Scorecard without gender -0.0012 (SE 0.0002); its gender slot is taken by "any bureau credit
  overdue now". All losses beat the noise: the columns carry real signal, small next to the 0.023 gap between the
  scorecard and LightGBM.
- **Gender (measured):** with gender, LightGBM approves 88.2% of women and 78.8% of men (default rates 7.0% vs
  10.1%), 90.4% vs 82.6% of those who repaid. Without gender: 87.4% vs 80.4%, good payers 89.7% vs 84.0%. The
  price: the model then overstates women's risk (7.29% predicted vs 7.00% observed) and understates men's (9.38%
  vs 10.14%), and the default rate among approved men rises from 5.90% to 6.11%. **Counterfactual:** switching
  only the gender value changes 4.4% of LightGBM decisions (scorecard 5.5% of women, 6.7% of men); a woman
  scored as a man gets +1.6 points of predicted risk on average.
- **Family (measured):** leaving out family status and children on top of gender costs 0.0004 more AUC and moves
  group approval rates by about one point at most.
- **Age (measured), the surprise:** approval rises steeply with age (under 25: 69.7%; 65+: 97.9%), in line with
  default rates (12.3% vs 3.7%). Leaving age out does not help the young: their approval falls to 64.5%, because
  other columns (employment years, pension status and more) still carry age, and the model now overstates their
  risk (14.0% predicted vs 12.3% observed). The scorecard, which has no age column (DAYS_BIRTH was dropped for its
  sign), already shows this: under-25 approval 58.8%, predicted 15.1% vs 12.3% observed.
- **Other gaps (measured, not acted on):** region rating 3 vs 1 (approval 76.4% vs 94.0%) and lower secondary vs
  higher education (78.7% vs 92.7%) are larger than the gender gap. Region is a place, not a personal attribute,
  but it can stand in for one; education was listed as possibly sensitive in EDA step 8. Whether to treat them as
  protected is a judgment call.
- **Decision (a judgment, adopted for v2):** use `lgbm_v1_no_gender_family` from now on (AUC 0.7773 vs
  0.7792). Gender and marital status are attributes many lenders may not use (for example under the US Equal
  Credit Opportunity Act; my reading, not legal advice), and gender alone decides about 4% of decisions. Keep age:
  removing it costs AUC and hurts the young applicants it would be meant to protect. Do not recalibrate by gender
  to fix the per-group gaps above: that would bring gender back in.

## 21. Kaggle submission files for every model
- **Run:** `pipeline/make_submissions.py` (about 7 s). Writes `submission/<model>.csv` for the 9 models with
  test predictions (not committed) and `analysis_output/submissions/manifest.csv`. The rule and logistic
  baselines got test predictions here for the first time (same definitions as `experiments/noise_check.py`; the
  logistic model refitted on all of train).
- **Checked (measured):** every file has the 48,744 test IDs in the order of sample_submission.csv, the header
  `SK_ID_CURR,TARGET`, no missing value and values between 0 and 1. The manifest's cross-validated AUCs match the
  earlier logs (rule 0.71647, logistic 0.7373, LightGBM v1 0.77921), a check that the right files were picked up.
- **Purpose:** an independent check only. The model choice is already made on our own validation; Kaggle scores
  will be reported next to the CV scores, whatever they are, and will not be used to pick or tune anything.
- **Not done yet:** uploading (done by hand from my own Kaggle account; at most 5 submissions a day).

## 22. Feature table v2 and LightGBM v2 (plan: apply public top-solution ideas, judged on my own CV)
- **Reference:** `lgbm_v1_no_gender_family` (AUC 0.7773), the model proposed in the fairness step. Every new group
  is added alone to this reference (one change at a time), same folds, same LightGBM settings.
- **Run:** `pipeline/make_features_v2.py` (about 16 s): `processed/features_v2.parquet`, the v1 columns (checked
  identical to features_v1) plus 52 new columns in four groups: `win_` recent behaviour and time windows (29),
  `stk_` loan stacking and other-lender history incl. bureau_balance (9), `term_` loan terms incl. the yearly
  interest rate implied by earlier approved loans (6), `ext_` external-score combinations (8). Then
  `pipeline/train_lgbm.py --features v2 --add <group> --drop gender,family` and `experiments/compare.py`.
- **Each group alone vs the reference (measured, 25 paired folds):** `win_` +0.0028 (SE 0.0005, 25 of 25 wins),
  `stk_` +0.0040 (25 of 25), `term_` +0.0035 (25 of 25), `ext_` +0.0002 (13 of 25, no clear difference: the trees
  already combine the three scores). `ext_` is dropped.
- **Drift check of the new columns (`eda/feature_drift.py --features v2`, measured):** the bureau_balance columns
  are the problem: `stk_bb_has_history` alone separates test from train with AUC 0.784, and the past-due columns
  are missing for 70% of train but 13-16% of test (EDA step 1 coverage gap). With them, the adversarial AUC rises
  from 0.957 to 0.968 and they become its most used features. Every other new column stays below 0.545 (largest:
  `term_credit_to_last_approved` 0.541); the history columns are again missing more often in train, as in v1.
- **stk_ without bureau_balance (measured):** +0.0038 vs the reference (25 of 25); adding the bureau_balance
  columns back gives +0.0002 (16 of 25, no clear difference). So they are left out: no gain, and their meaning
  differs between train and test. `--exclude stk_bb_` in `pipeline/train_lgbm.py`.
- **All three groups together, `lgbm_v2_combo` (186 columns, measured):** AUC 0.7850 all loans, 0.7845 cash;
  +0.0077 vs the reference (SE 0.0006, 25 of 25), and +0.0037 vs the best single group (25 of 25), so the groups
  add up rather than overlap.
- **Safety:** `pipeline/train_lgbm.py` and `pipeline/train_family.py` now refuse to write to a model name whose
  out-of-fold file exists; a test call with `--name lgbm_v1` was refused and the saved file kept its date.
- **Learning rate 0.02 instead of 0.05 (`lgbm_v2_combo_lr02`, measured):** AUC 0.7868, +0.0018 vs
  `lgbm_v2_combo` (SE 0.0003, 25 of 25); about 2.5 times the trees (final model 1,131), 250 s for 25 fits.
  A first attempt failed before training (zsh did not split a shell variable holding the arguments); nothing was
  written.
- **Average of 3 LightGBM fits per fold (`lgbm_v2_combo_lr02_x3`, seeds s, s+1000, s+2000, measured):** AUC
  0.7882 all loans, 0.7877 cash; +0.0015 vs one fit (SE 0.0002, 25 of 25); 757 s for the 25 folds. This is the
  v2 candidate: +0.0109 over the fairness reference `lgbm_v1_no_gender_family` and +0.0090 over `lgbm_v1` (which
  still used gender and family).
- **Exploratory 1-seed mode (to save time):** `--seeds 0` writes out-of-fold files for seed 0 only
  (no final model); `experiments/compare.py` then compares the 5 seed-0 folds and needs 4 of 5 wins (the same
  72% share as 18 of 25), and marks the verdict "exploratory". Checked: an existing 25-fold comparison file is
  reproduced exactly by the changed code. A winner from 1-seed runs is confirmed with 5 seeds before adoption.
- **A run stopped by me:** the first XGBoost / MLP launch forgot `--exclude stk_bb_` (189 instead of 186
  columns), so the comparison with LightGBM would not have used the same data. Stopped within a minute; no file
  had been written; `--exclude` was added to `pipeline/train_family.py` and the run restarted.
- **Other model families, 1 seed (exploratory, same 186 columns, same rules, no tuning; measured on the 5
  seed-0 folds against `lgbm_v2_combo`, which has the same learning rate 0.05 and a single fit):**
  XGBoost (`pipeline/train_family.py --model xgb`, 45 s) AUC 0.7851 vs LightGBM 0.7844, +0.0007 (SE 0.0004, 3 of
  5 wins): no clear difference. MLP (two hidden layers 256 / 128, 28 s) AUC 0.7681, -0.0163 (0 of 5): clearly
  worse; its best epoch was the 3rd or 4th, so it overfits fast with these settings. A first MLP run failed (a
  missing-value indicator met a column with missing values only in the hold-out); fixed with
  `error_on_new=False`. **Decision (within noise, keep the simpler):** XGBoost is within noise of LightGBM and no simpler, so
  LightGBM stays; the MLP result matches the public experience that plain neural nets trail boosted trees here.
  Not confirmed with 5 seeds: neither model is adopted, so there is nothing to confirm.
- **Fairness reference for v2 (`lgbm_v2_combo_with_gender`, same 186 columns plus gender and family, learning
  rate 0.05, measured):** AUC 0.7866. Leaving gender and family out (`lgbm_v2_combo`) costs 0.0016 (SE 0.0004, 0
  of 25), about the same as in v1 (0.0019). The v2 candidate without them (`lgbm_v2_combo_lr02_x3`) is 0.0016
  above this reference (25 of 25), but that comparison mixes two changes (lower learning rate and averaging).
- **Calibration of the v2 candidate (`analysis/calibration.py lgbm_v1 lgbm_v2_combo_lr02_x3 --tag v2`, measured):**
  raw ECE 0.42 points (v1: 0.40); cross-fitted isotonic brings it to 0.10 (v1: 0.10) at an AUC cost of 0.0006.
  Same picture as v1, so the same decision holds: rank with raw scores, use isotonic probabilities for money.
- **Fairness of the v2 candidate (`analysis/fairness.py ... --tag v2`, decline the riskiest 15%, measured):** women
  87.3% approved, men 80.6% (with gender: 88.1% vs 79.1%); under 25: 69.5%, 65+: 97.9%, as in v1.
- **Reason codes of the v2 candidate (`analysis/reason_codes.py ... --tag v2`, measured; the LightGBM part is the
  average of 3 fits at learning rate 0.02, SHAP averaged; the scorecard keeps the v1 columns without gender and
  family):** counterfactual risk drop top-1 14.0 points (74.1% of declines flip to approve), top-2 7.9 (43.8%),
  top-3 5.2 (29.7%), control 0.2 (5.9%), in that order in all 5 folds. A sensitive attribute is in the top 3 for
  0.45% of declines (age only; v1 with gender: 11.5%). "External credit scores" is the top reason for 71.7%
  (v1: 77.2%); the new reason "Loans still running at Home Credit" is the top reason for 2.2% and in the top 3 for
  10.6%. LightGBM and the scorecard (points below the approved average) share 2.2 of 3 reasons (v1: 2.32).
- **Submission files:** `pipeline/make_submissions.py --tag v2` adds the v2 models (`submission/`, 18 models;
  `analysis_output/submissions/manifest_v2.csv`). The first manifest is unchanged. XGBoost and MLP have no test
  predictions (1-seed exploratory runs write none).
- **Decision:** adopt `lgbm_v2_combo_lr02_x3` (CV AUC 0.7882 all loans, 0.7877 cash) as the model
  for the business step; keep `scorecard_full_no_gender` as the explainable comparison.

## 23. Business cut-off, project step 7
- **Run:** `analysis/business.py` (about 8 s, saved out-of-fold predictions only). Output
  `analysis_output/business/`. Models: `lgbm_v2_combo_lr02_x3` (the model chosen in the LightGBM v2 entry),
  `scorecard_full_no_gender` (explainable comparison), `lgbm_v1` (earlier model, with gender). Probabilities are the
  cross-fitted isotonic ones; profit is realised on the training applicants' actual outcomes, mean of 5 seeds.
- **Assumptions (not data):** a repaid loan earns a net margin of 20% of the credit, a loan with payment
  difficulties loses 60% of it (LGD); break-even probability 0.20 / (0.20 + 0.60) = 25%. Grid: margin 10/20/30%,
  LGD 40/60/80%. Currency not disclosed.
- **Base case (measured under the assumptions):** approving everyone earns 83,782 per applicant. The break-even
  rule with LightGBM v2 approves 93.9%, the difficulty rate among the approved falls from 8.07% to 6.28%, and the
  profit is 86,351 per applicant: +2,569 (+3.1%), seed sd 28. A cut-off chosen by cross-fitting (best on four
  folds, applied to the fifth) gives the same (94.1% approved, +2,556): the calibrated probabilities are good
  enough to set the cut-off directly from the money assumptions.
- **What the better model is worth (base case):** LightGBM v2 +2,569 per applicant vs approving everyone,
  scorecard +1,388, LightGBM v1 +2,273. So v2 earns about 1,180 per applicant more than the scorecard and about
  300 more than v1, far above the seed sd (18 to 43). The order v2 > v1 > scorecard holds in all 9 assumption pairs.
- **Sensitivity (measured):** the best approval rate depends strongly on the assumptions: from 77.5% (margin 10%,
  LGD 80%; break-even 11%) to 98.9% (margin 30%, LGD 40%; break-even 43%). The model is worth most when margins are
  thin and losses high (+11,001 per applicant vs approving everyone at margin 10%, LGD 80%) and little when margins
  are wide (+319 at 30% / 40%).
- **Compared with the provisional rule used since step 6 (decline 15%):** under the base case the profit-optimal
  policy declines only about 6%; at 85% approval the profit is 84,129 per applicant, 2,222 below the break-even
  rule (approval curve).
- **Who is approved under the base-case rule (LightGBM v2, measured):** women 95.0%, men 91.7%; under 25: 87.3%,
  65+: 99.5%. The scorecard is harder on the young (under 25: 83.5%) although it ranks less well overall.
- **Limits:** the target is "payment difficulties", not a measured loss, so LGD stands in for an unknown mix of
  late payers and true losses; only previously approved applicants are in the data (no reject inference); profit
  ignores loan term and timing. All business numbers are proposals until tested live.

## 24. Summary table of every model and the late-submission set
- **Run:** `experiments/model_summary.py` (about 10 s): `analysis_output/model_summary.csv`, one row per model
  (20 models incl. the rule) with its cross-validated AUC recomputed from the saved out-of-fold files. The values
  match the earlier log entries.
- **Submission set:** 18 files in `submission/` plus `submission/UPLOAD_ORDER.txt` (built from the summary table,
  in a suggested order of 5 a day), zipped for upload. Uploaded by hand from my own Kaggle account.

## 25. Kaggle late submissions, the independent check
- **Input:** `analysis_output/kaggle/kaggle_scores.csv`, private and public AUC of 11 LightGBM submissions,
  typed by hand from the Kaggle submissions page. **Run:**
  `experiments/kaggle_check.py`, output `analysis_output/kaggle/kaggle_vs_cv.csv` and `kaggle_steps.csv`.
  Nothing was chosen or changed because of these scores.
- **CV predicts the private score well (measured):** the chosen model `lgbm_v2_combo_lr02_x3` scores 0.78782 on
  the private leaderboard against 0.78771 CV on cash loans (+0.0001). Over the 11 models, private minus CV is
  +0.0009 on average, and the rank correlation of CV and private is 0.964. The test set is a different population
  (adversarial AUC 0.96), yet the validation held.
- **Public is higher than private for every model** (+0.0007 to +0.0069): the public part is only 20% of the test
  set, so it is the noisier of the two; private is the one to report.
- **Every step we adopted on CV also gains on private (measured, 10 steps, same sign in all 10):** new features
  +0.0068 (CV +0.0075), learning rate 0.02 +0.0023 (CV +0.0017), averaging 3 fits +0.0010 (CV +0.0015).
  Rough noise scale for one paired gain on the private part: 0.0005 to 0.0017 (CV bootstrap SE scaled to 38,995
  applicants; an assumption, see the script). Gains beyond 2 such units: 6 of 10.
- **Two differences in size, not in direction (reported, not acted on):** leaving gender and family out costs
  0.0034 on private in v2 (CV 0.0016), so the cost of the fairness choice may be about twice what CV says; the
  bureau_balance columns gain +0.0017 on private (CV +0.0002), plausibly because they are present for 99.9% of test
  credits. Acting on either would be tuning on the test set, so the chosen model stays as it is.
- **For scale (Kaggle leaderboard):** first place 0.80570 private, 99th place 0.79865; the chosen
  model's 0.78782 is below the top 99 (exact rank not checked).

## 26. Feature table v3 and SHAP feature reduction (aim: best result with fewer features)
- **Run:** `pipeline/make_features_v3.py` (about 23 s): `processed/features_v3.parquet`, v2 unchanged plus 542
  columns: `agg_` systematic aggregation of every side table (508), `last_` recent applications in detail (23),
  `trend_` changes over time (7), `load_` total debt burden (4). No gender, family or bureau_balance columns used
  in training (728 candidate columns).
- **All 728 columns, 1 seed (exploratory, `lgbm_v3_all_s0`, learning rate 0.05):** AUC 0.7908 vs 0.7844 for
  `lgbm_v2_combo` on the same 5 folds, +0.0064 (SE 0.0005, 5 of 5). About 130 s per fit, 4.8 GB peak memory per
  process.
- **SHAP reduction inside each training fold (`--shap-top K`, `homecredit.models.lgbm.shap_select`):** fit on
  the training part, mean |TreeSHAP| on 20,000 of its rows, keep the K largest, refit, score the validation fold.
  No validation row is used for the choice. 1 seed, vs all 728 columns: K=100 -0.0020 (0 of 5, worse); K=200
  -0.0009 (1 of 5); K=350 -0.0009 (2 of 5, no clear difference). The kept columns vary between folds (K=200: 124
  kept in all 5 folds, 295 in at least one). **Reading:** the trees already ignore weak columns, so SHAP
  reduction does not add accuracy here; its value is a smaller, simpler model at a small cost.
- **5 seeds (measured, learning rate 0.05, single fit):** all 728 columns (`lgbm_v3_all`) AUC 0.7904 all loans,
  0.7897 cash, +0.0054 vs `lgbm_v2_combo` (SE 0.0005, 25 of 25). SHAP top 350 (`lgbm_v3_shap350`) 0.7903 / 0.7897,
  +0.0053 (25 of 25). The two are the same: -0.0001 (SE 0.0004, 13 of 25 wins). **Decision (within noise, keep the simpler):**
  keep the SHAP-350 version, half the columns for the same accuracy. Run time 513 s (all) vs 892 s (the selection
  fit is extra).
- **Final v3 candidate (`lgbm_v3_shap350_lr02_x3`, measured):** SHAP top 350 chosen inside each fold, learning
  rate 0.02, 3 fits per fold: AUC 0.7939 all loans, 0.7933 cash. +0.0036 vs `lgbm_v3_shap350` (SE 0.0003, 25 of
  25) and +0.0057 vs the v2 model `lgbm_v2_combo_lr02_x3` (SE 0.0005, 25 of 25). 2,597 s for the 25 folds.
  Not yet done for v3: Kaggle submission, calibration, reason codes, fairness, business. Note (from an external
  review): K was picked on the seed-0 folds that are part of the 25-fold confirmation, a small forking path.

## 27. README rewritten (v2) and charts
- `README.md` rewritten in full around the chosen v2 model (a story, summary first); v3 is mentioned only as
  work in progress. Charts: `analysis/figures.py` (code in `homecredit/figures.py`) writes six PNGs to `figures/`
  from saved result files only.
- Two extra paired comparisons for README claims (measured): `lgbm_v2_combo_lr02_x3` vs `lgbm_v1` +0.0090 (SE
  0.0006, 25 of 25), vs `scorecard_full_no_gender` +0.0330 (SE 0.0008, 25 of 25).

## 28. Fixes after an external review (4 reviewers with no prior context; v2 only)
- **Review:** four independent AI reviewers, each with no prior context, read a clean clone of the repository (no Kaggle
  data): a credit-risk
  hiring manager, a non-technical reader, a senior data scientist, and a code reviewer who also ran read-only
  checks. No leakage of a validation row's own label was found. I re-checked every major finding myself; all held.
- **Calibration metric corrected (measured):** the pooled ECE of a cross-fitted isotonic map is biased low (pooled
  0.094 points for v2 isotonic, below the 0.160 that a perfectly calibrated model would show). New:
  `evaluate.ece_by_fold` and `ece_floor`, written by `analysis/calibration.py` to `ece_by_fold_v2.csv` (earlier
  files unchanged). Mean over 5 seeds, all loans: v2 raw 0.48 points per fold vs a noise floor of 0.35; isotonic
  0.36 vs 0.34, i.e. within noise. Brier and log loss barely change (0.06588 vs 0.06587). So the raw model was
  only slightly miscalibrated and isotonic removes that; the earlier "0.42 to 0.10" overstated the gain.
- **Kaggle check:** the hand-typed `kaggle_scores.csv` is now always read from the repository (`Paths.kaggle_scores`),
  so `run_all.sh` with an out-root no longer stops there; the rough private SE now scales with the 278,232 train
  cash loans (it used all 307,511, about 5% too large). New `kaggle_steps_v2.csv`: 7 of 10 steps beyond 2 rough SE
  (was 6 of 10). `kaggle_vs_cv_v2.csv` is identical to the earlier file.
- **run_all.sh:** added the two comparisons the README uses (`lgbm_v1` and `scorecard_full_no_gender` vs the chosen
  model); v3 steps moved to `run_v3.sh` (v3 is not part of the README yet). `pipeline/train_lgbm.py` now saves every
  fit of an average (`<name>_final.txt`, `<name>_1_final.txt`, ...); earlier model files are unchanged.
- **Business v2 (`analysis/business.py`, now writes `analysis_output/business_v2/`; the first version's files are
  kept, measured):** cut-offs now fall between distinct probability values (isotonic ties); 95% bootstrap
  intervals (1,000 resamples of applicants); six attributes by group; `lgbm_v1_no_gender_family` added.
  - Break-even rule unchanged: 93.9% approved, +2,569 per applicant vs approving everyone (95% CI 2,381 to 2,744).
    Cross-fitted cut-off: 93.9%, +2,558; difference to the rule -11 (CI -23 to +2), so the same.
  - The cost of explainability alone (same v1 columns, no gender or family): LightGBM v1 vs scorecard +811 per
    applicant (CI 693 to 945). The v2 features and settings add +369 (CI 293 to 446). The earlier README figure of
    1,180 (v2 vs scorecard, CI 1,050 to 1,328) mixed the two.
  - Groups under the rule: lowest adverse impact ratio 0.877 (under 25: 87.3% vs 65+: 99.5%); region rating 3
    0.909; lower secondary education 0.928; men 0.965. All above the 0.8 screening value (a US employment
    heuristic, not a legal test for credit). Women's predicted rate 7.35% vs 7.00% observed; men's 9.47% vs 10.14%.
  - The population is applicants Home Credit approved at the time, so the rule tightens the current policy by
    about 6%; nothing can be said about applicants who were declined then.
- **Possible repeat applicants (`experiments/duplicate_check.py`, 1 seed, exploratory, measured):** 12,411 train
  rows in 5,991 groups share birth date, ID-document date and gender (a guess at "same person"); 4,871 groups are
  split across seed-0 folds; when both of a pair are in train, both defaulted 1.11% vs 0.45% if independent. The
  plain v2 LightGBM on grouped folds (each group in one fold) scores AUC 0.7856 vs 0.7843 on the usual folds (pooled
  out-of-fold AUC), and 0.7748 vs 0.7744 on the repeated rows. No sign that the usual split flatters the CV.
- **Proxy check (`analysis/proxy_check.py`, LightGBM 5 folds seed 0, no TARGET, measured):** from the chosen
  model's own 186 columns, gender is predicted with AUC 0.907 (occupation type 33% of the gain, organisation type,
  car age); married or civil marriage 0.752; and, without the age columns, under 25 with 0.992 (ID-document date
  38%) and 55 or over with 0.969 (years employed). So dropping gender and family does not hide them from the model;
  the choice is a legal and policy one, and the same argument used to keep age (removing it hurt the young) also
  applies to gender (without it women's risk is overstated: 7.35% predicted vs 7.00% observed, business_v2).
- **Error analysis on the chosen model (`analysis/error_analysis.py --model lgbm_v2_combo_lr02_x3 --tag v2`,
  measured):** same picture as v1. Weakest groups: under 25 (AUC 0.745), pensioners (0.758), 55+ (0.759-0.760),
  longest-term loans (0.752); a defaulter aged 65+ costs 2.1 times the average share of wrong pairs. The safest
  quarter of defaulters averages 4.2% predicted risk vs 7.1% for good payers. The stacking signals that v1 missed
  are now absorbed: 2+ loans still running: observed minus predicted -0.20 points (v1: +0.65); 2+ cancelled
  applications +0.14 (v1: +0.38); 2+ new other-lender credits +0.78 (SE 0.44; v1: +1.11).
- **Reason codes at the proposed cut-off (`analysis/reason_codes.py ... --decline-rate 0.06 --with-fourth --tag
  v2_cut06`, about 11 min, measured):** 5.7% of applicants declined (threshold from each training part). Replacing
  the 1st / 2nd / 3rd / 4th reason group with an approved applicant's values lowers risk by 20.1 / 11.2 / 7.4 / 5.0
  points and turns 82.8% / 56.9% / 40.8% / 29.6% of declines into approvals; a random low-impact group: 0.6 points,
  8.5%. Order held in all 5 folds, including 4th > random. External scores are the first reason for 81.2% of
  declines; age is in the top 3 for 0.10%. LightGBM and scorecard (approved-average method) share 2.24 of 3 reasons.
  Caveat (review): this shows the reasons are faithful to the model, not that they are causal; the SHAP values of
  an average of 3 models are averaged in log-odds while the prediction averages probabilities (a small mismatch).
- **README rewritten after the review:** glossary, units, symmetric fairness framing with the proxy check, the
  explainability cost on the same columns (811, not 1,180), per-fold calibration numbers, "within sampling error"
  instead of "exactly", honest tuning statement, reject inference next to the business result, a "putting it to
  work" section (proposals), and the error analysis and reason codes of the chosen model. Charts and notebook
  updated to the new result files (calibration chart now shows the raw out-of-fold model only).

## 29. Second review round (three new AI reviewers, no code review) and fixes
- **Review:** a hiring manager, a non-technical reader and a senior data scientist read a fresh copy of the repo.
  All saw a clear improvement; no critical issue, no leakage. Main points: wording that still overclaimed (the
  "810" in the summary, two cut-offs mixed, "no inflation" for repeat applicants, "missed defaulters" partly by
  construction), and tables a credit team expects. One reviewer thought the "rerun in progress" line was stale; it
  was not (the only rerun report then was from the v1 stage).
- **Credit tables (`analysis/credit_tables.py`, measured):** score bands of the chosen model (10 equal bands; top
  band 30.5% difficulty rate, bottom 0.8%); KS 0.435 (sd 0.0007) vs scorecard 0.382. Swap-set at the break-even
  rule, seed 0: 1.39% approved by LightGBM only (18.2% difficulties), 3.29% by the scorecard only (31.1%). PSI
  train vs Kaggle test: model score 0.005; inputs: annuity_to_credit 0.999, EXT_SOURCE_1 0.085, the rest below 0.05.
- **Direct effect of age and gender at a 6% decline rule (`analysis/sensitive_counterfactual.py`, v2 columns,
  learning rate 0.05, one fit, seed-0 folds; refits match the saved predictions exactly):** setting every
  applicant's age to 45 changes 1.0% of decisions overall and 6.4% for under 25, whose decline rate rises from
  11.7% to 17.7% (risk +2.5 points): conditional on the other inputs, a young age lowers risk. Switching gender in
  the model that has gender changes 2.0% of decisions (women 4.35% declined to 6.07%; men 8.48% to 6.40%).
- **README revised:** one cut-off for decisions (about 6%) with the 15% figure only as "stricter"; money given a
  scale (83,782 to 86,351 per applicant); the 1,180 vs 810 split explained; good-payer approval gap (9.5 points,
  under 25 vs 65+); age decision argued with the new counterfactual; legal framing without naming a country;
  repeat-applicant check called inconclusive at about 0.001; Kaggle coverage stated (11 of 18 files scored);
  score-band table, swap-set and PSI added. `kaggle_scores.csv` source note reworded (values unchanged).
