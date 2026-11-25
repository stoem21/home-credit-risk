"""Fairness (project step 8): what do the sensitive attributes cost or buy, and how do decisions differ by group?

Models compared (all trained with the same folds and settings; only the input columns differ):
  lgbm_v1, lgbm_v1_no_gender, lgbm_v1_no_gender_family, lgbm_v1_no_sensitive (no gender, family or age columns);
  scorecard_full, scorecard_full_no_gender (the scorecard uses no age or family column anyway).

  1. Accuracy cost: paired comparison of each variant against its full model (same rule as experiments/compare.py).
  2. Decisions by group, under the provisional rule used since step 6: decline the riskiest 15% (per seed, the
     85th percentile of that model's out-of-fold predictions; same approval rate for every model, so groups can be
     compared across models). Per group: approval rate; approval rate among applicants who repaid (good payers)
     and among defaulters; default rate among the approved; mean predicted vs observed default rate. Mean over 5
     seeds. The real cut-off comes from the business step.
  3. Gender counterfactual: the full models are refitted on the seed-0 training folds (identical to the saved
     models; checked against the saved predictions) and each validation applicant is scored again with only the
     gender value switched (F <-> M). Share of decisions that change, and the average change in predicted risk.

Groups smaller than 1,000 applicants (gender XNA, family "Unknown", education "Academic degree") are kept in the
tables but left out of the disparity summary, where a handful of applicants would make the ratios noise.

The "four-fifths" ratio (lowest group approval rate / highest) is shown as a common screening number from US
employment practice; it is not a legal test for credit, and no threshold here is a legal judgment.

Usage:
    .venv/bin/python -m analysis.fairness [--jobs 5] [--threads 2] [--out-root DIR]
    .venv/bin/python -m analysis.fairness --reference FULL_MODEL --variants MODEL[,MODEL] --tag v2
        (parts 1 and 2 for other saved models; no counterfactual)
Outputs: analysis_output/fairness/accuracy_cost.csv, group_decisions.csv, disparity_summary.csv,
         gender_counterfactual.csv
"""

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from homecredit.data import N_SEEDS, load_features, load_folds, read_raw, scopes
from homecredit.evaluate import load_oof, paired_compare
from homecredit.models import lgbm
from homecredit.models import scorecard as sc
from homecredit.paths import Paths, add_out_root

DECLINE_RATE = 0.15
MIN_GROUP = 1000
VARIANTS = {"lgbm_v1": ["lgbm_v1", "lgbm_v1_no_gender", "lgbm_v1_no_gender_family", "lgbm_v1_no_sensitive"],
            "scorecard_full": ["scorecard_full", "scorecard_full_no_gender"]}
SWAP = {"F": "M", "M": "F", "XNA": "XNA"}


def groups(app: pd.DataFrame) -> dict[str, pd.Series]:
    age = -app.DAYS_BIRTH / 365.25
    return {
        "gender": app.CODE_GENDER.astype(str),
        "age band": pd.cut(age, [0, 25, 35, 45, 55, 65, 100], right=False,
                           labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"]).astype(str),
        "family status": app.NAME_FAMILY_STATUS.astype(str),
        "children": app.CNT_CHILDREN.clip(upper=3).map(lambda v: "3+" if v >= 3 else str(int(v))),
        "education": app.NAME_EDUCATION_TYPE.astype(str),
        "region rating": app.REGION_RATING_CLIENT.astype(str),
    }


def counterfactual_fold(model: str, k: int, out_root, threads: int):
    P = Paths(out_root)
    tr, _, cols = load_features(P, categories=(model == "lgbm_v1"))
    fo = load_folds(P, tr.SK_ID_CURR).fold_s0.values
    trn, va = np.where(fo != k)[0], np.where(fo == k)[0]
    y = tr.TARGET.values
    Xva = tr.iloc[va]
    Xsw = Xva.copy()
    Xsw["CODE_GENDER"] = Xva.CODE_GENDER.astype(object).map(SWAP).astype(Xva.CODE_GENDER.dtype)
    if model == "lgbm_v1":
        bst = lgbm.fit(tr[cols].iloc[trn], y[trn].astype(int), seed=0, threads=threads)
        p = bst.predict(Xva[cols], num_iteration=bst.best_iteration)
        p_sw = bst.predict(Xsw[cols], num_iteration=bst.best_iteration)
    else:
        specs, selected, lr, _ = sc.fit_scorecard(tr.iloc[trn], y[trn].astype(float), cols)
        p, p_sw = sc.predict(Xva, specs, selected, lr), sc.predict(Xsw, specs, selected, lr)
    return model, va, p, p_sw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--reference", default=None, help="another full model (with all columns) to compare against")
    ap.add_argument("--variants", default=None, help="comma list of models trained without some columns")
    ap.add_argument("--tag", default="", help="write to analysis_output/fairness_<tag>/ (earlier results stay)")
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    out = P.out("fairness" + (f"_{args.tag}" if args.tag else ""))
    variants = VARIANTS
    if args.reference:
        variants = {args.reference: [args.reference, *args.variants.split(",")]}
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)

    app = read_raw(P, "application_train")
    y = app.TARGET.values.astype(int)
    folds = load_folds(P, app.SK_ID_CURR)
    masks = scopes(app.NAME_CONTRACT_TYPE)
    oofs = {m: load_oof(P, m, app.SK_ID_CURR) for fam in variants.values() for m in fam}

    # ---- 1. accuracy cost ----
    cost = []
    for ref, fam in variants.items():
        for m in fam[1:]:
            for r in paired_compare(y, oofs[ref], oofs[m], folds, masks):
                cost.append({"reference": ref, "variant": m, **r})
    cost = pd.DataFrame(cost)
    cost.to_csv(out / "accuracy_cost.csv", index=False)
    print(cost.to_string(index=False), "\n")

    # ---- 2. decisions by group ----
    grp = groups(app)
    rows = []
    for m, oof in oofs.items():
        approve = np.column_stack([oof[f"s{s}"].values < np.quantile(oof[f"s{s}"].values, 1 - DECLINE_RATE)
                                   for s in range(N_SEEDS)])
        p_mean = np.column_stack([oof[f"s{s}"].values for s in range(N_SEEDS)]).mean(axis=1)
        for attr, lab in grp.items():
            for g in sorted(lab.unique()):
                gm = (lab == g).values
                good, bad = gm & (y == 0), gm & (y == 1)
                appr = approve[gm].mean(axis=0)
                rows.append({
                    "model": m, "attribute": attr, "group": g, "applicants": int(gm.sum()),
                    "default_rate_pct": round(100 * y[gm].mean(), 2), "mean_pred_pct": round(100 * p_mean[gm].mean(), 2),
                    "approval_pct": round(100 * appr.mean(), 2), "approval_pct_sd_seeds": round(100 * appr.std(ddof=1), 2),
                    "approval_pct_good_payers": round(100 * approve[good].mean(), 2),
                    "approval_pct_defaulters": round(100 * approve[bad].mean(), 2) if bad.any() else np.nan,
                    "default_pct_among_approved": round(100 * np.mean(
                        [y[gm & approve[:, s]].mean() for s in range(N_SEEDS)]), 2)})
    dec = pd.DataFrame(rows)
    dec.to_csv(out / "group_decisions.csv", index=False)

    summ = []
    for (m, attr), d in dec[dec.applicants >= MIN_GROUP].groupby(["model", "attribute"], sort=False):
        lo, hi = d.loc[d.approval_pct.idxmin()], d.loc[d.approval_pct.idxmax()]
        summ.append({"model": m, "attribute": attr,
                     "lowest_group": lo.group, "lowest_approval_pct": lo.approval_pct,
                     "highest_group": hi.group, "highest_approval_pct": hi.approval_pct,
                     "approval_ratio_low_high": round(lo.approval_pct / hi.approval_pct, 3),
                     "good_payer_approval_gap_pts": round(d.approval_pct_good_payers.max()
                                                         - d.approval_pct_good_payers.min(), 2),
                     "max_calibration_gap_pts": round((d.default_rate_pct - d.mean_pred_pct).abs().max(), 2)})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "disparity_summary.csv", index=False)
    print(summ.to_string(index=False), "\n")
    show = dec[(dec.attribute.isin(["gender", "age band"])) & (dec.applicants >= MIN_GROUP)]
    print(show.drop(columns=["approval_pct_sd_seeds"]).to_string(index=False), "\n")

    # ---- 3. gender counterfactual (only for the v1 models, refitted here) ----
    if args.reference:
        print("gender counterfactual skipped: it refits the v1 models only")
        return
    res = Parallel(n_jobs=args.jobs)(delayed(counterfactual_fold)(m, k, args.out_root, args.threads)
                                     for m in VARIANTS for k in range(5))
    cf = []
    gender = grp["gender"].values
    for m in VARIANTS:
        p, p_sw = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for mm, va, a, b in res:
            if mm == m:
                p[va], p_sw[va] = a, b
        saved = oofs[m].s0.values
        thr = np.quantile(saved, 1 - DECLINE_RATE)
        d0, d1 = p >= thr, p_sw >= thr
        for g in ["F", "M"]:
            gm = gender == g
            cf.append({"model": m, "gender": g, "applicants": int(gm.sum()),
                       "max_abs_diff_vs_saved_predictions": float(np.abs(p - saved).max()),
                       "mean_risk_change_pts_if_switched": round(100 * (p_sw - p)[gm].mean(), 3),
                       "pct_decisions_changed": round(100 * (d0 != d1)[gm].mean(), 2),
                       "pct_approved_then_declined": round(100 * (~d0 & d1)[gm].mean(), 2),
                       "pct_declined_then_approved": round(100 * (d0 & ~d1)[gm].mean(), 2)})
    cf = pd.DataFrame(cf)
    cf.to_csv(out / "gender_counterfactual.csv", index=False)
    print(cf.to_string(index=False))


if __name__ == "__main__":
    main()
