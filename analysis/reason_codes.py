"""Reason codes (project step 6): why was an applicant declined, and do the stated reasons really drive it?

Setting. Seed-0 folds; for each fold, a LightGBM and a scorecard are fitted on the training part (same code and
settings as homecredit/models/lgbm.py and scorecard.py) and explain the validation part, so no applicant is
explained by a model that saw their outcome.

Provisional decline rule: decline the riskiest 15% (threshold = 85th percentile of the model's predictions on its
own training part). The real cut-off is chosen in the business step; 15% is only a working value.

Columns are grouped into reasons a person can read (GROUPS below), because "EXT_SOURCE_2 = 0.31" means nothing to
an applicant and many columns describe the same thing.
  - scorecard: for each variable, points below the best bin of that variable ("points lost"); summed per group.
  - LightGBM: TreeSHAP contributions (LightGBM pred_contrib, in log-odds); summed per group (SHAP is additive).
  The top 3 groups with the largest loss / positive contribution are the reason codes.

Counterfactual check (lesson from the previous project: a large SHAP share is not proof of impact on decisions).
For each declined applicant (LightGBM), the columns of one reason group are replaced by those of 5 randomly drawn
approved applicants from the training part, one at a time. We record how much the predicted risk falls and how
often the decision flips to approve. Tested: top-1, top-2 and top-3 reason, and as a control a random group that
was not among the top 3. If the reasons are meaningful, top-1 should move the decision most and the control least.

Examples file: five applicants declined by both models, with their reasons. It holds no applicant ID and no
outcome, so it shows no individual record of the competition data.

Usage:
    .venv/bin/python -m analysis.reason_codes [--decline-rate 0.15] [--jobs 5] [--out-root DIR]
        [--features v2 --add win,stk,term --exclude stk_bb_ --drop gender,family --learning-rate 0.02
         --n-models 3 --tag v2]   (another LightGBM setting; the scorecard keeps the v1 columns minus --drop)
Outputs: analysis_output/reasons/*.csv
"""

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from homecredit.data import drop_columns, load_features, load_folds, select_groups
from homecredit.models import lgbm
from homecredit.models import scorecard as sc
from homecredit.paths import Paths, add_out_root

N_DONORS = 5
TOP = 3
SENSITIVE = {"Age [sensitive]", "Gender [sensitive]", "Family [sensitive]"}
NOT_A_FAIR_REASON = {"Application process (day, hour, companion)", "Contact details"}

LOAN = {"AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "AMT_INCOME_TOTAL", "credit_to_income", "annuity_to_income",
        "credit_to_goods", "annuity_to_credit", "NAME_CONTRACT_TYPE"}
EMPLOY = {"DAYS_EMPLOYED", "employed_years", "no_employment_code", "NAME_INCOME_TYPE", "OCCUPATION_TYPE",
          "ORGANIZATION_TYPE"}
HOUSING = {"FLAG_OWN_REALTY", "NAME_HOUSING_TYPE", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "TOTALAREA_MODE",
           "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"}


def group_of(c: str) -> str:
    # feature table v3 groups (pipeline/make_features_v3.py)
    if c.startswith(("agg_bur", "trend_bur")):
        return "Loans with other lenders"
    if c.startswith(("agg_prev", "last_")):
        return "Earlier Home Credit applications"
    if c.startswith(("agg_inst", "agg_pos", "trend_inst")):
        return "Repayment of earlier Home Credit loans"
    if c.startswith(("agg_cc", "trend_cc")):
        return "Credit card use"
    if c.startswith("load_"):
        return "Total debt burden"
    # feature table v2 groups (pipeline/make_features_v2.py)
    if c.startswith(("win_prev_", "win_last_", "term_prev_")):
        return "Earlier Home Credit applications"
    if c.startswith(("win_inst_", "win_pos_")):
        return "Repayment of earlier Home Credit loans"
    if c.startswith("win_cc_"):
        return "Credit card use"
    if c.startswith("stk_hc_"):
        return "Loans still running at Home Credit"
    if c.startswith("stk_"):
        return "Loans with other lenders"
    if c.startswith("term_"):
        return "Loan size and terms vs income"
    if c.startswith(("EXT_SOURCE", "ext_")):
        return "External credit scores"
    if c.startswith(("inst_", "pos_")):
        return "Repayment of earlier Home Credit loans"
    if c.startswith("prev_"):
        return "Earlier Home Credit applications"
    if c.startswith("cc_"):
        return "Credit card use"
    if c.startswith("bureau_"):
        return "Loans with other lenders"
    if c.startswith("AMT_REQ_CREDIT_BUREAU"):
        return "Recent credit enquiries"
    if c in LOAN:
        return "Loan size and terms vs income"
    if c in EMPLOY:
        return "Employment and income type"
    if c in {"DAYS_BIRTH", "age_years"}:
        return "Age [sensitive]"
    if c == "CODE_GENDER":
        return "Gender [sensitive]"
    if c in {"NAME_FAMILY_STATUS", "CNT_CHILDREN", "CNT_FAM_MEMBERS"}:
        return "Family [sensitive]"
    if c == "NAME_EDUCATION_TYPE":
        return "Education"
    if c.startswith(("REGION_", "REG_", "LIVE_")):
        return "Region and address"
    if c in HOUSING or c.endswith(("_AVG", "_MODE", "_MEDI")):
        return "Housing"
    if c in {"FLAG_OWN_CAR", "OWN_CAR_AGE"}:
        return "Car ownership"
    if c.startswith("FLAG_DOCUMENT") or c in {"DAYS_ID_PUBLISH", "DAYS_REGISTRATION"}:
        return "Documents and ID"
    if c.startswith("FLAG_") or c == "DAYS_LAST_PHONE_CHANGE":
        return "Contact details"
    if "SOCIAL_CIRCLE" in c:
        return "Defaults in social circle"
    if c in {"WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START", "NAME_TYPE_SUITE"}:
        return "Application process (day, hour, companion)"
    return "Other"


def top_groups(mat: np.ndarray, groups: list[str], k: int = TOP) -> list[list[str]]:
    """Per row, the k groups with the largest positive value (fewer if fewer are positive)."""
    order = np.argsort(-mat, axis=1)[:, :k]
    return [[groups[j] for j in row if mat[i, j] > 0] for i, row in enumerate(order)]


_cache = {}


class Averaged:
    """The mean of n LightGBM boosters (n = 1: a single model), with the same predict interface."""

    def __init__(self, boosters):
        self.b = boosters

    def predict(self, X, pred_contrib=False):
        return np.mean([b.predict(X, num_iteration=b.best_iteration, pred_contrib=pred_contrib) for b in self.b],
                       axis=0)


def columns(P, cfg):
    """LightGBM columns and scorecard columns for a model setting (cfg: features, add, exclude, drop)."""
    tr, _, cols = load_features(P, cfg["features"], categories=True)
    lgb_cols = [c for c in drop_columns(select_groups(cols, cfg["add"]), cfg["drop"])
                if not (cfg["exclude"] and c.startswith(cfg["exclude"]))]
    sc_cols = drop_columns(select_groups(cols, None), cfg["drop"])  # the scorecard keeps the v1 candidate columns
    return tr, lgb_cols, sc_cols


def run_fold(k: int, decline_rate: float, threads: int, out_root, cfg: dict):
    if "tr" not in _cache:
        P = Paths(out_root)
        tr, cols, sc_cols = columns(P, cfg)
        _cache.update(tr=tr, cols=cols, sc_cols=sc_cols, fo=load_folds(P, tr.SK_ID_CURR))
    tr, cols, sc_cols, fo = _cache["tr"], _cache["cols"], _cache["sc_cols"], _cache["fo"]
    groups = sorted(set(group_of(c) for c in cols))  # from all candidate columns, the same in every fold
    f = fo.fold_s0.values
    trn, va = np.where(f != k)[0], np.where(f == k)[0]
    y = tr.TARGET.values.astype(int)
    if cfg.get("shap_top"):  # the same in-fold SHAP selection as pipeline/train_lgbm.py --shap-top
        cols = lgbm.shap_select(tr[cols].iloc[trn], y[trn], cfg["shap_top"], 0, threads)
    gidx = {g: [cols.index(c) for c in cols if group_of(c) == g] for g in groups}
    Xtr, Xva = tr[cols].iloc[trn], tr[cols].iloc[va]

    # ---- LightGBM ----
    bst = Averaged([lgbm.fit(Xtr, y[trn], seed=1000 * i, threads=threads, learning_rate=cfg["learning_rate"])
                    for i in range(cfg["n_models"])])
    p_tr = bst.predict(Xtr)
    p_va = bst.predict(Xva)
    thr = np.quantile(p_tr, 1 - decline_rate)
    contrib = bst.predict(Xva, pred_contrib=True)[:, :-1]  # last column = expected value; SHAP averages linearly
    shap_g = np.column_stack([contrib[:, gidx[g]].sum(axis=1) for g in groups])

    # ---- Scorecard ----
    yf = y[trn].astype(float)
    specs, selected, lr, _ = sc.fit_scorecard(tr.iloc[trn], yf, sc_cols)
    factor = sc.FACTOR
    # Two common ways to turn points into reasons:
    #   lost     = points below the best bin of each variable ("points below maximum");
    #   lost_avg = points below the average points of approved applicants (training part), floored at 0.
    lost = np.zeros((len(va), len(groups)))
    lost_avg = np.zeros((len(va), len(groups)))
    sc_logit = np.full(len(va), lr.intercept_[0])
    sc_logit_tr = np.full(len(trn), lr.intercept_[0])
    pts_tr = {}
    for c, b in zip(selected, lr.coef_[0]):
        w_va = sc.transform(tr[c].iloc[va], specs[c])
        w_tr = sc.transform(tr[c].iloc[trn], specs[c])
        sc_logit += b * w_va
        sc_logit_tr += b * w_tr
        pts_tr[c] = -factor * b * w_tr
        valid = specs[c]["woe"][specs[c]["n"] > 0]
        best = (-factor * b * valid).max()
        lost[:, groups.index(group_of(c))] += best - (-factor * b * w_va)
    sc_p = 1 / (1 + np.exp(-sc_logit))
    sc_p_tr = 1 / (1 + np.exp(-sc_logit_tr))
    sc_thr = np.quantile(sc_p_tr, 1 - decline_rate)
    approved_sc = sc_p_tr < sc_thr
    for c, b in zip(selected, lr.coef_[0]):
        ref = pts_tr[c][approved_sc].mean()
        pts_va = -factor * b * sc.transform(tr[c].iloc[va], specs[c])
        lost_avg[:, groups.index(group_of(c))] += np.maximum(ref - pts_va, 0)

    # ---- Counterfactual on LightGBM declines ----
    rng = np.random.default_rng(k)
    dec = np.where(p_va >= thr)[0]
    approved_tr = np.where(p_tr < thr)[0]
    top_lgb = top_groups(shap_g[dec], groups)
    top4 = top_groups(shap_g[dec], groups, k=4)
    ranks = ["top1", "top2", "top3", "control"] + (["top4"] if cfg.get("with_fourth") else [])
    cf_rows = []
    Xdec = Xva.iloc[dec]
    for rank in ranks:
        chosen = []
        for i, tg in enumerate(top_lgb):
            if rank == "top4":  # a fairer control: the next reason after the top 3, not a random unused one
                chosen.append(top4[i][3] if len(top4[i]) > 3 else None)
            elif rank == "control":
                others = [g for g in groups if g not in tg and gidx[g]]  # groups the model uses
                chosen.append(others[rng.integers(len(others))])
            else:
                r = int(rank[-1]) - 1
                chosen.append(tg[r] if r < len(tg) else None)
        drops, flips = np.full((len(dec), N_DONORS), np.nan), np.full((len(dec), N_DONORS), np.nan)
        for g in groups:
            rows = np.array([i for i, cg in enumerate(chosen) if cg == g])
            if not len(rows):
                continue
            gcols = [cols[j] for j in gidx[g]]
            for d in range(N_DONORS):
                donors = Xtr.iloc[rng.choice(approved_tr, len(rows))]
                Xcf = Xdec.iloc[rows].copy()
                for c in gcols:
                    Xcf[c] = donors[c].values
                pcf = bst.predict(Xcf)
                drops[rows, d] = p_va[dec][rows] - pcf
                flips[rows, d] = (pcf < thr).astype(float)
        ok = ~np.isnan(drops[:, 0])
        cf_rows.append({"fold": k, "rank": rank, "declined": len(dec), "tested": int(ok.sum()),
                        "mean_risk_drop_pts": 100 * np.nanmean(drops),
                        "flip_to_approve_pct": 100 * np.nanmean(flips)})

    return {"k": k, "va": va, "p_va": p_va, "thr": thr, "shap_g": shap_g, "sc_p": sc_p, "sc_thr": sc_thr,
            "lost": lost, "lost_avg": lost_avg, "groups": groups, "cf": cf_rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decline-rate", type=float, default=0.15)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--features", default="v1")
    ap.add_argument("--add", default=None)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--drop", default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--n-models", type=int, default=1)
    ap.add_argument("--shap-top", type=int, default=None)
    ap.add_argument("--with-fourth", action="store_true",
                    help="also test the 4th-ranked reason group (a control matched closer in importance)")
    ap.add_argument("--tag", default="", help="write to analysis_output/reasons_<tag>/ (earlier results stay)")
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    out = P.out("reasons" + (f"_{args.tag}" if args.tag else ""))
    cfg = {"features": args.features, "add": args.add, "exclude": args.exclude, "drop": args.drop,
           "learning_rate": args.learning_rate, "n_models": args.n_models, "shap_top": args.shap_top,
           "with_fourth": args.with_fourth}

    tr, cols, _ = columns(P, cfg)
    pd.DataFrame({"column": cols, "reason_group": [group_of(c) for c in cols]}).to_csv(
        out / "reason_groups.csv", index=False)

    res = Parallel(n_jobs=args.jobs)(
        delayed(run_fold)(k, args.decline_rate, args.threads, args.out_root, cfg) for k in range(5))
    groups = res[0]["groups"]
    n = len(tr)
    p, sc_p = np.zeros(n), np.zeros(n)
    lgb_dec, sc_dec = np.zeros(n, bool), np.zeros(n, bool)
    shap_g, lost = np.zeros((n, len(groups))), np.zeros((n, len(groups)))
    lost_avg = np.zeros((n, len(groups)))
    for r in res:
        va = r["va"]
        p[va], sc_p[va] = r["p_va"], r["sc_p"]
        lgb_dec[va], sc_dec[va] = r["p_va"] >= r["thr"], r["sc_p"] >= r["sc_thr"]
        shap_g[va], lost[va], lost_avg[va] = r["shap_g"], r["lost"], r["lost_avg"]
    y = tr.TARGET.values

    top_l = top_groups(shap_g, groups)
    top_s = top_groups(lost, groups)
    top_sa = top_groups(lost_avg, groups)
    methods = [("lightgbm (SHAP)", lgb_dec, top_l), ("scorecard (points below max)", sc_dec, top_s),
               ("scorecard (points below approved average)", sc_dec, top_sa)]

    # 1. Which reasons come first, among each model's own declines
    share = []
    for model, dec, tops in methods:
        idx = np.where(dec)[0]
        first = pd.Series([tops[i][0] if tops[i] else "(none)" for i in idx]).value_counts(normalize=True)
        any3 = pd.Series([g for i in idx for g in tops[i]]).value_counts() / len(idx)
        for g in sorted(set(first.index) | set(any3.index)):
            share.append({"model": model, "reason_group": g,
                          "pct_declines_top1": round(100 * first.get(g, 0), 2),
                          "pct_declines_in_top3": round(100 * any3.get(g, 0), 2)})
        print(f"{model}: declined {dec.sum():,} ({100 * dec.mean():.1f}%), default rate among declined "
              f"{100 * y[dec].mean():.2f}% vs approved {100 * y[~dec].mean():.2f}%")
    share_df = pd.DataFrame(share).sort_values(["model", "pct_declines_top1"], ascending=[True, False])
    share_df.to_csv(out / "top_reason_share.csv", index=False)
    print(share_df.to_string(index=False), "\n")

    # 2. Sensitive or unfair reasons among the top 3
    flag = []
    for model, dec, tops in methods:
        idx = np.where(dec)[0]
        for label, bad in [("sensitive (age, gender, family)", SENSITIVE),
                           ("not a fair reason (application day/hour, contact details)", NOT_A_FAIR_REASON)]:
            flag.append({"model": model, "flag": label,
                         "pct_declines_top1": round(100 * np.mean([bool(tops[i]) and tops[i][0] in bad for i in idx]), 2),
                         "pct_declines_any_top3": round(100 * np.mean([any(g in bad for g in tops[i]) for i in idx]), 2)})
    flag_df = pd.DataFrame(flag)
    flag_df.to_csv(out / "sensitive_reasons.csv", index=False)
    print(flag_df.to_string(index=False), "\n")

    # 3. Do the two models give the same reasons? (applicants declined by both)
    both = np.where(lgb_dec & sc_dec)[0]
    agree = pd.DataFrame([{
        "scorecard_method": name,
        "declined_by_lightgbm": int(lgb_dec.sum()), "declined_by_scorecard": int(sc_dec.sum()),
        "declined_by_both": len(both),
        "pct_same_top1": round(100 * np.mean([bool(top_l[i]) and bool(ts[i]) and top_l[i][0] == ts[i][0]
                                              for i in both]), 2),
        "mean_shared_of_top3": round(np.mean([len(set(top_l[i]) & set(ts[i])) for i in both]), 2)}
        for name, ts in [("points below max", top_s), ("points below approved average", top_sa)]])
    agree.to_csv(out / "model_agreement.csv", index=False)
    print(agree.to_string(index=False), "\n")

    # 4. Counterfactual check
    cf = pd.DataFrame([row for r in res for row in r["cf"]])
    cf.to_csv(out / "counterfactual_by_fold.csv", index=False)
    cf_sum = cf.groupby("rank", sort=False).apply(lambda d: pd.Series({
        "tested": int(d.tested.sum()),
        "mean_risk_drop_pts": np.average(d.mean_risk_drop_pts, weights=d.tested),
        "flip_to_approve_pct": np.average(d.flip_to_approve_pct, weights=d.tested),
        "folds_where_top_beats_next": None}), include_groups=False).reset_index()
    order = cf.pivot_table(index="fold", columns="rank", values="mean_risk_drop_pts")
    cf_sum.loc[cf_sum["rank"] == "top1", "folds_where_top_beats_next"] = int((order.top1 > order.top2).sum())
    cf_sum.loc[cf_sum["rank"] == "top2", "folds_where_top_beats_next"] = int((order.top2 > order.top3).sum())
    cf_sum.loc[cf_sum["rank"] == "top3", "folds_where_top_beats_next"] = int((order.top3 > order.control).sum())
    if "top4" in order:
        cf_sum.loc[cf_sum["rank"] == "top4", "folds_where_top_beats_next"] = int((order.top4 > order.control).sum())
    cf_sum.to_csv(out / "counterfactual_summary.csv", index=False)
    print(cf_sum.round(3).to_string(index=False), "\n")

    # 5. Examples: five applicants declined by both models, with their reasons
    rng = np.random.default_rng(0)
    ex = []
    for j, i in enumerate(rng.choice(both, 5, replace=False)):
        ex.append({"example": j + 1,
                   "lightgbm_risk_pct": round(100 * p[i], 1), "scorecard_risk_pct": round(100 * sc_p[i], 1),
                   "lightgbm_reasons": " | ".join(top_l[i]),
                   "scorecard_reasons_below_max": " | ".join(top_s[i]),
                   "scorecard_reasons_below_avg": " | ".join(top_sa[i])})
    ex_df = pd.DataFrame(ex)
    ex_df.to_csv(out / "examples_no_ids.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(ex_df.to_string(index=False))


if __name__ == "__main__":
    main()
