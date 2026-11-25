"""Feature table v3: v2 plus four more groups, built for breadth; the model later keeps only the most useful
columns, chosen inside each training fold with SHAP (pipeline/train_lgbm.py --shap-top).

Ideas: systematic aggregation of every side table (public top solutions), the most recent applications in more
detail, time trends, and the total debt burden (our own error analysis found loan stacking under-predicted).
All side-table records are dated before the application; no feature uses TARGET; bureau_balance is not used
(its coverage differs between train and test, EDA step 1 and the v2 drift check). 365243 "no date" codes in
previous_application are set to missing before aggregating.

Groups (prefix -> what):
  agg_   every numeric column of bureau, previous_application, installments (one row per instalment),
         POS_CASH_balance and credit_card_balance, summarised per applicant (mean, max, min, sum where it makes
         sense), for all records and for meaningful subsets (active / closed credits; approved / refused
         applications; the last 12 months or 365 days); shares of the main categories of previous applications
         and bureau credit types.
  last_  the 3 and 5 most recent Home Credit applications (amounts, term, refusal share), the first application,
         and more fields of the most recent one (channel, goods, reject reason, client type).
  trend_ change over time: late payment and amounts paid in the last 6 months vs the 6 before; card use in the
         last 3 months vs months 4-12; bureau credits opened in the last year vs the year before.
  load_  total debt burden: this loan's instalment plus the instalments of active other-lender credits and of
         Home Credit loans still running, relative to income; number of loans running everywhere.
New numeric columns are stored as float32 to keep the table small enough for 5 parallel workers.

Usage:
    .venv/bin/python -m pipeline.make_features_v3 [--out-root DIR]
Output: processed/features_v3.parquet (all v2 columns unchanged, plus the new groups)
"""

import argparse

import numpy as np
import pandas as pd

from homecredit.data import CODE, read_raw
from homecredit.paths import Paths, add_out_root
from pipeline.make_features_v2 import per_installment


def agg(df: pd.DataFrame, cols: list[str], stats: list[str], prefix: str) -> pd.DataFrame:
    g = df.groupby("SK_ID_CURR")[cols].agg(stats)
    g.columns = [f"{prefix}_{c}_{s}" for c, s in g.columns]
    return g


def shares(df: pd.DataFrame, col: str, prefix: str, top: int) -> pd.DataFrame:
    keep = df[col].value_counts().index[:top]
    d = pd.get_dummies(df[col].where(df[col].isin(keep), "other").astype(str), prefix=f"{prefix}_{col}", dtype=float)
    d["SK_ID_CURR"] = df.SK_ID_CURR.values
    return d.groupby("SK_ID_CURR").mean()


def bureau_agg(P: Paths) -> pd.DataFrame:
    b = read_raw(P, "bureau")
    b["debt_ratio"] = b.AMT_CREDIT_SUM_DEBT / b.AMT_CREDIT_SUM.replace(0, np.nan)
    b["overdue_ratio"] = b.AMT_CREDIT_SUM_OVERDUE / b.AMT_CREDIT_SUM_DEBT.replace(0, np.nan)
    b["planned_days"] = b.DAYS_CREDIT_ENDDATE - b.DAYS_CREDIT
    num = ["DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT", "AMT_CREDIT_MAX_OVERDUE",
           "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT",
           "AMT_CREDIT_SUM_OVERDUE", "DAYS_CREDIT_UPDATE", "AMT_ANNUITY", "debt_ratio", "overdue_ratio", "planned_days"]
    parts = [agg(b, num, ["mean", "max", "min", "sum"], "agg_bur"),
             agg(b[b.CREDIT_ACTIVE.eq("Active")], num, ["mean", "max", "sum"], "agg_bur_act"),
             agg(b[b.CREDIT_ACTIVE.eq("Closed")], ["DAYS_CREDIT", "DAYS_ENDDATE_FACT", "AMT_CREDIT_SUM",
                                                   "AMT_CREDIT_MAX_OVERDUE", "planned_days"],
                 ["mean", "max", "min"], "agg_bur_cls"),
             shares(b, "CREDIT_TYPE", "agg_bur", 6), shares(b, "CREDIT_ACTIVE", "agg_bur", 3)]
    return pd.concat(parts, axis=1)


def prev_clean(P: Paths) -> pd.DataFrame:
    p = read_raw(P, "previous_application")
    for c in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        p[c] = p[c].replace(CODE, np.nan)
    p["app_to_credit"] = p.AMT_APPLICATION / p.AMT_CREDIT.replace(0, np.nan)
    p["credit_to_goods"] = p.AMT_CREDIT / p.AMT_GOODS_PRICE.replace(0, np.nan)
    p["annuity_to_credit"] = p.AMT_ANNUITY / p.AMT_CREDIT.replace(0, np.nan)
    p["planned_days"] = p.DAYS_LAST_DUE_1ST_VERSION - p.DAYS_FIRST_DUE
    p["end_shift"] = p.DAYS_LAST_DUE - p.DAYS_LAST_DUE_1ST_VERSION  # ended early (<0) or late
    return p


def prev_agg(p: pd.DataFrame) -> pd.DataFrame:
    num = ["AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT", "AMT_GOODS_PRICE",
           "HOUR_APPR_PROCESS_START", "RATE_DOWN_PAYMENT", "DAYS_DECISION", "CNT_PAYMENT", "DAYS_FIRST_DRAWING",
           "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION", "SELLERPLACE_AREA",
           "app_to_credit", "credit_to_goods", "annuity_to_credit", "planned_days", "end_shift"]
    parts = [agg(p, num, ["mean", "max", "min"], "agg_prev"),
             agg(p[p.NAME_CONTRACT_STATUS.eq("Approved")], num, ["mean", "max", "min"], "agg_prev_appr"),
             agg(p[p.NAME_CONTRACT_STATUS.eq("Refused")], ["AMT_APPLICATION", "AMT_CREDIT", "DAYS_DECISION",
                                                          "CNT_PAYMENT", "app_to_credit"],
                 ["mean", "max", "min"], "agg_prev_ref")]
    for col, top in [("NAME_CONTRACT_STATUS", 4), ("NAME_YIELD_GROUP", 5), ("NAME_CONTRACT_TYPE", 3),
                     ("CODE_REJECT_REASON", 6), ("NAME_PORTFOLIO", 5), ("CHANNEL_TYPE", 6), ("NAME_PRODUCT_TYPE", 3),
                     ("NAME_CLIENT_TYPE", 3), ("NAME_GOODS_CATEGORY", 8)]:
        parts.append(shares(p, col, "agg_prev", top))
    return pd.concat(parts, axis=1)


def inst_agg(P: Paths) -> pd.DataFrame:
    per = per_installment(P)
    per["days_early"] = (per.due_day - per.last_pay).clip(lower=0)
    per["paid_diff"] = per.amt_due - per.amt_paid
    num = ["days_late", "days_early", "paid_ratio", "paid_diff", "amt_due", "amt_paid", "late", "late30"]
    r = per[per.due_day >= -365]
    by_loan = per.groupby(["SK_ID_CURR", "SK_ID_PREV"]).agg(loan_max_late=("days_late", "max"),
                                                           loan_late_share=("late", "mean")).reset_index()
    parts = [agg(per, num, ["mean", "max", "min", "sum", "std"], "agg_inst"),
             agg(r, num, ["mean", "max", "sum"], "agg_inst_365d"),
             agg(by_loan, ["loan_max_late", "loan_late_share"], ["mean", "max"], "agg_inst_loan")]
    # trends: last 6 months vs the 6 months before
    a, b = per[per.due_day >= -182], per[(per.due_day < -182) & (per.due_day >= -365)]
    ta = a.groupby("SK_ID_CURR")[["days_late", "late", "amt_paid"]].mean()
    tb = b.groupby("SK_ID_CURR")[["days_late", "late", "amt_paid"]].mean()
    trend = (ta - tb).add_prefix("trend_inst_").add_suffix("_6m_minus_prev6m")
    return pd.concat(parts + [trend], axis=1)


def pos_agg(P: Paths) -> pd.DataFrame:
    pos = read_raw(P, "POS_CASH_balance")
    num = ["MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "SK_DPD", "SK_DPD_DEF"]
    r = pos[pos.MONTHS_BALANCE >= -12]
    pos["completed"] = pos.NAME_CONTRACT_STATUS.eq("Completed").astype(float)
    return pd.concat([agg(pos, num + ["completed"], ["mean", "max", "min"], "agg_pos"),
                      agg(r, ["CNT_INSTALMENT_FUTURE", "SK_DPD", "SK_DPD_DEF"], ["mean", "max"], "agg_pos_12m")],
                     axis=1)


def cc_agg(P: Paths) -> pd.DataFrame:
    cc = read_raw(P, "credit_card_balance")
    cc["util"] = cc.AMT_BALANCE / cc.AMT_CREDIT_LIMIT_ACTUAL.replace(0, np.nan)
    cc["pay_to_min"] = cc.AMT_PAYMENT_CURRENT / cc.AMT_INST_MIN_REGULARITY.replace(0, np.nan)
    num = [c for c in cc.columns if c not in ("SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_STATUS")]
    r6 = cc[cc.MONTHS_BALANCE >= -6]
    a, b = cc[cc.MONTHS_BALANCE >= -3], cc[(cc.MONTHS_BALANCE < -3) & (cc.MONTHS_BALANCE >= -12)]
    trend = (a.groupby("SK_ID_CURR")[["util", "AMT_BALANCE", "AMT_DRAWINGS_CURRENT"]].mean()
             - b.groupby("SK_ID_CURR")[["util", "AMT_BALANCE", "AMT_DRAWINGS_CURRENT"]].mean())
    trend = trend.add_prefix("trend_cc_").add_suffix("_3m_minus_4to12m")
    return pd.concat([agg(cc, num, ["mean", "max", "min", "sum"], "agg_cc"),
                      agg(r6, ["util", "AMT_BALANCE", "AMT_DRAWINGS_CURRENT", "AMT_PAYMENT_CURRENT", "pay_to_min",
                               "SK_DPD", "SK_DPD_DEF"], ["mean", "max"], "agg_cc_6m"), trend], axis=1)


def last_feats(p: pd.DataFrame) -> pd.DataFrame:
    p = p.sort_values(["SK_ID_CURR", "DAYS_DECISION", "SK_ID_PREV"], ascending=[True, False, False])
    p["rank"] = p.groupby("SK_ID_CURR").cumcount()
    p["refused"] = p.NAME_CONTRACT_STATUS.eq("Refused").astype(float)
    p["approved"] = p.NAME_CONTRACT_STATUS.eq("Approved").astype(float)
    parts = []
    for n in (3, 5):
        r = p[p["rank"] < n]
        g = r.groupby("SK_ID_CURR")[["AMT_CREDIT", "AMT_APPLICATION", "AMT_ANNUITY", "CNT_PAYMENT", "refused",
                                     "approved", "app_to_credit"]].mean()
        parts.append(g.add_prefix(f"last{n}_").add_suffix("_mean").add_prefix("last_"))
    first = p.sort_values("DAYS_DECISION").groupby("SK_ID_CURR").head(1).set_index("SK_ID_CURR")
    last = p[p["rank"] == 0].set_index("SK_ID_CURR")
    extra = pd.DataFrame({
        "last_first_days_decision": -first.DAYS_DECISION,
        "last_first_amt_credit": first.AMT_CREDIT,
        "last_credit_growth_last_vs_first": last.AMT_CREDIT / first.AMT_CREDIT.replace(0, np.nan),
        "last_channel": last.CHANNEL_TYPE.astype(object),
        "last_goods": last.NAME_GOODS_CATEGORY.astype(object),
        "last_reject_reason": last.CODE_REJECT_REASON.astype(object),
        "last_client_type": last.NAME_CLIENT_TYPE.astype(object),
        "last_portfolio": last.NAME_PORTFOLIO.astype(object),
    })
    same_day = p.groupby(["SK_ID_CURR", "DAYS_DECISION"]).size().groupby("SK_ID_CURR").max()
    extra["last_max_applications_same_day"] = same_day
    return pd.concat(parts + [extra], axis=1)


def bureau_trend(P: Paths) -> pd.DataFrame:
    b = read_raw(P, "bureau", ["SK_ID_CURR", "DAYS_CREDIT", "AMT_CREDIT_SUM"])
    y1 = b[b.DAYS_CREDIT >= -365].groupby("SK_ID_CURR").size()
    y2 = b[(b.DAYS_CREDIT < -365) & (b.DAYS_CREDIT >= -730)].groupby("SK_ID_CURR").size()
    ids = b.SK_ID_CURR.unique()
    t = pd.DataFrame(index=pd.Index(ids, name="SK_ID_CURR"))
    t["trend_bur_new_credits_last_year_minus_year_before"] = y1.reindex(ids).fillna(0) - y2.reindex(ids).fillna(0)
    return t


def load_feats(P: Paths, app: pd.DataFrame) -> pd.DataFrame:
    a = app.set_index("SK_ID_CURR")
    b = read_raw(P, "bureau", ["SK_ID_CURR", "CREDIT_ACTIVE", "AMT_ANNUITY", "AMT_CREDIT_SUM_DEBT"])
    act = b[b.CREDIT_ACTIVE.eq("Active")]
    bur_ann = act.groupby("SK_ID_CURR").AMT_ANNUITY.sum().reindex(a.index).fillna(0)
    bur_n = act.groupby("SK_ID_CURR").size().reindex(a.index).fillna(0)
    pos = read_raw(P, "POS_CASH_balance", ["SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE", "CNT_INSTALMENT_FUTURE"])
    last = pos.sort_values("MONTHS_BALANCE").groupby("SK_ID_PREV").tail(1)
    run = last[(last.MONTHS_BALANCE >= -2) & (last.CNT_INSTALMENT_FUTURE > 0)]
    prev = read_raw(P, "previous_application", ["SK_ID_PREV", "AMT_ANNUITY"])
    run = run.merge(prev, on="SK_ID_PREV", how="left")
    hc_ann = run.groupby("SK_ID_CURR").AMT_ANNUITY.sum().reindex(a.index).fillna(0)
    hc_n = run.groupby("SK_ID_CURR").size().reindex(a.index).fillna(0)
    income = a.AMT_INCOME_TOTAL
    return pd.DataFrame({
        "load_total_annuity_to_income": (a.AMT_ANNUITY.fillna(0) + bur_ann + hc_ann) / income,
        "load_existing_annuity_to_income": (bur_ann + hc_ann) / income,
        "load_loans_running_n": bur_n + hc_n,
        "load_bureau_debt_to_this_credit": act.groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.sum().reindex(a.index)
        / a.AMT_CREDIT,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)

    v2 = pd.read_parquet(P.features("v2")).set_index("SK_ID_CURR")
    app = read_raw(P, "application_train")[["SK_ID_CURR", "AMT_ANNUITY", "AMT_CREDIT", "AMT_INCOME_TOTAL"]]
    app = pd.concat([app, read_raw(P, "application_test")[["SK_ID_CURR", "AMT_ANNUITY", "AMT_CREDIT",
                                                            "AMT_INCOME_TOTAL"]]], ignore_index=True)
    p = prev_clean(P)
    inst = inst_agg(P)
    cc = cc_agg(P)
    new = {
        "agg_": [bureau_agg(P), prev_agg(p), inst[[c for c in inst.columns if c.startswith("agg_")]], pos_agg(P),
                 cc[[c for c in cc.columns if c.startswith("agg_")]]],
        "last_": [last_feats(p)],
        "trend_": [inst[[c for c in inst.columns if c.startswith("trend_")]],
                   cc[[c for c in cc.columns if c.startswith("trend_")]], bureau_trend(P)],
        "load_": [load_feats(P, app)],
    }
    feats = v2
    counts = {}
    for prefix, parts in new.items():
        g = pd.concat(parts, axis=1)
        assert all(c.startswith(prefix) for c in g.columns), prefix
        g = g.reindex(feats.index)
        for c in g.columns:
            if pd.api.types.is_numeric_dtype(g[c]):
                g[c] = g[c].replace([np.inf, -np.inf], np.nan).astype("float32")
        counts[prefix] = g.shape[1]
        feats = feats.join(g)
    feats = feats.reset_index()
    feats.to_parquet(P.features("v3"), index=False)
    print(f"rows {len(feats):,}; columns {feats.shape[1]} (v2: {v2.shape[1] + 1}); new per group: {counts}")


if __name__ == "__main__":
    main()
