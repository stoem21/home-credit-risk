"""Feature table v2: v1 plus four new groups of history features, each with its own prefix so it can be tested
alone against the same reference (one change at a time). Ideas come from our own error analysis and one-applicant
walk-through (recent behaviour, loan stacking) and from public top solutions of the competition (time windows,
last application, estimated interest rate, external-score combinations), judged only on our own validation.

All side-table records are dated before the application (DAYS_* and MONTHS_BALANCE are negative), so no feature
looks into the future. No feature uses TARGET.

Groups (prefix -> what):
  win_   recent behaviour, time windows: Home Credit applications in the last 90 / 365 days (all, refused,
         cancelled); the most recent application (status, contract type, product, yield group, amounts, term,
         days since) and the most recent approved one; instalments in the last 90 / 180 / 365 days (late share,
         mean days late, share paid under the amount due, paid / due); share of instalments over 30 days late;
         POS months past due in the last 12 months; card use in the last 3 months vs all months, and months
         past due in the last 12.
  stk_   loan stacking and other-lender history: credits opened with other lenders in the last 90 / 365 days;
         active other-lender credit and debt relative to income; Home Credit POS / cash loans still running
         (seen in the last 2 months with instalments left) and instalments left; bureau_balance (monthly
         status of other-lender credits): has any history, any month past due in the last 12 months, share of
         known months past due. Note: bureau_balance covers 36% of train bureau credits but 99.9% of test
         (EDA step 1), so these columns are checked for drift before use.
  term_  loan terms: yearly interest rate implied by amount, instalment and number of payments of earlier
         approved Home Credit loans (mean, most recent); amount granted vs amount asked in earlier applications;
         this application's credit and instalment vs the most recent approved loan; down payment (goods price -
         credit).
  ext_   combinations of the three external scores: mean, min, max, spread, product, number available, and
         credit / income and annuity / income divided by the mean score.

Usage:
    .venv/bin/python -m pipeline.make_features_v2 [--out-root DIR]
Output: processed/features_v2.parquet (the v1 columns are checked to be identical to features_v1.parquet)
"""

import argparse

import numpy as np
import pandas as pd

from homecredit.data import read_raw
from homecredit.paths import Paths, add_out_root
from pipeline.make_features import application, bureau_feats, cc_feats, installment_feats, pos_feats, prev_feats


def yearly_rate(credit, annuity, n):
    """Monthly rate r with annuity = credit * r / (1 - (1 + r)^-n), by bisection; returned as a yearly rate.
    NaN when the inputs cannot define a positive rate (no term, zero amounts, or instalments summing to less
    than the credit)."""
    credit, annuity, n = (np.asarray(a, dtype=float) for a in (credit, annuity, n))
    ok = (n > 0) & (credit > 0) & (annuity > 0) & (annuity * n > credit)
    lo, hi = np.full(credit.shape, 1e-9), np.full(credit.shape, 1.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        with np.errstate(all="ignore"):
            pay = credit * mid / (1 - (1 + mid) ** -n)
        too_low = pay < annuity
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    r = (lo + hi) / 2
    return np.where(ok, (1 + r) ** 12 - 1, np.nan)


def per_installment(P: Paths) -> pd.DataFrame:
    """One row per instalment (split payments combined), as in v1."""
    i = read_raw(P, "installments_payments")
    per = i.groupby(["SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"], sort=False).agg(
        SK_ID_CURR=("SK_ID_CURR", "first"), due_day=("DAYS_INSTALMENT", "min"),
        amt_due=("AMT_INSTALMENT", "max"), amt_paid=("AMT_PAYMENT", "sum"), last_pay=("DAYS_ENTRY_PAYMENT", "max"))
    days_late = (per.last_pay - per.due_day).clip(lower=0)
    per["late"] = ((days_late > 0) | per.last_pay.isna()).astype(float)
    per["days_late"] = days_late
    per["late30"] = ((days_late > 30) | per.last_pay.isna()).astype(float)
    per["underpaid"] = (per.amt_paid < per.amt_due - 1.0).astype(float)
    per["paid_ratio"] = per.amt_paid / per.amt_due.replace(0, np.nan)
    return per


def window_feats(P: Paths) -> pd.DataFrame:
    p = read_raw(P, "previous_application")
    f = {}
    for d in (90, 365):
        w = p[p.DAYS_DECISION >= -d]
        g = w.groupby("SK_ID_CURR")
        f[f"win_prev_n_{d}d"] = g.size()
        f[f"win_prev_refused_{d}d"] = w.NAME_CONTRACT_STATUS.eq("Refused").groupby(w.SK_ID_CURR).sum()
        f[f"win_prev_canceled_{d}d"] = w.NAME_CONTRACT_STATUS.eq("Canceled").groupby(w.SK_ID_CURR).sum()
    last = p.sort_values(["SK_ID_CURR", "DAYS_DECISION", "SK_ID_PREV"]).groupby("SK_ID_CURR").tail(1).set_index(
        "SK_ID_CURR")
    for c, name in [("NAME_CONTRACT_STATUS", "status"), ("NAME_CONTRACT_TYPE", "type"),
                    ("PRODUCT_COMBINATION", "product"), ("NAME_YIELD_GROUP", "yield_group")]:
        f[f"win_last_{name}"] = last[c].astype(object)
    f["win_last_amt_credit"] = last.AMT_CREDIT
    f["win_last_cnt_payment"] = last.CNT_PAYMENT
    f["win_last_days_decision"] = -last.DAYS_DECISION
    appr = p[p.NAME_CONTRACT_STATUS.eq("Approved")]
    f["win_last_approved_days"] = -appr.groupby("SK_ID_CURR").DAYS_DECISION.max()
    out = pd.DataFrame(f)

    per = per_installment(P)
    w_rows = {}
    for d in (90, 180, 365):
        w = per[per.due_day >= -d]
        g = w.groupby("SK_ID_CURR")
        w_rows[f"win_inst_late_share_{d}d"] = g.late.mean()
        w_rows[f"win_inst_days_late_mean_{d}d"] = g.days_late.mean()
    w = per[per.due_day >= -365]
    g = w.groupby("SK_ID_CURR")
    w_rows["win_inst_underpaid_share_365d"] = g.underpaid.mean()
    w_rows["win_inst_paid_ratio_365d"] = g.paid_ratio.mean()
    w_rows["win_inst_n_365d"] = g.size()
    w_rows["win_inst_late30_share"] = per.groupby("SK_ID_CURR").late30.mean()
    out = out.join(pd.DataFrame(w_rows), how="outer")

    pos = read_raw(P, "POS_CASH_balance", ["SK_ID_CURR", "MONTHS_BALANCE", "SK_DPD_DEF"])
    r = pos[pos.MONTHS_BALANCE >= -12]
    out = out.join((r.SK_DPD_DEF > 0).groupby(r.SK_ID_CURR).mean().rename("win_pos_dpd_share_12m"), how="outer")

    cc = read_raw(P, "credit_card_balance", ["SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE",
                                             "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_CURRENT", "SK_DPD_DEF"])
    cc["util"] = cc.AMT_BALANCE / cc.AMT_CREDIT_LIMIT_ACTUAL.replace(0, np.nan)
    r3, r12 = cc[cc.MONTHS_BALANCE >= -3], cc[cc.MONTHS_BALANCE >= -12]
    c = pd.DataFrame({
        "win_cc_util_3m": r3.groupby("SK_ID_CURR").util.mean(),
        "win_cc_drawings_3m": r3.groupby("SK_ID_CURR").AMT_DRAWINGS_CURRENT.mean(),
        "win_cc_dpd_share_12m": (r12.SK_DPD_DEF > 0).groupby(r12.SK_ID_CURR).mean(),
    })
    c["win_cc_util_3m_minus_all"] = c.win_cc_util_3m - cc.groupby("SK_ID_CURR").util.mean()
    return out.join(c, how="outer")


def stacking_feats(P: Paths, income: pd.Series) -> pd.DataFrame:
    b = read_raw(P, "bureau")
    f = {}
    for d in (90, 365):
        f[f"stk_bureau_new_{d}d"] = (b.DAYS_CREDIT >= -d).groupby(b.SK_ID_CURR).sum()
    act = b[b.CREDIT_ACTIVE.eq("Active")]
    out = pd.DataFrame(f)
    out["stk_bureau_active_credit_to_income"] = act.groupby("SK_ID_CURR").AMT_CREDIT_SUM.sum() / income
    out["stk_bureau_active_debt_to_income"] = act.groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.sum() / income

    pos = read_raw(P, "POS_CASH_balance", ["SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE", "CNT_INSTALMENT_FUTURE"])
    last = pos.sort_values("MONTHS_BALANCE").groupby("SK_ID_PREV").tail(1)
    run = last[(last.MONTHS_BALANCE >= -2) & (last.CNT_INSTALMENT_FUTURE > 0)]
    out = out.join(pd.DataFrame({"stk_hc_running_n": run.groupby("SK_ID_CURR").size(),
                                 "stk_hc_instalments_left": run.groupby("SK_ID_CURR").CNT_INSTALMENT_FUTURE.sum()}),
                   how="outer")

    bb = read_raw(P, "bureau_balance")
    bb = bb.merge(b[["SK_ID_BUREAU", "SK_ID_CURR"]], on="SK_ID_BUREAU")
    st = bb.STATUS.astype(str)
    known = ~st.isin(["X"])
    dpd = st.isin(["1", "2", "3", "4", "5"])
    r = bb.MONTHS_BALANCE >= -12
    out = out.join(pd.DataFrame({
        "stk_bb_has_history": bb.groupby("SK_ID_CURR").size().gt(0).astype(float),
        "stk_bb_dpd_any_12m": (dpd & r).groupby(bb.SK_ID_CURR).any().astype(float),
        "stk_bb_dpd_share": dpd[known].groupby(bb.SK_ID_CURR[known]).mean(),
    }), how="outer")
    return out


def term_feats(P: Paths, app: pd.DataFrame) -> pd.DataFrame:
    p = read_raw(P, "previous_application")
    appr = p[p.NAME_CONTRACT_STATUS.eq("Approved")].copy()
    appr["rate"] = yearly_rate(appr.AMT_CREDIT, appr.AMT_ANNUITY, appr.CNT_PAYMENT)
    last = appr.sort_values(["SK_ID_CURR", "DAYS_DECISION", "SK_ID_PREV"]).groupby("SK_ID_CURR").tail(1).set_index(
        "SK_ID_CURR")
    asked = p[p.AMT_APPLICATION > 0]
    a = app.set_index("SK_ID_CURR")
    out = pd.DataFrame({
        "term_prev_rate_mean": appr.groupby("SK_ID_CURR").rate.mean(),
        "term_prev_rate_last": last.rate,
        "term_prev_granted_to_asked": (asked.AMT_CREDIT / asked.AMT_APPLICATION).groupby(asked.SK_ID_CURR).mean(),
    })
    out = out.reindex(a.index)
    out["term_credit_to_last_approved"] = a.AMT_CREDIT / last.AMT_CREDIT.replace(0, np.nan).reindex(a.index)
    out["term_annuity_to_last_approved"] = a.AMT_ANNUITY / last.AMT_ANNUITY.replace(0, np.nan).reindex(a.index)
    out["term_down_payment"] = a.AMT_GOODS_PRICE - a.AMT_CREDIT
    return out


def ext_feats(app: pd.DataFrame) -> pd.DataFrame:
    e = app.set_index("SK_ID_CURR")[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    a = app.set_index("SK_ID_CURR")
    m = e.mean(axis=1)
    return pd.DataFrame({
        "ext_mean": m, "ext_min": e.min(axis=1), "ext_max": e.max(axis=1),
        "ext_spread": e.max(axis=1) - e.min(axis=1), "ext_prod": e.prod(axis=1, min_count=3),
        "ext_n": e.notna().sum(axis=1).astype(float),
        "ext_credit_to_income_per_score": (a.AMT_CREDIT / a.AMT_INCOME_TOTAL) / m,
        "ext_annuity_to_income_per_score": (a.AMT_ANNUITY / a.AMT_INCOME_TOTAL) / m,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    data = P.raw

    # v1 part, rebuilt with the v1 code and checked against the saved table
    app = application(data)
    hist = [bureau_feats(data), prev_feats(data), installment_feats(data), pos_feats(data), cc_feats(data)]
    feats = app.set_index("SK_ID_CURR")
    for h in hist:
        feats = feats.join(h)
    for c in ["bureau_n", "bureau_active_n", "prev_n", "prev_approved_n", "inst_n", "pos_months_n", "cc_has_card"]:
        feats[c] = feats[c].fillna(0)
    v1 = pd.read_parquet(P.features("v1")).set_index("SK_ID_CURR")
    pd.testing.assert_frame_equal(feats, v1)

    income = app.set_index("SK_ID_CURR").AMT_INCOME_TOTAL
    groups = {"win_": window_feats(P), "stk_": stacking_feats(P, income), "term_": term_feats(P, app),
              "ext_": ext_feats(app)}
    for prefix, g in groups.items():
        assert all(c.startswith(prefix) for c in g.columns)
        feats = feats.join(g.reindex(feats.index))
    # "No record" counts are 0, not missing
    zero = ["win_prev_n_90d", "win_prev_refused_90d", "win_prev_canceled_90d", "win_prev_n_365d",
            "win_prev_refused_365d", "win_prev_canceled_365d", "win_inst_n_365d", "stk_bureau_new_90d",
            "stk_bureau_new_365d", "stk_hc_running_n", "stk_hc_instalments_left", "stk_bb_has_history"]
    feats[zero] = feats[zero].fillna(0)
    feats = feats.reset_index()
    feats.to_parquet(P.features("v2"), index=False)
    tr = feats[feats.split == "train"]
    for prefix, g in groups.items():
        cols = list(g.columns)
        print(f"{prefix}: {len(cols)} columns")
        num = [c for c in cols if pd.api.types.is_numeric_dtype(tr[c])]
        print(tr[num].describe().T[["count", "mean", "50%", "max"]].round(3).to_string())
    print(f"rows {len(feats):,}; columns {feats.shape[1]}")


if __name__ == "__main__":
    main()
