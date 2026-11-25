"""Feature table v1: one row per applicant (train and test), application columns plus simple history summaries.

Application part:
  - every application column as it is, except the 365243 code in DAYS_EMPLOYED (EDA step 4), which becomes NaN
    plus a flag `no_employment_code` (pensioners / unemployed);
  - a few ratios a credit analyst would compute (credit / income, annuity / income, credit / goods price,
    annuity / credit as a rough inverse of the term), age and years employed.

History part (all records are dated before the application; EDA steps 4 and 7):
  - bureau: number of credits, active credits, any overdue now, max days overdue now, active debt, debt / credit
    on active credits, days since the most recent credit;
  - previous_application: number of applications, approved count, share refused, days since the last decision;
  - installments_payments: rows are first combined per installment (a payment can be split over rows), then
    share paid late, mean and max days late, share underpaid, share late in the last 365 days;
  - POS_CASH_balance: months of history, any month with days past due after tolerance (SK_DPD_DEF > 0);
  - credit_card_balance: has a card, mean utilisation (balance / limit), any month past due after tolerance.
  bureau_balance is left out on purpose: it covers 36% of train bureau credits but 99.9% of test (EDA step 1).

All monthly / payment tables are summarised by SK_ID_CURR directly, so rows whose SK_ID_PREV is missing from
previous_application are kept (EDA step 1).

Usage:
    .venv/bin/python -m pipeline.make_features [--out-root DIR]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from homecredit.data import CODE
from homecredit.paths import Paths, add_out_root


def application(data: Path) -> pd.DataFrame:
    tr = pd.read_parquet(data / "application_train.parquet").assign(split="train")
    te = pd.read_parquet(data / "application_test.parquet").assign(split="test")
    app = pd.concat([tr, te], ignore_index=True)
    app["no_employment_code"] = app.DAYS_EMPLOYED.eq(CODE).astype(float)
    app["DAYS_EMPLOYED"] = app.DAYS_EMPLOYED.replace(CODE, np.nan)
    app["age_years"] = -app.DAYS_BIRTH / 365.25
    app["employed_years"] = -app.DAYS_EMPLOYED / 365.25
    app["credit_to_income"] = app.AMT_CREDIT / app.AMT_INCOME_TOTAL
    app["annuity_to_income"] = app.AMT_ANNUITY / app.AMT_INCOME_TOTAL
    app["credit_to_goods"] = app.AMT_CREDIT / app.AMT_GOODS_PRICE
    app["annuity_to_credit"] = app.AMT_ANNUITY / app.AMT_CREDIT
    return app


def bureau_feats(data: Path) -> pd.DataFrame:
    b = pd.read_parquet(data / "bureau.parquet")
    active = b.CREDIT_ACTIVE.eq("Active")
    g = b.groupby("SK_ID_CURR")
    f = pd.DataFrame({
        "bureau_n": g.size(),
        "bureau_active_n": active.groupby(b.SK_ID_CURR).sum(),
        "bureau_overdue_any": (b.CREDIT_DAY_OVERDUE > 0).groupby(b.SK_ID_CURR).any().astype(float),
        "bureau_overdue_days_max": g.CREDIT_DAY_OVERDUE.max(),
        "bureau_days_since_last_credit": -g.DAYS_CREDIT.max(),
    })
    ab = b[active]
    debt = ab.groupby("SK_ID_CURR").AMT_CREDIT_SUM_DEBT.sum()
    credit = ab.groupby("SK_ID_CURR").AMT_CREDIT_SUM.sum()
    f["bureau_active_debt"] = debt
    f["bureau_active_debt_to_credit"] = (debt / credit.replace(0, np.nan))
    return f


def prev_feats(data: Path) -> pd.DataFrame:
    p = pd.read_parquet(data / "previous_application.parquet",
                        columns=["SK_ID_CURR", "NAME_CONTRACT_STATUS", "DAYS_DECISION"])
    g = p.groupby("SK_ID_CURR")
    return pd.DataFrame({
        "prev_n": g.size(),
        "prev_approved_n": p.NAME_CONTRACT_STATUS.eq("Approved").groupby(p.SK_ID_CURR).sum(),
        "prev_refused_share": p.NAME_CONTRACT_STATUS.eq("Refused").groupby(p.SK_ID_CURR).mean(),
        "prev_days_since_last_decision": -g.DAYS_DECISION.max(),
    })


def installment_feats(data: Path) -> pd.DataFrame:
    i = pd.read_parquet(data / "installments_payments.parquet")
    per = i.groupby(["SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"], sort=False).agg(
        SK_ID_CURR=("SK_ID_CURR", "first"), due_day=("DAYS_INSTALMENT", "min"),
        amt_due=("AMT_INSTALMENT", "max"), amt_paid=("AMT_PAYMENT", "sum"), last_pay=("DAYS_ENTRY_PAYMENT", "max"))
    days_late = (per.last_pay - per.due_day).clip(lower=0)
    # An installment with no payment recorded at all counts as late (days late stays unknown).
    per["late"] = ((days_late > 0) | per.last_pay.isna()).astype(float)
    per["days_late"] = days_late
    per["underpaid"] = (per.amt_paid < per.amt_due - 1.0).astype(float)  # 1 unit tolerance for rounding
    per["late_last_year"] = per.late.where(per.due_day >= -365)
    g = per.groupby("SK_ID_CURR")
    return pd.DataFrame({
        "inst_n": g.size(),
        "inst_late_share": g.late.mean(),
        "inst_days_late_mean": g.days_late.mean(),
        "inst_days_late_max": g.days_late.max(),
        "inst_underpaid_share": g.underpaid.mean(),
        "inst_late_share_last_year": g.late_last_year.mean(),
    })


def pos_feats(data: Path) -> pd.DataFrame:
    p = pd.read_parquet(data / "POS_CASH_balance.parquet", columns=["SK_ID_CURR", "SK_DPD_DEF"])
    return pd.DataFrame({
        "pos_months_n": p.groupby("SK_ID_CURR").size(),
        "pos_dpd_any": (p.SK_DPD_DEF > 0).groupby(p.SK_ID_CURR).any().astype(float),
    })


def cc_feats(data: Path) -> pd.DataFrame:
    c = pd.read_parquet(data / "credit_card_balance.parquet",
                        columns=["SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD_DEF"])
    util = c.AMT_BALANCE / c.AMT_CREDIT_LIMIT_ACTUAL.replace(0, np.nan)
    return pd.DataFrame({
        "cc_has_card": c.groupby("SK_ID_CURR").size().gt(0).astype(float),
        "cc_util_mean": util.groupby(c.SK_ID_CURR).mean(),
        "cc_dpd_any": (c.SK_DPD_DEF > 0).groupby(c.SK_ID_CURR).any().astype(float),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    data = P.raw

    app = application(data)
    hist = [bureau_feats(data), prev_feats(data), installment_feats(data), pos_feats(data), cc_feats(data)]
    feats = app.set_index("SK_ID_CURR")
    for h in hist:
        feats = feats.join(h)
    # "No record" counts are 0, not missing; ratios and shares stay NaN when there is no record.
    for c in ["bureau_n", "bureau_active_n", "prev_n", "prev_approved_n", "inst_n", "pos_months_n", "cc_has_card"]:
        feats[c] = feats[c].fillna(0)
    feats = feats.reset_index()
    feats.to_parquet(P.features("v1"), index=False)
    hist_cols = [c for h in hist for c in h.columns]
    print(f"rows {len(feats):,}; application columns {app.shape[1]}; history columns {len(hist_cols)}")
    print(feats.loc[feats.split == "train", hist_cols].describe().T[["count", "mean", "50%", "max"]].round(3)
          .to_string())


if __name__ == "__main__":
    main()
