"""EDA: follow one real applicant through every table.

Summary tables hide how the data is built. This prints one train applicant's application, every credit with other
lenders (bureau, with the monthly status history of bureau_balance), every earlier Home Credit application, and the
monthly / payment records behind them (POS_CASH_balance, credit_card_balance, installments_payments). It then
recomputes a few of that applicant's features by hand from the rows shown and checks them against
processed/features_v1.parquet, and shows what the models predicted for them (out-of-fold, seed 0).

The applicant is drawn at random (fixed seed) among train applicants who have rows in every side table.

Output goes to the terminal only. Nothing is saved: these are individual records of the competition data, which
must not be redistributed (Rule 7.B), so the write-up describes what the record taught, not the record itself.

Usage:
    .venv/bin/python -m eda.one_applicant [--target 1] [--seed 0] [--out-root DIR]
"""

import argparse

import numpy as np
import pandas as pd

from homecredit.data import read_raw
from homecredit.evaluate import load_oof
from homecredit.paths import Paths, add_out_root

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)


def title(t: str) -> None:
    print(f"\n{'=' * 8} {t} {'=' * (90 - len(t))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1, help="draw a defaulter (1) or a non-defaulter (0)")
    ap.add_argument("--seed", type=int, default=0)
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)

    app = read_raw(P, "application_train")
    side = {t: read_raw(P, t) for t in ["bureau", "previous_application", "POS_CASH_balance",
                                         "credit_card_balance", "installments_payments"]}
    bb = read_raw(P, "bureau_balance")
    has_bb = side["bureau"].loc[side["bureau"].SK_ID_BUREAU.isin(bb.SK_ID_BUREAU), "SK_ID_CURR"]
    eligible = set(app.SK_ID_CURR[app.TARGET == args.target]) & set(has_bb)
    for df in side.values():
        eligible &= set(df.SK_ID_CURR)
    rng = np.random.default_rng(args.seed)
    sid = int(rng.choice(sorted(eligible)))
    print(f"{len(eligible):,} train applicants with TARGET={args.target} have rows in every side table; drew one.")

    # ---- application ----
    a = app[app.SK_ID_CURR == sid].iloc[0]
    title("application_train (non-missing values; building-description columns *_AVG/_MODE/_MEDI left out)")
    keep = [c for c in app.columns if pd.notna(a[c]) and not c.endswith(("_AVG", "_MODE", "_MEDI"))
            and not (c.startswith("FLAG_DOCUMENT") and a[c] == 0)]
    print(a[keep].to_string())

    # ---- bureau + bureau_balance ----
    b = side["bureau"][side["bureau"].SK_ID_CURR == sid].sort_values("DAYS_CREDIT")
    title(f"bureau: {len(b)} credits with other lenders (DAYS_* = days before this application)")
    print(b.drop(columns="SK_ID_CURR").to_string(index=False))
    title("bureau_balance: monthly status per credit, oldest month left (0 = no DPD, 1-5 = DPD bands, C = closed, "
          "X = unknown)")
    for bid in b.SK_ID_BUREAU:
        h = bb[bb.SK_ID_BUREAU == bid].sort_values("MONTHS_BALANCE")
        if len(h):
            print(f"{bid}: months {h.MONTHS_BALANCE.min()}..{h.MONTHS_BALANCE.max()}  {''.join(h.STATUS.astype(str))}")
        else:
            print(f"{bid}: no monthly history")

    # ---- previous applications ----
    p = side["previous_application"][side["previous_application"].SK_ID_CURR == sid].sort_values("DAYS_DECISION")
    title(f"previous_application: {len(p)} earlier Home Credit applications")
    cols = ["SK_ID_PREV", "NAME_CONTRACT_TYPE", "NAME_CONTRACT_STATUS", "CODE_REJECT_REASON", "AMT_APPLICATION",
            "AMT_CREDIT", "AMT_ANNUITY", "CNT_PAYMENT", "DAYS_DECISION", "DAYS_FIRST_DUE", "DAYS_LAST_DUE",
            "DAYS_TERMINATION", "NAME_YIELD_GROUP"]
    print(p[cols].to_string(index=False))

    # ---- monthly / payment tables, per earlier loan ----
    pos = side["POS_CASH_balance"][side["POS_CASH_balance"].SK_ID_CURR == sid]
    cc = side["credit_card_balance"][side["credit_card_balance"].SK_ID_CURR == sid]
    ins = side["installments_payments"][side["installments_payments"].SK_ID_CURR == sid]
    title("POS_CASH_balance per earlier loan (months of history, last status, max days past due)")
    print(pos.groupby("SK_ID_PREV").agg(months=("MONTHS_BALANCE", "size"), first=("MONTHS_BALANCE", "min"),
                                        last=("MONTHS_BALANCE", "max"), last_status=("NAME_CONTRACT_STATUS", "last"),
                                        max_dpd=("SK_DPD", "max"), max_dpd_def=("SK_DPD_DEF", "max")).to_string())
    title("credit_card_balance per card (months, limit, balance, max days past due)")
    print(cc.groupby("SK_ID_PREV").agg(months=("MONTHS_BALANCE", "size"), limit_max=("AMT_CREDIT_LIMIT_ACTUAL", "max"),
                                       balance_mean=("AMT_BALANCE", "mean"), max_dpd=("SK_DPD", "max"),
                                       max_dpd_def=("SK_DPD_DEF", "max")).round(0).to_string())
    title("installments_payments: rows per earlier loan, and installments paid in more than one row")
    per = ins.groupby(["SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"]).agg(
        rows=("AMT_PAYMENT", "size"), due=("DAYS_INSTALMENT", "min"), amt_due=("AMT_INSTALMENT", "max"),
        paid=("AMT_PAYMENT", "sum"), last_pay=("DAYS_ENTRY_PAYMENT", "max")).reset_index()
    per["days_late"] = (per.last_pay - per.due).clip(lower=0)
    print(per.groupby("SK_ID_PREV").agg(installments=("rows", "size"), rows=("rows", "sum"),
                                        split=("rows", lambda r: int((r > 1).sum())),
                                        late=("days_late", lambda d: int((d > 0).sum())),
                                        max_days_late=("days_late", "max")).to_string())
    title("installments paid late, if any (one line per installment)")
    late = per[(per.days_late > 0) | per.last_pay.isna()]
    print(late.to_string(index=False) if len(late) else "none")

    # ---- recompute a few features by hand and compare with the feature table ----
    f = pd.read_parquet(P.features("v1"))
    f = f[f.SK_ID_CURR == sid].iloc[0]
    late_flag = ((per.days_late > 0) | per.last_pay.isna()).astype(float)
    by_hand = {
        "bureau_n": len(b),
        "bureau_active_n": int(b.CREDIT_ACTIVE.eq("Active").sum()),
        "prev_n": len(p),
        "prev_approved_n": int(p.NAME_CONTRACT_STATUS.eq("Approved").sum()),
        "inst_n": len(per),
        "inst_late_share": late_flag.mean(),
        "inst_days_late_max": per.days_late.max(),
        "pos_months_n": len(pos),
        "cc_has_card": float(len(cc) > 0),
        "credit_to_income": a.AMT_CREDIT / a.AMT_INCOME_TOTAL,
    }
    title("features recomputed by hand from the rows above vs processed/features_v1.parquet")
    chk = pd.DataFrame({"by_hand": by_hand, "feature_table": {k: f[k] for k in by_hand}})
    chk["match"] = np.isclose(chk.by_hand.astype(float), chk.feature_table.astype(float), equal_nan=True)
    print(chk.to_string())

    # ---- what the models said ----
    title("model predictions for this applicant (out-of-fold, seed 0) and their rank among all train applicants")
    for name in ["logreg_baseline", "scorecard_full", "lgbm_v1"]:
        o = load_oof(P, name, app.SK_ID_CURR)
        v = o.s0.values
        i = int(np.where(o.SK_ID_CURR.values == sid)[0][0])
        print(f"{name:16s} predicted risk {100 * v[i]:5.1f}%   riskier than {100 * (v < v[i]).mean():5.1f}% of "
              f"train applicants")
    print(f"outcome: TARGET = {int(a.TARGET)}")


if __name__ == "__main__":
    main()
