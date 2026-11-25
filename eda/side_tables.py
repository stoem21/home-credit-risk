"""EDA step 7: what the side tables hold, and simple applicant-level signals from them.

  - Time span of each table (in months or days before the application; there are no absolute dates).
  - Status shares (bureau CREDIT_ACTIVE, bureau_balance STATUS, previous_application NAME_CONTRACT_STATUS, ...).
  - Payment behaviour in installments_payments: late (paid after the due day) and short (paid less than due).
  - Default rate in train by a few simple applicant-level summaries, as a first hint of which history helps:
    share of late installments, share of refused previous applications, any bureau credit currently overdue,
    any month with days past due in POS / credit card history.

Usage:
    .venv/bin/python -m eda.side_tables [--out-root DIR]
"""

import argparse

import numpy as np
import pandas as pd

from homecredit.data import CODE
from homecredit.paths import Paths, add_out_root



def main() -> None:
    ap = argparse.ArgumentParser()
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    data, out = P.raw, P.out("eda")

    def rd(name, cols=None):
        return pd.read_parquet(data / f"{name}.parquet", columns=cols)

    tr = rd("application_train", ["SK_ID_CURR", "TARGET"]).set_index("SK_ID_CURR").TARGET
    bureau = rd("bureau")
    bb = rd("bureau_balance")
    prev = rd("previous_application")
    pos = rd("POS_CASH_balance")
    cc = rd("credit_card_balance")
    ins = rd("installments_payments")

    # ---- time spans ----
    spans = []

    def span(table, col, s, unit):
        s = s[s != CODE].dropna()
        spans.append({"table": table, "column": col, "unit": unit, "min": s.min(), "p01": s.quantile(0.01),
                      "median": s.median(), "max": s.max()})

    span("bureau", "DAYS_CREDIT", bureau.DAYS_CREDIT, "days")
    span("bureau_balance", "MONTHS_BALANCE", bb.MONTHS_BALANCE, "months")
    span("previous_application", "DAYS_DECISION", prev.DAYS_DECISION, "days")
    span("POS_CASH_balance", "MONTHS_BALANCE", pos.MONTHS_BALANCE, "months")
    span("credit_card_balance", "MONTHS_BALANCE", cc.MONTHS_BALANCE, "months")
    span("installments_payments", "DAYS_INSTALMENT", ins.DAYS_INSTALMENT, "days")
    span_df = pd.DataFrame(spans)
    span_df.to_csv(out / "side_time_spans.csv", index=False)
    print(span_df.to_string(index=False), "\n")

    # ---- status shares ----
    st = []
    for table, df, col in [("bureau", bureau, "CREDIT_ACTIVE"), ("bureau", bureau, "CREDIT_CURRENCY"),
                           ("bureau_balance", bb, "STATUS"),
                           ("previous_application", prev, "NAME_CONTRACT_STATUS"),
                           ("previous_application", prev, "NAME_CONTRACT_TYPE"),
                           ("POS_CASH_balance", pos, "NAME_CONTRACT_STATUS"),
                           ("credit_card_balance", cc, "NAME_CONTRACT_STATUS")]:
        for v, p in df[col].value_counts(normalize=True, dropna=False).head(8).items():
            st.append({"table": table, "column": col, "value": v, "pct_rows": round(100 * p, 2)})
    st_df = pd.DataFrame(st)
    st_df.to_csv(out / "side_status_shares.csv", index=False)
    print(st_df.to_string(index=False), "\n")

    # ---- installment payment behaviour ----
    paid = ins.DAYS_ENTRY_PAYMENT.notna()
    late = paid & (ins.DAYS_ENTRY_PAYMENT > ins.DAYS_INSTALMENT)
    short = ins.AMT_PAYMENT < ins.AMT_INSTALMENT
    beh = pd.DataFrame([
        {"measure": "rows with no payment recorded", "pct_rows": round(100 * (~paid).mean(), 2)},
        {"measure": "paid after the due day", "pct_rows": round(100 * late.mean(), 2)},
        {"measure": "paid less than the instalment (row level)", "pct_rows": round(100 * short.mean(), 2)},
    ])
    beh.to_csv(out / "installments_behaviour.csv", index=False)
    print(beh.to_string(index=False), "\n")

    # ---- applicant-level signals vs default rate (train) ----
    sig = []

    def add_signal(name, values, bins, labels):
        v = tr.index.to_series().map(values)
        grp = pd.cut(v, bins=bins, labels=labels, include_lowest=True).astype(object).fillna("no history")
        g = tr.groupby(grp.values).agg(["size", "mean"])
        for lab in ["no history", *labels]:
            if lab in g.index:
                sig.append({"signal": name, "group": lab, "applicants": int(g.loc[lab, "size"]),
                            "pct_of_train": round(100 * g.loc[lab, "size"] / len(tr), 2),
                            "default_rate_pct": round(100 * g.loc[lab, "mean"], 2)})

    late_share = late.groupby(ins.SK_ID_CURR).mean()
    add_signal("share of installments paid late", late_share, [0, 1e-9, 0.05, 0.2, 1],
               ["0%", "0-5%", "5-20%", ">20%"])
    refused = (prev.NAME_CONTRACT_STATUS == "Refused").groupby(prev.SK_ID_CURR).mean()
    add_signal("share of previous applications refused", refused, [0, 1e-9, 0.25, 0.5, 1],
               ["0%", "0-25%", "25-50%", ">50%"])
    overdue = (bureau.CREDIT_DAY_OVERDUE > 0).groupby(bureau.SK_ID_CURR).any().astype(float)
    add_signal("any bureau credit overdue now", overdue, [0, 0.5, 1], ["no", "yes"])
    pos_dpd = (pos.SK_DPD_DEF > 0).groupby(pos.SK_ID_CURR).any().astype(float)
    add_signal("any POS month with DPD (tolerance applied)", pos_dpd, [0, 0.5, 1], ["no", "yes"])
    cc_dpd = (cc.SK_DPD_DEF > 0).groupby(cc.SK_ID_CURR).any().astype(float)
    add_signal("any credit-card month with DPD (tolerance applied)", cc_dpd, [0, 0.5, 1], ["no", "yes"])
    sig_df = pd.DataFrame(sig)
    sig_df.to_csv(out / "side_signals_vs_default.csv", index=False)
    print(sig_df.to_string(index=False))


if __name__ == "__main__":
    main()
