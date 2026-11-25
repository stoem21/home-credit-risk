"""EDA step 4: odd values and placeholder codes.

  - Every numeric column of every table: min, max, most common value and its share, how often the value 365243
    appears (a "365243 days" value is 1,000 years, so it can only be a code, not a real duration), and for DAYS_*
    columns how many values are positive (DAYS_* count days relative to the application, so positive = after it).
  - Every text column: how often the placeholders "XNA" and "XAP" appear.
  - application: what DAYS_EMPLOYED = 365243 means (by income type, default rate); extreme incomes; and two guesses
    from step 3: are missing credit-bureau enquiry counts the applicants without bureau rows, and is missing
    occupation mostly pensioners?

Usage:
    .venv/bin/python -m eda.quirks [--out-root DIR]
"""

import argparse

import pandas as pd

from homecredit.data import CODE, is_text
from homecredit.paths import Paths, add_out_root

TABLES = [
    "application_train",
    "application_test",
    "bureau",
    "bureau_balance",
    "previous_application",
    "POS_CASH_balance",
    "credit_card_balance",
    "installments_payments",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    data, out = P.raw, P.out("eda")

    num_rows, txt_rows = [], []
    for t in TABLES:
        df = pd.read_parquet(data / f"{t}.parquet")
        for c in df.columns:
            s = df[c]
            if c.startswith("SK_ID") or c == "TARGET":
                continue
            if is_text(s):
                vc = s.value_counts()
                txt_rows.append(
                    {"table": t, "column": c, "n_categories": s.nunique(),
                     "n_XNA": int(vc.get("XNA", 0)), "n_XAP": int(vc.get("XAP", 0)),
                     "pct_XNA_or_XAP": round(100 * (vc.get("XNA", 0) + vc.get("XAP", 0)) / len(s), 2)}
                )
                continue
            nn = s.dropna()
            top = nn.value_counts().head(1)
            num_rows.append(
                {"table": t, "column": c, "min": nn.min(), "max": nn.max(),
                 "top_value": top.index[0] if len(top) else None,
                 "top_value_pct": round(100 * top.iloc[0] / len(s), 2) if len(top) else None,
                 "n_365243": int((nn == CODE).sum()),
                 "n_positive_days": int((nn > 0).sum()) if c.startswith("DAYS_") else None,
                 "n_positive_days_excl_365243": int(((nn > 0) & (nn != CODE)).sum()) if c.startswith("DAYS_") else None}
            )
    num = pd.DataFrame(num_rows)
    txt = pd.DataFrame(txt_rows)
    num.to_csv(out / "numeric_profile.csv", index=False)
    txt.to_csv(out / "placeholder_values.csv", index=False)

    print("columns containing 365243:")
    print(num[num.n_365243 > 0][["table", "column", "n_365243", "top_value_pct"]].to_string(index=False), "\n")
    print("DAYS_ columns with positive values other than 365243:")
    pos = num[num.n_positive_days_excl_365243.fillna(0) > 0]
    print(pos[["table", "column", "min", "max", "n_positive_days_excl_365243"]].to_string(index=False), "\n")
    print("text columns with XNA / XAP:")
    print(txt[txt.pct_XNA_or_XAP > 0].to_string(index=False), "\n")

    # ---- application_train specifics ----
    tr = pd.read_parquet(data / "application_train.parquet")
    checks = []

    def add(check: str, group: str, n: int, rate) -> None:
        checks.append({"check": check, "group": group, "applicants": n,
                       "pct_of_train": round(100 * n / len(tr), 2),
                       "default_rate_pct": None if rate is None else round(100 * rate, 2)})

    emp = tr.DAYS_EMPLOYED == CODE
    add("DAYS_EMPLOYED", "= 365243", int(emp.sum()), tr.TARGET[emp].mean())
    add("DAYS_EMPLOYED", "real value", int((~emp).sum()), tr.TARGET[~emp].mean())
    for it, n in tr.NAME_INCOME_TYPE[emp].value_counts().items():
        add("DAYS_EMPLOYED = 365243, by income type", it, int(n), tr.TARGET[emp & (tr.NAME_INCOME_TYPE == it)].mean())
    pens = tr.NAME_INCOME_TYPE == "Pensioner"
    add("Pensioners with DAYS_EMPLOYED = 365243", f"{int((pens & emp).sum())} of {int(pens.sum())} pensioners",
        int((pens & emp).sum()), None)

    gx = tr.CODE_GENDER == "XNA"
    add("CODE_GENDER", "XNA", int(gx.sum()), tr.TARGET[gx].mean() if gx.any() else None)

    inc = tr.AMT_INCOME_TOTAL
    for q in [0.5, 0.99, 0.999]:
        add("AMT_INCOME_TOTAL quantile", f"q{q}: {inc.quantile(q):,.0f}", int((inc > inc.quantile(q)).sum()), None)
    add("AMT_INCOME_TOTAL max", f"{inc.max():,.0f}", int((inc == inc.max()).sum()), tr.TARGET[inc == inc.max()].mean())

    # Guess 1 from step 3: missing enquiry counts = no bureau rows?
    has_bureau = tr.SK_ID_CURR.isin(pd.read_parquet(data / "bureau.parquet", columns=["SK_ID_CURR"]).SK_ID_CURR)
    enq_na = tr.AMT_REQ_CREDIT_BUREAU_YEAR.isna()
    for e in [True, False]:
        for b in [True, False]:
            m = (enq_na == e) & (has_bureau == b)
            add("enquiry count missing x has bureau rows", f"enquiry_missing={e}, has_bureau={b}", int(m.sum()),
                tr.TARGET[m].mean() if m.any() else None)

    # Guess 2 from step 3: missing occupation = mostly pensioners?
    occ_na = tr.OCCUPATION_TYPE.isna()
    for it, n in tr.NAME_INCOME_TYPE[occ_na].value_counts().head(4).items():
        add("OCCUPATION_TYPE missing, by income type", it, int(n), tr.TARGET[occ_na & (tr.NAME_INCOME_TYPE == it)].mean())

    chk = pd.DataFrame(checks)
    chk.to_csv(out / "quirk_checks.csv", index=False)
    print(chk.to_string(index=False))


if __name__ == "__main__":
    main()
