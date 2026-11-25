"""EDA steps 5 and 6: how test differs from train, whether there is a time order, and leakage checks.

Step 5 (train vs test, time):
  - For every application column, how well it alone separates test rows from train rows:
    numeric columns by AUC (0.5 = same distribution; reported as max(AUC, 1-AUC), missing values left out),
    text columns by total variation distance between category shares (0 = same shares, 1 = no overlap).
    Missingness itself is compared as well. A random split would give values near 0.5 / 0 everywhere.
  - Is there any absolute date? The tables only hold days relative to the application, so we check whether
    SK_ID_CURR could be a time proxy: are train and test IDs interleaved, and does the default rate drift with ID?

Step 6 (leakage):
  - AUC of each application column alone for TARGET (max(AUC, 1-AUC)). A column far stronger than the rest would
    need a look: it could hold information recorded after the loan was granted.
  - Rows in side tables dated after the application (positive DAYS_*), other than the 365243 code: counted in
    step 4 (numeric_profile.csv); here we only list which of them describe events rather than planned dates.

Usage:
    .venv/bin/python -m eda.drift_leakage [--out-root DIR]
"""

import argparse

import numpy as np
import pandas as pd

from homecredit.data import CODE, is_text
from homecredit.evaluate import auc
from homecredit.paths import Paths, add_out_root



def main() -> None:
    ap = argparse.ArgumentParser()
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    data, out = P.raw, P.out("eda")

    tr = pd.read_parquet(data / "application_train.parquet")
    te = pd.read_parquet(data / "application_test.parquet")
    # The 365243 code is replaced by NaN for these comparisons (step 4: it marks pensioners / unemployed).
    for df in (tr, te):
        df["DAYS_EMPLOYED"] = df.DAYS_EMPLOYED.replace(CODE, np.nan)
    cols = [c for c in te.columns if c != "SK_ID_CURR"]
    both = pd.concat([tr[cols], te[cols]], ignore_index=True)
    is_test = np.r_[np.zeros(len(tr)), np.ones(len(te))].astype(int)
    y = tr.TARGET.values

    rows = []
    for c in cols:
        s = both[c]
        miss_tr, miss_te = tr[c].isna().mean(), te[c].isna().mean()
        if is_text(s):
            p = tr[c].fillna("(missing)").value_counts(normalize=True)
            q = te[c].fillna("(missing)").value_counts(normalize=True)
            tvd = 0.5 * p.subtract(q, fill_value=0).abs().sum()
            rows.append({"column": c, "kind": "text", "drift_auc": None, "drift_tvd": round(tvd, 4),
                         "target_auc": None, "pct_missing_train": round(100 * miss_tr, 2),
                         "pct_missing_test": round(100 * miss_te, 2)})
        else:
            v = s.astype(float).values
            a_drift = auc(is_test, v)
            a_tgt = auc(y, tr[c].astype(float).values)
            rows.append({"column": c, "kind": "numeric",
                         "drift_auc": round(max(a_drift, 1 - a_drift), 4),
                         "drift_tvd": None,
                         "target_auc": round(max(a_tgt, 1 - a_tgt), 4),
                         "target_auc_direction": "higher = more defaults" if a_tgt >= 0.5 else "higher = fewer defaults",
                         "pct_missing_train": round(100 * miss_tr, 2), "pct_missing_test": round(100 * miss_te, 2)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "column_drift_and_target_auc.csv", index=False)

    print("numeric columns that best separate test from train (0.5 = no difference):")
    print(res[res.kind == "numeric"].sort_values("drift_auc", ascending=False)
          [["column", "drift_auc", "pct_missing_train", "pct_missing_test"]].head(10).to_string(index=False), "\n")
    print("text columns with the largest share differences (TVD):")
    print(res[res.kind == "text"].sort_values("drift_tvd", ascending=False)
          [["column", "drift_tvd"]].head(6).to_string(index=False), "\n")
    print("strongest single columns for TARGET (AUC, numeric only):")
    print(res[res.kind == "numeric"].sort_values("target_auc", ascending=False)
          [["column", "target_auc", "target_auc_direction", "pct_missing_train"]].head(12).to_string(index=False), "\n")

    # Detail for the two biggest shifts: value shares in train vs test.
    det = []
    for c, vals in [("NAME_CONTRACT_TYPE", None), ("AMT_REQ_CREDIT_BUREAU_QRT", [0, 1, 2])]:
        for name, df in [("train", tr), ("test", te)]:
            s = df[c]
            if vals is not None:
                s = s.where(s.isna() | s.isin(vals), other=f">{max(vals)}").astype("object")
            for v, p in s.fillna("(missing)").value_counts(normalize=True).items():
                det.append({"column": c, "value": str(v), "split": name, "pct": round(100 * p, 2)})
    det_df = pd.DataFrame(det).pivot_table(index=["column", "value"], columns="split", values="pct").reset_index()
    det_df.to_csv(out / "drift_details.csv", index=False)
    print(det_df.to_string(index=False), "\n")

    # ---- Is SK_ID_CURR a time proxy? ----
    ids = pd.DataFrame({"id": np.r_[tr.SK_ID_CURR, te.SK_ID_CURR], "is_test": is_test})
    ids["decile"] = pd.qcut(ids.id, 10, labels=False) + 1
    id_tab = ids.groupby("decile").agg(id_min=("id", "min"), id_max=("id", "max"), pct_test=("is_test", "mean"))
    tr_dec = pd.qcut(tr.SK_ID_CURR, 10, labels=False) + 1
    id_tab["train_default_rate_pct"] = (tr.TARGET.groupby(tr_dec.values).mean() * 100).round(2).values
    id_tab["pct_test"] = (id_tab.pct_test * 100).round(2)
    id_tab.to_csv(out / "id_order_check.csv")
    print("SK_ID_CURR deciles (train+test): share of test rows, and train default rate by train ID decile")
    print(id_tab.to_string(), "\n")
    a_id = auc(y, tr.SK_ID_CURR.astype(float).values)
    a_id_test = auc(ids.is_test.values, ids.id.astype(float).values)
    print(f"AUC of SK_ID_CURR for TARGET: {a_id:.4f}; for test vs train: {a_id_test:.4f}")
    pd.DataFrame([{"what": "SK_ID_CURR AUC for TARGET (train)", "auc": round(a_id, 4)},
                  {"what": "SK_ID_CURR AUC for test vs train", "auc": round(a_id_test, 4)}]).to_csv(
        out / "id_auc.csv", index=False)


if __name__ == "__main__":
    main()
