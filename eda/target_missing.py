"""EDA steps 2 and 3: the target, and missing values.

Step 2 (target): overall default rate in application_train, and the default rate by every category of every
text column (count and rate, so small categories are visible as small).

Step 3 (missing values):
  - share missing per column, train vs test, for every table (side tables split by the applicant's split);
  - groups of application columns that tend to be missing together (missing-indicator correlation >= 0.9);
  - default rate when a column is missing vs present (application_train), to see whether "missing" carries signal.

Missing here means NaN in the file. Text placeholders such as "XNA" are not counted; that is EDA step 4.

Usage:
    .venv/bin/python -m eda.target_missing [--out-root DIR]
"""

import argparse

import numpy as np
import pandas as pd

from homecredit.data import is_text
from homecredit.paths import Paths, add_out_root

SIDE_TABLES = [
    "bureau",
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

    train = pd.read_parquet(data / "application_train.parquet")
    test = pd.read_parquet(data / "application_test.parquet")
    y = train.TARGET

    # ---- Step 2: target ----
    overall = pd.DataFrame(
        [{"applicants": len(y), "defaults": int(y.sum()), "default_rate_pct": round(100 * y.mean(), 2),
          "target_values": ",".join(map(str, sorted(y.unique())))}]
    )
    overall.to_csv(out / "target_overall.csv", index=False)
    print(overall.to_string(index=False), "\n")

    text_cols = [c for c in train.columns if is_text(train[c])]
    rows = []
    for c in text_cols:
        g = y.groupby(train[c].fillna("(missing)")).agg(["size", "mean"])
        for cat, r in g.iterrows():
            rows.append(
                {"column": c, "category": cat, "applicants": int(r["size"]),
                 "pct_of_train": round(100 * r["size"] / len(y), 2), "default_rate_pct": round(100 * r["mean"], 2)}
            )
    by_cat = pd.DataFrame(rows)
    by_cat.to_csv(out / "target_by_category.csv", index=False)
    print(by_cat[by_cat.column == "NAME_CONTRACT_TYPE"].to_string(index=False), "\n")

    # ---- Step 3a: share missing per column, train vs test ----
    split = pd.concat([pd.Series("train", index=train.SK_ID_CURR), pd.Series("test", index=test.SK_ID_CURR)])
    miss = []

    def add_missing(table: str, df: pd.DataFrame, s: pd.Series) -> None:
        for sp in ["train", "test"]:
            part = df[s.values == sp]
            pct = part.isna().mean() * 100
            for c, p in pct.items():
                miss.append({"table": table, "column": c, "split": sp, "pct_missing": round(p, 2)})

    feat_cols = [c for c in test.columns if c != "SK_ID_CURR"]
    app = pd.concat([train[feat_cols], test[feat_cols]], ignore_index=True)
    app_split = pd.Series(["train"] * len(train) + ["test"] * len(test))
    add_missing("application", app, app_split)
    for t in SIDE_TABLES:
        df = pd.read_parquet(data / f"{t}.parquet")
        add_missing(t, df.drop(columns=["SK_ID_CURR"]), df.SK_ID_CURR.map(split))
    miss_df = (
        pd.DataFrame(miss)
        .pivot_table(index=["table", "column"], columns="split", values="pct_missing", sort=False)
        .reset_index()[["table", "column", "train", "test"]]
        .rename(columns={"train": "pct_missing_train", "test": "pct_missing_test"})
    )
    miss_df["gap_pts"] = (miss_df.pct_missing_test - miss_df.pct_missing_train).round(2)
    miss_df.to_csv(out / "missing_by_column.csv", index=False)

    summary = (
        miss_df.assign(any_missing=miss_df.pct_missing_train > 0)
        .groupby("table", sort=False)
        .agg(columns=("column", "size"), with_missing=("any_missing", "sum"),
             over_50pct=("pct_missing_train", lambda s: int((s > 50).sum())),
             max_abs_gap_pts=("gap_pts", lambda s: s.abs().max()))
    )
    print(summary.to_string(), "\n")
    print("largest train/test gaps:")
    print(miss_df.reindex(miss_df.gap_pts.abs().sort_values(ascending=False).index).head(10).to_string(index=False))
    print()

    # ---- Step 3b: application columns missing together ----
    m = train[feat_cols].isna()
    m = m.loc[:, m.any()]
    corr = m.astype(float).corr().fillna(0)
    unassigned = list(corr.columns)
    groups = []
    while unassigned:
        c = unassigned[0]
        members = [d for d in unassigned if corr.loc[c, d] >= 0.9]
        groups.append(members)
        unassigned = [d for d in unassigned if d not in members]
    grp_rows = []
    for i, members in enumerate(sorted(groups, key=len, reverse=True)):
        all_missing = m[members].all(axis=1)
        grp_rows.append(
            {"group": i + 1, "n_columns": len(members), "pct_rows_all_missing": round(100 * all_missing.mean(), 2),
             "example_columns": ", ".join(members[:4]) + (" ..." if len(members) > 4 else ""),
             "columns": ", ".join(members)}
        )
    grp_df = pd.DataFrame(grp_rows)
    grp_df.to_csv(out / "missing_groups.csv", index=False)
    print(grp_df.drop(columns="columns").head(8).to_string(index=False), "\n")

    # ---- Step 3c: default rate when missing vs present ----
    mvd = []
    for c in m.columns:
        is_na = m[c]
        mvd.append(
            {"column": c, "n_missing": int(is_na.sum()), "pct_missing": round(100 * is_na.mean(), 2),
             "default_rate_missing_pct": round(100 * y[is_na].mean(), 2),
             "default_rate_present_pct": round(100 * y[~is_na].mean(), 2)}
        )
    mvd_df = pd.DataFrame(mvd)
    mvd_df["diff_pts"] = (mvd_df.default_rate_missing_pct - mvd_df.default_rate_present_pct).round(2)
    # Rates from a handful of rows are noise: rank only columns with at least MIN_MISSING missing rows,
    # and keep the rest at the bottom of the file, flagged.
    MIN_MISSING = 1000
    mvd_df["enough_rows"] = mvd_df.n_missing >= MIN_MISSING
    mvd_df = mvd_df.assign(_k=-mvd_df.diff_pts.abs()).sort_values(["enough_rows", "_k"], ascending=[False, True])
    mvd_df = mvd_df.drop(columns="_k")
    mvd_df.to_csv(out / "missing_vs_default.csv", index=False)
    print(f"columns with >= {MIN_MISSING} missing rows, largest gaps first:")
    print(mvd_df[mvd_df.enough_rows].head(12).to_string(index=False))
    print("\nEXT_SOURCE columns:")
    print(mvd_df[mvd_df.column.str.startswith("EXT_SOURCE")].to_string(index=False))


if __name__ == "__main__":
    main()
