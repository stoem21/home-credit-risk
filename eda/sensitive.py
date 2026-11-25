"""EDA step 8: inventory of columns that may be sensitive for fairness.

Only describes them: share of applicants and default rate per group, train and test shares. Whether a column is
treated as a protected attribute, and whether the model may use it, is decided later (fairness step) with evidence.

Candidates: gender, age, family status, number of children, education, and the region rating (a region-level
attribute that can act as a proxy for where someone lives).

Usage:
    .venv/bin/python -m eda.sensitive [--out-root DIR]
"""

import argparse

import pandas as pd

from homecredit.paths import Paths, add_out_root


def main() -> None:
    ap = argparse.ArgumentParser()
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    data, out = P.raw, P.out("eda")

    tr = pd.read_parquet(data / "application_train.parquet")
    te = pd.read_parquet(data / "application_test.parquet")

    def groups(df: pd.DataFrame) -> pd.DataFrame:
        age = -df.DAYS_BIRTH / 365.25
        return pd.DataFrame({
            "gender (CODE_GENDER)": df.CODE_GENDER,
            "age band (from DAYS_BIRTH)": pd.cut(age, [0, 25, 35, 45, 55, 65, 100],
                                                 labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
                                                 right=False).astype(str),
            "family status (NAME_FAMILY_STATUS)": df.NAME_FAMILY_STATUS,
            "children (CNT_CHILDREN)": df.CNT_CHILDREN.clip(upper=3).map(lambda v: "3+" if v >= 3 else str(int(v))),
            "education (NAME_EDUCATION_TYPE)": df.NAME_EDUCATION_TYPE,
            "region rating (REGION_RATING_CLIENT)": df.REGION_RATING_CLIENT.astype(str),
        })

    gtr, gte = groups(tr), groups(te)
    rows = []
    for col in gtr.columns:
        g = tr.TARGET.groupby(gtr[col]).agg(["size", "mean"])
        test_share = gte[col].value_counts(normalize=True)
        for v, r in g.iterrows():
            rows.append({"attribute": col, "group": v, "train_applicants": int(r["size"]),
                         "pct_of_train": round(100 * r["size"] / len(tr), 2),
                         "pct_of_test": round(100 * test_share.get(v, 0), 2),
                         "default_rate_pct": round(100 * r["mean"], 2)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "sensitive_inventory.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
