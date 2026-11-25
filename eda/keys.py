"""EDA step 1: table structure and keys.

Answers, from the Parquet copies of the raw data:
  1. Are the ID columns unique where they should be (one row per loan / per previous credit)?
  2. Do train and test share any applicant? Does sample_submission list exactly the test IDs?
  3. Do the side tables join to the main table? The column description marks SK_ID_CURR as "hashed" in some
     side tables, so we measure how many of their IDs are found in application_train/test.
  4. How many history rows does each applicant have, and how many have none at all?

Usage:
    .venv/bin/python -m eda.keys [--out-root DIR]

Writes keys_uniqueness.csv, join_coverage.csv and rows_per_applicant.csv to the output folder.
"""

import argparse
from pathlib import Path

import pandas as pd

from homecredit.paths import Paths, add_out_root


def load(data: Path, name: str, cols: list[str]) -> pd.DataFrame:
    return pd.read_parquet(data / f"{name}.parquet", columns=cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    data, out = P.raw, P.out("eda")

    train = load(data, "application_train", ["SK_ID_CURR"])
    test = load(data, "application_test", ["SK_ID_CURR"])
    sub = load(data, "sample_submission", ["SK_ID_CURR"])
    bureau = load(data, "bureau", ["SK_ID_CURR", "SK_ID_BUREAU"])
    bb = load(data, "bureau_balance", ["SK_ID_BUREAU"])
    prev = load(data, "previous_application", ["SK_ID_CURR", "SK_ID_PREV"])
    side_prev = {
        name: load(data, name, ["SK_ID_CURR", "SK_ID_PREV"])
        for name in ["POS_CASH_balance", "credit_card_balance", "installments_payments"]
    }

    train_ids, test_ids = set(train.SK_ID_CURR), set(test.SK_ID_CURR)
    app_ids = train_ids | test_ids

    # 1-2. Uniqueness and overlaps
    uniq = []

    def add_unique(table: str, col: str, s: pd.Series) -> None:
        uniq.append(
            {"table": table, "column": col, "rows": len(s), "distinct": s.nunique(), "missing": int(s.isna().sum()),
             "unique": s.is_unique}
        )

    add_unique("application_train", "SK_ID_CURR", train.SK_ID_CURR)
    add_unique("application_test", "SK_ID_CURR", test.SK_ID_CURR)
    add_unique("sample_submission", "SK_ID_CURR", sub.SK_ID_CURR)
    add_unique("bureau", "SK_ID_BUREAU", bureau.SK_ID_BUREAU)
    add_unique("previous_application", "SK_ID_PREV", prev.SK_ID_PREV)
    uniq_df = pd.DataFrame(uniq)
    uniq_df.to_csv(out / "keys_uniqueness.csv", index=False)
    print(uniq_df.to_string(index=False), "\n")
    print("train/test shared SK_ID_CURR:", len(train_ids & test_ids))
    print("sample_submission IDs == test IDs:", set(sub.SK_ID_CURR) == test_ids, "\n")

    # 3. Join coverage. Two directions per link:
    #    child_match_pct: share of the side table's distinct keys found in the parent table (low = broken/hashed IDs)
    #    train_pct / test_pct: share of applicants with at least one row in the side table
    cov = []

    def add_cov(child: str, key: str, child_keys: pd.Series, parent: str, parent_keys: set) -> None:
        ck = set(child_keys.dropna())
        cov.append(
            {"child": child, "key": key, "parent": parent, "child_distinct_keys": len(ck),
             "child_match_pct": round(100 * len(ck & parent_keys) / len(ck), 2)}
        )

    add_cov("bureau", "SK_ID_CURR", bureau.SK_ID_CURR, "application_train+test", app_ids)
    add_cov("previous_application", "SK_ID_CURR", prev.SK_ID_CURR, "application_train+test", app_ids)
    for name, df in side_prev.items():
        add_cov(name, "SK_ID_CURR", df.SK_ID_CURR, "application_train+test", app_ids)
    add_cov("bureau_balance", "SK_ID_BUREAU", bb.SK_ID_BUREAU, "bureau", set(bureau.SK_ID_BUREAU))
    for name, df in side_prev.items():
        add_cov(name, "SK_ID_PREV", df.SK_ID_PREV, "previous_application", set(prev.SK_ID_PREV))
    cov_df = pd.DataFrame(cov)

    # Do the monthly tables agree with previous_application on which applicant owns each SK_ID_PREV?
    owner = prev.set_index("SK_ID_PREV").SK_ID_CURR
    agree = {}
    for name, df in side_prev.items():
        pairs = df.drop_duplicates()
        matched = pairs[pairs.SK_ID_PREV.isin(owner.index)]
        agree[name] = round(100 * (matched.SK_ID_CURR.values == owner.loc[matched.SK_ID_PREV].values).mean(), 2)
    cov_df["same_owner_as_prev_pct"] = cov_df.apply(
        lambda r: agree.get(r.child) if r.key == "SK_ID_PREV" else None, axis=1
    )
    cov_df.to_csv(out / "join_coverage.csv", index=False)
    print(cov_df.to_string(index=False), "\n")

    # 4. Rows per applicant (train and test separately), including applicants with zero rows.
    # bureau_balance is counted through bureau: months of history per applicant.
    bb_per_bureau = bb.SK_ID_BUREAU.value_counts()
    bb_curr = bureau.assign(n=bureau.SK_ID_BUREAU.map(bb_per_bureau).fillna(0)).groupby("SK_ID_CURR").n.sum()
    counts = {
        "bureau": bureau.SK_ID_CURR.value_counts(),
        "bureau_balance (via bureau)": bb_curr,
        "previous_application": prev.SK_ID_CURR.value_counts(),
        **{name: df.SK_ID_CURR.value_counts() for name, df in side_prev.items()},
    }
    rpa = []
    for table, vc in counts.items():
        for split, ids in [("train", train.SK_ID_CURR), ("test", test.SK_ID_CURR)]:
            n = ids.map(vc).fillna(0)
            q = n.quantile([0.5, 0.9, 0.99])
            rpa.append(
                {"table": table, "split": split, "applicants": len(n), "pct_with_none": round(100 * (n == 0).mean(), 2),
                 "mean": round(n.mean(), 2), "median": q[0.5], "p90": q[0.9], "p99": q[0.99], "max": int(n.max())}
            )
    rpa_df = pd.DataFrame(rpa)
    rpa_df.to_csv(out / "rows_per_applicant.csv", index=False)
    print(rpa_df.to_string(index=False), "\n")

    # 5. bureau_balance coverage differs a lot between train and test (found in step 4). Break it down:
    #    share of bureau credits with any monthly history, and per applicant whether none / some / all of their
    #    bureau credits have it. For train, also the default rate in each group.
    split = pd.concat(
        [pd.Series("train", index=train.SK_ID_CURR), pd.Series("test", index=test.SK_ID_CURR)]
    )
    b = bureau.assign(has_bb=bureau.SK_ID_BUREAU.isin(bb_per_bureau.index), split=bureau.SK_ID_CURR.map(split))
    share = b.groupby("SK_ID_CURR").has_bb.mean()
    target = pd.read_parquet(data / "application_train.parquet", columns=["SK_ID_CURR", "TARGET"])
    target = target.set_index("SK_ID_CURR").TARGET
    bbc = []
    for s, ids in [("train", train.SK_ID_CURR), ("test", test.SK_ID_CURR)]:
        sh = ids.map(share)  # NaN = applicant has no bureau rows
        grp = pd.Series("no bureau rows", index=ids.index)
        grp[sh == 0] = "bureau, no balance history"
        grp[(sh > 0) & (sh < 1)] = "bureau, some credits with history"
        grp[sh == 1] = "bureau, all credits with history"
        for g, idx in grp.groupby(grp).groups.items():
            bbc.append(
                {"split": s, "group": g, "applicants": len(idx), "pct_of_split": round(100 * len(idx) / len(ids), 2),
                 "default_rate_pct": round(100 * ids[idx].map(target).mean(), 2) if s == "train" else None}
            )
        bbc.append(
            {"split": s, "group": "(bureau credits with any balance history, % of credits)",
             "applicants": None, "pct_of_split": round(100 * b[b.split == s].has_bb.mean(), 2),
             "default_rate_pct": None}
        )
    bbc_df = pd.DataFrame(bbc)
    bbc_df.to_csv(out / "bureau_balance_coverage.csv", index=False)
    print(bbc_df.to_string(index=False))


if __name__ == "__main__":
    main()
