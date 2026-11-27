"""Loading rules shared by every script: application tables, folds, feature tables, scopes, column types."""

import numpy as np
import pandas as pd

from homecredit.paths import Paths

CODE = 365243  # "365243 days" placeholder (EDA step 4): pensioners / unemployed in DAYS_EMPLOYED
NOT_FEATURES = {"SK_ID_CURR", "TARGET", "split"}
HIST_PREFIX = ("bureau_", "prev_", "inst_", "pos_", "cc_")  # history summaries in the feature table
N_SEEDS = 5
N_FOLDS = 5

# Possibly sensitive columns, by attribute (EDA step 8). Used by --drop in the training scripts and by the fairness
# analysis. Which of them a lender may use is a judgment, made in the fairness step with evidence.
SENSITIVE_COLS = {
    "gender": ["CODE_GENDER"],
    "family": ["NAME_FAMILY_STATUS", "CNT_CHILDREN", "CNT_FAM_MEMBERS"],
    "age": ["DAYS_BIRTH", "age_years"],
}


def drop_columns(cols: list[str], drop: str | None) -> list[str]:
    """Remove the columns of the named sensitive attributes (comma list, e.g. "gender,family") from cols."""
    if not drop:
        return cols
    gone = {c for a in drop.split(",") for c in SENSITIVE_COLS[a]}
    return [c for c in cols if c not in gone]


def exclude_prefixes(cols: list[str], exclude: str | None) -> list[str]:
    """Remove columns starting with any of the comma-separated prefixes in `exclude` (e.g. "stk_bb_,load_total")."""
    if not exclude:
        return cols
    return [c for c in cols if not c.startswith(tuple(exclude.split(",")))]


# Feature groups added in feature table v2 (pipeline/make_features_v2.py), by short name -> column prefix.
NEW_GROUPS = {"win": "win_", "stk": "stk_", "term": "term_", "ext": "ext_",
              # feature table v3 (pipeline/make_features_v3.py)
              "agg": "agg_", "last": "last_", "trend": "trend_", "load": "load_",
              # feature table v3c (pipeline/make_features_rate.py)
              "rate": "rate_"}


def select_groups(cols: list[str], add: str | None) -> list[str]:
    """Keep the v1 columns plus the v2 groups named in `add` (comma list, e.g. "win,stk"; "all" = every group)."""
    names = list(NEW_GROUPS) if add == "all" else (add.split(",") if add else [])
    keep = tuple(NEW_GROUPS[n] for n in names)
    every = tuple(NEW_GROUPS.values())
    return [c for c in cols if not c.startswith(every) or (keep and c.startswith(keep))]


def is_text(s: pd.Series) -> bool:
    """Text-like column: object, pandas string or category dtype."""
    return s.dtype == object or str(s.dtype) in ("str", "category")


def read_raw(P: Paths, table: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(P.raw / f"{table}.parquet", columns=columns)


def load_target(P: Paths) -> pd.DataFrame:
    """SK_ID_CURR, TARGET and NAME_CONTRACT_TYPE of application_train, in file order."""
    return read_raw(P, "application_train", ["SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE"])


def scopes(contract_type: pd.Series) -> dict[str, np.ndarray]:
    """Row masks for the two scopes every model is scored on: all loans, and cash loans only (test is 99% cash)."""
    return {"all": np.ones(len(contract_type), bool), "cash": contract_type.eq("Cash loans").values}


def check_ids(df: pd.DataFrame, ids: pd.Series, what: str) -> None:
    assert (df.SK_ID_CURR.values == np.asarray(ids)).all(), f"{what}: SK_ID_CURR order differs from train"


def load_folds(P: Paths, ids: pd.Series) -> pd.DataFrame:
    """Folds table (SK_ID_CURR, fold_s0..fold_s4), checked to be in the same row order as `ids`."""
    fo = pd.read_parquet(P.folds)
    check_ids(fo, ids, "folds")
    return fo


def load_features(P: Paths, version: str = "v1", categories: bool = False):
    """Feature table split into train and test, plus the list of feature columns.

    categories=True turns text columns into pandas categories (LightGBM's own handling); the categories come from
    train + test values, which carries no target information.
    """
    f = pd.read_parquet(P.features(version))
    cols = [c for c in f.columns if c not in NOT_FEATURES]
    if categories:
        for c in cols:
            if f[c].dtype == object or str(f[c].dtype) == "str":
                f[c] = f[c].astype("category")
    tr = f[f.split == "train"].reset_index(drop=True)
    te = f[f.split == "test"].reset_index(drop=True)
    return tr, te, cols
