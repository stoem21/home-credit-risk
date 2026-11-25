"""Proxy check (fairness step, added after an external review): how much do the remaining columns reveal a
sensitive attribute that was left out?

Leaving gender and family status out of the model does not stop other columns from carrying them. Here LightGBM
(default settings, early stopping on an inner 10%, 5 folds, seed 0) learns to predict each attribute from the
columns the chosen v2 model uses, with that attribute's own columns removed:
  - gender (men vs women), from the chosen model's 186 columns (they already exclude gender and family);
  - married or civil marriage vs not, from the same columns;
  - age under 25 and age 55 or over, from the same columns minus the age columns (DAYS_BIRTH, age_years).
Out-of-fold AUC near 0.5 means the columns say little about the attribute; near 1.0 means they reveal it almost
fully. The columns that help most show where the proxies are. No TARGET is used.

Usage:
    .venv/bin/python -m analysis.proxy_check [--jobs 5] [--threads 2] [--out-root DIR]
Output: analysis_output/fairness_v2/proxy_check.csv, proxy_top_columns.csv
"""

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from homecredit.data import SENSITIVE_COLS, drop_columns, load_features, select_groups
from homecredit.evaluate import auc
from homecredit.models.lgbm import fit
from homecredit.paths import Paths, add_out_root


def setup(P):
    tr, _, cols = load_features(P, "v2", categories=True)
    cols = [c for c in drop_columns(select_groups(cols, "win,stk,term"), "gender,family") if not c.startswith("stk_bb_")]
    age = -tr.DAYS_BIRTH / 365.25
    married = tr.NAME_FAMILY_STATUS.astype(str).isin(["Married", "Civil marriage"])
    targets = {
        "gender: men vs women": (tr.CODE_GENDER.astype(str).eq("M").astype(int).values,
                                 tr.CODE_GENDER.astype(str).isin(["M", "F"]).values, cols),
        "family: married or civil marriage": (married.astype(int).values, np.ones(len(tr), bool), cols),
        "age: under 25": ((age < 25).astype(int).values, np.ones(len(tr), bool),
                          [c for c in cols if c not in SENSITIVE_COLS["age"]]),
        "age: 55 or over": ((age >= 55).astype(int).values, np.ones(len(tr), bool),
                            [c for c in cols if c not in SENSITIVE_COLS["age"]]),
    }
    return tr, targets


def run_fold(out_root, name, trn, va, threads):
    tr, targets = setup(Paths(out_root))
    t, keep, cols = targets[name]
    idx = np.where(keep)[0]
    X = tr[cols].iloc[idx]
    bst = fit(X.iloc[trn], t[idx][trn], 0, threads)
    return name, va, bst.predict(X.iloc[va], num_iteration=bst.best_iteration), \
        pd.Series(bst.feature_importance("gain"), index=cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    out = P.out("fairness_v2")
    tr, targets = setup(P)

    tasks = []
    for name, (t, keep, cols) in targets.items():
        idx = np.where(keep)[0]
        for trn, va in StratifiedKFold(5, shuffle=True, random_state=0).split(idx, t[idx]):
            tasks.append((name, trn, va))
    res = Parallel(n_jobs=args.jobs)(delayed(run_fold)(args.out_root, n, trn, va, args.threads)
                                     for n, trn, va in tasks)
    rows, tops = [], []
    for name, (t, keep, cols) in targets.items():
        yy = t[np.where(keep)[0]]
        p = np.empty(len(yy))
        imps = []
        for n, va, pv, imp in res:
            if n == name:
                p[va] = pv
                imps.append(imp)
        rows.append({"attribute": name, "rows": len(yy), "share_positive_pct": round(100 * yy.mean(), 2),
                     "columns_used": len(cols), "proxy_auc": round(auc(yy, p), 4)})
        imp = pd.concat(imps, axis=1).mean(axis=1)
        imp = (100 * imp / imp.sum()).sort_values(ascending=False).head(8)
        tops += [{"attribute": name, "rank": i + 1, "column": c, "share_of_gain_pct": round(v, 2)}
                 for i, (c, v) in enumerate(imp.items())]
    res_df = pd.DataFrame(rows)
    res_df.to_csv(out / "proxy_check.csv", index=False)
    pd.DataFrame(tops).to_csv(out / "proxy_top_columns.csv", index=False)
    pd.set_option("display.width", 200)
    print(res_df.to_string(index=False), "\n")
    print(pd.DataFrame(tops).to_string(index=False))


if __name__ == "__main__":
    main()
