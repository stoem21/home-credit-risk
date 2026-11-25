"""Does the same person appear more than once in train, and does splitting them across folds flatter the CV?

Found by an external code review: many train rows share the same birth date, ID-document date and gender. All
DAYS_* values count back from the application date, so equal values on the same rows suggest the same person
applying more than once (or a coincidence). A random fold split can put one application in training and the other
in validation, which could make validation easier than real use.

Check (exploratory, 1 seed): the chosen v2 feature set with the plain LightGBM setting (learning rate 0.05, one
fit; same as lgbm_v2_combo) is trained on grouped folds, where every pseudo-person stays in one fold
(StratifiedGroupKFold on TARGET, seed 0), and compared with the same model on the usual seed-0 folds (taken from
the saved lgbm_v2_combo out-of-fold predictions). AUC is reported for all rows, for the rows in a repeated group,
and for the others. The pseudo-person key is a guess at identity, not a confirmed ID.

Usage:
    .venv/bin/python -m experiments.duplicate_check [--jobs 5] [--threads 2] [--out-root DIR]
Output: analysis_output/validation/duplicate_check.csv
"""

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

from homecredit.data import drop_columns, load_features, load_folds, select_groups
from homecredit.evaluate import auc, load_oof
from homecredit.models.lgbm import fit
from homecredit.paths import Paths, add_out_root


def columns(P):
    tr, _, cols = load_features(P, "v2", categories=True)
    cols = [c for c in drop_columns(select_groups(cols, "win,stk,term"), "gender,family") if not c.startswith("stk_bb_")]
    return tr, cols


def run_fold(out_root, trn, va, threads):
    tr, cols = columns(Paths(out_root))
    y = tr.TARGET.values.astype(int)
    bst = fit(tr[cols].iloc[trn], y[trn], 0, threads)
    return va, bst.predict(tr[cols].iloc[va], num_iteration=bst.best_iteration)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    out = P.out("validation")

    tr, cols = columns(P)
    y = tr.TARGET.values.astype(int)
    key = (tr.DAYS_BIRTH.astype(str) + "_" + tr.DAYS_ID_PUBLISH.astype(str) + "_" + tr.CODE_GENDER.astype(str)).values
    size = pd.Series(key).map(pd.Series(key).value_counts()).values
    rep = size > 1
    folds = load_folds(P, tr.SK_ID_CURR)
    f0 = folds.fold_s0.values
    split = pd.DataFrame({"key": key[rep], "fold": f0[rep]}).groupby("key").fold.nunique()
    pairs = pd.DataFrame({"key": key, "y": y})[size == 2].groupby("key").y.agg(["sum"])
    both = (pairs["sum"] == 2).mean()
    expected = y[size == 2].mean() ** 2

    groups = pd.factorize(key)[0]
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    res = Parallel(n_jobs=args.jobs)(delayed(run_fold)(args.out_root, trn, va, args.threads)
                                     for trn, va in sgk.split(tr, y, groups))
    p_grp = np.empty(len(y))
    for va, p in res:
        p_grp[va] = p
    p_std = load_oof(P, "lgbm_v2_combo", tr.SK_ID_CURR).s0.values

    rows = [{"measure": "train rows in a repeated pseudo-person group", "value": int(rep.sum())},
            {"measure": "repeated groups", "value": int(len(split))},
            {"measure": "repeated groups split across seed-0 folds", "value": int((split > 1).sum())},
            {"measure": "pairs where both defaulted, % (observed)", "value": round(100 * both, 3)},
            {"measure": "pairs where both defaulted, % (if independent)", "value": round(100 * expected, 3)}]
    for name, m in [("all rows", np.ones(len(y), bool)), ("rows in a repeated group", rep),
                    ("other rows", ~rep)]:
        a_std, a_grp = auc(y[m], p_std[m]), auc(y[m], p_grp[m])
        rows += [{"measure": f"AUC usual folds, {name}", "value": round(a_std, 5)},
                 {"measure": f"AUC grouped folds, {name}", "value": round(a_grp, 5)},
                 {"measure": f"AUC grouped minus usual, {name}", "value": round(a_grp - a_std, 5)}]
    res_df = pd.DataFrame(rows)
    res_df.to_csv(out / "duplicate_check.csv", index=False)
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
