"""Validation folds: stratified 5-fold cross-validation on application_train, repeated with 5 seeds.

Why this design (EDA steps 5-6, EXPERIMENTS.md): the data has no absolute dates, so a split by time is not
possible, and SK_ID_CURR is not a time proxy. Folds are therefore random but stratified on TARGET and on
NAME_CONTRACT_TYPE, so every fold has the same default rate and the same share of cash vs revolving loans.

Each model is scored twice on every validation fold: on all loans, and on cash loans only, because the Kaggle
test set is 99% cash loans (train: 90%).

Five seeds let us measure how much a score moves just from the random split (noise) before comparing models.

Usage:
    .venv/bin/python -m pipeline.make_folds [--out-root DIR]

Output: one row per train applicant, columns SK_ID_CURR, fold_s0 ... fold_s4 (values 0-4).
"""

import argparse

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from homecredit.data import N_FOLDS, N_SEEDS, load_target
from homecredit.paths import Paths, add_out_root

SEEDS = list(range(N_SEEDS))


def main() -> None:
    ap = argparse.ArgumentParser()
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)

    tr = load_target(P)
    strata = tr.TARGET.astype(str) + "_" + tr.NAME_CONTRACT_TYPE
    folds = pd.DataFrame({"SK_ID_CURR": tr.SK_ID_CURR})
    for seed in SEEDS:
        col = f"fold_s{seed}"
        folds[col] = -1
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for k, (_, va) in enumerate(skf.split(tr, strata)):
            folds.loc[folds.index[va], col] = k
    assert (folds.filter(like="fold_") >= 0).all().all()
    P.processed.mkdir(parents=True, exist_ok=True)
    folds.to_parquet(P.folds, index=False)

    # Check: default rate and revolving share per fold (seed 0)
    chk = tr.assign(fold=folds.fold_s0, revolving=tr.NAME_CONTRACT_TYPE.eq("Revolving loans")).groupby("fold").agg(
        rows=("TARGET", "size"), default_rate_pct=("TARGET", "mean"), revolving_pct=("revolving", "mean"))
    chk[["default_rate_pct", "revolving_pct"]] *= 100
    print(chk.round(3).to_string())


if __name__ == "__main__":
    main()
