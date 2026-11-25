"""Measure validation noise before comparing any models (project step 3).

Two deliberately simple scorers, each scored on every validation fold of the 5 seeds x 5 folds in
processed/folds.parquet, on all loans and on cash loans only:

  rule  - no training: score = minus the mean of the available EXT_SOURCE_1/2/3 (the three strongest single
          columns in EDA step 6). Its spread across folds is pure evaluation noise (which rows land in a fold).
  logreg - logistic regression on a dozen application columns (listed in FEATURES below), fitted on the training
          folds. Its spread includes training noise too. This is also the first baseline of project step 4.

What the numbers are for:
  - fold_sd: how much one fold's AUC moves. Use it to judge a single-fold comparison (we will not do that).
  - seed_sd: how much the 5-fold mean AUC moves when only the random split changes. A change between two models
    smaller than about 2 x seed_sd is not distinguishable from noise.
  - Paired comparisons (same folds, model A vs B) are less noisy than this; they are used later.

It also saves the out-of-fold predictions of the logistic baseline (processed/oof/logreg_baseline.parquet), the
reference for later paired comparisons (experiments/compare.py).

Usage:
    .venv/bin/python -m experiments.noise_check [--out-root DIR]
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from homecredit.data import CODE, load_folds, read_raw, scopes
from homecredit.evaluate import auc, bootstrap_aucs, fold_aucs, save_oof
from homecredit.paths import Paths, add_out_root

EXT = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple, hand-picked application features. NaN are filled later with training-fold medians."""
    x = pd.DataFrame(index=df.index)
    for c in EXT:
        x[c] = df[c]
        x[c + "_missing"] = df[c].isna().astype(float)
    x["age_years"] = -df.DAYS_BIRTH / 365.25
    emp = df.DAYS_EMPLOYED.replace(CODE, np.nan)
    x["employed_years"] = -emp / 365.25
    x["no_employment_code"] = df.DAYS_EMPLOYED.eq(CODE).astype(float)  # pensioners / unemployed (EDA step 4)
    x["credit_to_income"] = df.AMT_CREDIT / df.AMT_INCOME_TOTAL
    x["annuity_to_income"] = df.AMT_ANNUITY / df.AMT_INCOME_TOTAL
    x["credit_to_goods"] = df.AMT_CREDIT / df.AMT_GOODS_PRICE
    x["revolving"] = df.NAME_CONTRACT_TYPE.eq("Revolving loans").astype(float)
    x["male"] = df.CODE_GENDER.eq("M").astype(float)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    out = P.out("validation")

    tr = read_raw(P, "application_train")
    folds = load_folds(P, tr.SK_ID_CURR)
    y = tr.TARGET.values
    masks = scopes(tr.NAME_CONTRACT_TYPE)

    rule_score = -tr[EXT].mean(axis=1).fillna(tr[EXT].mean(axis=1).mean()).values
    X = build_features(tr)

    oof_all = pd.DataFrame({"SK_ID_CURR": tr.SK_ID_CURR})
    seeds = [int(c.split("_s")[1]) for c in folds.columns if c.startswith("fold_")]
    for seed in seeds:
        f = folds[f"fold_s{seed}"].values
        oof_all[f"s{seed}"] = np.nan
        for k in sorted(np.unique(f)):
            va, trn = f == k, f != k
            med = X[trn].median()
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            model.fit(X[trn].fillna(med), y[trn])
            oof_all.loc[va, f"s{seed}"] = model.predict_proba(X[va].fillna(med))[:, 1]
        print(f"seed {seed} done", flush=True)
    rule_all = pd.DataFrame({f"s{s}": rule_score for s in seeds})

    # One row per seed x fold x model x scope, in that order.
    res = pd.concat([fold_aucs(y, o, folds, masks, seeds).assign(model=name)
                     for name, o in [("rule", rule_all), ("logreg", oof_all)]], ignore_index=True)
    order = {"rule": 0, "logreg": 1, "all": 0, "cash": 1}
    res = (res.assign(_m=res.model.map(order), _s=res.scope.map(order))
           .sort_values(["seed", "fold", "_m", "_s"])[["seed", "fold", "model", "scope", "auc"]]
           .reset_index(drop=True))
    res.to_csv(out / "noise_by_fold.csv", index=False)
    # Out-of-fold predictions of the logistic baseline, one column per seed, for paired comparisons.
    save_oof(P, "logreg_baseline", oof_all)

    seed_means = res.groupby(["model", "scope", "seed"]).auc.mean()
    summ = res.groupby(["model", "scope"]).auc.agg(mean_auc="mean").join(
        res.groupby(["model", "scope", "seed"]).auc.std().groupby(["model", "scope"]).mean().rename("fold_sd")
    ).join(seed_means.groupby(["model", "scope"]).agg(seed_sd="std", seed_mean_min="min", seed_mean_max="max"))
    summ = summ.round(5).reset_index()
    summ.to_csv(out / "noise_summary.csv", index=False)
    print(summ.to_string(index=False), "\n")

    # seed_sd turns out tiny for these two scorers: neither has training randomness, and the 5-fold mean covers
    # every row whatever the split. So two more measures:
    # (a) sampling noise: bootstrap standard error of the out-of-fold AUC (seed 0), i.e. how much the AUC would
    #     move on another sample of applicants of this size;
    # (b) paired noise: per-fold difference logreg - rule on the same folds, the way models will be compared.
    oof = oof_all.s0.values
    boot = bootstrap_aucs(y, [oof, rule_score])
    piv = res.pivot_table(index=["scope", "seed", "fold"], columns="model", values="auc")
    diff = (piv.logreg - piv.rule).groupby("scope")
    extra = pd.DataFrame([
        {"measure": "out-of-fold AUC, logreg, seed 0", "value": auc(y, oof)},
        {"measure": "bootstrap SE of OOF AUC, logreg (200 resamples)", "value": boot[:, 0].std()},
        {"measure": "bootstrap SE of AUC, rule (200 resamples)", "value": boot[:, 1].std()},
        {"measure": "bootstrap SE of the difference logreg - rule", "value": (boot[:, 0] - boot[:, 1]).std()},
        *[{"measure": f"paired fold diff logreg - rule ({s}): mean", "value": d.mean()} for s, d in diff],
        *[{"measure": f"paired fold diff logreg - rule ({s}): sd over 25 folds", "value": d.std()} for s, d in diff],
        *[{"measure": f"paired fold diff logreg - rule ({s}): folds won of 25", "value": float((d > 0).sum())}
          for s, d in diff],
    ])
    extra["value"] = extra.value.round(5)
    extra.to_csv(out / "noise_extra.csv", index=False)
    print(extra.to_string(index=False))


if __name__ == "__main__":
    main()
