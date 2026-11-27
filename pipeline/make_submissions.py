"""Kaggle submission files for every model, for the independent late-submission check.

The models were chosen on our own validation; Kaggle scores are reported as an outside check only and are not used
to pick or tune anything (project rule: no score-gaming).

  - Models with saved test predictions (processed/preds/<model>_test.parquet, from the final model trained on all
    of train) are written as they are.
  - The two baselines had no test predictions yet, so they are made here, with the same definitions as in
    experiments/noise_check.py:
      rule: 1 - mean of the available EXT_SOURCE_1/2/3 (no training; a missing mean takes the train average).
            Only the order matters for AUC; 1 - mean keeps the values between 0 and 1.
      logreg_baseline: the same 15 features and pipeline, fitted on all of train (missing values: train medians).

Every file is checked against sample_submission.csv: same IDs in the same order, 48,744 rows, header
SK_ID_CURR,TARGET, no missing value, every value between 0 and 1. A manifest lists each file with the model's
cross-validated AUC (mean of 25 folds), for comparison once the Kaggle scores are known.

Usage:
    .venv/bin/python -m pipeline.make_submissions [--tag v2] [--out-root DIR]
Outputs: submission/<model>.csv (not committed), processed/preds/rule_test.parquet,
         processed/preds/logreg_baseline_test.parquet, analysis_output/submissions/manifest.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.noise_check import EXT, build_features
from homecredit.data import load_folds, read_raw, scopes
from homecredit.evaluate import fold_aucs, load_oof
from homecredit.paths import Paths, add_out_root


def baselines(P: Paths) -> None:
    tr, te = read_raw(P, "application_train"), read_raw(P, "application_test")
    fill = tr[EXT].mean(axis=1).mean()
    P.preds_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"SK_ID_CURR": te.SK_ID_CURR, "TARGET": 1 - te[EXT].mean(axis=1).fillna(fill)}).to_parquet(
        P.test_preds("rule"), index=False)
    X, Xte = build_features(tr), build_features(te)
    med = X.median()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X.fillna(med), tr.TARGET.values)
    pd.DataFrame({"SK_ID_CURR": te.SK_ID_CURR, "TARGET": model.predict_proba(Xte.fillna(med))[:, 1]}).to_parquet(
        P.test_preds("logreg_baseline"), index=False)


def cv_auc(P: Paths, name: str, app: pd.DataFrame, folds: pd.DataFrame) -> dict:
    y = app.TARGET.values
    if name == "rule":
        score = 1 - app[EXT].mean(axis=1).fillna(app[EXT].mean(axis=1).mean()).values
        oof = pd.DataFrame({f"s{s}": score for s in range(5)})
    else:
        oof = load_oof(P, name, app.SK_ID_CURR)
    seeds = [s for s in range(5) if f"s{s}" in oof.columns]  # exploratory runs have one seed
    fa = fold_aucs(y, oof, folds, scopes(app.NAME_CONTRACT_TYPE), seeds)
    return {f"cv_auc_{k}": round(v, 5) for k, v in fa.groupby("scope").auc.mean().items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="manifest_<tag>.csv, so an earlier manifest stays as it was")
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    baselines(P)

    sample = read_raw(P, "sample_submission")
    app = read_raw(P, "application_train")
    folds = load_folds(P, app.SK_ID_CURR)
    sub_dir = P.root / "submission"
    sub_dir.mkdir(exist_ok=True)
    rows = []
    for f in sorted(P.preds_dir.glob("*_test.parquet")):
        name = f.name.removesuffix("_test.parquet")
        pr = pd.read_parquet(f).set_index("SK_ID_CURR").TARGET
        sub = pd.DataFrame({"SK_ID_CURR": sample.SK_ID_CURR.values, "TARGET": pr.reindex(sample.SK_ID_CURR).values})
        assert len(sub) == 48744 and len(pr) == len(sub) and pr.index.is_unique, name
        assert sub.TARGET.notna().all() and sub.TARGET.between(0, 1).all(), name
        path = sub_dir / f"{name}.csv"
        sub.to_csv(path, index=False)
        back = pd.read_csv(path)
        assert list(back.columns) == ["SK_ID_CURR", "TARGET"] and (back.SK_ID_CURR.values == sample.SK_ID_CURR.values).all()
        rows.append({"model": name, "file": f"submission/{name}.csv", "rows": len(sub),
                     "pred_min": round(sub.TARGET.min(), 5), "pred_mean": round(sub.TARGET.mean(), 5),
                     "pred_max": round(sub.TARGET.max(), 5), **cv_auc(P, name, app, folds)})
    man = pd.DataFrame(rows).sort_values("cv_auc_all").reset_index(drop=True)
    man.to_csv(P.out("submissions") / f"manifest{'_' + args.tag if args.tag else ''}.csv", index=False)
    pd.set_option("display.width", 250)
    print(man.to_string(index=False))


if __name__ == "__main__":
    main()
