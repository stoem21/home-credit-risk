"""What the model relies on overall (global SHAP), to describe its behaviour in the README.

Reason codes explain one applicant at a time. This script summarises the final model (trained on all of train, the
model used for the Kaggle predictions) over a random sample of 20,000 training applicants:
  1. share of the total mean |SHAP| per reason group (the same groups as the reason codes);
  2. the 15 columns with the largest mean |SHAP|, with the direction of their effect: the Spearman correlation
     between the column's value and its SHAP value (positive = higher value, higher predicted risk);
  3. for the 6 largest numeric columns, the mean SHAP value (in log-odds) per decile of the column, for the
     dependence chart in the notebook.
SHAP describes what the model uses, not what causes payment difficulties, and a large share is not proof of an
effect on decisions (the counterfactual test in analysis/reason_codes.py covers that).

The final model files are processed/<name>_final.txt (+ _1, _2 ... for an average of several fits); their own
feature names say which columns they use.

Usage:
    .venv/bin/python -m analysis.global_shap --model lgbm_v2_combo_lr02_x3 --features v2 --tag v2 [--out-root DIR]
Outputs: analysis_output/global_shap_<tag>/groups.csv, top_columns.csv, dependence.csv
"""

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.reason_codes import group_of
from homecredit.data import is_text, load_features
from homecredit.paths import Paths, add_out_root

ROWS = 20000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--tag", required=True)
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    out = P.out(f"global_shap_{args.tag}")

    files = [P.model_file(args.model)]
    i = 1
    while P.model_file(f"{args.model}_{i}").exists():
        files.append(P.model_file(f"{args.model}_{i}"))
        i += 1
    boosters = [lgb.Booster(model_file=str(f)) for f in files]
    cols = boosters[0].feature_name()
    assert all(b.feature_name() == cols for b in boosters)

    tr, _, _ = load_features(P, args.features, categories=True)
    # LightGBM saves feature names with spaces replaced by "_" (e.g. one-hot columns of text categories)
    names = {c.replace(" ", "_"): c for c in tr.columns}
    cols = [names.get(c, c) for c in cols]
    X = tr[cols].iloc[np.random.default_rng(0).choice(len(tr), ROWS, replace=False)]
    contrib = np.mean([b.predict(X, pred_contrib=True)[:, :-1] for b in boosters], axis=0)
    mean_abs = np.abs(contrib).mean(axis=0)
    total = mean_abs.sum()

    g = pd.DataFrame({"column": cols, "group": [group_of(c) for c in cols], "mean_abs_shap": mean_abs})
    groups = g.groupby("group").agg(columns=("column", "size"), mean_abs_shap=("mean_abs_shap", "sum"))
    groups["share_pct"] = (100 * groups.mean_abs_shap / total).round(1)
    groups = groups.sort_values("share_pct", ascending=False).reset_index()
    groups.to_csv(out / "groups.csv", index=False)

    top = g.sort_values("mean_abs_shap", ascending=False).head(15).copy()
    top["share_pct"] = (100 * top.mean_abs_shap / total).round(2)
    dirs = []
    for c in top.column:
        j = cols.index(c)
        if is_text(X[c]):
            dirs.append(np.nan)
            continue
        x = X[c].astype(float).values
        ok = ~np.isnan(x)
        dirs.append(round(spearmanr(x[ok], contrib[ok, j]).statistic, 2) if ok.sum() > 100 else np.nan)
    top["direction_spearman"] = dirs
    top.to_csv(out / "top_columns.csv", index=False)

    dep = []
    num_top = [c for c in top.column if not is_text(X[c])][:6]
    for c in num_top:
        j = cols.index(c)
        x = X[c].astype(float)
        ok = x.notna().values
        dec = pd.qcut(x[ok].rank(method="first"), 10, labels=False)
        d = pd.DataFrame({"x": x[ok].values, "shap": contrib[ok, j], "decile": dec.values}).groupby("decile").agg(
            x_mid=("x", "median"), mean_shap=("shap", "mean"), rows=("shap", "size")).reset_index()
        dep.append(d.assign(column=c))
        if (~ok).any():
            dep.append(pd.DataFrame({"decile": ["missing"], "x_mid": [np.nan], "mean_shap": [contrib[~ok, j].mean()],
                                     "rows": [int((~ok).sum())], "column": [c]}))
    pd.concat(dep, ignore_index=True).to_csv(out / "dependence.csv", index=False)

    pd.set_option("display.width", 200)
    print(groups.to_string(index=False), "\n")
    print(top[["column", "group", "share_pct", "direction_spearman"]].to_string(index=False))


if __name__ == "__main__":
    main()
