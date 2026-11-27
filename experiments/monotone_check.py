"""Does a model's risk move the expected way when one input changes? (EXPERIMENTS.md entry 34)

For each column with a business direction (pipeline/monotone_v3.json: +1 = higher value should mean higher risk,
-1 = lower risk), every applicant in a random sample of 20,000 training applicants is re-scored with that column
moved one small step the "worse" way (a tenth of the gap between the column's 10th and 90th percentile; 1 for
0/1 flags), everything else unchanged. A violation is an applicant whose predicted risk goes DOWN after the change
to a worse value (e.g. a lower external score giving a lower risk). A model trained with monotone constraints on
these columns should have none; an unconstrained model shows how often a reason code or a "what if" answer would
point the wrong way. Uses the final models (all of train), as in analysis/global_shap.py.

Usage:
    .venv/bin/python -m experiments.monotone_check --models A,B --features v3c --tag v3c [--out-root DIR]
Output: analysis_output/monotone_<tag>/violations.csv
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from homecredit.data import load_features
from homecredit.paths import Paths, add_out_root

ROWS = 20000


def boosters(P: Paths, name: str) -> list[lgb.Booster]:
    files = [P.model_file(name)]
    i = 1
    while P.model_file(f"{name}_{i}").exists():
        files.append(P.model_file(f"{name}_{i}"))
        i += 1
    return [lgb.Booster(model_file=str(f)) for f in files]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--monotone", default="pipeline/monotone_v3.json")
    ap.add_argument("--tag", required=True)
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    out = P.out(f"monotone_{args.tag}")
    direction = json.loads(Path(args.monotone).read_text())["columns"]

    tr, _, _ = load_features(P, args.features, categories=True)
    names = {c.replace(" ", "_"): c for c in tr.columns}
    sample = tr.iloc[np.random.default_rng(0).choice(len(tr), ROWS, replace=False)]
    rows = []
    for model in args.models.split(","):
        bs = boosters(P, model)
        cols = [names.get(c, c) for c in bs[0].feature_name()]
        X = sample[cols]

        def predict(df):
            return np.mean([b.predict(df) for b in bs], axis=0)

        base = predict(X)
        for c, d in direction.items():
            if c not in cols:
                continue
            x = X[c].astype(float)
            ok = x.notna().values
            vals = x[ok]
            step = 1.0 if set(vals.unique()) <= {0.0, 1.0} else (vals.quantile(0.9) - vals.quantile(0.1)) / 10
            if not step > 0:
                continue
            Xc = X.copy()
            Xc[c] = x + d * step  # one step the worse way
            moved = predict(Xc) - base
            wrong = (moved < -1e-9) & ok
            rows.append({"model": model, "column": c, "direction": d, "step": round(float(step), 4),
                         "applicants_with_value": int(ok.sum()),
                         "pct_risk_down_after_worse_value": round(100 * wrong.sum() / ok.sum(), 2),
                         "pct_risk_up": round(100 * ((moved > 1e-9) & ok).sum() / ok.sum(), 2),
                         "mean_change_pts": round(100 * moved[ok].mean(), 4)})
        print(model, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "violations.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))
    print(res.groupby("model").pct_risk_down_after_worse_value.agg(["mean", "max"]).round(2).to_string())


if __name__ == "__main__":
    main()
