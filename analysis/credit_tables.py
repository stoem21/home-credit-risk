"""Credit-risk tables a lending team expects (added after an external review): score bands with KS, a swap-set
between the chosen model and the scorecard, and population stability (PSI) between train and the Kaggle test set.

All from saved predictions; no model is trained here.

  1. Score bands: the chosen model's out-of-fold risk (raw, mean of the 5 seeds) cut into 10 equal-size bands, from
     safest to riskiest. Per band: applicants, payment-difficulty rate, and if one approved every band up to this
     one: cumulative approval and cumulative difficulty rate. KS (Kolmogorov-Smirnov) = the largest gap between the
     cumulative shares of good payers and of defaulters; banks report it next to AUC / Gini. Same for the scorecard.
  2. Swap-set at the proposed cut-off (break-even rule, approve if calibrated risk < 25%; seed 0): applicants
     approved by both models, by only one, or by neither, and their difficulty rates. It shows what the better
     model changes in practice.
  3. PSI (population stability index) between training applicants and the Kaggle test applicants, for the
     chosen model's score (train: out-of-fold seed 0; test: the final model's predictions) and its 10 most used
     inputs. Bins: training deciles (missing values as their own bin; text columns by category). Usual reading:
     below 0.1 stable, 0.1 to 0.25 a moderate shift, above 0.25 a large shift (a rule of thumb).

Usage:
    .venv/bin/python -m analysis.credit_tables [--out-root DIR]
    v3: --main lgbm_v3b_sh90_lr02_x3 --features v3b --tag v3
Outputs: analysis_output/credit_<tag>/score_bands.csv, ks.csv, swap_set.csv, psi.csv (default tag v2)
"""

import argparse

import numpy as np
import pandas as pd

from analysis.calibration import recalibrate
from homecredit.data import N_SEEDS, is_text, load_features, load_folds
from homecredit.evaluate import load_oof
from homecredit.paths import Paths, add_out_root

DEFAULT_MAIN, SCORECARD = "lgbm_v2_combo_lr02_x3", "scorecard_full_no_gender"
P_STAR = 0.20 / (0.20 + 0.60)  # break-even risk of the business base case (analysis/business.py)


def ks(y, p) -> float:
    order = np.argsort(p, kind="stable")
    yy = y[order]
    cum_bad = np.cumsum(yy) / yy.sum()
    cum_good = np.cumsum(1 - yy) / (1 - yy).sum()
    return float(np.max(np.abs(cum_good - cum_bad)))


def psi(expected: pd.Series, actual: pd.Series) -> float:
    if is_text(expected):
        e = expected.astype(object).fillna("(missing)").value_counts(normalize=True)
        a = actual.astype(object).fillna("(missing)").value_counts(normalize=True)
    else:
        x, z = expected.astype(float), actual.astype(float)
        edges = np.unique(np.nanquantile(x, np.linspace(0, 1, 11)[1:-1]))
        lab = lambda v: pd.Series(np.where(np.isnan(v), -1, np.searchsorted(edges, v, side="right")))
        e = lab(x.values).value_counts(normalize=True)
        a = lab(z.values).value_counts(normalize=True)
    e, a = e.align(a, fill_value=0)
    e, a = e.clip(lower=1e-4), a.clip(lower=1e-4)
    return float(((a - e) * np.log(a / e)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default=DEFAULT_MAIN, help="the chosen model")
    ap.add_argument("--features", default="v2", help="feature table of the chosen model (for PSI of its inputs)")
    ap.add_argument("--tag", default="v2", help="output folder credit_<tag>")
    add_out_root(ap)
    args = ap.parse_args()
    chosen = args.main
    P = Paths(args.out_root)
    out = P.out(f"credit_{args.tag}")
    pd.set_option("display.width", 220)

    tr, te, _ = load_features(P, args.features, categories=True)
    y = tr.TARGET.values.astype(int)
    folds = load_folds(P, tr.SK_ID_CURR)

    # ---- 1. score bands and KS ----
    bands, ks_rows = [], []
    for name in [chosen, SCORECARD]:
        oof = load_oof(P, name, tr.SK_ID_CURR)
        p = np.column_stack([oof[f"s{s}"].values for s in range(N_SEEDS)]).mean(axis=1)
        band = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False).values + 1
        cum_n = cum_bad = 0
        for b in range(1, 11):
            m = band == b
            cum_n += m.sum()
            cum_bad += y[m].sum()
            bands.append({"model": name, "band": b, "risk_from_pct": round(100 * p[m].min(), 2),
                          "risk_to_pct": round(100 * p[m].max(), 2), "applicants": int(m.sum()),
                          "difficulty_rate_pct": round(100 * y[m].mean(), 2),
                          "cum_approval_pct": round(100 * cum_n / len(y), 1),
                          "cum_difficulty_rate_pct": round(100 * cum_bad / cum_n, 2)})
        k = [ks(y, oof[f"s{s}"].values) for s in range(N_SEEDS)]
        ks_rows.append({"model": name, "ks_mean_of_seeds": round(np.mean(k), 4), "ks_sd": round(np.std(k, ddof=1), 4),
                        "ks_of_seed_mean_score": round(ks(y, p), 4)})
    bands, ks_df = pd.DataFrame(bands), pd.DataFrame(ks_rows)
    bands.to_csv(out / "score_bands.csv", index=False)
    ks_df.to_csv(out / "ks.csv", index=False)
    print(bands[bands.model == chosen].to_string(index=False), "\n")
    print(ks_df.to_string(index=False), "\n")

    # ---- 2. swap-set at the proposed cut-off (seed 0) ----
    f0 = folds.fold_s0.values
    appr = {}
    for name in [chosen, SCORECARD]:
        raw = load_oof(P, name, tr.SK_ID_CURR).s0.values
        appr[name] = recalibrate(raw, y, f0, "isotonic") < P_STAR
    a, b = appr[chosen], appr[SCORECARD]
    rows = []
    for label, m in [("approved by both", a & b), ("approved by LightGBM only", a & ~b),
                     ("approved by the scorecard only", ~a & b), ("declined by both", ~a & ~b)]:
        rows.append({"group": label, "applicants": int(m.sum()), "pct_of_applicants": round(100 * m.mean(), 2),
                     "difficulty_rate_pct": round(100 * y[m].mean(), 2)})
    swap = pd.DataFrame(rows)
    swap.to_csv(out / "swap_set.csv", index=False)
    print(swap.to_string(index=False), "\n")

    # ---- 3. PSI train vs Kaggle test ----
    score_tr = load_oof(P, chosen, tr.SK_ID_CURR).s0
    score_te = pd.read_parquet(P.test_preds(chosen)).TARGET
    rows = [{"what": "model score", "psi": round(psi(score_tr, score_te), 4)}]
    imp = pd.read_csv(P.results / "lgbm" / f"{chosen}_importance.csv").variable.head(10)
    for c in imp:
        rows.append({"what": c, "psi": round(psi(tr[c], te[c]), 4)})
    ps = pd.DataFrame(rows)
    ps.to_csv(out / "psi.csv", index=False)
    print(ps.to_string(index=False))


if __name__ == "__main__":
    main()
