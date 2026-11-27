"""Is the gain from the rate_ columns smaller on test-like applicants? (EXPERIMENTS.md entries 33 and 34)

On Kaggle the v3c model (with the estimated term and interest rate) gained less over v3b tuned than in validation.
One guess: test loans are shorter and dearer, so the rate_ columns help less there. This checks the guess inside
our own validation, using only saved out-of-fold predictions (no model is trained):
  - short cash loans: predicted term (rate_term_pred) at most 24 months, the kind of loan the test set has more of;
  - the 20% most test-like training applicants by the v1 adversarial model (processed/adversarial_v1.parquet);
  - everyone else (neither of the two).
For each part: AUC of each model per seed (out-of-fold), gain = B - A per seed, mean and sd over the 5 seeds.

Usage:
    .venv/bin/python -m experiments.test_like_gain [--a lgbm_v3b_sh90_lr02_x3_tuned]
        [--b lgbm_v3c_sh90_lr02_x3_tuned] [--out-root DIR]
Output: analysis_output/comparisons/test_like_gain_<a>_vs_<b>.csv
"""

import argparse

import numpy as np
import pandas as pd

from homecredit.data import load_features
from homecredit.evaluate import auc, load_oof
from homecredit.paths import Paths, add_out_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="lgbm_v3b_sh90_lr02_x3_tuned")
    ap.add_argument("--b", default="lgbm_v3c_sh90_lr02_x3_tuned")
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    tr, _, _ = load_features(P, "v3c")
    y = tr.TARGET.values.astype(int)
    a, b = load_oof(P, args.a, tr.SK_ID_CURR), load_oof(P, args.b, tr.SK_ID_CURR)
    seeds = [c for c in a.columns if c.startswith("s") and c in b.columns]

    adv = pd.read_parquet(P.processed / "adversarial_v1.parquet")
    adv = adv[adv.split == "train"].set_index("SK_ID_CURR").p_test.reindex(tr.SK_ID_CURR).values
    short = (tr.rate_term_pred <= 24).values  # missing (revolving loans) counts as not short
    testlike = adv >= np.quantile(adv, 0.8)
    parts = {"all loans": np.ones(len(tr), bool),
             "short cash loans (predicted term <= 24 months)": short,
             "20% most test-like (adversarial v1)": testlike,
             "rest (neither)": ~short & ~testlike}
    rows = []
    for name, m in parts.items():
        auc_a = np.array([auc(y[m], a[s].values[m]) for s in seeds])
        auc_b = np.array([auc(y[m], b[s].values[m]) for s in seeds])
        g = auc_b - auc_a
        rows.append({"part": name, "applicants": int(m.sum()), "default_rate_pct": round(100 * y[m].mean(), 2),
                     "auc_a": round(auc_a.mean(), 5), "auc_b": round(auc_b.mean(), 5),
                     "gain_mean": round(g.mean(), 5), "gain_sd_over_seeds": round(g.std(ddof=1), 5),
                     "seeds": len(seeds)})
    res = pd.DataFrame(rows)
    out = P.out("comparisons") / f"test_like_gain_{args.a}_vs_{args.b}.csv"
    res.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
