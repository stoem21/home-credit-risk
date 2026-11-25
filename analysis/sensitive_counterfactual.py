"""Direct effect of age and gender on decisions at the proposed cut-off (added after an external review).

Attribution (reason codes) spreads an attribute's effect over correlated columns, so "age is rarely a top reason"
does not say how much age changes decisions. Here the input itself is changed and the applicant re-scored:
  - age: the chosen v2 columns keep age (DAYS_BIRTH, age_years). Every validation applicant is re-scored as if aged
    45 (both age columns set to 45 years), everything else unchanged;
  - gender: the v2 model WITH gender and family (lgbm_v2_combo_with_gender's setting) re-scores each applicant with
    only the gender switched (F <-> M).
Model setting: the v2 columns with learning rate 0.05 and one fit (as lgbm_v2_combo / lgbm_v2_combo_with_gender),
refitted on the seed-0 training folds; checked against their saved out-of-fold predictions. The chosen model uses
the slower learning rate and an average of 3 fits; its behaviour should be close, but this is not measured.
Decision rule: decline the riskiest 6% (threshold = 94th percentile of each training part's predictions), about the
share the business step's break-even rule declines.
Note the direct effect is only part of the story: other columns carry age and gender (analysis/proxy_check.py).

Usage:
    .venv/bin/python -m analysis.sensitive_counterfactual [--jobs 5] [--threads 2] [--out-root DIR]
Output: analysis_output/fairness_v2/counterfactual_age_gender.csv
"""

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from homecredit.data import drop_columns, load_features, load_folds, select_groups
from homecredit.evaluate import load_oof
from homecredit.models.lgbm import fit
from homecredit.paths import Paths, add_out_root

DECLINE = 0.06
AGE = 45.0
SWAP = {"F": "M", "M": "F", "XNA": "XNA"}
SETTINGS = {"age (model without gender/family)": ("gender,family", "lgbm_v2_combo"),
            "gender (model with gender/family)": (None, "lgbm_v2_combo_with_gender")}


def columns(P, drop):
    tr, _, cols = load_features(P, "v2", categories=True)
    return tr, [c for c in drop_columns(select_groups(cols, "win,stk,term"), drop) if not c.startswith("stk_bb_")]


def run_fold(out_root, what, k, threads):
    drop, _ = SETTINGS[what]
    tr, cols = columns(Paths(out_root), drop)
    fo = load_folds(Paths(out_root), tr.SK_ID_CURR).fold_s0.values
    trn, va = np.where(fo != k)[0], np.where(fo == k)[0]
    y = tr.TARGET.values.astype(int)
    bst = fit(tr[cols].iloc[trn], y[trn], 0, threads)
    it = bst.best_iteration
    thr = np.quantile(bst.predict(tr[cols].iloc[trn], num_iteration=it), 1 - DECLINE)
    X = tr[cols].iloc[va]
    Xc = X.copy()
    if what.startswith("age"):
        Xc["age_years"] = AGE
        Xc["DAYS_BIRTH"] = -AGE * 365.25
    else:
        Xc["CODE_GENDER"] = X.CODE_GENDER.astype(object).map(SWAP).astype(X.CODE_GENDER.dtype)
    return what, va, bst.predict(X, num_iteration=it), bst.predict(Xc, num_iteration=it), thr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    out = P.out("fairness_v2")
    tr, _ = columns(P, None)
    age = -tr.DAYS_BIRTH / 365.25
    groups = {"age band": pd.cut(age, [0, 25, 35, 45, 55, 65, 100], right=False,
                                 labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"]).astype(str).values,
              "gender": tr.CODE_GENDER.astype(str).values}

    res = Parallel(n_jobs=args.jobs)(delayed(run_fold)(args.out_root, w, k, args.threads)
                                     for w in SETTINGS for k in range(5))
    rows = []
    for what, (_, saved) in SETTINGS.items():
        n = len(tr)
        p, pc, dec, decc = np.zeros(n), np.zeros(n), np.zeros(n, bool), np.zeros(n, bool)
        for w, va, a, b, thr in res:
            if w == what:
                p[va], pc[va], dec[va], decc[va] = a, b, a >= thr, b >= thr
        check = float(np.abs(p - load_oof(P, saved, tr.SK_ID_CURR).s0.values).max())
        attr = "age band" if what.startswith("age") else "gender"
        lab = groups[attr]
        for g in ["all"] + sorted(set(lab)):
            m = np.ones(n, bool) if g == "all" else lab == g
            if m.sum() < 1000:
                continue
            rows.append({"change": what, "group": g, "applicants": int(m.sum()),
                         "declined_pct": round(100 * dec[m].mean(), 2),
                         "declined_after_change_pct": round(100 * decc[m].mean(), 2),
                         "decisions_changed_pct": round(100 * (dec != decc)[m].mean(), 2),
                         "mean_risk_change_pts": round(100 * (pc - p)[m].mean(), 3),
                         "max_abs_diff_vs_saved_oof": check})
    res_df = pd.DataFrame(rows)
    res_df.to_csv(out / "counterfactual_age_gender.csv", index=False)
    pd.set_option("display.width", 220)
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
