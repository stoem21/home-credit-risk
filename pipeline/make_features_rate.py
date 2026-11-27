"""Feature table v3c: v3b plus an estimate of the term and interest rate of the loan applied for (group rate_).

Idea (from public write-ups of top solutions, and our error analysis: long-term loans are where the model is
weakest, AUC 0.758 in the longest fifth): the application gives the credit amount and the monthly instalment but
not the number of instalments, so the interest rate is hidden. Two unknowns (term, rate) fit one annuity equation,
so the term is predicted first, from earlier Home Credit cash loans whose term is known (CNT_PAYMENT), and the
rate then follows from the annuity formula.

Term model: LightGBM regression of CNT_PAYMENT on credit, instalment, goods price and their ratios, fitted on the
earlier cash-loan applications in previous_application (all applicants, train and test; no TARGET is used, so
nothing about the outcome can leak). Checked on 20% of applicants held out (grouped by applicant).
Revolving loans (credit cards) have no term: their rate_ columns are missing.

Columns (prefix rate_):
  rate_term_pred          predicted number of monthly instalments
  rate_yearly_pred        yearly interest rate implied by the predicted term
  rate_total_cost         total instalments / credit - 1 (share of the credit paid as interest and fees)
  rate_vs_prev            rate_yearly_pred minus the applicant's mean rate on earlier approved loans
                          (term_prev_rate_mean, v2); missing without earlier approved loans
  rate_term_vs_prev       predicted term / mean term of the applicant's earlier approved cash loans

Usage:
    .venv/bin/python -m pipeline.make_features_rate [--out-root DIR]
Output: processed/features_v3c.parquet (every v3b column unchanged, plus rate_),
        analysis_output/features/rate_term_model.csv (hold-out check), rate_drift.csv (train vs test)
"""

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from homecredit.data import read_raw
from homecredit.paths import Paths, add_out_root
from pipeline.make_features_v2 import yearly_rate

X_COLS = ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "annuity_to_credit", "goods_to_credit", "credit_log"]
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_child_samples=50, verbose=-1, seed=0)


def inputs(df: pd.DataFrame) -> pd.DataFrame:
    x = df[["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]].astype(float).copy()
    x["annuity_to_credit"] = x.AMT_ANNUITY / x.AMT_CREDIT
    x["goods_to_credit"] = x.AMT_GOODS_PRICE / x.AMT_CREDIT
    x["credit_log"] = np.log(x.AMT_CREDIT)
    return x[X_COLS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    out = P.out("features")

    p = read_raw(P, "previous_application", ["SK_ID_CURR", "NAME_CONTRACT_TYPE", "NAME_CONTRACT_STATUS",
                                             "CNT_PAYMENT", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"])
    cash = p[p.NAME_CONTRACT_TYPE.eq("Cash loans") & (p.CNT_PAYMENT > 0) & (p.AMT_CREDIT > 0)
             & (p.AMT_ANNUITY > 0)].reset_index(drop=True)
    X, y = inputs(cash), cash.CNT_PAYMENT.values.astype(float)

    # ---- hold-out check of the term model (20% of applicants) ----
    tr_i, ho_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, y, cash.SK_ID_CURR))
    dtr = lgb.Dataset(X.iloc[tr_i], y[tr_i])
    dho = lgb.Dataset(X.iloc[ho_i], y[ho_i], reference=dtr)
    bst = lgb.train({**PARAMS, "num_threads": args.threads}, dtr, 3000, valid_sets=[dho],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    n_trees = bst.best_iteration
    pred = bst.predict(X.iloc[ho_i], num_iteration=n_trees)
    err = np.abs(pred - y[ho_i])
    check = pd.DataFrame([{"rows_fit": len(tr_i), "rows_holdout": len(ho_i), "trees": n_trees,
                           "mae_months": round(err.mean(), 2), "median_abs_error_months": round(np.median(err), 2),
                           "share_within_3_months_pct": round(100 * (err <= 3).mean(), 1),
                           "r2": round(1 - ((pred - y[ho_i]) ** 2).sum() / ((y[ho_i] - y[ho_i].mean()) ** 2).sum(), 4),
                           "mae_if_always_median": round(np.abs(np.median(y[tr_i]) - y[ho_i]).mean(), 2)}])
    check.to_csv(out / "rate_term_model.csv", index=False)
    print(check.to_string(index=False), flush=True)

    # ---- final term model on all earlier cash loans, applied to the current applications ----
    bst = lgb.train({**PARAMS, "num_threads": args.threads}, lgb.Dataset(X, y), n_trees)
    f = pd.read_parquet(P.features("v3b"))
    app = pd.concat([read_raw(P, t, ["SK_ID_CURR", "NAME_CONTRACT_TYPE", "AMT_CREDIT", "AMT_ANNUITY",
                                     "AMT_GOODS_PRICE"]) for t in ["application_train", "application_test"]])
    app = app.set_index("SK_ID_CURR").reindex(f.SK_ID_CURR)
    is_cash = app.NAME_CONTRACT_TYPE.eq("Cash loans").values
    term = np.where(is_cash, bst.predict(inputs(app.reset_index())), np.nan)
    term = np.clip(term, 3, 84)
    credit, annuity = app.AMT_CREDIT.values.astype(float), app.AMT_ANNUITY.values.astype(float)
    rate = yearly_rate(credit, annuity, term)

    appr = p[p.NAME_CONTRACT_STATUS.eq("Approved") & p.NAME_CONTRACT_TYPE.eq("Cash loans") & (p.CNT_PAYMENT > 0)]
    prev_term = appr.groupby("SK_ID_CURR").CNT_PAYMENT.mean().reindex(f.SK_ID_CURR).values
    g = pd.DataFrame({
        "rate_term_pred": term,
        "rate_yearly_pred": rate,
        "rate_total_cost": annuity * term / credit - 1,
        "rate_vs_prev": rate - f.term_prev_rate_mean.values,
        "rate_term_vs_prev": term / prev_term,
    }).replace([np.inf, -np.inf], np.nan).astype("float32")
    f = pd.concat([f, g], axis=1)
    f.to_parquet(P.features("v3c"), index=False)

    # ---- train vs test: same distribution? (quantiles and missing share) ----
    rows = []
    for c in g.columns:
        a, b = g[c][f.split.eq("train").values], g[c][f.split.eq("test").values]
        rows.append({"column": c, "missing_train_pct": round(100 * a.isna().mean(), 1),
                     "missing_test_pct": round(100 * b.isna().mean(), 1),
                     **{f"train_q{q}": round(float(a.quantile(q / 100)), 3) for q in (10, 50, 90)},
                     **{f"test_q{q}": round(float(b.quantile(q / 100)), 3) for q in (10, 50, 90)}})
    drift = pd.DataFrame(rows)
    drift.to_csv(out / "rate_drift.csv", index=False)
    print(drift.to_string(index=False))
    print(f"features_v3c: {f.shape[1]} columns ({g.shape[1]} new)")


if __name__ == "__main__":
    main()
