"""Business step (project step 7): choose an approval cut-off from expected profit and loss.

Everything runs on out-of-fold predictions of the training applicants (whose outcomes are known), so every
applicant is scored by a model that never saw them. Probabilities are the cross-fitted isotonic ones (the
calibration step's decision: rank with raw scores, use calibrated probabilities for money).

Population: every training applicant was approved by Home Credit at the time (only granted loans have an
outcome). So "approve everyone" is the lender's current policy on this population, and any cut-off here can only
TIGHTEN that policy; whether to approve applicants who were declined in the past cannot be evaluated (no reject
inference).

Money model (ASSUMPTIONS, not data: the data has no prices, costs or recoveries):
  - an approved applicant who repays brings a net margin m x AMT_CREDIT (interest minus funding and running costs,
    over the life of the loan; loan term, timing and discounting are not modelled);
  - an approved applicant with payment difficulties (TARGET = 1) costs LGD x AMT_CREDIT (a stand-in for an
    unknown mix of late payers who cure and true losses);
  - a declined applicant brings nothing and costs nothing.
  Base case m = 0.20, LGD = 0.60 (guesses for a high-rate consumer lender: per train applicant, the average yearly
  rate implied by their earlier approved Home Credit loans has a median of about 54%, feature term_prev_rate_mean
  in pipeline/make_features_v2.py). A grid of m in {0.10, 0.20, 0.30} and LGD in {0.40, 0.60, 0.80} shows how much
  the answer depends on them. Amounts are in the data's currency, which is not disclosed; "per applicant" means
  per applicant in this population, over the life of the loan.
  Approving is worth it when the probability of difficulties p is below the break-even p* = m / (m + LGD) (base
  case 25%).

Version 2 (after an external review; the first version's files in analysis_output/business/ are kept):
  - cut-offs fall between distinct probability values (isotonic probabilities have many ties; the first version
    could split a block of tied values by row order);
  - 95% bootstrap intervals (applicants resampled) for the profit differences, not only the sd over seeds;
  - approval by group for all six attributes of EDA step 8, with the adverse impact ratio (group approval /
    highest group approval), approval among good payers, and predicted vs observed difficulty rate;
  - lgbm_v1_no_gender_family is added, so the scorecard (v1 columns) can be compared with LightGBM on the same
    columns: the cost of explainability alone, without the v2 features and settings.

Outputs (analysis_output/business_v2/):
  1. approval_curve.csv: per approval rate 50% ... 100%: cut-off, actual approval, difficulty rate among the
     approved, share of all defaulters declined, realised profit per applicant (base case); mean of 5 seeds.
  2. policies.csv: base case: approve everyone; break-even rule (approve if calibrated p < p*); a cut-off chosen
     by cross-fitting (best on the other four folds, applied to the fold).
  3. differences.csv: profit differences between policies / models with 95% bootstrap intervals.
  4. sensitivity.csv: the break-even rule for every (m, LGD) pair.
  5. policy_groups.csv: approval by group under the base-case break-even rule.

Usage:
    .venv/bin/python -m analysis.business [--tag v2] [--out-root DIR]
    v3: --main lgbm_v3b_sh90_lr02_x3 --also lgbm_v2_combo_lr02_x3 --tag v3 (adds the v3 vs v2 difference; the
        other models and pairs stay as in v2, with the main model in place of the v2 one)
"""

import argparse

import numpy as np
import pandas as pd

from analysis.calibration import recalibrate
from homecredit.data import N_SEEDS, load_folds, read_raw
from homecredit.evaluate import load_oof
from homecredit.paths import Paths, add_out_root

BASE_M, BASE_LGD = 0.20, 0.60
GRID_M, GRID_LGD = [0.10, 0.20, 0.30], [0.40, 0.60, 0.80]
RATES = np.round(np.arange(0.50, 1.0001, 0.05), 2)
MAIN = "lgbm_v2_combo_lr02_x3"
OTHERS = ["scorecard_full_no_gender", "lgbm_v1_no_gender_family", "lgbm_v1"]
N_BOOT = 1000
MIN_GROUP = 1000


def contribution(y, credit, m, lgd):
    """Realised profit of approving each applicant."""
    return np.where(y == 0, m * credit, -lgd * credit)


def best_cutoff(p, gain):
    """Cut-off (approve p <= cut-off) that maximises total gain, considered only at the end of each block of tied
    probabilities, so a block is never split. Approving nobody is allowed (-inf)."""
    order = np.argsort(p, kind="stable")
    ps = p[order]
    cum = np.cumsum(gain[order])
    ends = np.r_[np.where(np.diff(ps) > 0)[0], len(ps) - 1]  # last index of each distinct value
    k = int(np.argmax(cum[ends]))
    return ps[ends[k]] if cum[ends[k]] > 0 else -np.inf


def cutoff_for_rate(p, rate):
    """Distinct probability value whose 'approve p <= value' rule approves the share closest to `rate`."""
    vals, counts = np.unique(p, return_counts=True)
    share = np.cumsum(counts) / len(p)
    return vals[int(np.argmin(np.abs(share - rate)))]


def boot_ci(d, n_boot=N_BOOT, seed=0):
    """95% interval of the mean of per-applicant differences d, resampling applicants."""
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])


def groups(app: pd.DataFrame) -> dict[str, np.ndarray]:
    age = -app.DAYS_BIRTH / 365.25
    return {
        "gender": app.CODE_GENDER.astype(str).values,
        "age band": pd.cut(age, [0, 25, 35, 45, 55, 65, 100], right=False,
                           labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"]).astype(str).values,
        "family status": app.NAME_FAMILY_STATUS.astype(str).values,
        "children": app.CNT_CHILDREN.clip(upper=3).map(lambda v: "3+" if v >= 3 else str(int(v))).values,
        "education": app.NAME_EDUCATION_TYPE.astype(str).values,
        "region rating": app.REGION_RATING_CLIENT.astype(str).values,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2", help="output folder analysis_output/business_<tag>/")
    ap.add_argument("--main", default=MAIN, help="the chosen model")
    ap.add_argument("--also", default="", help="comma list of extra models compared with the main one")
    add_out_root(ap)
    args = ap.parse_args()
    main_m = args.main
    also = [m for m in args.also.split(",") if m]
    MODELS = [main_m] + also + OTHERS
    P = Paths(args.out_root)
    out = P.out(f"business_{args.tag}")
    pd.set_option("display.width", 250)

    app = read_raw(P, "application_train", ["SK_ID_CURR", "TARGET", "AMT_CREDIT", "CODE_GENDER", "DAYS_BIRTH",
                                             "NAME_FAMILY_STATUS", "CNT_CHILDREN", "NAME_EDUCATION_TYPE",
                                             "REGION_RATING_CLIENT"])
    y = app.TARGET.values.astype(int)
    credit = app.AMT_CREDIT.values.astype(float)
    n = len(y)
    folds = load_folds(P, app.SK_ID_CURR)
    base_gain = contribution(y, credit, BASE_M, BASE_LGD)
    p_star = BASE_M / (BASE_M + BASE_LGD)

    cal = {}
    for name in MODELS:
        oof = load_oof(P, name, app.SK_ID_CURR)
        cal[name] = [recalibrate(oof[f"s{s}"].values, y, folds[f"fold_s{s}"].values, "isotonic")
                     for s in range(N_SEEDS)]

    # ---- 1. approval curve ----
    curve = []
    for name in MODELS:
        for r in RATES:
            rows = []
            for p in cal[name]:
                cut = cutoff_for_rate(p, r)
                appr = p <= cut
                rows.append({"cutoff_p": cut, "actual_approval_pct": 100 * appr.mean(),
                             "bad_rate_approved_pct": 100 * y[appr].mean(),
                             "pct_defaulters_declined": 100 * (~appr & (y == 1)).sum() / y.sum(),
                             "profit_per_applicant": base_gain[appr].sum() / n})
            curve.append({"model": name, "approval_pct": round(100 * r), **pd.DataFrame(rows).mean().round(4)})
    curve = pd.DataFrame(curve)
    curve.to_csv(out / "approval_curve.csv", index=False)

    # ---- 2. policies under the base case ----
    decisions = {("(no model)", "approve everyone"): [np.ones(n, bool)] * N_SEEDS}
    for name in MODELS:
        decisions[(name, "break-even rule")] = [p < p_star for p in cal[name]]
        cv = []
        for s, p in enumerate(cal[name]):
            f = folds[f"fold_s{s}"].values
            appr = np.zeros(n, bool)
            for k in np.unique(f):
                appr[f == k] = p[f == k] <= best_cutoff(p[f != k], base_gain[f != k])
            cv.append(appr)
        decisions[(name, "cross-fitted best cut-off")] = cv
    pol = []
    for (name, policy), ds in decisions.items():
        m = pd.DataFrame([{"approval_pct": 100 * a.mean(), "bad_rate_approved_pct": 100 * y[a].mean(),
                           "profit_per_applicant": base_gain[a].sum() / n} for a in ds])
        pol.append({"model": name, "policy": policy, **m.mean().round(3),
                    "profit_sd_seeds": round(m.profit_per_applicant.std(ddof=1), 2) if len(set(map(id, ds))) > 1
                    else 0.0})
    pol = pd.DataFrame(pol)
    pol["profit_vs_approve_all"] = (pol.profit_per_applicant - pol.profit_per_applicant.iloc[0]).round(1)
    pol.to_csv(out / "policies.csv", index=False)
    print(f"base case: margin {BASE_M}, LGD {BASE_LGD}, break-even probability {p_star:.3f}")
    print(pol.to_string(index=False), "\n")

    # ---- 3. differences with bootstrap intervals (per-applicant gain, averaged over the 5 seeds) ----
    def per_applicant(key):
        return np.mean([base_gain * a for a in decisions[key]], axis=0)

    pairs = [((main_m, "break-even rule"), ("(no model)", "approve everyone")),
             ((main_m, "break-even rule"), ("scorecard_full_no_gender", "break-even rule")),
             (("lgbm_v1_no_gender_family", "break-even rule"), ("scorecard_full_no_gender", "break-even rule")),
             ((main_m, "break-even rule"), ("lgbm_v1_no_gender_family", "break-even rule")),
             ((main_m, "break-even rule"), ("lgbm_v1", "break-even rule")),
             ((main_m, "cross-fitted best cut-off"), (main_m, "break-even rule"))]
    pairs += [((main_m, "break-even rule"), (m, "break-even rule")) for m in also]
    diff = []
    for a, b in pairs:
        d = per_applicant(a) - per_applicant(b)
        lo, hi = boot_ci(d)
        diff.append({"policy_a": f"{a[0]} / {a[1]}", "policy_b": f"{b[0]} / {b[1]}",
                     "profit_diff_per_applicant": round(d.mean(), 1), "ci95_low": round(lo, 1),
                     "ci95_high": round(hi, 1)})
    diff = pd.DataFrame(diff)
    diff.to_csv(out / "differences.csv", index=False)
    print(diff.to_string(index=False), "\n")

    # ---- 4. sensitivity of the break-even rule ----
    sens = []
    for m in GRID_M:
        for lgd in GRID_LGD:
            gain = contribution(y, credit, m, lgd)
            ps = m / (m + lgd)
            for name in MODELS:
                rows = [{"approval_pct": 100 * (p < ps).mean(), "bad_rate_approved_pct": 100 * y[p < ps].mean(),
                         "profit_per_applicant": gain[p < ps].sum() / n} for p in cal[name]]
                sens.append({"margin": m, "lgd": lgd, "break_even_p": round(ps, 3), "model": name,
                             **pd.DataFrame(rows).mean().round(3), "profit_approve_all": round(gain.sum() / n, 1)})
    sens = pd.DataFrame(sens)
    sens["profit_vs_approve_all"] = (sens.profit_per_applicant - sens.profit_approve_all).round(1)
    sens.to_csv(out / "sensitivity.csv", index=False)

    # ---- 5. who is approved under the base-case rule ----
    pg = []
    for name in [main_m, "scorecard_full_no_gender"]:
        appr = np.column_stack(decisions[(name, "break-even rule")])
        p_mean = np.column_stack(cal[name]).mean(axis=1)
        for attr, lab in groups(app).items():
            rows = []
            for g in sorted(set(lab)):
                gm = lab == g
                if gm.sum() < MIN_GROUP:
                    continue
                rows.append({"model": name, "attribute": attr, "group": g, "applicants": int(gm.sum()),
                             "default_rate_pct": round(100 * y[gm].mean(), 2),
                             "mean_pred_pct": round(100 * p_mean[gm].mean(), 2),
                             "approval_pct": round(100 * appr[gm].mean(), 2),
                             "approval_pct_good_payers": round(100 * appr[gm & (y == 0)].mean(), 2),
                             "default_rate_approved_pct": round(100 * np.mean([y[gm & appr[:, s]].mean()
                                                                              for s in range(N_SEEDS)]), 2)})
            r = pd.DataFrame(rows)
            r["adverse_impact_ratio"] = (r.approval_pct / r.approval_pct.max()).round(3)
            pg.append(r)
    pg = pd.concat(pg, ignore_index=True)
    pg.to_csv(out / "policy_groups.csv", index=False)
    print(pg[pg.model == main_m].to_string(index=False))


if __name__ == "__main__":
    main()
