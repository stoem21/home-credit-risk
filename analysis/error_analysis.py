"""Error analysis of LightGBM v1: who does the model get wrong, what share of the
lost score do they cause, and which signals does it miss?

Everything uses the saved out-of-fold predictions (processed/oof/lgbm_v1.parquet), so each applicant is scored by
a model that never saw their outcome. No model is trained here.

  1. Segments. For each group (contract type, income type, age band, number of external scores, history
     available, loan-term band, credit-bureau enquiries, how "test-like" the applicant is, ...): size, default
     rate, mean predicted risk, AUC inside the group (mean and sd over the 5 seeds).
  2. Lost score. AUC = share of (defaulter, non-defaulter) pairs ranked the right way round. Every wrongly ordered
     pair (seed 0; ties count half) is charged twice: to the defaulter ("missed defaulter": ranked safer than a
     non-defaulter) and to the non-defaulter ("false alarm"). For each group: its share of the wrong pairs vs its
     share of defaulters / non-defaulters. A ratio above 1 means the group costs more AUC than its size.
  3. Missed defaulters. Defaulters in the safest quarter of predicted risk among defaulters vs the riskiest
     quarter (prediction = mean of the 5 seeds' out-of-fold values): how they differ on the main features.
  4. Candidate signals (hypotheses from eda/one_applicant.py, counted on all train applicants). Per bucket:
     observed default rate vs mean predicted risk. If the model already used the signal, the two would match in
     every bucket (up to the overall calibration gap, shown in the "all applicants" row). A clear gap suggests a
     feature worth testing in v2; it is not proof that the feature will raise AUC.

Usage:
    .venv/bin/python -m analysis.error_analysis [--model lgbm_v2_combo_lr02_x3 --tag v2] [--out-root DIR]
Outputs: analysis_output/errors/segments.csv, missed_defaulters.csv, candidate_signals.csv
"""

import argparse

import numpy as np
import pandas as pd

from homecredit.data import N_SEEDS, load_features, read_raw
from homecredit.evaluate import auc, load_oof
from homecredit.paths import Paths, add_out_root

MODEL = "lgbm_v1"  # default; --model chooses another saved out-of-fold file


def wrong_pairs(y: np.ndarray, p: np.ndarray):
    """Per row: number of wrongly ordered pairs it is part of (ties count half). Sum over defaulters = sum over
    non-defaulters = total wrong pairs; 1 - AUC = total / (n_defaulters x n_non_defaulters)."""
    d, n = np.where(y == 1)[0], np.where(y == 0)[0]
    pn, pd_ = np.sort(p[n]), np.sort(p[d])
    out = np.zeros(len(y))
    hi = len(pn) - np.searchsorted(pn, p[d], side="right")
    tie = np.searchsorted(pn, p[d], side="right") - np.searchsorted(pn, p[d], side="left")
    out[d] = hi + 0.5 * tie
    lo = np.searchsorted(pd_, p[n], side="left")
    tie = np.searchsorted(pd_, p[n], side="right") - np.searchsorted(pd_, p[n], side="left")
    out[n] = lo + 0.5 * tie
    return out


def segments(tr: pd.DataFrame, P: Paths) -> dict[str, pd.Series]:
    """Group label per train applicant, for several ways of splitting them."""
    s = {}
    s["contract type"] = tr.NAME_CONTRACT_TYPE.astype(str)
    inc = tr.NAME_INCOME_TYPE.astype(str)
    s["income type"] = inc.where(inc.isin(["Working", "Commercial associate", "Pensioner", "State servant"]), "Other")
    s["age band"] = pd.cut(tr.age_years, [0, 25, 35, 45, 55, 65, 100], right=False,
                           labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"]).astype(str)
    s["gender"] = tr.CODE_GENDER.astype(str)
    s["education"] = tr.NAME_EDUCATION_TYPE.astype(str)
    s["external scores available"] = tr[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].notna().sum(axis=1).astype(str)
    s["Home Credit history (installments)"] = np.where(tr.inst_n > 0, "yes", "no")
    s["other-lender history (bureau)"] = np.where(tr.bureau_n > 0, "yes", "no")
    s["bureau enquiries last quarter"] = tr.AMT_REQ_CREDIT_BUREAU_QRT.map(
        lambda v: "missing" if pd.isna(v) else ("0" if v == 0 else "1+"))
    term = tr.annuity_to_credit
    s["loan term band (annuity / credit quintile)"] = pd.qcut(
        term, 5, labels=["Q1 longest", "Q2", "Q3", "Q4", "Q5 shortest"]).astype(str)
    s["region rating"] = tr.REGION_RATING_CLIENT.astype(str)
    adv = pd.read_parquet(P.processed / "adversarial_v1.parquet")
    adv = adv[adv.split == "train"].set_index("SK_ID_CURR").p_test.reindex(tr.SK_ID_CURR).values
    s["test-likeness (adversarial model, train deciles)"] = pd.qcut(
        adv, 10, labels=[f"D{i}" for i in range(1, 11)]).astype(str)
    return {k: pd.Series(np.asarray(v, dtype=object), index=tr.index).fillna("missing").astype(str)
            for k, v in s.items()}


def candidate_signals(tr: pd.DataFrame, P: Paths) -> dict[str, pd.Series]:
    ids = tr.SK_ID_CURR
    prev = read_raw(P, "previous_application", ["SK_ID_CURR", "NAME_CONTRACT_STATUS", "DAYS_DECISION"])
    pos = read_raw(P, "POS_CASH_balance", ["SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE", "CNT_INSTALMENT_FUTURE"])
    bur = read_raw(P, "bureau", ["SK_ID_CURR", "SK_ID_BUREAU", "DAYS_CREDIT"])
    bb = read_raw(P, "bureau_balance")

    def count_bucket(counts: pd.Series, has_table: pd.Series, top: int) -> pd.Series:
        c = ids.map(counts).fillna(0)
        lab = c.clip(upper=top).astype(int).astype(str).where(c < top, f"{top}+")
        return lab.where(ids.isin(has_table), "no records")

    out = {}
    recent = prev[prev.DAYS_DECISION >= -90]
    out["Home Credit applications in last 90 days"] = count_bucket(
        recent.groupby("SK_ID_CURR").size(), prev.SK_ID_CURR, 3)
    canc = prev[(prev.DAYS_DECISION >= -365) & prev.NAME_CONTRACT_STATUS.eq("Canceled")]
    out["cancelled Home Credit applications in last 365 days"] = count_bucket(
        canc.groupby("SK_ID_CURR").size(), prev.SK_ID_CURR, 2)
    last = pos.sort_values("MONTHS_BALANCE").groupby("SK_ID_PREV").tail(1)
    running = last[(last.MONTHS_BALANCE >= -2) & (last.CNT_INSTALMENT_FUTURE > 0)]
    out["Home Credit POS / cash loans still running (seen in last 2 months)"] = count_bucket(
        running.groupby("SK_ID_CURR").size(), pos.SK_ID_CURR, 2)
    newb = bur[bur.DAYS_CREDIT >= -90]
    out["credits opened with other lenders in last 90 days"] = count_bucket(
        newb.groupby("SK_ID_CURR").size(), bur.SK_ID_CURR, 2)
    late = tr.inst_days_late_max
    lab = pd.Series("never late", index=tr.index)
    lab[late.isna()] = "no records"
    lab[(late > 0) & (late <= 5)] = "late, at most 5 days"
    lab[(late > 5) & (late <= 30)] = "late, 6-30 days"
    lab[late > 30] = "late, over 30 days"
    out["worst Home Credit instalment delay"] = lab
    bbr = bb[(bb.MONTHS_BALANCE >= -12)].merge(bur[["SK_ID_BUREAU", "SK_ID_CURR"]], on="SK_ID_BUREAU")
    dpd = bbr.STATUS.astype(str).isin(["1", "2", "3", "4", "5"]).groupby(bbr.SK_ID_CURR).any()
    lab = ids.map(dpd).map({True: "yes", False: "no"}).fillna("no monthly history")
    out["any other-lender month past due in last 12 months (bureau_balance)"] = lab
    return {k: pd.Series(np.asarray(v, dtype=object), index=tr.index).fillna("missing").astype(str)
            for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tag", default="", help="write to analysis_output/errors_<tag>/ (earlier results stay)")
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    out = P.out("errors" + (f"_{args.tag}" if args.tag else ""))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)

    tr, _, cols = load_features(P)
    y = tr.TARGET.values.astype(int)
    oof = load_oof(P, args.model, tr.SK_ID_CURR)
    preds = np.column_stack([oof[f"s{s}"].values for s in range(N_SEEDS)])
    p_mean = preds.mean(axis=1)
    wp = wrong_pairs(y, preds[:, 0])
    total = wp[y == 1].sum()
    assert np.isclose(total, wp[y == 0].sum())
    print(f"seed-0 AUC {auc(y, preds[:, 0]):.5f} = 1 - wrong pairs / all pairs "
          f"{1 - total / ((y == 1).sum() * (y == 0).sum()):.5f}")

    # ---- 1 + 2. segments ----
    rows = []
    for seg, lab in segments(tr, P).items():
        for g in sorted(lab.unique()):
            m = (lab == g).values
            aucs = [auc(y[m], preds[m, s]) for s in range(N_SEEDS)]
            rows.append({
                "segment": seg, "group": g, "applicants": int(m.sum()), "defaulters": int(y[m].sum()),
                "pct_of_train": round(100 * m.mean(), 2), "default_rate_pct": round(100 * y[m].mean(), 2),
                "mean_pred_pct": round(100 * p_mean[m].mean(), 2),
                "auc_within": round(np.mean(aucs), 4), "auc_within_sd_seeds": round(np.std(aucs, ddof=1), 4),
                "pct_of_defaulters": round(100 * y[m].sum() / y.sum(), 2),
                "pct_of_missed_defaulter_pairs": round(100 * wp[m & (y == 1)].sum() / total, 2),
                "pct_of_non_defaulters": round(100 * (m & (y == 0)).sum() / (y == 0).sum(), 2),
                "pct_of_false_alarm_pairs": round(100 * wp[m & (y == 0)].sum() / total, 2)})
    seg = pd.DataFrame(rows)
    seg["missed_ratio"] = (seg.pct_of_missed_defaulter_pairs / seg.pct_of_defaulters).round(2)
    seg["false_alarm_ratio"] = (seg.pct_of_false_alarm_pairs / seg.pct_of_non_defaulters).round(2)
    seg.to_csv(out / "segments.csv", index=False)
    print(seg.drop(columns=["pct_of_defaulters", "pct_of_non_defaulters"]).to_string(index=False), "\n")

    # ---- 3. missed vs caught defaulters ----
    d = np.where(y == 1)[0]
    q1, q3 = np.quantile(p_mean[d], [0.25, 0.75])
    missed, caught = d[p_mean[d] <= q1], d[p_mean[d] >= q3]
    non = np.where(y == 0)[0]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(tr[c]) and tr[c].nunique() > 2]
    prof = []
    for c in num:
        v = tr[c].astype(float)
        sd = v.std()
        if not sd > 0:
            continue
        prof.append({"feature": c, "median_missed": v.iloc[missed].median(), "median_caught": v.iloc[caught].median(),
                     "median_non_defaulters": v.iloc[non].median(),
                     "pct_missing_missed": round(100 * v.iloc[missed].isna().mean(), 1),
                     "pct_missing_caught": round(100 * v.iloc[caught].isna().mean(), 1),
                     "std_diff_missed_vs_caught": round((v.iloc[missed].mean() - v.iloc[caught].mean()) / sd, 3)})
    prof = pd.DataFrame(prof)
    prof = prof.reindex(prof.std_diff_missed_vs_caught.abs().sort_values(ascending=False).index)
    prof.insert(1, "mean_pred_missed_pct", round(100 * p_mean[missed].mean(), 2))
    prof.insert(2, "mean_pred_caught_pct", round(100 * p_mean[caught].mean(), 2))
    prof.to_csv(out / "missed_defaulters.csv", index=False)
    print(f"missed defaulters (safest quarter, mean predicted {100 * p_mean[missed].mean():.1f}%) vs caught "
          f"(riskiest quarter, {100 * p_mean[caught].mean():.1f}%); non-defaulters average "
          f"{100 * p_mean[non].mean():.1f}%")
    print(prof.head(15).round(3).to_string(index=False), "\n")

    # ---- 4. candidate signals ----
    rows = [{"signal": "all applicants", "bucket": "", "applicants": len(y), "pct_of_train": 100.0,
             "observed_default_pct": round(100 * y.mean(), 2), "mean_pred_pct": round(100 * p_mean.mean(), 2),
             "gap_pts": round(100 * (y.mean() - p_mean.mean()), 2),
             "se_pts": round(100 * np.sqrt(y.mean() * (1 - y.mean()) / len(y)), 2)}]
    for sig, lab in candidate_signals(tr, P).items():
        for g in sorted(lab.unique()):
            m = (lab == g).values
            o, pr = y[m].mean(), p_mean[m].mean()
            rows.append({"signal": sig, "bucket": g, "applicants": int(m.sum()), "pct_of_train": round(100 * m.mean(), 2),
                         "observed_default_pct": round(100 * o, 2), "mean_pred_pct": round(100 * pr, 2),
                         "gap_pts": round(100 * (o - pr), 2), "se_pts": round(100 * np.sqrt(o * (1 - o) / m.sum()), 2)})
    cs = pd.DataFrame(rows)
    cs.to_csv(out / "candidate_signals.csv", index=False)
    print(cs.to_string(index=False))


if __name__ == "__main__":
    main()
