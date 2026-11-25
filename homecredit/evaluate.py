"""The one place where models are scored: AUC, per-fold AUC, bootstrap noise, the paired adoption rule, and the
calibration measures. Every model, baseline and EDA check uses these functions; no copies elsewhere.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from homecredit.data import N_SEEDS, check_ids
from homecredit.paths import Paths

N_BOOT = 200  # bootstrap resamples of applicants
MIN_WINS = 18  # adoption rule: B must win at least 18 of the 25 fold x seed pairs ...
SE_MULT = 2.0  # ... and gain more than 2 x the bootstrap standard error of the gain
EPS = 1e-6
N_BINS = 20  # reliability table / ECE bins


# ---------------- ranking ----------------
def auc(y, score, weight=None) -> float:
    """ROC AUC (optionally weighted). Rows with a missing score are left out; NaN when only one class remains."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    ok = ~np.isnan(score)
    y, score = y[ok], score[ok]
    if len(y) == 0 or y.min() == y.max():
        return np.nan
    if weight is None:
        return float(roc_auc_score(y, score))
    return float(roc_auc_score(y, score, sample_weight=np.asarray(weight, dtype=float)[ok]))


def fold_aucs(y, oof: pd.DataFrame, folds: pd.DataFrame, masks: dict[str, np.ndarray],
              seeds=range(N_SEEDS)) -> pd.DataFrame:
    """AUC on every validation fold of every seed, for each scope mask. oof has columns s0..s4, folds fold_s0..4.

    Returns rows (scope, seed, fold, auc) in that loop order.
    """
    rows = []
    for scope, m in masks.items():
        for s in seeds:
            f = folds[f"fold_s{s}"].values
            p = oof[f"s{s}"].values
            for k in sorted(np.unique(f)):
                v = (f == k) & m
                rows.append({"scope": scope, "seed": s, "fold": int(k), "auc": auc(y[v], p[v])})
    return pd.DataFrame(rows)


def bootstrap_aucs(y, preds: list[np.ndarray], n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """AUC of each prediction vector on the same bootstrap resamples of applicants: array (n_boot, len(preds))."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(preds)))
    for b in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        out[b] = [auc(y[i], p[i]) for p in preds]
    return out


def paired_compare(y, oof_a: pd.DataFrame, oof_b: pd.DataFrame, folds: pd.DataFrame,
                   masks: dict[str, np.ndarray]) -> list[dict]:
    """Model B vs reference A on the same folds: mean AUCs, gain, bootstrap SE of the gain (seed-0 OOF), wins,
    and the verdict of the adoption rule (gain > 2 x SE and at least 18 of 25 wins).

    Exploratory runs may hold fewer seeds (e.g. only s0): then only the seeds both models have are compared and
    the win threshold keeps the same share (18/25 = 72%, rounded up: 4 of 5). Such a verdict is marked
    exploratory; a winner is confirmed with all 5 seeds before it is adopted (the more ideas are tried, the more
    likely one wins by chance)."""
    seeds = [s for s in range(N_SEEDS) if f"s{s}" in oof_a.columns and f"s{s}" in oof_b.columns]
    fa = fold_aucs(y, oof_a, folds, masks, seeds)
    fb = fold_aucs(y, oof_b, folds, masks, seeds)
    rows = []
    for scope, m in masks.items():
        a = fa.auc[fa.scope == scope].values
        b = fb.auc[fb.scope == scope].values
        gain = b - a
        boot = bootstrap_aucs(y[m], [oof_a.s0.values[m], oof_b.s0.values[m]])
        se = float(np.std(boot[:, 1] - boot[:, 0]))
        n = len(gain)
        need = MIN_WINS if n == 25 else int(np.ceil(MIN_WINS / 25 * n))
        b_better = gain.mean() > SE_MULT * se and (gain > 0).sum() >= need
        a_better = -gain.mean() > SE_MULT * se and (gain < 0).sum() >= need
        note = "" if n == 25 else f" [exploratory: {n} pairs]"
        rows.append({"scope": scope, "auc_a": round(a.mean(), 5), "auc_b": round(b.mean(), 5),
                     "gain_b_minus_a": round(gain.mean(), 5), "boot_se_gain": round(se, 5),
                     f"b_wins_of_{n}": int((gain > 0).sum()),
                     "verdict": ("B better (beats noise)" if b_better else
                                 ("A better (beats noise)" if a_better else "no clear difference")) + note})
    return rows


# ---------------- out-of-fold predictions ----------------
def save_oof(P: Paths, name: str, oof: pd.DataFrame) -> None:
    P.oof_dir.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(P.oof(name), index=False)


def load_oof(P: Paths, name: str, ids: pd.Series) -> pd.DataFrame:
    oof = pd.read_parquet(P.oof(name))
    check_ids(oof, ids, f"oof {name}")
    return oof


# ---------------- calibration ----------------
def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def slope_intercept(y, p):
    """Logistic regression of the outcome on logit(p). Slope 1 and intercept 0 = perfectly calibrated."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def reliability(y, p, n_bins: int = N_BINS) -> pd.DataFrame:
    """Equal-size bins of predicted probability: rows, mean predicted, observed default rate, range."""
    b = pd.qcut(pd.Series(p).rank(method="first"), n_bins, labels=False)
    return pd.DataFrame({"bin": b, "p": p, "y": y}).groupby("bin").agg(
        rows=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"), p_min=("p", "min"), p_max=("p", "max"))


def ece(y, p) -> float:
    """Expected calibration error: row-weighted mean |predicted - observed| over the reliability bins."""
    t = reliability(y, p)
    return float((t.rows * (t.mean_pred - t.observed).abs()).sum() / t.rows.sum())


def ece_by_fold(y, p, folds) -> float:
    """ECE measured inside each validation fold, then averaged over folds (points of probability x 100).

    Use this for cross-fitted recalibration: the pooled ECE of cross-fitted maps is biased low, because the map
    applied to one fold was fitted on the other folds' rows that fall in the same pooled bins.
    """
    return float(np.mean([100 * ece(y[folds == k], p[folds == k]) for k in np.unique(folds)]))


def ece_floor(p, folds, n_sim: int = 10, seed: int = 0) -> float:
    """What ece_by_fold would show for a perfectly calibrated model with these predictions: outcomes drawn from
    p itself, n_sim times. A measured ECE near this floor is within noise."""
    rng = np.random.default_rng(seed)
    return float(np.mean([ece_by_fold(rng.binomial(1, np.clip(p, 0, 1)), p, folds) for _ in range(n_sim)]))


def calibration_metrics(y, p) -> dict:
    s, i = slope_intercept(y, p)
    return {"auc": auc(y, p), "mean_pred_pct": 100 * p.mean(), "observed_pct": 100 * y.mean(),
            "slope": s, "intercept": i, "ece_pts": 100 * ece(y, p), "brier": brier_score_loss(y, p),
            "log_loss": log_loss(y, np.clip(p, EPS, 1 - EPS))}
