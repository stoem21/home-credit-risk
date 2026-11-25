"""Traditional credit scorecard: WOE binning + logistic regression (model code; CV and outputs in
pipeline/train_scorecard.py).

How it works, on the rows it is given (in CV: the training part only, so validation rows never influence bins):
  1. Binning. Numeric columns: 10 quantile bins, then neighbouring bins are merged until the default rate rises
     (or falls) steadily from bin to bin, and until every bin holds at least 1% of rows. Missing values get their
     own bin. Text columns: one bin per category; categories under 1% of rows are pooled as OTHER.
  2. WOE (weight of evidence) per bin = ln(share of all defaulters in the bin / share of all non-defaulters in
     the bin), with +0.5 smoothing. Positive WOE = riskier than average. Information value (IV) sums up how well a
     column separates defaulters from the rest.
  3. Selection, fixed in advance (not tuned): keep columns with IV >= 0.02, strongest first, skip a column whose
     WOE correlates above 0.7 with one already kept, stop at 30 columns.
  4. Logistic regression on the WOE values. Every coefficient must be positive (more risk in the bin -> more
     risk in the score); columns with a non-positive coefficient are dropped one at a time and the model refitted.

Points: 600 points = odds of 50 good : 1 bad, and every 20 points double the odds (PDO 20).
Points per bin = -factor x coefficient x WOE; score = base + sum.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from homecredit.data import is_text

N_FINE = 10
MIN_FRAC = 0.01
IV_MIN = 0.02
CORR_MAX = 0.7
MAX_VARS = 30
PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50
FACTOR = PDO / np.log(2)  # points per unit of log-odds
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


# ---------------- binning ----------------
def _woe_iv(tot: np.ndarray, bad: np.ndarray, B: float, G: float):
    good = tot - bad
    woe = np.log(((bad + 0.5) / (B + 0.5)) / ((good + 0.5) / (G + 0.5)))
    iv = float(np.sum((bad / B - good / G) * woe))
    return woe, iv


def fit_numeric(x: np.ndarray, y: np.ndarray) -> dict:
    m = ~np.isnan(x)
    xv, yv = x[m], y[m]
    B, G = y.sum(), len(y) - y.sum()
    edges = np.unique(np.quantile(xv, np.linspace(0, 1, N_FINE + 1)[1:-1])) if len(xv) else np.array([])
    idx = np.searchsorted(edges, xv, side="left")
    tot = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    bad = np.bincount(idx, weights=yv, minlength=len(edges) + 1)

    def merge(i):  # merge bin i with bin i+1
        nonlocal edges, tot, bad
        tot[i] += tot[i + 1]
        bad[i] += bad[i + 1]
        tot, bad, edges = np.delete(tot, i + 1), np.delete(bad, i + 1), np.delete(edges, i)

    while len(tot) > 1 and (tot == 0).any():
        i = int(np.argmax(tot == 0))
        merge(i if i < len(tot) - 1 else i - 1)
    if len(tot) > 1:
        r = bad / tot
        direction = np.sign(np.corrcoef(np.arange(len(r)), r)[0, 1]) or 1.0
        while len(tot) > 1:
            viol = np.where(np.diff(bad / tot) * direction < 0)[0]
            if not len(viol):
                break
            merge(int(viol[0]))
    while len(tot) > 1 and tot.min() < MIN_FRAC * len(y):
        i = int(np.argmin(tot))
        r = bad / tot
        if i == 0:
            merge(0)
        elif i == len(tot) - 1:
            merge(i - 1)
        else:
            merge(i - 1 if abs(r[i] - r[i - 1]) <= abs(r[i] - r[i + 1]) else i)
    n_miss, bad_miss = float((~m).sum()), float(y[~m].sum())
    woe, iv = _woe_iv(np.r_[tot, n_miss], np.r_[bad, bad_miss], B, G)
    if n_miss == 0:
        woe[-1] = 0.0  # no missing in training: an unseen missing value is scored as average
    labels = []
    lo = "-inf"
    for e in edges:
        labels.append(f"({lo}, {e:.4g}]")
        lo = f"{e:.4g}"
    labels.append(f"({lo}, inf)")
    labels.append("missing")
    return {"kind": "num", "edges": edges, "woe": woe, "iv": iv, "labels": labels,
            "n": np.r_[tot, n_miss], "bad": np.r_[bad, bad_miss]}


def fit_text(s: pd.Series, y: np.ndarray) -> dict:
    s = s.astype(object).fillna("(missing)").astype(str)
    vc = s.value_counts()
    small = set(vc[vc < MIN_FRAC * len(s)].index)
    s = s.where(~s.isin(small), "OTHER")
    g = pd.DataFrame({"c": s.values, "y": y}).groupby("c").y.agg(["size", "sum"])
    woe, iv = _woe_iv(g["size"].values.astype(float), g["sum"].values, y.sum(), len(y) - y.sum())
    return {"kind": "txt", "cats": list(g.index), "woe": woe, "iv": iv, "labels": list(g.index),
            "n": g["size"].values.astype(float), "bad": g["sum"].values.astype(float)}


def fit_feature(s: pd.Series, y: np.ndarray) -> dict:
    return fit_text(s, y) if is_text(s) else fit_numeric(s.astype(float).values, y)


def transform(s: pd.Series, spec: dict) -> np.ndarray:
    if spec["kind"] == "num":
        x = s.astype(float).values
        idx = np.searchsorted(spec["edges"], x, side="left")
        idx[np.isnan(x)] = len(spec["woe"]) - 1
        return spec["woe"][idx]
    lookup = dict(zip(spec["cats"], spec["woe"]))
    other = lookup.get("OTHER", 0.0)
    return s.astype(object).fillna("(missing)").astype(str).map(lookup).astype(float).fillna(other).values


# ---------------- selection + fit ----------------
def fit_scorecard(X: pd.DataFrame, y: np.ndarray, cols: list[str]):
    specs = {c: fit_feature(X[c], y) for c in cols}
    cand = sorted([c for c in cols if specs[c]["iv"] >= IV_MIN], key=lambda c: -specs[c]["iv"])
    W = pd.DataFrame({c: transform(X[c], specs[c]) for c in cand}, index=X.index)
    corr = W.sample(min(50000, len(W)), random_state=0).corr().abs().fillna(0)
    selected = []
    for c in cand:
        if all(corr.loc[c, s] <= CORR_MAX for s in selected):
            selected.append(c)
        if len(selected) == MAX_VARS:
            break
    dropped_sign = []
    while True:
        lr = LogisticRegression(max_iter=2000).fit(W[selected].values, y)
        coef = lr.coef_[0]
        if (coef > 0).all() or len(selected) == 1:
            break
        worst = selected[int(np.argmin(coef))]
        dropped_sign.append(worst)
        selected.remove(worst)
    return specs, selected, lr, dropped_sign


def predict(X: pd.DataFrame, specs, selected, lr) -> np.ndarray:
    W = np.column_stack([transform(X[c], specs[c]) for c in selected])
    return lr.predict_proba(W)[:, 1]
