"""LightGBM model code (CV and outputs in pipeline/train_lgbm.py).

Settings are fixed in advance, common starting values, not tuned:
  learning_rate 0.05, num_leaves 31, min_child_samples 100, feature_fraction 0.8, bagging_fraction 0.8
  (every iteration), lambda_l2 1.0, up to 5,000 trees, early stopping after 100 trees without gain.

Fold-safe early stopping: inside the rows given to fit(), 10% (stratified, fixed seed) are held out to choose the
number of trees. A validation fold is only used for scoring, never for stopping.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
MAX_TREES, PATIENCE = 5000, 100


def fit(X: pd.DataFrame, y: np.ndarray, seed: int, threads: int, learning_rate: float | None = None) -> lgb.Booster:
    """Fit with early stopping on an inner 10% hold-out; return the booster (use bst.best_iteration)."""
    i_fit, i_stop = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=seed)
    params = {**PARAMS, "seed": seed, "num_threads": threads}
    if learning_rate:
        params["learning_rate"] = learning_rate
    dfit = lgb.Dataset(X.iloc[i_fit], y[i_fit])
    dstop = lgb.Dataset(X.iloc[i_stop], y[i_stop], reference=dfit)
    return lgb.train(params, dfit, MAX_TREES, valid_sets=[dstop],
                     callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])


def fit_fixed(X: pd.DataFrame, y: np.ndarray, n_trees: int, seed: int, threads: int,
              learning_rate: float | None = None) -> lgb.Booster:
    """Fit a fixed number of trees on all rows given (final model)."""
    params = {**PARAMS, "seed": seed, "num_threads": threads}
    if learning_rate:
        params["learning_rate"] = learning_rate
    return lgb.train(params, lgb.Dataset(X, y), n_trees)


def shap_select(X: pd.DataFrame, y: np.ndarray, top: int, seed: int, threads: int, rows: int = 20000) -> list[str]:
    """Feature reduction with SHAP, using only the rows given (in CV: the training part of the fold).

    Fits the default model (learning rate 0.05, early stopping on an inner 10%), computes TreeSHAP contributions
    on a random sample of `rows` of the given rows, ranks columns by mean |SHAP| and returns the `top` columns in
    their original order. No validation row is used, so the selection cannot leak into the score.
    """
    bst = fit(X, y, seed, threads)
    idx = np.random.default_rng(seed).choice(len(X), min(rows, len(X)), replace=False)
    contrib = bst.predict(X.iloc[idx], num_iteration=bst.best_iteration, pred_contrib=True)[:, :-1]
    imp = np.abs(contrib).mean(axis=0)
    keep = set(np.argsort(-imp)[:top])
    return [c for i, c in enumerate(X.columns) if i in keep]
