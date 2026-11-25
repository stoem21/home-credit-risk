"""XGBoost model code (CV and outputs in pipeline/train_family.py).

Settings fixed in advance, common starting values chosen to mirror the LightGBM ones, not tuned:
  learning_rate 0.05, max_depth 6 (depth-wise trees, about as complex as 31 leaves), min_child_weight 5,
  subsample 0.8, colsample_bytree 0.8, lambda 1.0, histogram method, native categorical support,
  up to 5,000 rounds, early stopping after 100 rounds without gain on an inner 10% hold-out (same rule as LightGBM).
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

PARAMS = dict(objective="binary:logistic", eval_metric="auc", eta=0.05, max_depth=6, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", max_cat_to_onehot=1,
              verbosity=0)
MAX_ROUNDS, PATIENCE = 5000, 100


def fit(X: pd.DataFrame, y: np.ndarray, seed: int, threads: int):
    """Fit with early stopping on an inner 10% hold-out; returns (booster, best number of rounds)."""
    i_fit, i_stop = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=seed)
    params = {**PARAMS, "seed": seed, "nthread": threads}
    dfit = xgb.DMatrix(X.iloc[i_fit], y[i_fit], enable_categorical=True)
    dstop = xgb.DMatrix(X.iloc[i_stop], y[i_stop], enable_categorical=True)
    bst = xgb.train(params, dfit, MAX_ROUNDS, evals=[(dstop, "stop")], early_stopping_rounds=PATIENCE,
                    verbose_eval=False)
    return bst, bst.best_iteration + 1


def fit_fixed(X: pd.DataFrame, y: np.ndarray, n_rounds: int, seed: int, threads: int):
    params = {**PARAMS, "seed": seed, "nthread": threads}
    return xgb.train(params, xgb.DMatrix(X, y, enable_categorical=True), n_rounds)


def predict(bst, X: pd.DataFrame, n_rounds: int) -> np.ndarray:
    return bst.predict(xgb.DMatrix(X, enable_categorical=True), iteration_range=(0, n_rounds))
