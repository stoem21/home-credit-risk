"""Neural network (multi-layer perceptron) model code (CV and outputs in pipeline/train_family.py).

A plain feed-forward network, the simplest member of the neural-net family. Settings fixed in advance, not tuned:
  - inputs, fitted on the training rows only: numeric columns mapped to a normal shape by their ranks
    (quantile transform, 1,000 quantiles; robust to the very skewed amounts), missing values set to 0 plus one
    0/1 "was missing" column per numeric column that has missing values; text columns one-hot encoded,
    categories under 1% of rows pooled, missing as its own category;
  - network: two hidden layers of 256 and 128 ReLU units, Adam, learning rate 0.001, batch 512, L2 penalty 1e-4;
  - early stopping: one pass over the data per epoch, up to 40 epochs; after each epoch the AUC on an inner 10%
    hold-out of the training rows is measured; stop after 4 epochs without gain and keep the best epoch (the same
    idea as the trees' inner hold-out).
A deep version with a denoising autoencoder, as in some top public solutions, needs a GPU and is out of scope.
"""

import copy
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from homecredit.data import is_text

MAX_EPOCHS, PATIENCE = 40, 4


def make_prep(X: pd.DataFrame, seed: int) -> ColumnTransformer:
    num = [c for c in X.columns if not is_text(X[c])]
    txt = [c for c in X.columns if is_text(X[c])]
    num_pipe = make_union(
        make_pipeline(QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200_000,
                                          random_state=seed), SimpleImputer(strategy="constant", fill_value=0.0)),
        MissingIndicator(features="missing-only", error_on_new=False))
    txt_pipe = make_pipeline(SimpleImputer(strategy="constant", fill_value="(missing)"),
                             OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                           sparse_output=False))
    return ColumnTransformer([("num", num_pipe, num), ("txt", txt_pipe, txt)])


def _as_object(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if is_text(X[c]):
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
        else:
            X[c] = X[c].astype(float)
    return X


def _net(seed: int) -> MLPClassifier:
    return MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", alpha=1e-4, batch_size=512,
                         learning_rate_init=1e-3, random_state=seed)


def fit(X: pd.DataFrame, y: np.ndarray, seed: int):
    """Returns (preprocessor, network at its best epoch, best number of epochs)."""
    X = _as_object(X)
    i_fit, i_stop = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=seed)
    prep = make_prep(X, seed).fit(X.iloc[i_fit])
    A, B = prep.transform(X.iloc[i_fit]).astype(np.float32), prep.transform(X.iloc[i_stop]).astype(np.float32)
    net, best, best_auc, best_ep, wait = _net(seed), None, -1.0, 0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for ep in range(1, MAX_EPOCHS + 1):
            net.partial_fit(A, y[i_fit], classes=np.array([0, 1]))
            a = roc_auc_score(y[i_stop], net.predict_proba(B)[:, 1])
            if a > best_auc:
                best, best_auc, best_ep, wait = copy.deepcopy(net), a, ep, 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
    return prep, best, best_ep


def fit_fixed(X: pd.DataFrame, y: np.ndarray, n_epochs: int, seed: int):
    X = _as_object(X)
    prep = make_prep(X, seed).fit(X)
    A = prep.transform(X).astype(np.float32)
    net = _net(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for _ in range(n_epochs):
            net.partial_fit(A, y, classes=np.array([0, 1]))
    return prep, net


def predict(prep, net, X: pd.DataFrame) -> np.ndarray:
    return net.predict_proba(prep.transform(_as_object(X)).astype(np.float32))[:, 1]
