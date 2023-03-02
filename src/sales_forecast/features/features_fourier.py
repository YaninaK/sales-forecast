import logging
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


__all__ = ["get_fourier_features"]

logger = logging.getLogger()


PLOT = True
N = 10

SELECTED_TS_DIFF = {
    0: [2, 3, 4, 5, 7],
    1: [1, 4, 5, 7, 9],
    2: [2, 4, 5, 7, 8],
    3: [2, 3, 6, 7, 9],
    4: [1, 2, 4, 5, 8],
}


def get_fourier_features(
    X_scaled: pd.DataFrame,
    clusters: pd.DataFrame,
    selected_ts_diff: Optional[dict] = None,
):
    if selected_ts_diff is None:
        selected_ts_diff = SELECTED_TS_DIFF

    n_features = len(selected_ts_diff[0]) + 1
    X = np.empty((X_scaled.shape[0], X_scaled.shape[1], n_features))

    _, ts = fourier_transform(X_scaled, clusters)

    for i in range(clusters.shape[1]):
        X1 = np.expand_dims(ts[i][0], axis=0)

        cluster_ts = get_ts_difference(ts[i], False)
        cluster_selection = [cluster_ts[j] for j in selected_ts_diff[i]]
        X2 = np.vstack(cluster_selection)

        cluster_selection = np.concatenate([X1, X2], axis=0).T

        ind = clusters[clusters[i] == 1].index
        cluster_selection = np.expand_dims(cluster_selection, axis=0).repeat(
            len(ind), axis=0
        )
        X[:, ind, :] = np.moveaxis(cluster_selection, 0, 1)

    return X


def fourier_transform(
    X_scaled: pd.DataFrame,
    clusters: pd.DataFrame,
    n: Optional[int] = None,
) -> Tuple[dict, dict]:

    if n is None:
        n = N

    thresholds = {}
    ts = {}
    for i in range(clusters.shape[1]):
        ind = clusters.loc[clusters[i] == 1].index
        composite_ts_FFT = fft(X_scaled[ind].mean(axis=1).tolist())

        a = set(abs(composite_ts_FFT) * 1000 // 10 / 100)
        thresholds[i] = sorted(a, reverse=True)[:n]

        ts[i] = decompose_ts(composite_ts_FFT, thresholds[i], False)

    return thresholds, ts


def decompose_ts(composite_ts_FFT, thresholds, plot: Optional[bool] = None) -> dict:
    if plot is None:
        plot = PLOT

    ts = {}
    for i, threshold in enumerate(thresholds):
        frequencies = composite_ts_FFT.copy()
        cond = abs(frequencies) < threshold
        frequencies[cond] = 0
        composite_ts_inv = ifft(frequencies)
        ts[i] = np.real(composite_ts_inv)

        if plot:
            plt.figure(figsize=(16, 6))
            plt.plot(np.real(composite_ts_inv))
            plt.xlabel("time")
            plt.ylabel("sales")
            plt.title(f"threshold_{i}")

    return ts


def get_ts_difference(ts: dict, plot: Optional[bool] = None) -> dict:
    if plot is None:
        plot = PLOT

    ts_diff = {}
    ts_temp = ts[0]
    for i in range(1, len(ts)):
        ts_temp = ts[i] - ts[i - 1]
        ts_diff[i] = ts_temp

        if plot:
            plt.figure(figsize=(16, 6))
            plt.plot(ts_temp)
            plt.xlabel("time")
            plt.ylabel("sales")
            plt.title(f"Time series difference {i}: ts_{i} - ts_{i-1}")

    return ts_diff
