import logging
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ["get_fourier_features"]

logger = logging.getLogger()


THRESHOLDS = {
    0: [40, 35, 32, 28, 26, 24, 23, 22, 21, 20],
    1: [48, 44, 37, 28, 26, 23, 21, 20, 15, 14],
    2: [39, 33, 30, 28, 23, 19, 18, 17.5, 17.28, 17.26],
    3: [56, 35, 32, 25, 22, 18, 17, 16.6, 16.1, 15.75],
    4: [51, 31, 29.9, 29.5, 24.5, 19, 17, 14, 13.6, 13.4],
}
PLOT = False


def get_fourier_features(
    clusters: pd.DataFrame, X_scaled: pd.DataFrame, thresholds: Optional[dict] = None
):
    if thresholds is None:
        thresholds = THRESHOLDS

    n1 = X_scaled.shape[0]
    n2 = X_scaled.shape[1]
    n_features = 6
    X_fourier = np.zeros((n1, n2, n_features))

    ts_, diff_ = get_fourier_ts(X_scaled, clusters, thresholds)
    selected_ts = select_fourier_ts(ts_, diff_)

    for i in range(clusters.shape[1]):
        ind = clusters.loc[clusters[i] == 1].index
        x = np.asarray(selected_ts[i]).T
        X_fourier[:, ind, :] = np.repeat(x, len(ind), axis=0).reshape(
            n1, len(ind), n_features
        )

    return X_fourier


def get_fourier_ts(
    X_scaled: pd.DataFrame, clusters: pd.DataFrame, thresholds: dict
) -> Tuple[list, list]:
    ts_ = []
    diff_ = []
    for i in range(clusters.shape[1]):
        ind = clusters.loc[clusters[i] == 1].index

        composite_y_value = X_scaled[ind].mean(axis=1).tolist()
        composite_y_FFT = get_composite_y_FFT(composite_y_value)
        ts = decompose_ts(composite_y_FFT, thresholds[i])
        diff_1 = get_ts_difference(ts)

        ts_.append(ts)
        diff_.append(diff_1)

    return ts_, diff_


def get_composite_y_FFT(composite_y_value: list, plot: Optional[bool] = None):
    if plot is None:
        plot = PLOT

    composite_y_FFT = fft(composite_y_value)

    if plot:
        N = len(composite_y_value)
        T = N
        t = np.linspace(0, N, T)
        f = fftfreq(len(t), np.diff(t)[0])

        plt.figure(figsize=(20, 4))
        plt.plot(f[: N // 2], np.abs(composite_y_FFT[: N // 2]))
        plt.xlabel("Frequency [Hz]", fontsize=16)
        plt.ylabel("Amplitude", fontsize=16)
        plt.title("Frequency domain of the signal", fontsize=16)

    return composite_y_FFT


def decompose_ts(
    composite_y_FFT, thresholds: Optional[dict] = None, plot: Optional[bool] = None
) -> dict:
    if thresholds is None:
        thresholds = THRESHOLDS
    if plot is None:
        plot = PLOT

    ts = {}
    for i, threshold in enumerate(thresholds):
        frequencies = composite_y_FFT.copy()
        cond = abs(frequencies) < threshold
        frequencies[cond] = 0
        composite_y_inv = ifft(frequencies)
        ts[i] = np.real(composite_y_inv)

        if plot:
            plt.figure(figsize=(16, 6))
            plt.plot(np.real(composite_y_inv))
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


def select_fourier_ts(ts_: list, diff_: list) -> dict:
    selected_ts = {}
    selected_ts[0] = [
        ts_[0][0],
        diff_[0][2],
        diff_[0][3],
        diff_[0][4],
        diff_[0][5],
        diff_[0][7],
    ]
    selected_ts[1] = [
        ts_[1][0],
        diff_[1][1],
        diff_[1][4],
        diff_[1][5],
        diff_[1][7],
        diff_[1][9],
    ]
    selected_ts[2] = [
        ts_[2][0],
        diff_[2][2],
        diff_[2][4],
        diff_[2][5],
        diff_[2][7],
        diff_[2][8],
    ]
    selected_ts[3] = [
        ts_[3][0],
        diff_[3][1],
        diff_[3][2],
        diff_[3][3],
        diff_[3][7],
        diff_[3][9],
    ]
    selected_ts[4] = [
        ts_[4][0],
        diff_[4][1],
        diff_[4][2],
        diff_[4][4],
        diff_[4][5],
        diff_[4][8],
    ]

    return selected_ts
