import logging
import pandas as pd
import numpy as np
import pywt
from typing import Optional


__all__ = ["generate_wavelet_features"]

logger = logging.getLogger()

N_WAVELET_FEATURES = 8


def get_wavelet_features(X_scaled: pd.DataFrame, n_features: Optional[int] = None):
    if n_features is None:
        n_features = N_WAVELET_FEATURES

    logging.info("Generating wavelet features ...")

    X = np.empty((X_scaled.shape[0], X_scaled.shape[1], n_features))
    scales = np.arange(1, n_features + 1)

    for i in range(X_scaled.shape[1]):
        [coefficients, _] = pywt.cwt(X_scaled[:, i], scales=scales, wavelet="cmor")
        X[:, i, :] = coefficients.T

    return X
