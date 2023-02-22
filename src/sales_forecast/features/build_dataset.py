import logging
import pandas as pd
import numpy as np

__all__ = ["build_dataset"]

logger = logging.getLogger()


def get_features(
    X_scaled: pd.DataFrame,
    X_past_scaled: pd.DataFrame,
    extracted_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Builds dataset for LSTM model
    """

    logging.info("Building dataset for LSTM model...")

    time_seq = X_scaled.shape[0]
    n_shops = X_scaled.shape[1]
    X1 = np.empty((time_seq, n_shops, 2))
    X1[:, :, 0] = X_scaled
    X1[:, :, 1] = X_past_scaled

    n_features = extracted_features.shape[1]
    X2 = np.empty((time_seq, n_shops, n_features))
    for i in range(X2.shape[0]):
        X2[i, :, :] = extracted_features

    X = np.concatenate([X1, X2], axis=2)

    return X
