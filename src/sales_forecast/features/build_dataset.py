import logging
import pandas as pd
import numpy as np
import scipy.stats as st


__all__ = ["build_dataset"]

logger = logging.getLogger()


def get_features(
    X_scaled: pd.DataFrame,
    X_past_scaled: pd.DataFrame,
    clusters: pd.DataFrame,
    extracted_features_scaled: np.array,
) -> np.array:
    """
    Builds dataset for LSTM model
    """

    logging.info("Building dataset for LSTM model...")

    time_seq = X_scaled.shape[0] - 1

    a = X_scaled.values
    b = X_past_scaled.values
    X1 = np.vstack([a, b, a - b]).reshape(3, -1, X_scaled.shape[1])
    X1 = np.moveaxis(X1, 0, -1)[-time_seq:, :, :]

    X2 = get_1st_differences(X_scaled)
    X3 = get_2nd_differences(X_scaled)
    X4 = get_moments(X_scaled)
    X5 = get_rolling_means(X_scaled)

    X6 = clusters.values.repeat(time_seq, axis=0).reshape(
        clusters.shape[0], -1, clusters.shape[1]
    )
    X6 = np.moveaxis(X6, 1, 0)

    X7 = extracted_features_scaled.repeat(time_seq, axis=0).reshape(
        extracted_features_scaled.shape[0], -1, extracted_features_scaled.shape[1]
    )
    X7 = np.moveaxis(X7, 1, 0)

    return np.concatenate([X1, X2, X3, X4, X5, X6, X7], axis=-1)


def get_1st_differences(X_scaled: pd.DataFrame, periods=["D", "W", "M"]) -> np.array:
    df = pd.DataFrame(index=X_scaled.index[1:])
    for t in periods:
        a1 = X_scaled.resample(t).median()
        X1 = a1.values[1:, :] - a1.values[:-1, :]
        X2 = (a1.values[1:, :] - a1.values[:-1, :]) / a1.values[:-1, :]
        X = pd.DataFrame(np.hstack([X1, X2]), index=a1.index[1:])
        df = df.merge(X, on="dt", how="left").bfill().ffill()

        X = df.values.reshape(len(df), -1, X_scaled.shape[1])

    return np.moveaxis(X, 1, -1)


def get_2nd_differences(X_scaled: pd.DataFrame, periods=["W", "M"]) -> np.array:
    df = pd.DataFrame(index=X_scaled.index[1:])
    for t in periods:
        a1 = X_scaled.resample(t).median()
        X = a1.values[1:, :] - a1.values[:-1, :]
        X = pd.DataFrame(X[1:, :] - X[:-1, :], index=a1.index[2:])
        df = df.merge(X, on="dt", how="left").bfill().ffill()

        X = df.values.reshape(len(df), -1, X_scaled.shape[1])

    return np.moveaxis(X, 1, -1)


def get_moments(X_scaled: pd.DataFrame, periods=["W", "M"]) -> np.array:
    df = pd.DataFrame(index=X_scaled.index)
    for t in periods:
        f = X_scaled.resample(t)
        df = pd.concat(
            [
                df,
                f.median(),
                f.mean(),
                f.std(),
                f.mean() / f.std(),
                f.agg(st.skew),
                f.agg(st.kurtosis),
            ],
            axis=1,
        ).bfill()
    df = pd.DataFrame(index=X_scaled.index[1:]).merge(df, on="dt", how="left")

    X = df.values.reshape(len(df), -1, X_scaled.shape[1])

    return np.moveaxis(X, 1, -1)


def get_rolling_means(X_scaled: pd.DataFrame, windows=[6, 24]) -> np.array:
    df = pd.DataFrame(index=X_scaled.index)
    for w in windows:
        f = X_scaled.rolling(w)
        df = pd.concat([df, X_scaled.rolling(w).mean()], axis=1).bfill()
    df = pd.DataFrame(index=X_scaled.index[1:]).merge(df, on="dt", how="left")

    X = df.values.reshape(len(df), -1, X_scaled.shape[1])

    return np.moveaxis(X, 1, -1)
