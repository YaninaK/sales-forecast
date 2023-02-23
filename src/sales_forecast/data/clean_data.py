import logging
import pandas as pd
import numpy as np
from typing import Optional

__all__ = ["clean_data"]

logger = logging.getLogger()


def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:

    logging.info("Cleaning outliers...")

    stats = df.describe()
    iqr = stats.loc["75%"] - stats.loc["25%"]
    outliers_level_lower = stats.loc["25%"] - 1.5 * iqr
    outliers_level_upper = stats.loc["75%"] + 1.5 * iqr

    n_outliers = (df > outliers_level_upper).sum().sum() + (
        df < outliers_level_lower
    ).sum().sum()

    for i in range(df.shape[1]):
        df[i] = df[i].clip(outliers_level_lower[i], outliers_level_upper[i])

    print(f"Number of outliers = {n_outliers}")

    return df
