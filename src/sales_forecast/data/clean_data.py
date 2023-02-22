import logging
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Optional

__all__ = ["clean_data"]

logger = logging.getLogger()

N_SHOPS = 20


def impute_with_medians(
    missing_dates: list, cols: list, df: pd.DataFrame
) -> pd.DataFrame:
    """
    Imputes values in missing_dates with medians for particular year-month-weekday
    """
    logging.info("Imputeing missing values with medians...")

    medians = pd.DataFrame(index=cols, columns=missing_dates)
    for i in missing_dates:
        cond_1 = df.index.year == i.year
        cond_2 = df.index.month == i.month
        cond_3 = df.index.weekday == i.weekday()
        medians.loc[:, i] = df.loc[cond_1 & cond_2 & cond_3, cols].median()

        df.loc[df.index.isin(missing_dates), cols] = medians.T

    return df


def impute_by_regression(correlated_shops: dict, df: pd.DataFrame) -> pd.DataFrame:

    logging.info("Imputeing missing values by regression...")

    for i in correlated_shops.keys():
        cols = correlated_shops[i]
        model = LinearRegression()
        model.fit(df.loc[~df[i].isnull(), cols], df.loc[~df[i].isnull(), i])
        predictions = model.predict(df.loc[df[i].isnull(), cols])
        df.loc[df[i].isnull(), i] = predictions

    return df


def clean_outliers(df: pd.DataFrame, n_shops: Optional[int] = None) -> pd.DataFrame:

    logging.info("Cleaning outliers...")

    if n_shops is None:
        n_shops = N_SHOPS

    stats = df.describe()
    iqr = stats.loc["75%"] - stats.loc["25%"]
    outliers_level_lower = stats.loc["25%"] - 1.5 * iqr
    outliers_level_upper = stats.loc["75%"] + 1.5 * iqr

    n_outliers = (df > outliers_level_upper).sum().sum() + (
        df < outliers_level_lower
    ).sum().sum()

    for i in range(n_shops):
        df = np.clip(df, outliers_level_lower[i], outliers_level_upper[i])

    print(f"Number of outliers = {n_outliers}")

    return df
