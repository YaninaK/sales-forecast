import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Optional

__all__ = ["impute_data"]

logger = logging.getLogger()


def impute(data: pd.DataFrame) -> pd.DataFrame:

    logging.info("Imputing data...")

    n_shops = data.shape[1]
    missing_dates = [pd.to_datetime(t) for t in ["2016-06-19", "2017-06-04"]]
    cols = [i for i in range(n_shops) if not i in [0, 1, 11, 12, 15]]
    data = impute_with_medians(missing_dates, cols=cols, df=data)

    missing_dates = [pd.to_datetime(t) for t in ["2016-01-05", "2017-01-06"]]
    cols = [i for i in [4, 8, 10, 12, 13, 17, 18]]
    data = impute_with_medians(missing_dates, cols=cols, df=data)

    missing_dates = [pd.to_datetime("2016-11-01")]
    data = impute_with_medians(missing_dates, cols=[7], df=data)

    correlated_shops = {
        3: [0, 1, 15, 19],
        11: [0, 1, 3, 15, 19],
        12: [0, 1, 3, 11, 15, 19],
        14: [0, 1, 3, 11, 12, 15, 19],
        18: [0, 1, 3, 11, 12, 14, 15, 19],
    }
    data = impute_by_regression(correlated_shops, data)

    correlated_shops = {2: [7, 9, 16]}
    data = impute_by_regression(correlated_shops, data)

    return data


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
