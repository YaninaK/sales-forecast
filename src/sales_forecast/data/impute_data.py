import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Optional

__all__ = ["impute_data"]

logger = logging.getLogger()


IMPUTE_DATE_SHOPS = [
    [
        ["2016-06-19", "2017-06-04"],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18, 19],
    ],
    [["2016-01-05", "2017-01-06"], [4, 8, 10, 12, 13, 17, 18]],
    [["2016-11-01"], [7]],
]

CORRELATED_SHOPS = {
    0: {
        3: [0, 1, 15, 19],
        11: [0, 1, 3, 15, 19],
        12: [0, 1, 3, 11, 15, 19],
        14: [0, 1, 3, 11, 12, 15, 19],
        18: [0, 1, 3, 11, 12, 14, 15, 19],
    },
    1: {2: [7, 9, 16]},
}


def impute(
    data: pd.DataFrame,
    impute_dates_shops: Optional[list] = None,
    correlated_shops: Optional[dict] = None,
) -> pd.DataFrame:

    if impute_dates_shops is None:
        impute_dates_shops = IMPUTE_DATE_SHOPS
    if correlated_shops is None:
        correlated_shops = CORRELATED_SHOPS

    for missing_dates, shops in impute_dates_shops:
        missing_dates = [pd.to_datetime(t) for t in missing_dates]
        data = impute_with_medians(missing_dates, shops, data)

    for i in range(len(correlated_shops)):
        data = impute_by_regression(correlated_shops[i], data)

    return data


def impute_with_medians(missing_dates, cols, df: pd.DataFrame) -> pd.DataFrame:
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
