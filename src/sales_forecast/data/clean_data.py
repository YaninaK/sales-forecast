import pandas as pd
from sklearn.linear_model import LinearRegression


def impute_with_medians(
    missing_dates: list, cols: list, df: pd.DataFrame
) -> pd.DataFrame:
    """
    Imputes values in missing_dates with medians for particular year-month-weekday
    """
    medians = pd.DataFrame(index=cols, columns=missing_dates)
    for i in missing_dates:
        cond_1 = df.index.year == i.year
        cond_2 = df.index.month == i.month
        cond_3 = df.index.weekday == i.weekday()
        medians.loc[:, i] = df.loc[cond_1 & cond_2 & cond_3, cols].median()

        df.loc[df.index.isin(missing_dates), cols] = medians.T

    return df


def impute_by_regression(correlated_shops: dict, df: pd.DataFrame) -> pd.DataFrame:
    for i in correlated_shops.keys():
        cols = correlated_shops[i]
        model = LinearRegression()
        model.fit(df.loc[~df[i].isnull(), cols], df.loc[~df[i].isnull(), i])
        predictions = model.predict(df.loc[df[i].isnull(), cols])
        df.loc[df[i].isnull(), i] = predictions

    return df
