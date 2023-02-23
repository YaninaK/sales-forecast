import logging
import pandas as pd
import numpy as np
from typing import Optional

from .impute_data import impute_with_medians


logger = logging.getLogger(__name__)

__all__ = ["load_train_dataset"]


LOCAL_PATH = ""
DATA_PATH = "data/01_raw/train.parquet.gzip"


def load_data(
    local_path: Optional[str] = None, data_path: Optional[str] = None
) -> pd.DataFrame:

    logging.info(f"Reading dataset from {data_path}...")

    if local_path is None:
        local_path = LOCAL_PATH
    if data_path is None:
        data_path = local_path + DATA_PATH

    data = pd.read_parquet(data_path)

    return data


def get_dataset(train: pd.DataFrame) -> pd.DataFrame:
    """
    Add missing dates
    Switch Jan and Feb 2016 days-off from Tuesdays to Wednesdays
    Drop days-off
    Impute data in missing dates with medians for particular year-month-weekday
    """
    n_shops = train["id"].nunique()
    train["dt"] = pd.to_datetime(train["dt"])
    data = pd.pivot_table(
        train, values="target", index=["dt"], columns=["id"], aggfunc=np.sum
    ).reset_index(level=-1)
    data.columns = ["dt"] + list(range(n_shops))
    data.set_index("dt", inplace=True)

    # Add missing dates
    calendar_days = pd.date_range(start=data.index[0], end=data.index[-1])
    df = (
        pd.DataFrame(calendar_days, columns=["dt"])
        .merge(data, on="dt", how="left")
        .set_index("dt")
    )
    # Switch Jan and Feb 2016 days-off from Tuesdays to Wednesdays
    ind = df[:59][df[:59].index.weekday == 1].index
    tmp = df[:59][df[:59].index.weekday == 2]
    tmp.index = ind
    df[:59][df[:59].index.weekday == 1] = tmp

    # Drop days-off
    data = df.drop(df[df.index.weekday == 2].index)

    # Impute data in missing dates with medians for particular year-month-weekday
    missing_dates = sorted(list(set(data.index) - set(train["dt"])))[8:]
    data = impute_with_medians(missing_dates, range(n_shops), data)

    return data
