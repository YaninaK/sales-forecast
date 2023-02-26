import logging
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings
from typing import Optional

__all__ = ["get_tsfresh_features"]

logger = logging.getLogger()

N_SHOPS = 20


def get_tsfresh_features(ts: pd.DataFrame) -> pd.DataFrame:

    logging.info("Generating tsfresh features...")

    settings_comprehensive = settings.ComprehensiveFCParameters()
    tsfresh_dataset = get_tsfresh_dataset(ts)

    extracted_features = extract_features(
        tsfresh_dataset,
        column_id="id",
        column_sort="time",
        impute_function=impute,
        default_fc_parameters=settings_comprehensive,
    )
    for i in extracted_features.columns:
        col = extracted_features[i]
        if len(col.unique()) == 1:
            del extracted_features[i]

    return extracted_features


def get_tsfresh_dataset(
    ts: pd.DataFrame, n_shops: Optional[int] = None
) -> pd.DataFrame:
    if n_shops is None:
        n_shops = N_SHOPS

    df_tsfresh = pd.DataFrame(columns=["id", "time", "sales"])
    for i in range(n_shops):
        df = pd.DataFrame(columns=["id", "time", "sales"])
        df["sales"] = ts.iloc[:, i].values
        df["id"] = i
        df["time"] = ts.index.tolist()
        df_tsfresh = pd.concat([df_tsfresh, df], axis=0)

    return df_tsfresh
