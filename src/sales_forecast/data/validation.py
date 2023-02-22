import logging
import pandas as pd
from typing import Optional, Tuple

__all__ = ["train_validation_split"]

logger = logging.getLogger()

OUTPUT_SEQUENCE_LENGTH = 27
SHIFT = 1
N = 155


def train_validation_split(
    data: pd.DataFrame,
    output_sequence_length: Optional[int] = None,
    shift: Optional[int] = None,
    n: Optional[int] = None,
) -> Tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    1st half of 2016 and 1st half of 2017

    Input:
    data - dataset for split
    output_sequence_length - the length of forecasting period
    shift - shift in weekdays in 2017 compared to 2016
    n - the number of days of the year available for modeling

    Output:
    train_df valid_df - train and validation dataset
    train_df_past valid_df_past - parts of 2016 data to be used as features
    """

    logging.info("Train-validation split...")

    if output_sequence_length is None:
        output_sequence_length = OUTPUT_SEQUENCE_LENGTH
    if shift is None:
        shift = SHIFT
    if n is None:
        n = N

    train_df = data.iloc[-n : -output_sequence_length - shift, :]
    valid_df = data.iloc[-output_sequence_length - shift : -shift, :]

    train_df_past = data.iloc[shift:n, :]
    valid_df_past = data.iloc[n : n + output_sequence_length, :].fillna(0)

    return train_df, valid_df, train_df_past, valid_df_past
