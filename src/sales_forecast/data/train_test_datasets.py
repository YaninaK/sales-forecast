import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple


__all__ = ["build_train_datset_for_LSTM_model"]

logger = logging.getLogger()

INPUT_SEQUENCE_LENGTH = 54
OUTPUT_SEQUENCE_LENGTH = 27
FINAL = False
DEFAULT_RANDOM_SEED = 25


def get_train_dataset(
    X,
    input_sequence_length: Optional[int] = None,
    output_sequence_length: Optional[int] = None,
    final: Optional[bool] = None,
    seed: Optional[int] = None,
):

    logging.info("Generating train datasets for LSTM model...")

    if input_sequence_length is None:
        input_sequence_length = INPUT_SEQUENCE_LENGTH
    if output_sequence_length is None:
        output_sequence_length = OUTPUT_SEQUENCE_LENGTH
    if final is None:
        final = FINAL
    if seed is None:
        seed = DEFAULT_RANDOM_SEED

    np.random.seed(seed)

    if final:
        n_train = X.shape[0]
    else:
        n_train = X.shape[0] - output_sequence_length

    X_train, y_train = [], []
    n = n_train - (input_sequence_length + output_sequence_length)
    list_train = list(range(n))
    np.random.shuffle(list_train)

    for t0 in list_train:
        t1 = t0 + input_sequence_length
        t2 = t1 + output_sequence_length
        for j in range(X.shape[1]):
            input_ = X[t0:t1, j, :]
            output_ = X[t1:t2, j, :]
            X_train.append(input_)
            y_train.append(output_)

    return np.array(X_train), np.array(y_train)
