import logging
import pandas as pd
from typing import Optional


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
