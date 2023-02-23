import logging
import pandas as pd
import numpy as np
import joblib
from typing import Optional


__all__ = ["save_artifacts"]

logger = logging.getLogger()


PATH = ""
FOLDER_1 = "data/02_intermediate/"
TRAIN_DF_PATH = FOLDER_1 + "train_df.parquet.gzip"
VALID_DF_PATH = FOLDER_1 + "valid_df.parquet.gzip"
TRAIN_DF_PAST_PATH = FOLDER_1 + "train_df_past.parquet.gzip"
VALID_DF_PAST_PATH = FOLDER_1 + "valid_df_past.parquet.gzip"

FOLDER_2 = "data/03_primary/"
SCALER_JS_PATH = FOLDER_2 + "scaler_js.joblib"

FOLDER_3 = "data/04_feature/"
CLUSTERS_PATH = FOLDER_3 + "clusters.parquet.gzip"
EXTRACTED_FEATURES_PATH = FOLDER_3 + "extracted_features.parquet.gzip"
TRAIN_DATASET_PATH = FOLDER_3 + "train_dataset"


def save_time_split(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    train_df_past: pd.DataFrame,
    valid_df_past: pd.DataFrame,
    path: Optional[str] = None,
    train_df_path: Optional[str] = None,
    valid_df_path: Optional[str] = None,
    train_df_past_path: Optional[str] = None,
    valid_df_past_path: Optional[str] = None,
):
    if path is None:
        path = PATH
    if train_df_path is None:
        train_df_path = path + TRAIN_DF_PATH
    if valid_df_path is None:
        valid_df_path = path + VALID_DF_PATH
    if train_df_past_path is None:
        train_df_past_path = path + TRAIN_DF_PAST_PATH
    if valid_df_past_path is None:
        valid_df_past_path = path + VALID_DF_PAST_PATH

    cols = [str(i) for i in range(train_df.shape[1])]
    train_df.columns = cols
    train_df.to_parquet(train_df_path, compression="gzip")
    valid_df.columns = cols
    valid_df.to_parquet(valid_df_path, compression="gzip")
    train_df_past.columns = cols
    train_df_past.to_parquet(train_df_past_path, compression="gzip")
    valid_df_past.columns = cols
    valid_df_past.to_parquet(valid_df_past_path, compression="gzip")


def save_scaler_js(
    scaler_js, path: Optional[str] = None, scaler_js_path: Optional[str] = None
):
    if path is None:
        path = PATH
    if scaler_js_path is None:
        scaler_js_path = path + SCALER_JS_PATH

    joblib.dump(scaler_js, scaler_js_path, compress=3)


def save_clusters(
    clusters: pd.DataFrame,
    path: Optional[str] = None,
    clusters_path: Optional[str] = None,
):
    if path is None:
        path = PATH
    if clusters_path is None:
        clusters_path = path + CLUSTERS_PATH

    clusters.to_parquet(clusters_path, compression="gzip")


def save_extracted_features(
    extracted_features: pd.DataFrame,
    path: Optional[str] = None,
    extracted_features_path: Optional[str] = None,
):
    if path is None:
        path = PATH
    if extracted_features_path is None:
        extracted_features_path = path + EXTRACTED_FEATURES_PATH


def save_train_dataset(
    X, path: Optional[str] = None, train_dataset_path: Optional[str] = None
):
    if path is None:
        path = PATH
    if train_dataset_path is None:
        train_dataset_path = path + TRAIN_DATASET_PATH

    np.save(train_dataset_path, X)
