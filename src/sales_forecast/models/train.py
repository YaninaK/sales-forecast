import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "sales_forecast"))

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional

from data.make_dataset import get_dataset
from data.impute_data import impute
from data.validation import train_validation_split
from data.johnson_su_transformation import JohnsonSU
from data.clean_data import clean_outliers
from features.time_series_clusters import get_clusters
from features.features_tsfresh import get_tsfresh_features
from features.build_dataset import get_features
from .save_artifacts import (
    save_time_split,
    save_scaler_js,
    save_clusters,
    save_extracted_features,
    save_train_dataset,
)

SAVE_ARTIFACTS = True


def data_preprocessing_pipeline(
    data: pd.DataFrame,
    save_artifacts: Optional[bool] = None,
) -> pd.DataFrame:

    if save_artifacts is None:
        save_artifacts = SAVE_ARTIFACTS

    data = get_dataset(data)
    data = impute(data)
    train_df, valid_df, train_df_past, valid_df_past = train_validation_split(data)

    scaler_js = JohnsonSU()
    scaler_js.fit(train_df.astype(float))
    train_df = scaler_js.transform(train_df.astype(float))
    valid_df = scaler_js.transform(valid_df.astype(float))
    X_scaled = pd.concat([train_df, valid_df], axis=0)
    X_scaled = clean_outliers(X_scaled)

    scaler_js_past = JohnsonSU()
    scaler_js_past.fit(train_df_past.astype(float))
    X_past_scaled = scaler_js_past.transform(train_df_past.astype(float))

    clusters = get_clusters(X_scaled)

    extracted_features = get_tsfresh_features(X_scaled)
    extracted_features_combined = pd.concat([clusters, extracted_features], axis=1)
    sd_scaler = StandardScaler()
    extracted_features_scaled = sd_scaler.fit_transform(extracted_features_combined)

    X = get_features(X_scaled, X_past_scaled, clusters, extracted_features_scaled)

    if save_artifacts:
        save_time_split(train_df, valid_df, train_df_past, valid_df_past)
        save_scaler_js(scaler_js)
        save_clusters(clusters)
        save_extracted_features(extracted_features)
        save_train_dataset(X)

    return X
