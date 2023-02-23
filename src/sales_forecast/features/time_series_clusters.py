import logging
import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from typing import Optional

__all__ = ["generate_clusters"]

logger = logging.getLogger()

DEFAULT_RANDOM_SEED = 3
N_CLUSTERS = 5


def get_clusters(
    X_scaled, n_clusters: Optional[int] = None, seed: Optional[int] = None
) -> pd.DataFrame:

    if seed is None:
        seed = DEFAULT_RANDOM_SEED
    if n_clusters is None:
        n_clusters = N_CLUSTERS

    ts_dtw = TimeSeriesKMeans(
        n_clusters=n_clusters, metric="dtw", n_jobs=6, max_iter=10, random_state=seed
    )
    ts_dtw.fit(X_scaled.T)

    n_shops = X_scaled.shape[1]
    clusters = pd.DataFrame(
        data=ts_dtw.labels_, index=range(n_shops), columns=["cluster"]
    )
    clusters = pd.get_dummies(clusters["cluster"])

    return clusters
