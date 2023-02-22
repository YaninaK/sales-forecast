import logging
import pandas as pd
import math
import scipy.stats as st
from typing import Optional

__all__ = ["Johnson_SU_transformation"]

logger = logging.getLogger()

N_SHOPS = 20


class JohnsonSU:
    """
    Johnson SU transformation according to
    https://en.wikipedia.org/wiki/Johnson%27s_SU-distribution
    """

    def __init__(self, n_shops: Optional[int] = None):
        if n_shops is None:
            n_shops = N_SHOPS

        self.johnsonsu_params = dict()
        self.n_shops = n_shops

    def johnsonsu_transform(self, x, a, b, loc, scale):
        return a + b * math.asinh((x - loc) / scale)

    def johnsonsu_inv_transform(self, x, a, b, loc, scale):
        return loc + scale * math.sinh((x - a) / b)

    def fit(self, X):
        X = X.copy()
        for i in range(n_shops):
            self.johnsonsu_params[i] = st.johnsonsu.fit(X[i])

    def transform(self, X):
        X = X.copy()
        for i in range(n_shops):
            a, b, loc, scale = self.johnsonsu_params[i]
            X[i] = X[i].apply(self.johnsonsu_transform, args=(a, b, loc, scale))

        return X

    def inv_transform(self, X):
        X = X.copy()
        for i in range(n_shops):
            a, b, loc, scale = self.johnsonsu_params[i]
            X[i] = X[i].apply(self.johnsonsu_inv_transform, args=(a, b, loc, scale))

        return X
