import pandas as pd
import math
import scipy.stats as st


class JohnsonSU:
    """
    Johnson SU transformation according to
    https://en.wikipedia.org/wiki/Johnson%27s_SU-distribution
    """

    def __init__(self):
        self.johnsonsu_params = dict()

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
