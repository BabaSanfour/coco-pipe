import pandas as pd
from sklearn import config_context
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class SubjectStandardScaler(BaseEstimator, TransformerMixin):
    """
    Standardization Strategy:
    1. Global standardization across all samples (StandardScaler).
    2. Per-Subject centering (Mean subtraction within each subject).
    """

    def __init__(self):
        self.global_scaler = StandardScaler()
        with config_context(enable_metadata_routing=True):
            self.set_transform_request(groups=True)

    def fit(self, X, y=None, groups=None):
        self.global_scaler.fit(X)
        return self

    def transform(self, X, groups=None):
        X_scaled = self.global_scaler.transform(X)
        if groups is None:
            return X_scaled
        df = pd.DataFrame(X_scaled)
        means = df.groupby(groups).transform("mean")
        return X_scaled - means.values

    def fit_transform(self, X, y=None, groups=None):
        return self.fit(X, y, groups).transform(X, groups=groups)
