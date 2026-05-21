import numpy as np
import pandas as pd
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

    def fit(self, X, y=None, groups=None):
        self.global_scaler.fit(X)
        return self

    def transform(self, X, groups=None):
        X_scaled = self.global_scaler.transform(X)
        target_groups = groups if groups is not None else getattr(self, '_temp_groups', None)
        
        if target_groups is None:
            return X_scaled
            
        # Convert to DF to perform grouped operation
        # Note: transform('mean') already aligns with the index of the DF
        df = pd.DataFrame(X_scaled)
        means = df.groupby(target_groups).transform('mean')
        
        # Symmetrical centering: x - mean(x_group)
        X_centered = X_scaled - means.values
        return X_centered

    def fit_transform(self, X, y=None, groups=None):
        return self.fit(X, y, groups).transform(X, groups=groups)
