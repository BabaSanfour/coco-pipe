from abc import ABC, abstractmethod
import os
import joblib

class BaseReducer(ABC):
    """
    Abstract base class for dimensionality reduction reducers.
    Provides fit, transform, fit_transform, save, and load methods.
    """
    @abstractmethod
    def fit(self, X, y=None):
        """Learn any model parameters from X (and optional y)."""
        pass

    @abstractmethod
    def transform(self, X):
        """Project X into the reduced space."""
        pass

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        self.fit(X, y=y)
        return self.transform(X)

    def save(self, filepath: str):
        """
        Persist this reducer’s fitted state to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        """
        Load a previously saved reducer from disk.
        """
        return joblib.load(filepath)