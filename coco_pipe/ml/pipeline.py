"""
coco_pipe/ml/pipeline.py
----------------
Wrapper for ML pipelines.

Author: Hamza Abdelhedi <hamza.abdelhedii@gmail.com>
Date: 2025-05-18
Version: 0.0.1
License: TBD
"""
from .classification import ClassificationPipeline
from .regression    import RegressionPipeline

class MLPipeline:
    """
    Wrapper for ML pipelines.

    Parameters
    ----------
        :X: np.ndarray, Feature matrix, array-like of shape (n_samples, n_features)
        :y: np.ndarray, Target vector, array-like of shape (n_samples,)
        :config: dict, Configuration dictionary

    Returns
    -------
        :dict, Results dictionary
    """
    def __init__(self, X, y, config):
        self.X = X
        self.y = y
        self.config = config
        self.task = config.get("task")
        if self.task == "regression":
            self.pipeline = RegressionPipeline
        elif self.task == "classification"   :
            self.pipeline = ClassificationPipeline
        else:
            raise ValueError(f"Invalid task: {self.task}")

        # Instantiate and run pipeline
        # Extract cv_kwargs first to avoid duplicate parameters
        cv_kwargs = self.config.get("cv_kwargs", {})
        # Remove cv parameters from cv_kwargs if they exist to avoid duplicates
        cv_kwargs.pop("cv_strategy", None)
        cv_kwargs.pop("n_splits", None)
        cv_kwargs.pop("random_state", None)
        self.cv_kwargs = cv_kwargs

    def run(self):
        """
        Run the ML pipeline.
        """

        pipeline = self.pipeline(
            X=self.X, y=self.y,
            analysis_type=self.config.get("analysis_type", "baseline"),
            models=self.config.get("models", "all"),
            metrics=self.config.get("metrics", None),
            random_state=self.config.get("random_state", 42),
            cv_strategy=self.config.get("cv_strategy", "stratified"),
            n_splits=self.config.get("n_splits", 5),
            n_features=self.config.get("n_features", None),
            direction=self.config.get("direction", "forward"),
            search_type=self.config.get("search_type", "grid"),
            n_iter=self.config.get("n_iter", 100),
            scoring=self.config.get("scoring", None),
            n_jobs=self.config.get("n_jobs", -1),
            save_intermediate=self.config.get("save_intermediate", False),
            results_dir=self.config.get("results_dir", "results"),
            results_file=self.config.get("results_file", "results"),
            cv_kwargs=self.cv_kwargs
        )
        return pipeline.run()