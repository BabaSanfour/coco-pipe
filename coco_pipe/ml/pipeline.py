from .classification import ClassificationPipeline
from .regression    import RegressionPipeline

class MLPipeline:
    def __init__(self, X, y, config):
        self.X = X
        self.y = y
        self.config = config

    def run(self):
        task = self.config.get("task")
        
        if task == "regression":
            pipeline = RegressionPipeline
        elif task == "classification"   :
            pipeline = ClassificationPipeline
        else:
            raise ValueError(f"Invalid task: {task}")

        # Instantiate and run pipeline
        # Extract cv_kwargs first to avoid duplicate parameters
        cv_kwargs = self.config.get("cv_kwargs", {})
        # Remove cv parameters from cv_kwargs if they exist to avoid duplicates
        cv_kwargs.pop("cv_strategy", None)
        cv_kwargs.pop("n_splits", None)
        cv_kwargs.pop("random_state", None)

        pipeline = pipeline(
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
            cv_kwargs=cv_kwargs
        )
        return pipeline.run()