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
            P = RegressionPipeline
        elif task == "classification"   :
            P = ClassificationPipeline
        else:
            raise ValueError(f"Invalid task: {task}")

        # Instantiate and run pipeline
        pipeline = P(
            X=self.X, y=self.y,
            models=self.config.get("models", "all"),
            metrics=self.config.get("metrics", None), 
            random_state=self.config.get("random_state", 42),
            cv_kwargs=self.config.get("cv_kwargs"),
            n_jobs=self.config.get("n_jobs", -1),
        )
        return pipeline.execute(type=self.config.get("type", "baseline"),
                              n_features=self.config.get("n_features"),
                              direction=self.config.get("direction"), 
                              search_type=self.config.get("search_type"),
                              n_iter=self.config.get("n_iter"),
                              scoring=self.config.get("scoring"),
                              )