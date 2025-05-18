import yaml
import pandas as pd
from coco_pipe.io import select_features
from coco_pipe.ml.pipeline import MLPipeline

def main():
    # 0) Load config & data
    cfg = yaml.safe_load(open("experiment.yaml"))
    df  = pd.read_csv(cfg["csv_path"])

    # 1) Select features & target
    X, y = select_features(
        df,
        target_columns=cfg["target_columns"],
        covariates=cfg.get("covariates"),
        spatial_units=cfg.get("spatial_units"),
        feature_names=cfg.get("feature_names", "all"),
        row_filter=cfg.get("row_filter"),
    )

    # 2) Pick and run the right pipeline
    pipeline = MLPipeline(X=X, y=y, config=cfg)
    results = pipeline.run()

    # 3) Dump `results` wherever you like
    pd.to_pickle(results, cfg.get("results_file", "results.pkl"))

if __name__ == "__main__":
    main()