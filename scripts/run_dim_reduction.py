import argparse
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

from coco_pipe.dim_reduction import DimReductionPipeline
from coco_pipe.io.load import load

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def run_job(cfg: dict) -> dict:
    pipeline = DimReductionPipeline(**cfg)
    out = pipeline.execute()
    return {"method": pipeline.method, "output": str(out)}

def main():
    parser = argparse.ArgumentParser(
        description="Run Dim Reduction on M/EEG, CSV, or Embedding data."
    )

    parser.add_argument(
        "--config", "-c", required=True, type=Path, help="Path to YAML config file"
    )
    args = parser.parse_args()
    if not args.config.exists():
        logger.error(f"Config file {args.config} does not exist.")
        return

    cfg = yaml.safe_load(args.config.read_text())
    data_cfg = cfg.get("data", {})
    reducers_cfg = cfg.get("reducers", [])
    parallel = cfg.get("parallel", False)
    summary_path = Path(cfg.get("summary_path", "dr_summary.json"))

    if not reducers_cfg:
        logger.error("No reducers defined under 'reducers' in config")
        return

    # Build job configs
    jobs = []
    for r in reducers_cfg:
        job = {
            "type":           data_cfg.get("type", "embeddings"),
            "method":         r["method"],
            "data_path":      data_cfg["data_path"],
            "n_components":   r.get("n_components", 2),
            "reducer_kwargs": r.get("reducer_kwargs", {}),
            "task":           data_cfg.get("task"),
            "run":            data_cfg.get("run"),
            "processing":     data_cfg.get("processing"),
            "subjects":       data_cfg.get("subjects"),
            "max_seg":        data_cfg.get("max_seg"),
            "sensorwise":     data_cfg.get("sensorwise", False),
            "flatten":        data_cfg.get("flatten", False),
        }
        
        # Handle sessions parameter - can be a string or a list
        if "sessions" in data_cfg:
            job["session"] = data_cfg["sessions"] 
        elif "session" in data_cfg:
            job["session"] = data_cfg["session"]
            
        jobs.append(job)

    results = []
    if parallel and len(jobs) > 1:
        logger.info(f"Running {len(jobs)} jobs in parallel")
        with ProcessPoolExecutor() as exe:
            future_to_job = {exe.submit(run_job, job): job for job in jobs}
            for fut in as_completed(future_to_job):
                job = future_to_job[fut]
                try:
                    res = fut.result()
                    logger.info(f"{res['method']} → {res['output']}")
                except Exception as e:
                    logger.error(f"Job {job['method']} failed: {e}")
                    res = {"method": job["method"], "error": str(e)}
                results.append(res)
    else:
        logger.info(f"Running {len(jobs)} jobs sequentially")
        for job in jobs:
            try:
                res = run_job(job)
                logger.info(f"{res['method']} → {res['output']}")
            except Exception as e:
                logger.error(f"Job {job['method']} failed: {e}")
                res = {"method": job["method"], "error": str(e)}
            results.append(res)

    logger.info(f"Writing summary to {summary_path}")
    summary_path.write_text(json.dumps(results, indent=2))
    logger.info("All tasks complete.")


if __name__ == "__main__":
    main()