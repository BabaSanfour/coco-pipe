from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._constants import RESULT_SCHEMA_VERSION
from ._diagnostics import (
    confusion_matrix_frame,
    curve_score_groups,
    prediction_rows,
    proba_matrix,
    scalar_prediction_frame,
    score_rows,
)

logger = logging.getLogger(__name__)


class ExperimentResult:
    """
    Unified Container for Experiment Results.

    Provides tidy data views for easier analysis, visualization, and
    statistical assessment of decoding performance across multiple models,
    folds, and temporal coordinates.

    Examples
    --------
    >>> result = Experiment(config).run(X, y)
    >>> summary_df = result.summary()
    >>> preds_df = result.get_predictions()
    """

    def __init__(
        self,
        raw_results: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        time_axis: Optional[Sequence[Any]] = None,
        schema_version: str = RESULT_SCHEMA_VERSION,
    ):
        """
        Initialize the ExperimentResult container.

        This object serves as the primary interface for exploring, visualizing,
        and validating decoding results. It encapsulates raw metrics,
        predictions, and cross-validation splits into a unified structure.

        Parameters
        ----------
        raw_results : dict
            The raw results dictionary returned by the Experiment engine,
            keyed by model name.
        config : dict, optional
            The configuration dictionary used for the experiment.
        meta : dict, optional
            Additional metadata (e.g., sample IDs, unit of inference, versions).
        time_axis : sequence, optional
            The scientific time points (e.g., in seconds or ms) corresponding
            to the temporal coordinates in the results.
        schema_version : str, optional
            The version of the result schema for forward compatibility.
        """
        self.raw = raw_results
        self.config = config or {}
        self.meta = meta or {}
        self.schema_version = schema_version

        # Explicit time axis resolution
        self._time_axis_cache = None
        if time_axis is not None:
            self._time_axis_cache = list(time_axis)
        elif "time_axis" in self.meta:
            t = self.meta["time_axis"]
            self._time_axis_cache = list(t) if t is not None else None

    @property
    def time_axis(self) -> Optional[list[Any]]:
        """The scientific time points for temporal decoding results."""
        return self._time_axis_cache

    def to_payload(self, serializable: bool = False) -> Dict[str, Any]:
        """
        Return the result payload for persistence or transmission.

        Converts the internal state into a dictionary containing the schema
        version, configuration, metadata, and raw model results.

        Parameters
        ----------
        serializable : bool, default=False
            If True, recursively converts all NumPy arrays, integers, floats,
            and booleans into standard Python primitives (lists, ints, etc.)
            suitable for JSON serialization.

        Returns
        -------
        payload : dict
            The consolidated result payload.

        See Also
        --------
        ExperimentResult.save : Persist results to disk.
        """
        payload = {
            "schema_version": self.schema_version,
            "config": self.config,
            "meta": self.meta,
            "results": self.raw,
        }
        return make_serializable(payload) if serializable else payload

    def save(self, path: Optional[Union[str, Path, Any]] = None, indent: int = 2):
        """
        Save results to a file, auto-detecting the format from the extension.

        Supports both binary formats (via joblib) for speed and disk space,
        and JSON format for interoperability and human-readability.

        Parameters
        ----------
        path : str or Path, optional
            The destination path.
            - If None, uses 'output_dir' from the experiment config.
            - If a directory, generates a timestamped filename with a '.pkl' extension.
            - If a file path ending in '.json', performs JSON serialization.
            - Otherwise, uses joblib binary serialization.
        indent : int, default=2
            JSON indentation level (only applicable for .json files).

        Returns
        -------
        path : Path
            The path where the results were saved.

        See Also
        --------
        ExperimentResult.load : Load results from disk.
        Experiment.save_results : Experiment-level wrapper.
        """
        from datetime import datetime

        if path is None:
            path = self.config.get("output_dir", ".")

        path = Path(path)
        if path.suffix == "" or path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
            tag = self.config.get("tag", "result")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = path / f"{tag}_{ts}.pkl"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".json":
            import json

            payload = self.to_payload(serializable=True)
            with open(path, "w") as f:
                json.dump(payload, f, indent=indent)
        else:
            import joblib

            joblib.dump(self.to_payload(), path)

        return path

    @classmethod
    def load(cls, path: Union[str, Path, Any]) -> "ExperimentResult":
        """
        Load results from a file (auto-detects JSON or Pickle).

        Reconstructs an ExperimentResult instance from a previously saved
        payload on disk.

        Parameters
        ----------
        path : str or Path
            The path to the result file.

        Returns
        -------
        result : ExperimentResult
            The rehydrated result container.

        Raises
        ------
        FileNotFoundError
            If the specified path does not exist.
        ValueError
            If the file format is unrecognized or corrupted.

        See Also
        --------
        ExperimentResult.save : Persist results to disk.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {path}")

        if path.suffix == ".json":
            import json

            with open(path, "r") as f:
                payload = json.load(f)
        else:
            import joblib

            payload = joblib.load(path)

        return cls(
            raw_results=payload["results"],
            config=payload.get("config"),
            meta=payload.get("meta"),
            schema_version=payload.get("schema_version", RESULT_SCHEMA_VERSION),
        )

    def summary(self) -> pd.DataFrame:
        """
        Get a high-level summary of performance (Mean/Std and Stats).

        Aggregates results across all models and folds into a single
        benchmarking table.

        Scientific Rationale
        --------------------
        A summary table provides a concise overview of model performance
        expectations and their reliability. By including standard deviations
        and p-values alongside means, it allows for immediate identification
        of significant decoding effects and model stability.

        Returns
        -------
        summary_df : pd.DataFrame
            DataFrame with models as index and scalar metrics as columns.
            Includes p-values and significance markers ('*') if statistical
            assessments were executed.

        Examples
        --------
        >>> # df = result.summary()
        >>> # print(df[['accuracy_mean', 'accuracy_p_val']])

        See Also
        --------
        ExperimentResult.get_detailed_scores : Get fold-level results.
        ExperimentResult.get_temporal_score_summary : Temporal-resolved summary.
        """
        rows = []
        for model, res in self.raw.items():
            if "error" in res:
                continue

            # 1. Performance Metrics
            row = {"Model": model}
            for metric, stats in res.get("metrics", {}).items():
                mean = np.asarray(stats["mean"])
                std = np.asarray(stats["std"])
                if mean.ndim == 0 and std.ndim == 0:
                    row[f"{metric}_mean"] = float(mean)
                    row[f"{metric}_std"] = float(std)

            # 2. Statistical Assessment (if available)
            stats_rows = res.get("statistical_assessment", [])
            for s in stats_rows:
                # Only include scalar stats (where Time/TrainTime are None)
                if s.get("Time") is None and s.get("TrainTime") is None:
                    m = s.get("Metric")
                    p_val = s.get("PValue")
                    if p_val is not None:
                        row[f"{m}_p_val"] = p_val
                        # Add significance marker
                        if s.get("Significant"):
                            row[f"{m}_sig"] = "*"

            if len(row) > 1:
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("Model")
        # Sort columns to group Mean/Std/P-val together for each metric
        cols = sorted(df.columns)
        return df[cols]

    def get_detailed_scores(self, model: Optional[str] = None) -> pd.DataFrame:
        """
        Get fold-level scores for all models or a specific model in long format.

        Expands results into a 'tidy' format where each row represents a
        single score for one fold, model, and metric.

        Parameters
        ----------
        model : str, optional
            The name of the model to filter by. Default is None (all models).

        Returns
        -------
        scores_df : pd.DataFrame
            Tidy DataFrame with columns: Model, Fold, Metric, and Value.
            Includes temporal coordinates if the data is time-resolved.

        See Also
        --------
        ExperimentResult.summary : Mean/Std aggregate view.
        """
        rows = []
        target_models = [model] if model is not None else self.raw.keys()
        for m_name in target_models:
            res = self.raw.get(m_name, {})
            if "error" in res:
                continue
            metrics_data = res["metrics"]
            n_folds = len(next(iter(metrics_data.values()))["folds"])
            for fold_idx in range(n_folds):
                for metric, stats in metrics_data.items():
                    rows.extend(
                        score_rows(
                            m_name,
                            fold_idx,
                            metric,
                            stats["folds"][fold_idx],
                            time_axis=self.time_axis,
                        )
                    )
        return pd.DataFrame(rows)

    def get_temporal_score_summary(self, model: Optional[str] = None) -> pd.DataFrame:
        """
        Get temporal metric means/stds and significance across folds.

        Averages performance metrics across cross-validation folds for each
        temporal coordinate (Time or TrainTime/TestTime pair).

        Scientific Rationale
        --------------------
        Temporal decoding and time-generalization analysis yield multi-dimensional
        performance arrays. Aggregating these across folds provides an estimate of
        the central tendency and variance of the model's ability to decode at
        specific latency points. Integrating p-values into this view allows for
        the identification of 'significant' time windows.

        Parameters
        ----------
        model : str, optional
            The name of the model to filter by. Default is None (all models).

        Returns
        -------
        summary_df : pd.DataFrame
            DataFrame in long format with Model, Metric, Time coordinates,
            Mean, Std, and PValue/Significant columns if statistical
            assessments were executed.

        See Also
        --------
        ExperimentResult.summary : Scalar-only summary view.
        ExperimentResult.get_generalization_matrix : 2D matrix view of TG results.
        """
        rows = []
        columns = [
            "Model",
            "Metric",
            "Time",
            "TrainTime",
            "TestTime",
            "Mean",
            "Std",
            "PValue",
            "Significant",
        ]

        target_models = [model] if model is not None else self.raw.keys()
        for m_name in target_models:
            res = self.raw.get(m_name, {})
            if "error" in res:
                continue

            # 1. Base Metrics (Mean/Std across folds)
            for metric, stats in res.get("metrics", {}).items():
                folds = [np.asarray(fold) for fold in stats.get("folds", [])]
                if not folds or folds[0].ndim == 0:
                    continue
                stack = np.stack(folds)
                mean = np.nanmean(stack, axis=0)
                std = np.nanstd(stack, axis=0)

                # Prepare stats lookup for this model/metric
                # Using a dict for O(1) lookup: (Time) or (TrainTime, TestTime) -> row
                stats_rows = res.get("statistical_assessment", [])
                stats_lookup = {}
                for s in stats_rows:
                    if s.get("Metric") == metric:
                        key = (s.get("Time"), s.get("TrainTime"), s.get("TestTime"))
                        stats_lookup[key] = s

                if mean.ndim == 1:
                    for t_idx, val in enumerate(mean):
                        t_val = self._time_value(t_idx)
                        s_row = stats_lookup.get((t_val, None, None), {})
                        rows.append(
                            {
                                "Model": m_name,
                                "Metric": metric,
                                "Time": t_val,
                                "Mean": val,
                                "Std": std[t_idx],
                                "PValue": s_row.get("PValue"),
                                "Significant": s_row.get("Significant", False),
                            }
                        )
                elif mean.ndim == 2:
                    for t_tr in range(mean.shape[0]):
                        tr_val = self._time_value(t_tr)
                        for t_te in range(mean.shape[1]):
                            te_val = self._time_value(t_te)
                            s_row = stats_lookup.get((None, tr_val, te_val), {})
                            rows.append(
                                {
                                    "Model": m_name,
                                    "Metric": metric,
                                    "TrainTime": tr_val,
                                    "TestTime": te_val,
                                    "Mean": mean[t_tr, t_te],
                                    "Std": std[t_tr, t_te],
                                    "PValue": s_row.get("PValue"),
                                    "Significant": s_row.get("Significant", False),
                                }
                            )
        return pd.DataFrame(rows, columns=columns)

    def get_predictions(self, model: Optional[str] = None) -> pd.DataFrame:
        """
        Get concatenated predictions for all models or a specific model.

        Converts nested prediction dictionaries from all folds into a single
        flattened DataFrame.

        Parameters
        ----------
        model : str, optional
            The name of the model to filter by. Default is None (all models).

        Returns
        -------
        predictions_df : pd.DataFrame
            Tidy DataFrame of predictions. Includes SampleID, y_true, y_pred,
            y_score, and probability columns if available.

        See Also
        --------
        ExperimentResult.get_splits : Membership of samples in each fold.
        """
        rows = []
        time_axis = self.time_axis
        target_models = [model] if model is not None else self.raw.keys()

        for m_name in target_models:
            res = self.raw.get(m_name, {})
            if "error" in res or "predictions" not in res:
                continue
            for fold_idx, preds in enumerate(res["predictions"]):
                rows.extend(
                    prediction_rows(m_name, fold_idx, preds, time_axis=time_axis)
                )
        return pd.DataFrame(rows)

    def get_splits(self, model: Optional[str] = None) -> pd.DataFrame:
        """
        Get outer-CV train/test membership in long format for all models.

        Tracks which samples were used for training and testing in each fold
        of the cross-validation procedure.

        Parameters
        ----------
        model : str, optional
            The name of the model to filter by. Default is None (all models).

        Returns
        -------
        splits_df : pd.DataFrame
            DataFrame with Model, Fold, Set (train/test), SampleIndex, SampleID,
            and associated metadata columns (e.g., Subject, Session).

        See Also
        --------
        ExperimentResult.get_predictions : Link predictions to splits via SampleID.
        """
        from ._diagnostics import optional_values

        all_rows = []
        target_models = [model] if model is not None else self.raw.keys()
        for m_name in target_models:
            res = self.raw.get(m_name, {})
            if "error" in res:
                continue
            for fold_idx, split in enumerate(res.get("splits", [])):
                for set_name, idx_key, id_key, group_key, meta_key in [
                    (
                        "train",
                        "train_idx",
                        "train_sample_id",
                        "train_group",
                        "train_metadata",
                    ),
                    (
                        "test",
                        "test_idx",
                        "test_sample_id",
                        "test_group",
                        "test_metadata",
                    ),
                ]:
                    indices = np.asarray(split[idx_key])
                    n = len(indices)
                    if n == 0:
                        continue

                    ids = np.asarray(split[id_key])
                    groups = optional_values(split.get(group_key), n)
                    metadata = split.get(meta_key) or {}

                    # Flatten metadata into columns
                    meta_arrays = {
                        k: np.asarray(v, dtype=object)[:n] for k, v in metadata.items()
                    }

                    for i in range(n):
                        row = {
                            "Model": m_name,
                            "Fold": fold_idx,
                            "Set": set_name,
                            "SampleIndex": indices[i],
                            "SampleID": ids[i],
                            "Group": groups[i],
                        }
                        # Add metadata columns
                        for k, v_arr in meta_arrays.items():
                            row[k] = v_arr[i]
                        all_rows.append(row)

        if not all_rows:
            return pd.DataFrame()

        return pd.DataFrame(all_rows)

    def get_fit_diagnostics(self, model: Optional[str] = None) -> pd.DataFrame:
        """
        Get fold-level timing and warning diagnostics for all models.

        Aggregates operational metrics such as execution time and runtime
        warnings encountered during the model fit and predict stages.

        Scientific Rationale
        --------------------
        Runtime diagnostics are critical for identifying computational
        bottlenecks and ensuring model validity. Long training times may
        suggest the need for dimensionality reduction, while consistent
        warnings (e.g., convergence failures) can signal that model
        hyperparameters are poorly suited to the dataset.

        Parameters
        ----------
        model : str, optional
            The name of the model to filter by. Default is None (all models).

        Returns
        -------
        diagnostics_df : pd.DataFrame
            DataFrame with Model, Fold, and timing columns (FitTime, PredictTime,
            ScoreTime, TotalTime). If warnings were captured, includes Stage,
            WarningCategory, and WarningMessage columns.

        Examples
        --------
        >>> diagnostics = result.get_fit_diagnostics()
        >>> # Identify the slowest model
        >>> slow_model = diagnostics.groupby("Model")["TotalTime"].mean().idxmax()

        See Also
        --------
        ExperimentResult.summary : General performance summary.
        """
        rows = []
        columns = [
            "Model",
            "Fold",
            "FitTime",
            "PredictTime",
            "ScoreTime",
            "TotalTime",
            "Stage",
            "WarningCategory",
            "WarningMessage",
        ]
        target_models = [model] if model is not None else self.raw.keys()
        for m_name in target_models:
            res = self.raw.get(m_name, {})
            if "error" in res:
                continue
            for fold_idx, diag in enumerate(res.get("diagnostics", [])):
                base = {
                    "Model": m_name,
                    "Fold": fold_idx,
                    "FitTime": diag.get("fit_time"),
                    "PredictTime": diag.get("predict_time"),
                    "ScoreTime": diag.get("score_time"),
                    "TotalTime": diag.get("total_time"),
                }
                warnings_ = diag.get("warnings") or []
                if not warnings_:
                    rows.append(
                        {
                            **base,
                            "Stage": None,
                            "WarningCategory": None,
                            "WarningMessage": None,
                        }
                    )
                    continue
                for w in warnings_:
                    rows.append(
                        {
                            **base,
                            "Stage": w.get("stage"),
                            "WarningCategory": w.get("category"),
                            "WarningMessage": w.get("message"),
                        }
                    )
        return pd.DataFrame(rows, columns=columns)

    def _build_confusion_df(
        self,
        model: Optional[str],
        labels: Optional[Sequence[Any]],
        normalize: Optional[str],
        group_cols: list[str],
    ) -> pd.DataFrame:
        """Shared logic for building confusion matrix DataFrames."""
        preds = scalar_prediction_frame(self.get_predictions(model=model))
        if preds.empty:
            cols = group_cols + ["TrueLabel", "PredictedLabel", "Value"]
            return pd.DataFrame(columns=cols)

        if labels is None:
            labels = sorted(
                pd.unique(pd.concat([preds["y_true"], preds["y_pred"]])).tolist()
            )

        return confusion_matrix_frame(preds, labels, normalize, group_cols=group_cols)

    def get_confusion_matrices(
        self,
        model: Optional[str] = None,
        labels: Optional[Sequence[Any]] = None,
        normalize: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get fold-level confusion matrices in long format.

        Computes the confusion between true and predicted labels for each
        cross-validation fold.

        Scientific Rationale
        --------------------
        Confusion matrices provide a granular view of model errors, identifying
        specific classes that are frequently misidentified. Analyzing these
        per-fold allows for assessing the consistency of error patterns across
        different data splits.

        Parameters
        ----------
        model : str, optional
            The name of the model to filter by. Default is None (all models).
        labels : sequence of any, optional
            The list of labels to use for the matrix axes. If None, uses all
            labels present in the predictions.
        normalize : {'true', 'pred', 'all'}, optional
            Normalization strategy:
            - 'true': Normalize by true labels (rows).
            - 'pred': Normalize by predicted labels (columns).
            - 'all': Normalize by total number of samples.

        Returns
        -------
        confusion_df : pd.DataFrame
            Tidy DataFrame with Model, Fold, TrueLabel, PredictedLabel, and Value.

        See Also
        --------
        ExperimentResult.get_pooled_confusion_matrix : Aggregate across folds.
        """
        return self._build_confusion_df(
            model=model,
            labels=labels,
            normalize=normalize,
            group_cols=["Model", "Fold"],
        )

    def get_confusion_counts(
        self, model: Optional[str] = None, labels: Optional[Sequence[Any]] = None
    ) -> pd.DataFrame:
        """
        Get unnormalized per-fold confusion counts.

        Equivalent to `get_confusion_matrices(normalize=None)`.

        Parameters
        ----------
        model : str, optional
            The name of the model to filter by.
        labels : sequence of any, optional
            The list of labels to use.

        Returns
        -------
        counts_df : pd.DataFrame
            Unnormalized confusion counts.
        """
        return self.get_confusion_matrices(model=model, labels=labels, normalize=None)

    def get_pooled_confusion_matrix(
        self,
        model: Optional[str] = None,
        labels: Optional[Sequence[Any]] = None,
        normalize: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get pooled out-of-fold confusion matrices in long format.

        Aggregates predictions from all cross-validation folds before
        calculating the confusion matrix.

        Parameters
        ----------
        model : str, optional
            The name of the model to filter by.
        labels : sequence of any, optional
            The list of labels to use.
        normalize : {'true', 'pred', 'all'}, optional
            Normalization strategy.

        Returns
        -------
        confusion_df : pd.DataFrame
            Pooled confusion matrix with Model, TrueLabel, PredictedLabel,
            and Value.

        See Also
        --------
        ExperimentResult.get_confusion_matrices : Fold-level view.
        """
        return self._build_confusion_df(
            model=model, labels=labels, normalize=normalize, group_cols=["Model"]
        )

    def get_roc_curve(
        self, model: Optional[str] = None, pos_label: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Get binary or one-vs-rest ROC curve coordinates.

        Calculates False Positive Rate (FPR) and True Positive Rate (TPR) at
        various thresholds for each fold. For multiclass problems, computes
        One-vs-Rest (OvR) curves for each class.

        Scientific Rationale
        --------------------
        Receiver Operating Characteristic (ROC) curves illustrate the
        diagnostic ability of a classifier as its discrimination threshold is
        varied. Analyzing the spread of these curves across folds helps in
        assessing the robustness of the model's probabilistic rankings.

        Parameters
        ----------
        model : str, optional
            The model name to filter by.
        pos_label : any, optional
            The label to treat as the positive class in binary cases. If None,
            uses the second class in alphabetical order.

        Returns
        -------
        roc_df : pd.DataFrame
            DataFrame with Model, Fold, Class, Threshold, FPR, and TPR.

        See Also
        --------
        ExperimentResult.get_roc_auc_summary : Aggregate AUC metrics.
        """
        from sklearn.metrics import roc_curve

        frames = []
        preds = scalar_prediction_frame(self.get_predictions(model=model))
        for m_name, f_idx, label, y_binary, y_score in curve_score_groups(
            preds, model=model, pos_label=pos_label
        ):
            fpr, tpr, thresholds = roc_curve(y_binary, y_score, pos_label=True)
            df_c = pd.DataFrame(
                {
                    "Model": m_name,
                    "Fold": f_idx,
                    "Class": label,
                    "Threshold": thresholds,
                    "FPR": fpr,
                    "TPR": tpr,
                }
            )
            frames.append(df_c)

        if not frames:
            return pd.DataFrame(
                columns=["Model", "Fold", "Class", "Threshold", "FPR", "TPR"]
            )
        return pd.concat(frames, ignore_index=True)

    def get_pr_curve(
        self, model: Optional[str] = None, pos_label: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Get binary or one-vs-rest precision-recall curve coordinates.

        Calculates Precision and Recall at various thresholds for each fold.
        For multiclass problems, computes One-vs-Rest (OvR) curves for each class.

        Scientific Rationale
        --------------------
        Precision-Recall (PR) curves are often more informative than ROC
        curves for imbalanced datasets, as they focus on the model's
        performance on the minority (positive) class.

        Parameters
        ----------
        model : str, optional
            The model name to filter by.
        pos_label : any, optional
            The label to treat as positive.

        Returns
        -------
        pr_df : pd.DataFrame
            DataFrame with Model, Fold, Class, Threshold, Precision, and Recall.

        See Also
        --------
        ExperimentResult.get_pr_auc_summary : Average Precision summary.
        """
        from sklearn.metrics import precision_recall_curve

        frames = []
        preds = scalar_prediction_frame(self.get_predictions(model=model))
        for m_name, f_idx, label, y_binary, y_score in curve_score_groups(
            preds, model=model, pos_label=pos_label
        ):
            precision, recall, thresholds = precision_recall_curve(
                y_binary, y_score, pos_label=True
            )
            # thresholds is 1 element shorter than precision/recall
            thresh_vals = np.append(thresholds, np.nan)
            df_c = pd.DataFrame(
                {
                    "Model": m_name,
                    "Fold": f_idx,
                    "Class": label,
                    "Threshold": thresh_vals,
                    "Precision": precision,
                    "Recall": recall,
                }
            )
            frames.append(df_c)

        if not frames:
            return pd.DataFrame(
                columns=["Model", "Fold", "Class", "Threshold", "Precision", "Recall"]
            )
        return pd.concat(frames, ignore_index=True)

    def get_roc_auc_summary(self, model: Optional[str] = None) -> pd.DataFrame:
        """
        Get summary ROC-AUC metrics across models and folds.

        Calculates the Area Under the ROC Curve for each fold, using macro- and
        weighted-averaging for multiclass tasks.

        Parameters
        ----------
        model : str, optional
            Model name to filter by.

        Returns
        -------
        auc_df : pd.DataFrame
            Summary with Model, Fold, MacroROCAUC, and WeightedROCAUC.

        See Also
        --------
        ExperimentResult.get_roc_curve : Detailed curve coordinates.
        """
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import LabelBinarizer

        rows = []
        preds = scalar_prediction_frame(self.get_predictions(model=model))
        if preds.empty:
            return pd.DataFrame(
                columns=["Model", "Fold", "MacroROCAUC", "WeightedROCAUC"]
            )

        proba_cols = sorted(
            [col for col in preds.columns if col.startswith("y_proba_")],
            key=lambda v: int(v.rsplit("_", 1)[-1]),
        )
        if not proba_cols:
            return pd.DataFrame(
                columns=["Model", "Fold", "MacroROCAUC", "WeightedROCAUC"]
            )

        for (m_name, f_idx), group in preds.groupby(["Model", "Fold"]):
            y_true = group["y_true"].to_numpy()
            y_proba = group[proba_cols].to_numpy()

            lb = LabelBinarizer()
            y_true_bin = lb.fit_transform(y_true)
            if y_true_bin.shape[1] == 1:
                score = roc_auc_score(y_true_bin, y_proba[:, -1])
                macro = weighted = score
            else:
                macro = roc_auc_score(
                    y_true_bin, y_proba, multi_class="ovr", average="macro"
                )
                weighted = roc_auc_score(
                    y_true_bin, y_proba, multi_class="ovr", average="weighted"
                )

            rows.append(
                {
                    "Model": m_name,
                    "Fold": f_idx,
                    "MacroROCAUC": float(macro),
                    "WeightedROCAUC": float(weighted),
                }
            )
        return pd.DataFrame(rows)

    def get_pr_auc_summary(self, model: Optional[str] = None) -> pd.DataFrame:
        """
        Get summary PR-AUC (Average Precision) metrics across models and folds.

        Calculates the Area Under the Precision-Recall Curve for each fold.

        Parameters
        ----------
        model : str, optional
            Model name to filter by.

        Returns
        -------
        auc_df : pd.DataFrame
            Summary with Model, Fold, MacroPRAUC, and WeightedPRAUC.

        See Also
        --------
        ExperimentResult.get_pr_curve : Detailed curve coordinates.
        """
        from sklearn.metrics import average_precision_score
        from sklearn.preprocessing import LabelBinarizer

        rows = []
        preds = scalar_prediction_frame(self.get_predictions(model=model))
        if preds.empty:
            return pd.DataFrame(
                columns=["Model", "Fold", "MacroPRAUC", "WeightedPRAUC"]
            )

        proba_cols = sorted(
            [col for col in preds.columns if col.startswith("y_proba_")],
            key=lambda v: int(v.rsplit("_", 1)[-1]),
        )
        if not proba_cols:
            return pd.DataFrame(
                columns=["Model", "Fold", "MacroPRAUC", "WeightedPRAUC"]
            )

        for (m_name, f_idx), group in preds.groupby(["Model", "Fold"]):
            y_true = group["y_true"].to_numpy()
            y_proba = group[proba_cols].to_numpy()

            lb = LabelBinarizer()
            y_true_bin = lb.fit_transform(y_true)
            if y_true_bin.shape[1] == 1:
                score = average_precision_score(y_true_bin, y_proba[:, -1])
                macro = weighted = score
            else:
                macro = average_precision_score(y_true_bin, y_proba, average="macro")
                weighted = average_precision_score(
                    y_true_bin, y_proba, average="weighted"
                )

            rows.append(
                {
                    "Model": m_name,
                    "Fold": f_idx,
                    "MacroPRAUC": float(macro),
                    "WeightedPRAUC": float(weighted),
                }
            )
        return pd.DataFrame(rows)

    def get_calibration_curve(
        self,
        model: Optional[str] = None,
        n_bins: int = 5,
        pos_label: Optional[Any] = None,
        strategy: str = "uniform",
    ) -> pd.DataFrame:
        """
        Get binary reliability/calibration curve coordinates.

        Calculates the fraction of positive samples vs. mean predicted
        probabilities for each probability bin.

        Scientific Rationale
        --------------------
        A well-calibrated classifier provides probabilistic outputs that
        reflect the true likelihood of the predicted event. Calibration
        curves (reliability diagrams) are essential for assessing whether
        predicted probabilities can be interpreted as confidence levels.

        Parameters
        ----------
        model : str, optional
            The model name to filter by.
        n_bins : int, default=5
            Number of bins to use for the calibration curve.
        pos_label : any, optional
            The label to treat as positive.
        strategy : {'uniform', 'quantile'}, default='uniform'
            Strategy used to define the widths of the bins.
            - 'uniform': Bins have identical widths.
            - 'quantile': Bins have the same number of samples.

        Returns
        -------
        calibration_df : pd.DataFrame
            DataFrame with Model, Fold, Class, MeanPredictedProbability,
            and FractionPositive.

        See Also
        --------
        ExperimentResult.get_probability_diagnostics : Brier score and Log Loss.
        """
        from sklearn.calibration import calibration_curve

        frames = []
        preds = scalar_prediction_frame(self.get_predictions(model=model))
        for m_name, f_idx, label, y_binary, y_score in curve_score_groups(
            preds, model=model, require_probability=True, pos_label=pos_label
        ):
            p_true, p_pred = calibration_curve(
                y_binary.astype(int), y_score, n_bins=n_bins, strategy=strategy
            )
            df_c = pd.DataFrame(
                {
                    "Model": m_name,
                    "Fold": f_idx,
                    "Class": label,
                    "MeanPredictedProbability": p_pred,
                    "FractionPositive": p_true,
                }
            )
            frames.append(df_c)

        if not frames:
            return pd.DataFrame(
                columns=[
                    "Model",
                    "Fold",
                    "Class",
                    "MeanPredictedProbability",
                    "FractionPositive",
                ]
            )
        return pd.concat(frames, ignore_index=True)

    def get_probability_diagnostics(self, model: Optional[str] = None) -> pd.DataFrame:
        """
        Get fold-level log-loss and Brier summaries when probabilities exist.

        Computes summary metrics that penalize poor probability calibration
        and high-uncertainty predictions.

        Parameters
        ----------
        model : str, optional
            The model name to filter by.

        Returns
        -------
        diagnostics_df : pd.DataFrame
            DataFrame in long format with Model, Fold, Metric, Class, and Value.
            Metrics include 'log_loss', 'brier_score_ovr', and 'brier_score_macro'.

        See Also
        --------
        ExperimentResult.get_calibration_curve : Visual calibration view.
        """
        from sklearn.metrics import log_loss

        rows = []
        preds = scalar_prediction_frame(self.get_predictions(model=model))
        if preds.empty:
            return pd.DataFrame(columns=["Model", "Fold", "Metric", "Class", "Value"])

        for (m_name, f_idx), group in preds.groupby(["Model", "Fold"]):
            y_true = group["y_true"].to_numpy()
            unique_labels = sorted(pd.unique(y_true).tolist())
            y_proba = proba_matrix(group, len(unique_labels))
            if y_proba is None:
                continue

            # 1. Log Loss (Overall)
            try:
                ll = log_loss(y_true, y_proba, labels=unique_labels)
                rows.append(
                    {
                        "Model": m_name,
                        "Fold": f_idx,
                        "Metric": "log_loss",
                        "Class": None,
                        "Value": float(ll),
                    }
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"log_loss skipped for {m_name} fold {f_idx}: {e}")

            # 2. Brier Scores (Vectorized)
            # Create binary matrix [n_samples, n_classes]
            y_binary = (y_true[:, None] == np.array(unique_labels)).astype(float)
            # Brier Score = Mean Squared Error per class
            brier_ovr = np.mean((y_binary - y_proba) ** 2, axis=0)

            for c_idx, label in enumerate(unique_labels):
                rows.append(
                    {
                        "Model": m_name,
                        "Fold": f_idx,
                        "Metric": "brier_score_ovr",
                        "Class": label,
                        "Value": float(brier_ovr[c_idx]),
                    }
                )

            # 3. Macro Brier Score
            rows.append(
                {
                    "Model": m_name,
                    "Fold": f_idx,
                    "Metric": "brier_score_macro",
                    "Class": None,
                    "Value": float(np.mean(brier_ovr)),
                }
            )

        return pd.DataFrame(rows)

    def get_statistical_assessment(
        self,
        lightweight: bool = False,
        metric: str = "accuracy",
        unit: Optional[str] = None,
        n_permutations: int = 1000,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get finite-sample statistical assessment rows in long form.

        Returns p-values and significance markers for model performance,
        supporting both full-pipeline and post-hoc permutation methods.

        Scientific Rationale
        --------------------
        Statistical significance in decoding ensures that observed performance
        deltas are not due to chance fluctuations. This method allows
        accessing pre-calculated results from the full experimental pipeline
        (the gold standard) or running a faster post-hoc permutation test
        directly on stored predictions.

        Parameters
        ----------
        lightweight : bool, default=False
            If True, perform a post-hoc label permutation on out-of-fold
            predictions. Fast but does not account for pipeline leakage
            (e.g., in tuning).
            If False, returns results from the full-pipeline assessment if
            they were computed during the experiment.
        metric : str, default='accuracy'
            Metric to use for the assessment.
        unit : str, optional
            The level of independence for the permutation test (e.g., 'subject').
        n_permutations : int, default=1000
            Number of permutations for the lightweight assessment.
        random_state : int, optional
            Seed for reproducible permutations.

        Returns
        -------
        stats_df : pd.DataFrame
            Tidy DataFrame with Model, Metric, Observed, PValue, and Significance.

        See Also
        --------
        coco_pipe.decoding.stats.run_statistical_assessment : Underlying engine.
        """
        u_type = self._resolve_inference_unit(unit)
        rows = []

        # 1. Pull pre-calculated results from the full pipeline if requested
        if not lightweight:
            for model, res in self.raw.items():
                if "error" in res:
                    continue
                stats_rows = res.get("statistical_assessment", [])
                for s in stats_rows:
                    row = dict(s)
                    row["Model"] = model
                    rows.append(row)

        # 2. Fallback to lightweight if no stats found
        if not rows and not lightweight:
            logger.info(
                "Full-pipeline statistical assessment results not found. "
                "Falling back to lightweight post-hoc assessment."
            )
            lightweight = True

        # 3. Lightweight post-hoc permutation assessment
        if lightweight:
            from .stats import assess_post_hoc_permutation

            for model, res in self.raw.items():
                if "error" in res:
                    continue
                try:
                    df_l = assess_post_hoc_permutation(
                        res,
                        metric=metric,
                        unit=u_type,
                        n_permutations=n_permutations,
                        random_state=random_state,
                    )
                    df_l["Model"] = model
                    rows.extend(df_l.to_dict("records"))
                except Exception as e:
                    logger.warning(f"Lightweight assessment failed for {model}: {e}")

        if not rows:
            return pd.DataFrame()

        # Consistent column ordering
        cols = [
            "Model",
            "Metric",
            "Observed",
            "PValue",
            "Significant",
            "NullMethod",
            "NPermutations",
            "InferentialUnit",
            "Time",
            "TrainTime",
            "TestTime",
            "NullLower",
            "NullUpper",
        ]
        df = pd.DataFrame(rows)
        # Sort and filter columns to match what's present
        present_cols = [c for c in cols if c in df.columns]
        return df[present_cols]

    def get_statistical_nulls(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Return stored statistical null distributions, when configured.

        Accesses the empirical null distributions (e.g., from permutation
        tests) stored during the experiment.

        Parameters
        ----------
        model : str, optional
            Model name to filter by. Default is None (all models).

        Returns
        -------
        nulls : dict
            Dictionary mapping model names to their null distribution payloads,
            containing coordinates and permuted score arrays.

        See Also
        --------
        ExperimentResult.get_statistical_assessment : P-values derived from these nulls.
        """
        nulls = {}
        for m_name, res in self.raw.items():
            if model is not None and m_name != model:
                continue
            if "error" in res:
                continue
            if "statistical_nulls" in res:
                nulls[m_name] = res["statistical_nulls"]
        return nulls

    def get_model_artifacts(self, model: Optional[str] = None) -> pd.DataFrame:
        """
        Return fold-level model artifact metadata in long form.

        Accesses non-metric outputs stored by models, such as learned
        coefficients, intercept values, or class labels.

        Parameters
        ----------
        model : str, optional
            The model name to filter by. Default is None (all models).

        Returns
        -------
        artifacts_df : pd.DataFrame
            DataFrame with Model, Fold, ArtifactType, Key, and Value.

        See Also
        --------
        ExperimentResult.get_feature_importances : Specifically for importances.
        """
        rows = []
        cols = ["Model", "Fold", "ArtifactType", "Key", "Value"]
        for m_name, res in self.raw.items():
            if model is not None and m_name != model:
                continue
            if "error" in res:
                continue
            for f_idx, m in enumerate(res.get("metadata", [])):
                artifacts = m.get("artifacts", {})
                for a_type, payload in artifacts.items():
                    if isinstance(payload, dict):
                        for k, v in payload.items():
                            rows.append(
                                {
                                    "Model": m_name,
                                    "Fold": f_idx,
                                    "ArtifactType": a_type,
                                    "Key": k,
                                    "Value": v,
                                }
                            )
                    else:
                        rows.append(
                            {
                                "Model": model,
                                "Fold": f_idx,
                                "ArtifactType": "model",
                                "Key": a_type,
                                "Value": payload,
                            }
                        )
        return pd.DataFrame(rows, columns=cols)

    def get_bootstrap_confidence_intervals(
        self,
        metric: str = "accuracy",
        model: Optional[str] = None,
        unit: Optional[str] = None,
        n_bootstraps: int = 1000,
        ci: float = 0.95,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Bootstrap metric confidence intervals over configured inference units.

        Estimates the uncertainty of a performance metric by resampling
        independent units (e.g., subjects) with replacement.

        Scientific Rationale
        --------------------
        Bootstrapping provides a non-parametric estimate of the sampling
        distribution of a metric. By resampling at the 'unit' level, we
        account for within-unit correlations (e.g., multiple trials from the
        same subject) and provide more realistic uncertainty bounds than
        sample-level analytical methods.

        Parameters
        ----------
        metric : str, default='accuracy'
            The metric to estimate uncertainty for.
        model : str, optional
            The model name to filter by.
        unit : str, optional
            The level of independence for resampling (e.g., 'subject').
        n_bootstraps : int, default=1000
            Number of bootstrap iterations.
        ci : float, default=0.95
            Confidence interval level (e.g., 0.95 for 95% CI).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        bootstrap_df : pd.DataFrame
            DataFrame with Model, Metric, Estimate (observed), CILower, and CIUpper.

        See Also
        --------
        coco_pipe.decoding.stats.assess_bootstrap_ci : Underlying engine.
        """
        from .stats import assess_bootstrap_ci

        u_type = self._resolve_inference_unit(unit)
        rows = []
        for m_name, res in self.raw.items():
            if model is not None and m_name != model:
                continue
            if "error" in res:
                continue

            try:
                df_b = assess_bootstrap_ci(
                    res,
                    metric=metric,
                    unit=u_type,
                    n_bootstraps=n_bootstraps,
                    ci=ci,
                    random_state=random_state,
                )
                df_b["Model"] = m_name
                rows.extend(df_b.to_dict("records"))
            except Exception as e:
                logger.warning(f"Bootstrap failed for {m_name}: {e}")

        if not rows:
            return pd.DataFrame()

        cols = [
            "Model",
            "Metric",
            "Estimate",
            "CILower",
            "CIUpper",
            "Unit",
            "NUnits",
            "NBootstraps",
        ]
        return pd.DataFrame(rows)[cols]

    def compare_models(
        self,
        models: Optional[Sequence[str]] = None,
        metric: str = "accuracy",
        unit: Optional[str] = None,
        n_permutations: int = 1000,
        correction: str = "fdr_bh",
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Perform exhaustive pairwise comparisons between multiple models.

        Automatically applies p-value correction (e.g., FDR) for the multiple
        comparisons performed across all pairs of models.

        Scientific Rationale
        --------------------
        Benchmarking multiple models requires controlling for the 'multiple
        comparisons problem'—the increased risk of Type I errors (false
        positives) when testing many hypotheses. This method automates the
        pairwise testing and subsequent error-rate control.

        Parameters
        ----------
        models : list of str, optional
            List of model names to compare. Default is all models in the result.
        metric : str, default='accuracy'
            Metric to use for comparison.
        unit : str, optional
            Level of independence for permutation testing (e.g., 'subject').
        n_permutations : int, default=1000
            Number of permutations for each paired test.
        correction : str, default='fdr_bh'
            Multiple comparison correction method (e.g., 'bonferroni', 'fdr_bh').
        random_state : int, optional
            Random seed for permutations.

        Returns
        -------
        comparison_df : pd.DataFrame
            DataFrame containing ModelA, ModelB, Difference, and corrected
            PValue (PValueCorrected).

        See Also
        --------
        ExperimentResult.compare_models_paired : Underlying paired test.
        """
        from itertools import combinations

        if models is None:
            models = sorted(self.raw.keys())

        if len(models) < 2:
            raise ValueError("Need at least two models to perform a comparison.")

        all_results = []
        # 1. Run all pairwise comparisons
        for m_a, m_b in combinations(models, 2):
            try:
                res = self.compare_models_paired(
                    model_a=m_a,
                    model_b=m_b,
                    metric=metric,
                    unit=unit,
                    n_permutations=n_permutations,
                    random_state=random_state,
                )
                all_results.append(res)
            except Exception as e:
                logger.warning(f"Comparison failed for {m_a} vs {m_b}: {e}")

        if not all_results:
            return pd.DataFrame()

        df = pd.concat(all_results, ignore_index=True)

        # 2. Apply multiple comparison correction
        from .stats import apply_multiple_comparison_correction

        return apply_multiple_comparison_correction(df, method=correction)

    def compare_models_paired(
        self,
        model_a: str,
        model_b: str,
        metric: str = "accuracy",
        unit: Optional[str] = None,
        n_permutations: int = 1000,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Paired model comparison using outer-fold predictions on shared samples.

        Performs a within-unit permutation test (e.g., swapping model labels
        per subject) to determine if the performance difference is significant.

        Scientific Rationale
        --------------------
        Paired tests are generally more powerful than independent-sample tests
        because they control for unit-specific baseline variance (e.g., a subject
        who is overall 'harder' to decode). By aligning predictions at the
        sample level across models, we ensure a valid paired comparison.

        Parameters
        ----------
        model_a, model_b : str
            The names of the two models to compare.
        metric : str, default='accuracy'
            Metric to use for comparison.
        unit : str, optional
            Level of independence (e.g., 'subject').
        n_permutations : int, default=1000
            Number of permutations for the test.
        random_state : int, optional
            Random seed for reproducible permutations.

        Returns
        -------
        paired_df : pd.DataFrame
            DataFrame with ModelA, ModelB, ScoreA, ScoreB, Difference, and PValue.

        See Also
        --------
        coco_pipe.decoding.stats.assess_paired_comparison : Underlying engine.
        """
        from .stats import assess_paired_comparison

        u_type = self._resolve_inference_unit(unit)
        preds = scalar_prediction_frame(self.get_predictions())
        a = preds[preds["Model"] == model_a]
        b = preds[preds["Model"] == model_b]

        if a.empty or b.empty:
            raise ValueError(f"One or both models not found: {model_a}, {model_b}")

        # Merge to find shared samples across all relevant coordinates
        merge_cols = ["SampleID", "y_true", "Fold"]
        for col in ["Time", "TrainTime", "TestTime"]:
            if col in a and col in b:
                merge_cols.append(col)

        merged = a.merge(b, on=merge_cols, suffixes=("_A", "_B"))
        if merged.empty:
            raise ValueError("No overlapping samples found between the two models.")

        df_res = assess_paired_comparison(
            merged,
            metric=metric,
            unit=u_type,
            n_permutations=n_permutations,
            random_state=random_state,
        )

        df_res["ModelA"] = model_a
        df_res["ModelB"] = model_b
        df_res["Unit"] = u_type

        # Reorder columns
        cols = [
            "ModelA",
            "ModelB",
            "Metric",
            "Unit",
            "NUnits",
            "ScoreA",
            "ScoreB",
            "Difference",
            "PValue",
            "Significant",
        ]
        # Preserve temporal columns if present
        for c in ["Time", "TrainTime", "TestTime"]:
            if c in df_res.columns:
                cols.insert(4, c)

        return df_res[cols]

    def get_feature_importances(
        self, model: Optional[str] = None, fold_level: bool = False
    ) -> pd.DataFrame:
        """
        Get feature importances in long format.

        Aggregates relative feature contributions (e.g., coefficients,
        Gini importance) across all folds.

        Scientific Rationale
        --------------------
        Feature importances identify the data dimensions that drive the model's
        predictions. Analyzing these across folds ensures that identified
        features are robust and not artifacts of a specific data split. Ranking
        features provides a prioritized list for subsequent biological
        interpretation.

        Parameters
        ----------
        model : str, optional
            The model name to filter by. Default is None (all models).
        fold_level : bool, default=False
            - If True: Returns importance for each fold individually.
            - If False: Returns the mean and standard deviation across folds.

        Returns
        -------
        importances_df : pd.DataFrame
            DataFrame with Model, FeatureName, and Importance (or Mean/Std).
            Includes a 'Rank' column based on the importance magnitude.

        See Also
        --------
        ExperimentResult.get_selected_features : If feature selection was used.
        """

        frames = []
        for m_name, res in self.raw.items():
            if model is not None and m_name != model:
                continue
            if "error" in res:
                continue

            imp = res.get("importances")
            if not imp:
                continue

            if fold_level:
                raw = np.asarray(imp.get("raw", []), dtype=float)
                if raw.ndim != 2:
                    continue
                n_feats = raw.shape[1]
                f_names = imp.get("feature_names")
                if f_names is None or len(f_names) != n_feats:
                    f_names = [f"feature_{i}" for i in range(n_feats)]

                df_m = pd.DataFrame(raw, columns=f_names)
                df_m.index.name = "Fold"
                df_m = df_m.reset_index().melt(
                    id_vars="Fold", var_name="FeatureName", value_name="Importance"
                )
                df_m["Model"] = m_name
                # Reconstruct Feature Index
                name_to_idx = {name: i for i, name in enumerate(f_names)}
                df_m["Feature"] = df_m["FeatureName"].map(name_to_idx)
                frames.append(df_m)
            else:
                means = np.asarray(imp.get("mean", []), dtype=float).ravel()
                if len(means) == 0:
                    continue
                stds = np.asarray(imp.get("std", []), dtype=float).ravel()
                if len(stds) != len(means):
                    stds = np.full(len(means), np.nan)

                f_names = imp.get("feature_names")
                if f_names is None or len(f_names) != len(means):
                    f_names = [f"feature_{i}" for i in range(len(means))]
                df_m = pd.DataFrame(
                    {
                        "Model": m_name,
                        "Feature": np.arange(len(means)),
                        "FeatureName": f_names,
                        "Mean": means,
                        "Std": stds,
                    }
                )
                frames.append(df_m)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)

        # Add ranks
        group_cols = ["Model"]
        if fold_level:
            group_cols.append("Fold")
            val_col = "Importance"
        else:
            val_col = "Mean"

        df["Rank"] = df.groupby(group_cols)[val_col].rank(ascending=False, method="min")

        # Ensure stable column order for API consistency
        if fold_level:
            ordered_cols = [
                "Model",
                "Fold",
                "Feature",
                "FeatureName",
                "Importance",
                "Rank",
            ]
        else:
            ordered_cols = ["Model", "Feature", "FeatureName", "Mean", "Std", "Rank"]

        return df[ordered_cols]

    def _resolve_inference_unit(self, unit: Optional[str]) -> str:
        if unit is not None:
            return unit
        return self.meta.get("inferential_unit") or "sample"

    def _time_value(self, index: int) -> Any:
        from ._diagnostics import time_value as get_time_val

        return get_time_val(index, self.time_axis)

    def get_best_params(self, model: Optional[str] = None) -> pd.DataFrame:
        """
        Get the best hyperparameters selected per fold.

        If hyperparameter tuning was enabled, returns the optimal
        configuration found in each cross-validation outer fold.

        Parameters
        ----------
        model : str, optional
            The name of the model to filter by. Default is None (all models).

        Returns
        -------
        params_df : pd.DataFrame
            DataFrame with Model, Fold, Param, and Value.

        See Also
        --------
        ExperimentResult.get_search_results : Detailed tuning diagnostics.
        """
        rows = []
        for m_name, res in self.raw.items():
            if model is not None and m_name != model:
                continue
            if "error" in res:
                continue
            if "metadata" in res:
                for f_idx, meta in enumerate(res["metadata"]):
                    if "best_params" in meta:
                        for p_name, p_val in meta["best_params"].items():
                            rows.append(
                                {
                                    "Model": m_name,
                                    "Fold": f_idx,
                                    "Param": p_name,
                                    "Value": p_val,
                                }
                            )
        return pd.DataFrame(rows)

    def get_search_results(self, model: Optional[str] = None) -> pd.DataFrame:
        """
        Get compact hyperparameter-search diagnostics in long form.

        Provides a summary of all candidate configurations evaluated during
        tuning, including their mean performance and ranking.

        Parameters
        ----------
        model : str, optional
            The model name to filter by. Default is None (all models).

        Returns
        -------
        search_df : pd.DataFrame
            DataFrame with Model, Fold, Candidate, Rank, MeanTestScore,
            StdTestScore, and Params.

        See Also
        --------
        ExperimentResult.get_best_params : Just the winner per fold.
        """
        rows = []
        for m_name, res in self.raw.items():
            if model is not None and m_name != model:
                continue
            if "error" in res:
                continue
            for f_idx, meta in enumerate(res.get("metadata", [])):
                for s_row in meta.get("search_results", []):
                    rows.append(
                        {
                            "Model": m_name,
                            "Fold": f_idx,
                            "Candidate": s_row.get("candidate"),
                            "Rank": s_row.get("rank_test_score"),
                            "MeanTestScore": s_row.get("mean_test_score"),
                            "StdTestScore": s_row.get("std_test_score"),
                            "Params": s_row.get("params"),
                        }
                    )
        cols = [
            "Model",
            "Fold",
            "Candidate",
            "Rank",
            "MeanTestScore",
            "StdTestScore",
            "Params",
        ]
        return pd.DataFrame(rows, columns=cols)

    def get_selected_features(self, model: Optional[str] = None) -> pd.DataFrame:
        """
        Get fold-level selected feature masks in long format.

        Returns a boolean mask indicating which features were retained by
        automated feature selection in each fold.

        Parameters
        ----------
        model : str, optional
            The model name to filter by. Default is None (all models).

        Returns
        -------
        features_df : pd.DataFrame
            DataFrame with Model, Fold, FeatureName, and Selected status.
            Includes an 'Order' column if recursive or sequential selection
            was used.

        See Also
        --------
        ExperimentResult.get_feature_stability : Cross-fold selection consistency.
        """
        frames = []
        for m_name, res in self.raw.items():
            if model is not None and m_name != model:
                continue
            if "error" in res:
                continue
            for f_idx, meta in enumerate(res.get("metadata", [])):
                if "selected_features" not in meta:
                    continue

                mask = np.asarray(meta["selected_features"], dtype=bool)
                order = meta.get("selection_order")
                f_names = meta.get("feature_names")
                if f_names is None or len(f_names) != len(mask):
                    f_names = [f"feature_{idx}" for idx in range(len(mask))]

                df_f = pd.DataFrame(
                    {
                        "Model": m_name,
                        "Fold": f_idx,
                        "Feature": np.arange(len(mask)),
                        "FeatureName": f_names,
                        "Selected": mask,
                    }
                )

                if order is not None:
                    # Resolve order (Rank)
                    order_arr = np.asarray(order)
                    if len(order_arr) == len(mask):
                        df_f["Order"] = order_arr
                    elif len(order_arr) < len(mask):
                        # Order is a list of selected indices
                        order_map = {idx: i + 1 for i, idx in enumerate(order_arr)}
                        df_f["Order"] = df_f["Feature"].map(order_map)
                else:
                    df_f["Order"] = np.nan

                frames.append(df_f)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def get_feature_scores(self, model: Optional[str] = None) -> pd.DataFrame:
        """
        Get fold-level feature-selection scores.

        Accesses raw univariate or multivariate scores (e.g., F-values,
        p-values, or internal selector scores) used for feature ranking.

        Parameters
        ----------
        model : str, optional
            The model name to filter by. Default is None (all models).

        Returns
        -------
        scores_df : pd.DataFrame
            DataFrame with Model, Fold, FeatureName, Score, and PValue
            (if available).

        See Also
        --------
        ExperimentResult.get_selected_features : Final binary selection mask.
        """
        frames = []
        for m_name, res in self.raw.items():
            if model is not None and m_name != model:
                continue
            if "error" in res:
                continue
            for f_idx, meta in enumerate(res.get("metadata", [])):
                if "feature_scores" not in meta:
                    continue

                scores = np.asarray(meta["feature_scores"], dtype=float)
                pvals = meta.get("feature_pvalues")
                f_names = meta.get("feature_names")
                if f_names is None or len(f_names) != len(scores):
                    f_names = [f"feature_{idx}" for idx in range(len(scores))]

                sel = meta.get("selected_features")

                df_f = pd.DataFrame(
                    {
                        "Model": m_name,
                        "Fold": f_idx,
                        "Feature": np.arange(len(scores)),
                        "FeatureName": f_names,
                        "Selector": meta.get("feature_selection_method"),
                        "Score": scores,
                    }
                )

                if pvals is not None and len(pvals) == len(scores):
                    df_f["PValue"] = np.asarray(pvals, dtype=float)
                else:
                    df_f["PValue"] = np.nan

                if sel is not None and len(sel) == len(scores):
                    df_f["Selected"] = np.asarray(sel, dtype=bool)
                else:
                    df_f["Selected"] = np.nan

                frames.append(df_f)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def get_feature_stability(self, model: Optional[str] = None) -> pd.DataFrame:
        """
        Analyze feature selection stability across folds.

        Calculates the frequency with which each feature was selected across
        the cross-validation procedure.

        Scientific Rationale
        --------------------
        Stability analysis helps distinguish robust predictors from features
        that are selected due to noise in specific data splits. High stability
        (e.g., > 90% of folds) provides strong evidence for the relevance of
        a feature to the decoding task.

        Parameters
        ----------
        model : str, optional
            The model name to filter by. Default is None (all models).

        Returns
        -------
        stability_df : pd.DataFrame
            DataFrame with Model, FeatureName, SelectionFrequency (0.0 to 1.0),
            and NFolds.

        See Also
        --------
        ExperimentResult.get_selected_features : Fold-level selection data.
        """
        frames = []
        for m_name, res in self.raw.items():
            if model is not None and m_name != model:
                continue
            if "error" in res:
                continue
            if "metadata" in res:
                masks = []
                f_names = None
                for meta in res["metadata"]:
                    if "selected_features" in meta:
                        masks.append(meta["selected_features"])
                        if f_names is None and "feature_names" in meta:
                            f_names = meta["feature_names"]

                if not masks:
                    continue

                stack = np.vstack(masks)  # [n_folds, n_features]
                n_folds = stack.shape[0]
                freq = np.mean(stack, axis=0)

                if f_names is None or len(f_names) != stack.shape[1]:
                    f_names = [f"feature_{i}" for i in range(stack.shape[1])]

                df_m = pd.DataFrame(
                    {
                        "Model": m_name,
                        "Feature": np.arange(len(freq)),
                        "FeatureName": f_names,
                        "SelectionFrequency": freq,
                        "NFolds": n_folds,
                    }
                )
                frames.append(df_m)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def get_generalization_matrix(
        self, model: Optional[str] = None, metric: str = "accuracy"
    ) -> pd.DataFrame:
        """
        Get Generalization Matrix (Train Time x Test Time) averaged across folds.

        Computes the cross-temporal performance matrix for Time-Generalization
        analysis.

        Scientific Rationale
        --------------------
        Temporal Generalization (TG) analysis reveals the dynamics of neural
        representations. By training a classifier at one time point and
        testing it across all others, we can identify whether a neural
        pattern is transient, sustained, or reoccurring.

        Parameters
        ----------
        model : str, optional
            The model name to filter by. Default is None (all models).
            If None, returns a long-format DataFrame suitable for plotting.
        metric : str, default='accuracy'
            The metric to retrieve.

        Returns
        -------
        gen_df : pd.DataFrame
            - If model is specified: A square matrix (2D DataFrame) with
              TrainTime as index and TestTime as columns.
            - If model is None: A tidy long-format DataFrame with Model,
              Metric, TrainTime, TestTime, and Value.

        See Also
        --------
        ExperimentResult.get_temporal_score_summary : Linear temporal summary.
        """
        frames = []
        for m_name, res in self.raw.items():
            if model is not None and m_name != model:
                continue
            if "error" in res:
                continue

            metrics_data = res.get("metrics", {})
            if metric not in metrics_data:
                # Fallback to first available metric if requested one is missing
                if not metrics_data:
                    continue
                metric = next(iter(metrics_data.keys()))

            fold_scores = metrics_data[metric].get("folds", [])
            valid_matrices = [
                s for s in fold_scores if isinstance(s, np.ndarray) and s.ndim == 2
            ]

            if valid_matrices:
                stack = np.stack(valid_matrices)
                mean = np.nanmean(stack, axis=0)

                labels = self.time_axis
                if labels is None or len(labels) != mean.shape[0]:
                    labels = list(range(mean.shape[0]))

                df_gen = pd.DataFrame(mean, index=labels, columns=labels)
                df_gen.index.name = "TrainTime"
                df_gen.columns.name = "TestTime"

                if model is not None:
                    return df_gen

                df_gen = df_gen.stack().reset_index(name="Value")
                df_gen["Model"] = m_name
                df_gen["Metric"] = metric
                frames.append(df_gen)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)


def make_serializable(obj: Any) -> Any:
    """Recursively convert NumPy types to JSON-safe Python primitives."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32, np.int16, np.integer)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.floating)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj
