"""
Lightweight public interfaces for decoding estimator families.
============================================================

These protocols define the structural contracts for models and extractors
used within the decoding pipeline. Since they are runtime-checkable,
the pipeline can verify capabilities without strict inheritance from
scikit-learn base classes.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DecoderEstimator(Protocol):
    """
    Protocol for scikit-learn-compatible decoding estimators.

    This interface defines the minimal set of methods required for an
    estimator to be integrated into the cross-validation engine.
    """

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> DecoderEstimator:
        """
        Fit the estimator to the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,), optional
            Target values (class labels in classification, real numbers in
            regression).
        **fit_params : dict
            Parameters to pass to the underlying fit method.

        Returns
        -------
        self : DecoderEstimator
            The fitted estimator.
        """
        ...  # pragma: no cover

    def predict(self, X: Any) -> Any:
        """
        Predict targets for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted target values per sample.
        """
        ...  # pragma: no cover

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        ...  # pragma: no cover

    def set_params(self, **params: Any) -> DecoderEstimator:
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : DecoderEstimator
            The estimator instance.
        """
        ...  # pragma: no cover


@runtime_checkable
class EmbeddingExtractor(Protocol):
    """
    Interface for pretrained or frozen feature extraction backbones.

    Embedding extractors typically represent foundation models or frozen
    neural networks that transform raw data into a fixed-dimensional
    vector space before classical decoding.
    """

    def transform(self, X: Any) -> Any:
        """
        Extract features from the provided data.

        Parameters
        ----------
        X : array-like
            The raw data to be transformed.

        Returns
        -------
        embeddings : array-like
            The extracted feature vectors.
        """
        ...  # pragma: no cover

    def get_embedding_info(self) -> dict[str, Any]:
        """
        Return technical metadata about the extractor and its output space.

        Returns
        -------
        info : dict
            A dictionary containing provider name, model name, pooling
            strategy, and output dimensionality.
        """
        ...  # pragma: no cover


@runtime_checkable
class NeuralTrainable(Protocol):
    """
    Interface for trainable neural estimators with diagnostic metadata.

    This protocol exposes internal training states and histories for
    reporting and verification purposes.
    """

    def get_training_history(self) -> list[dict[str, Any]]:
        """
        Get the step-by-step training history (e.g., loss per epoch).

        Returns
        -------
        history : list of dict
            A list of diagnostic records, one per training iteration.
        """
        ...  # pragma: no cover

    def get_checkpoint_manifest(self) -> dict[str, Any]:
        """
        Get information about saved model checkpoints.

        Returns
        -------
        manifest : dict
            Metadata including checkpoint paths and best-epoch indices.
        """
        ...  # pragma: no cover

    def get_model_card_info(self) -> dict[str, Any]:
        """
        Get high-level model card metadata for the artifact registry.

        Returns
        -------
        info : dict
            Information about model architecture, training configuration,
            and hyperparameters.
        """
        ...  # pragma: no cover

    def get_failure_diagnostics(self) -> dict[str, Any]:
        """
        Get technical diagnostics if training failed or diverged.

        Returns
        -------
        diagnostics : dict
            Information about gradients, NaN detection, or hardware state.
        """
        ...  # pragma: no cover

    def get_artifact_metadata(self) -> dict[str, Any]:
        """
        Aggregate all diagnostic metadata into a single dictionary.

        Returns
        -------
        metadata : dict
            A serializable dictionary containing history, model card, and checkpoints.
        """
        ...  # pragma: no cover


@runtime_checkable
class StagedTrainable(Protocol):
    """
    Interface for estimators that support multi-stage training schedules.
    """

    def set_train_stage(self, stage: str) -> StagedTrainable:
        """
        Configure the active training stage (e.g., 'pretrain', 'finetune').

        Parameters
        ----------
        stage : str
            The name of the training stage to activate.

        Returns
        -------
        self : StagedTrainable
            The estimator instance.
        """
        ...  # pragma: no cover

    def get_train_stage(self) -> str:
        """
        Get the name of the currently active training stage.

        Returns
        -------
        stage : str
            The active stage name.
        """
        ...  # pragma: no cover
