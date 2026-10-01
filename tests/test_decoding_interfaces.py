import ast
from pathlib import Path

from coco_pipe.decoding.interfaces import (
    DecoderEstimator,
    EmbeddingExtractor,
    NeuralTrainable,
)


def test_decoding_package_does_not_import_diagnostics():
    """Keep diagnostics -> decoding as a one-way dependency."""
    decoding_root = Path(__file__).parents[1] / "coco_pipe" / "decoding"
    violations: list[str] = []
    for path in decoding_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(
                    alias.name == "coco_pipe.diagnostics"
                    or alias.name.startswith("coco_pipe.diagnostics.")
                    for alias in node.names
                ):
                    violations.append(str(path.relative_to(decoding_root)))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports_diagnostics = (
                    module == "coco_pipe.diagnostics"
                    or module.startswith("coco_pipe.diagnostics.")
                    or (node.level > 0 and module == "diagnostics")
                    or (node.level > 0 and module.startswith("diagnostics."))
                )
                if imports_diagnostics:
                    violations.append(str(path.relative_to(decoding_root)))
    assert violations == []


def test_decoder_estimator_protocol():
    class ValidDecoder:
        def fit(self, X, y=None, **kwargs):
            return self

        def predict(self, X):
            return X

        def get_params(self, deep=True):
            return {}

        def set_params(self, **params):
            return self

    class InvalidDecoder:
        def fit(self, X, y=None):
            return self

        # Missing predict
        def get_params(self, deep=True):
            return {}

        def set_params(self, **params):
            return self

    assert isinstance(ValidDecoder(), DecoderEstimator)
    assert not isinstance(InvalidDecoder(), DecoderEstimator)


def test_embedding_extractor_protocol():
    class ValidExtractor:
        def transform(self, X):
            return X

        def get_embedding_info(self):
            return {}

    class InvalidExtractor:
        def transform(self, X):
            return X

        # Missing get_embedding_info

    assert isinstance(ValidExtractor(), EmbeddingExtractor)
    assert not isinstance(InvalidExtractor(), EmbeddingExtractor)


def test_neural_trainable_protocol():
    class ValidNeural:
        def get_training_history(self):
            return []

        def get_checkpoint_manifest(self):
            return {}

        def get_model_card_info(self):
            return {}

        def get_failure_diagnostics(self):
            return {}

        def get_artifact_metadata(self):
            return {}

    class PartialNeural:
        def get_training_history(self):
            return []

        # Missing others

    assert isinstance(ValidNeural(), NeuralTrainable)
    assert not isinstance(PartialNeural(), NeuralTrainable)
