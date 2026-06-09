from coco_pipe.decoding.interfaces import (
    DecoderEstimator,
    EmbeddingExtractor,
    NeuralTrainable,
    StagedTrainable,
)


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


def test_staged_trainable_protocol():
    class ValidStaged:
        def set_train_stage(self, stage: str):
            return self

        def get_train_stage(self):
            return "pretrain"

    class InvalidStaged:
        def something_else(self):
            pass

    assert isinstance(ValidStaged(), StagedTrainable)
    assert not isinstance(InvalidStaged(), StagedTrainable)
