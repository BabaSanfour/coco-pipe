# Temporal trajectory alignment

`TemporalProcrustesAlignment` puts participant-specific PCA trajectories into
a common coordinate system. It expects an
`(observation, feature, time)` array and one participant ID per observation.

The transform is deliberately small: it fits a shared PCA/template, fits one
PCA per participant, and uses an orthogonal Procrustes rotation to map each
participant's grand-mean temporal trajectory onto the template. It never uses
the target labels.

## Decoding integration

Enable alignment on an ordinary temporal decoding experiment:

```python
from coco_pipe.decoding import (
    Experiment,
    ExperimentConfig,
    TemporalAlignmentConfig,
)

config = ExperimentConfig(
    task="classification",
    models={"sliding": temporal_decoder},
    metrics=["balanced_accuracy"],
    cv=subject_aware_cv,
    temporal_alignment=TemporalAlignmentConfig(
        enabled=True,
        n_components=30,
    ),
)

result = Experiment(config).run(X, y, groups=participant_ids, time_axis=times)
```

`Experiment` applies the alignment separately in every outer cross-validation
fold, before fitting the temporal estimator. The shared PCA and template see
only that fold's training observations.

## The transductive assumption

For a participant present during `fit`, `transform` reuses the stored PCA and
rotation. For a wholly unseen participant, coco-pipe estimates that
participant's PCA and rotation from the unlabelled transform batch. This is
transductive domain adaptation: it does not use test labels, but it does use
the available test-participant observations.

Use it when that calibration policy matches the scientific question, and
report it explicitly. It is not blind single-trial generalization to a new
participant. The current configuration names this supported policy with
`adaptation="transductive"`.

## Direct use

The same transform can be used outside decoding for descriptive analyses:

```python
from coco_pipe.transforms import TemporalProcrustesAlignment

aligner = TemporalProcrustesAlignment(n_components=3, random_state=42)
aligned = aligner.fit_transform(X, groups=participant_ids)
```

Direct use does not create cross-validation boundaries for you. For predictive
evaluation, prefer the `ExperimentConfig` integration above.
