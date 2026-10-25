# Subject alignment

Subject-alignment transforms reduce stable subject identity in learned
representations. They are useful when subject-specific offsets or covariance
structure dominate the scientific label of interest.

These transforms are experimental. Alignment changes the representation and
can remove label signal when label and subject are confounded. Always compare
label decoding and subject diagnostics before and after alignment.

## Direct use

All transforms require an explicit observation-aligned `groups` vector:

```python
from coco_pipe.transforms import make_subject_transform

aligner = make_subject_transform("ea_coral", shrinkage=True)
aligned = aligner.fit_transform(embedding, groups=subject_ids)
print(aligner.fingerprint())
```

The registry normalizes method names and rejects unknown methods. A fingerprint
is a stable hash of the transform name and constructor parameters, suitable for
artifact provenance and cache keys.

## Decoding integration

For classical 2-D inputs, configure alignment through `ErasureConfig`:

```python
from coco_pipe.decoding import CVConfig, ErasureConfig, Experiment, ExperimentConfig
from coco_pipe.decoding.configs import LogisticRegressionConfig

config = ExperimentConfig(
    task="classification",
    models={"lr": LogisticRegressionConfig(max_iter=1000)},
    metrics=["balanced_accuracy"],
    cv=CVConfig(strategy="group_kfold", n_splits=5),
    erasure=ErasureConfig(enabled=True, method="leace"),
)

result = Experiment(config).run(X, y, groups=subject_ids)
```

`Experiment` inserts the transform as the first estimator-pipeline step, before
feature scaling, and routes `groups` through scikit-learn metadata routing. The
transform is therefore fit separately inside every outer fold and inside model
selection workflows.

`ErasureConfig` accepts the vector methods `leace`, `ea_mean`, and `ea_coral`.
`ra` changes 3-D token tensors into 2-D tangent vectors and must be applied in a
token/covariance workflow before ordinary 2-D decoding.

`ea_mean` and `ea_coral` align each test subject on its own samples, so they
need the test subjects' groups at prediction. Hyperparameter search scores its
inner folds, and calibration predicts, without groups, so the configuration
refuses EA erasure with `tuning` or `calibration`. LEACE's projector is fixed
after fitting and needs no groups, so it works with both.

## Fit scope and leakage

The class attribute `fold_local` describes whether a transform intrinsically
requires training-fold fitting; it does not change `Experiment` behavior.
`Experiment` always places enabled vector erasure inside the fold-local
pipeline.

| Method | `fold_local` | Unseen subject at `transform` | Precomputation guidance |
|---|---:|---|---|
| `leace` | `True` | Uses the projector learned from training subjects | Never fit on the full evaluation dataset |
| `ea_mean` | `False` | Estimates that subject's mean from the transform batch | Precompute only for subject-disjoint evaluation |
| `ea_coral` | `False` | Estimates that subject's mean/covariance from the transform batch | Precompute only for subject-disjoint evaluation |
| `ra` | `False` | Estimates that subject's Riemannian reference from the transform batch | Precompute only for subject-disjoint evaluation |

For observation-level CV where the same subject appears in training and test,
fitting or precomputing alignment globally uses held-out observations. Keep the
transform inside the estimator pipeline. For subject-disjoint group CV,
per-subject Euclidean or Riemannian statistics do not mix training and test
subjects; precomputation may be used as a deliberate compute optimization.

Estimating statistics from all observations of an unseen test subject is
unsupervised but transductive. If sessions or runs are the intended independent
unit, compute alignment within that acquisition policy rather than pooling
across it silently.

## LEACE erasure

`LeaceEraser` accepts a 2-D embedding matrix and erases the directions in
which subjects differ most. During `fit` it:

1. estimates the total covariance `S_t` (Ledoit-Wolf) and the
   between-subject covariance `S_b` of the subject means;
2. gives every direction `v` its subject eta^2, the share of its variance that
   lies between subjects (`S_b v = eta^2 S_t v`);
3. keeps the `r` directions with eta^2 of at least `eta2` (default 0.5), at
   most `max_rank` of them (default: no limit);
4. builds LEACE's oblique projector that erases those `r` directions:
   `P = I - S_t V_r V_r.T`, with `V_r.T S_t V_r = I`.

`transform` applies `P` around the training mean. The projector is fixed after
`fit`, so it also applies to subjects it has not seen.

Full LEACE, which erases every direction in which subject means differ, leaves
nothing once there are more subjects than features; limiting the rank keeps
the rest of the space. `fit` refuses to erase 95% of the feature directions
or more.

`eta2` leaves alone the directions that are mostly within-subject
variance, and its number of directions then follows the model: e.g. 5 for one
embedding, 49 for another. `max_rank` bounds that number; `eta2=0` with
`max_rank=5` erases exactly the 5 directions of largest eta^2.

When the windows belong to conditions (e.g. eyes open and closed), pass them to
`fit` as `condition`: `S_b` is then taken from the subject x condition cell
means around their condition's mean. A subject with more windows of one
condition does not count that as identity, and a difference between conditions
that the subjects share is kept. In `Experiment`, name the column of
`sample_metadata` that holds them:

```python
ErasureConfig(enabled=True, method="leace", condition_column="condition")
```

Each fold then fits the eraser with the conditions of its training samples.

Fitted attributes:

- `rank_`: number of directions erased;
- `eta2_`: subject eta^2 of every direction, largest first;
- `components_`: every direction as a row, in the same order;
- `n_subjects_`: fitted subject count.

### What the erased directions carry

A direction in which subjects differ can follow a trait of theirs, such as age
or diagnosis, and erasing it erases the trait too. `direction_traits` scores
the windows on the eraser's directions, averages them per subject, and gives
the share of the variance of these means that each trait explains (`r2`: the
squared correlation for a numeric trait, eta^2 across levels otherwise):

```python
from coco_pipe.diagnostics import direction_traits
from coco_pipe.viz import plot_heatmap

traits = participants.set_index("subject")[["age", "sex", "group"]]
table = direction_traits(eraser, X, subject_ids, traits, n_directions=30)
plot_heatmap(
    table.pivot(index="trait", columns="direction", values="r2"),
    vmin=0, vmax=1, colorbar_label="R²",
)
```

The table also holds each direction's `eta2` and whether it is `erased`.

## Euclidean alignment

`EuclideanAlign(mode="mean")` subtracts each subject mean.

`EuclideanAlign(mode="coral")` additionally left-whitens each subject with an
inverse covariance square root. Ledoit-Wolf shrinkage is enabled by default and
is recommended for high-dimensional embeddings or small subject sample counts.
With `shrinkage=False`, rank-deficient covariance directions are projected to
zero.

The transform stores fitted subject means in `mu_` and whitening matrices in
`w_`. At transform time, known subjects use stored statistics; unseen subjects
derive statistics from their provided transform batch.

`EuclideanAlign` also takes foundation-model tokens, `(window, ..., feature)`.
A subject's mean and covariance are then taken over all its tokens, subjects are
aligned one at a time, and the output keeps the shape and float dtype of the
input. Pooling tokens after `ea_mean` gives the same vectors as `ea_mean` on the
pooled vectors. Token covariances can be rank-deficient: a final LayerNorm, as
in LaBraM and LUNA, leaves one direction without variance, which whitening
leaves at zero.

`channel_axis` centers each channel's tokens on the subject's mean for that
channel, e.g. `EuclideanAlign("coral", channel_axis=1)` for
`(window, channel, time_patch, feature)` tokens. `"coral"` then whitens every
channel with the subject's one covariance of the centered tokens, so the
channels keep one feature basis. In mean mode the window and patch vectors are
those of plain `ea_mean`; the channel vectors and the tokens lose each
subject's mean per channel, its average topography in the model's space.

## Riemannian alignment

`RiemannAlign` expects `(window, token, feature)` tensors. Each window is
converted to the feature covariance of its tokens, centered on their mean, so it
describes how the tokens co-vary and not where they sit:

```text
Zc = Z - Z.mean(axis=0)
C = Zc.T @ Zc / (n_tokens - 1)
C_regularized = (1 - shrinkage) * C + shrinkage * trace(C) / d * I
```

A window's covariance has rank at most `n_tokens - 1`, often below `d` (LUNA:
64 tokens of 256 features), so `shrinkage` (default 0.1) moves it toward the
identity scaled to its mean variance: the Riemannian mean and the tangent map
need positive definite matrices.

The method computes a Riemannian mean reference per subject, recenters the
subject matrices, and maps them to symmetric tangent vectors. An input feature
dimension `d` produces `d * (d + 1) / 2` output features, exposed as
`n_output_features_` after fitting.

The helper `tokens_to_covariances` performs only the tensor-to-matrix step and
is useful when a downstream method consumes SPD matrices directly.

## Choosing a method

- Use `leace` when the target is a 2-D embedding and the goal is to remove
  the directions that carry subject identity most, with a projector learned on
  training subjects only.
- Use `ea_mean` when subject offsets dominate but within-subject geometry should
  otherwise remain unchanged.
- Use `ea_coral` when both offsets and subject-specific covariance/scaling are
  nuisance structure.
- Use `ra` for token sequences where covariance geometry is itself the desired
  representation.

No transform guarantees preservation of the scientific label. If labels are
constant within subject, label and subject information are partly confounded;
aggressive alignment can necessarily remove both.

## Auditing the result

Use the design-aware variance report before and after alignment:

```python
from coco_pipe.diagnostics import variance_decomposition_report

before = variance_decomposition_report(X, subject_ids, labels)
after = variance_decomposition_report(aligned, subject_ids, labels)
```

Compare at least:

- the additive label, subject-within-label, and residual fractions;
- marginal subject excess over its permutation null;
- subject-probe balanced accuracy and chance;
- label decoding under leakage-safe subject-aware CV;
- LEACE's `rank_` and `eta2_`, and the traits its erased directions carry
  (`direction_traits`), when applicable.

See [Subject and label variance diagnostics](../diagnostics/variance.md) for the
metric definitions, scaling policy, permutation nulls, and block-aware subject
probe.

## API summary

```python
from coco_pipe.transforms import (
    EuclideanAlign,
    LeaceEraser,
    RiemannAlign,
    TOKEN_TRANSFORMS,
    VECTOR_TRANSFORMS,
    make_subject_transform,
    tokens_to_covariances,
)
```

The generated {doc}`API reference
<../api/coco_pipe/transforms/subject_alignment/index>` contains constructor
signatures and fitted attributes.
