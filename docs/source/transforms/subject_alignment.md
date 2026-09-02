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

`LeaceEraser` accepts a 2-D embedding matrix. During `fit` it:

1. centers and optionally Ledoit-Wolf-whitens the embedding;
2. one-hot encodes subject identity;
3. finds the whitened cross-covariance directions that linearly predict
   subject;
4. constructs a projection onto the orthogonal complement of those directions.

`transform` applies the learned projection and restores the training mean.
Useful fitted attributes include:

- `rank_`: removed subject-subspace rank;
- `cond_`: covariance condition estimate;
- `n_subjects_`: fitted subject count;
- `degenerate_`: whether the removed rank approaches the feature dimension.

`degenerate_=True` is a warning that erasure may destroy most of the embedding.
This commonly occurs when the number of independent subject directions is near
or above the embedding dimension. It should be reported, not silently ignored.

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

## Riemannian alignment

`RiemannAlign` expects `(window, token, feature)` tensors. Each window is
converted to a regularized feature second-moment matrix:

```text
C = Z.T @ Z / n_tokens
C_regularized = (1 - shrinkage) * C + shrinkage * trace(C) / d * I
```

The method computes a Riemannian mean reference per subject, recenters the
subject matrices, and maps them to symmetric tangent vectors. An input feature
dimension `d` produces `d * (d + 1) / 2` output features, exposed as
`n_output_features_` after fitting.

The helper `tokens_to_covariances` performs only the tensor-to-matrix step and
is useful when a downstream method consumes SPD matrices directly.

## Choosing a method

- Use `leace` when the target is a 2-D embedding and the goal is specifically
  to remove linearly decodable subject identity.
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
- LEACE's `degenerate_` and removed `rank_`, when applicable.

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
