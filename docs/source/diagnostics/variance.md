# Subject and label variance diagnostics

`coco_pipe.diagnostics.variance_decomposition_report` summarizes how an
embedding varies with subject identity and a scientific label. It is intended
for representation auditing and before/after comparisons of subject-alignment
transforms. It is not a replacement for a prespecified inferential model.

## Basic use

With arrays, subject and label vectors are positional arguments:

```python
from coco_pipe.diagnostics import variance_decomposition_report

report = variance_decomposition_report(
    embedding,
    subject_ids,
    diagnosis,
)
```

With a `DataContainer`, coordinate roles are explicit. The function does not
guess names such as `subject`, `study_id`, or `participant_id`:

```python
report = variance_decomposition_report(
    container,
    subject="study_id",
    label="diagnosis",
    probe_blocks="session",
)
```

`container.y` is the only implicit label source. Named values are resolved
through `DataContainer.obs_table()`, so only one-dimensional,
observation-aligned coordinates are accepted. Containers are flattened with
`obs` first before the calculations run.

## Design-aware partitions

The report selects a decomposition from the observed subject-label layout.

### Subject nested within label

When every subject has exactly one label, the additive partition is

```text
label_fraction
+ subject_within_label_fraction
+ residual_fraction
= 1
```

This is the nested model

```text
embedding = label + subject(label) + residual.
```

The label effect uses label means across recordings for the descriptive SS
partition. `omega2_label_subject_level` is instead calculated from equally
weighted subject means, so repeated recordings are not treated as independent
evidence for the label effect. `partial_omega2_subject_within_label` uses the
recording-level residual as the error term for the nested subject effect.

This error hierarchy follows the standard nested-ANOVA structure described in
the [NIST two-way nested ANOVA guide](https://www.itl.nist.gov/div898/handbook/ppc/section2/ppc233.htm).

### Label crossed with subject

When at least one subject has multiple labels, the report fits additive
fixed-effect indicator models and returns

```text
unique_label_fraction
+ unique_subject_fraction
+ shared_or_confounding_fraction
+ residual_fraction
= 1
```

The unique terms are extra sums of squares: the improvement from adding one
factor to a model that already contains the other. The shared term exposes
non-orthogonality or confounding instead of assigning it silently to label or
subject. It can be negative in suppressor designs. The corresponding effect
sizes are named `partial_omega2_label` and `partial_omega2_subject` to make the
fixed-effect, adjusted interpretation explicit.

For confirmatory inference with subjects treated as a sampled population, fit
a mixed-effects model chosen for the acquisition design. The crossed report is
a descriptive embedding partition, not a REML variance-component estimate.

## Marginal diagnostics are not a decomposition

The report also includes `marginal_label_eta2` and
`between_subject_eta2`. Each is a valid one-factor descriptive diagnostic, but
they overlap and must not be added or subtracted to derive a residual. In a
nested design, between-subject variation already contains the label effect.

Earlier versions subtracted both marginal effects and clipped negative
residuals to zero. That metric was removed because it double-counted label
variation.

## Feature scaling

`feature_scaling="zscore"` is the default. Each nonconstant embedding feature
is standardized across observations before sums of squares are pooled. This
makes the partition invariant to feature units and prevents high-variance
coordinates from dominating solely because of scale. Constant features become
zero and are reported in `n_constant_features`.

`DataContainer.zscore(dim="obs")` is not used internally for two reasons:

1. the report accepts both arrays and containers and should apply one identical
   numerical path after input adaptation;
2. the container method divides by `std + eps`, while the diagnostic needs to
   identify constant features explicitly and set only those features to zero.

Use `feature_scaling="none"` when the embedding's native feature variances have
a scientific meaning. The selected policy is repeated in the
`feature_scaling` output column.

## Permutation nulls

The default null uses 200 hierarchy-preserving permutations of the observed
embedding rather than synthetic IID Gaussian features.

- For nested labels, label assignments are permuted across subjects.
- For crossed labels, labels are permuted within subject.
- Subject identities are permuted within label when estimating the marginal
  subject null.

This preserves observed feature covariance and distribution while respecting
the main repeated-measure structure. The report includes null means, null
standard deviations, excess-over-null ratios, and finite-permutation p-values.
These are exploratory Monte Carlo diagnostics and are not corrected for
multiple testing.

## Subject-identity probe

The probe delegates scaling, cross-validation, estimation, and balanced
accuracy to `Experiment.run`. By default it performs stratified
observation-level CV and therefore measures whether held-out observations from
known subjects remain identifiable.

When recordings contain sessions, runs, or other acquisition clusters, pass
`probe_blocks`. The probe then uses stratified group CV over complete
subject-block combinations, preventing observations from the same acquisition
block from appearing in both training and test folds. Subjects without enough
observations or blocks for the requested number of folds are excluded, and the
remaining sample count is capped per subject by `probe_cap`.

## Output metadata and limitations

Every metric row includes:

- `design`
- `feature_scaling`
- observation, feature, subject, and label counts
- constant-feature count
- null method and permutation count
- probe split unit
- status and, when unavailable, a reason

Additional interpretation notes are stored in `report.attrs["notes"]`.
Inputs containing missing subject/label values, nonfinite features, or zero
total variance are rejected instead of returning misleading fractions.

The report remains scale-policy dependent, permutation validity still depends
on exchangeability, and subject-probe accuracy can reflect any stable
subject-specific signal. Those choices are exposed rather than hidden so the
result can be interpreted against the acquisition design.

## Implementation note: categorical design matrices

For crossed layouts, `_dummy_matrix` creates a full indicator block with one
row per observation and one column per categorical level. Subject and label
blocks are combined into ordinary least-squares design matrices. Keeping every
level makes fitted values independent of reference-category coding;
`numpy.linalg.lstsq` handles the redundant intercept. Reduced and full model
residual sums of squares then produce the adjusted unique effects.
