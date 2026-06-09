# `mne-features` vs `coco-pipe.descriptors`: Full Assessment

## Scope

This assessment compares the current `coco-pipe.descriptors` stack against
[`mne-features`](https://github.com/mne-tools/mne-features) at four levels:

1. Coverage
2. Provenance
3. Validity and stability
4. Runtime and workflow fit

No descriptor extraction behavior in `coco-pipe` was changed for this study.
The comparison was implemented as standalone tooling in
[scripts/compare_descriptors_vs_mne_features.py](/Users/hamzaabdelhedi/Projects/packages/coco-pipe/scripts/compare_descriptors_vs_mne_features.py),
with regression coverage in
[tests/test_compare_descriptors_vs_mne_features.py](/Users/hamzaabdelhedi/Projects/packages/coco-pipe/tests/test_compare_descriptors_vs_mne_features.py).

## Data sources used for this assessment

This document is grounded in three concrete outputs from the comparison harness:

1. Coverage/provenance inventory generated from the harness record set
2. Validity/stability review rows generated from the same inventory
3. Executed benchmark rows from the harness for the representative overlap case:
   - synthetic medium dataset
   - `64 epochs x 19 channels x 512 samples`
   - `overlap_combined`

The harness is capable of running a larger benchmark matrix, including multiple
synthetic sizes, parallel sweeps, and a real-data micro-case. For this
assessment, the hard numbers below come from the executed representative overlap
benchmark plus the generated coverage/review inventories.

## Completed script outputs

Two standalone scripts were executed for this update.

### 1. Coverage/provenance/assessment harness

Command:

```bash
./.venv_coco_pipe/bin/python scripts/compare_descriptors_vs_mne_features.py \
  --skip-benchmarks \
  --output-dir /tmp/descriptors_vs_mne_features_skip
```

Observed stdout:

```json
{
  "coverage_rows": 60,
  "review_rows": 73,
  "benchmark_rows": 0,
  "output_dir": "/tmp/descriptors_vs_mne_features_skip"
}
```

### 2. Numeric agreement script

Command:

```bash
./.venv_coco_pipe/bin/python scripts/compare_overlap_numeric_agreement.py \
  --output-dir /tmp/numeric_agreement_report
```

Observed stdout:

```json
{
  "rows": 17,
  "output_dir": "/tmp/numeric_agreement_report"
}
```

The rest of this document incorporates those script outputs plus the executed
representative benchmark rows from the comparison harness benchmark path.

## Environment and package versions

### `coco-pipe` benchmark environment

- Python: `3.11.12`
- `coco-pipe`: `0.0.1`
- `mne`: `1.11.0`
- `antropy`: `0.2.1`
- `neurokit2`: `0.2.13`
- `specparam`: `2.0.0rc6`
- `numpy`: `2.2.6`
- `scipy`: `1.14.1`
- `pydantic`: `2.12.5`

### `mne-features` benchmark environment

- Python: `3.14.3`
- `mne-features`: `0.3.2`
- `mne`: `1.11.0`
- `numpy`: `2.4.4`
- `scipy`: `1.17.1`
- `PyWavelets`: `1.9.0`
- `numba`: `0.64.0`
- `scikit-learn`: `1.8.0`

## API structure comparison

## `coco-pipe.descriptors`

Primary API shape:

- config-driven via `DescriptorConfig`
- execution via `DescriptorPipeline.extract(X, sfreq, channel_names)`
- explicit post-extraction grouping via `DescriptorPipeline.pool_channels(...)`
- result structure:
  - `X`
  - `descriptor_names`
  - `failures`

Architectural characteristics:

- family-aware config surface:
  - `bands`
  - `parametric`
  - `complexity`
- runtime planner with shared PSD reuse
- explicit precision/runtime knobs
- structured failure collection
- sensor-first output contract with optional pooled derivation
- naming designed for downstream aggregation and study workflows

Implication:

- This is a descriptor system designed as part of an EEG pipeline, not just a
  feature function catalog.

## `mne-features`

Primary API shape:

- function-driven via `extract_features(...)`
- sklearn-style transformer via `FeatureExtractor`
- feature selection by flat function names
- parameterization through `funcs_params`
- output is a flat feature matrix

Architectural characteristics:

- strong classical feature catalog
- univariate and bivariate feature functions
- sklearn `FeatureUnion`-style composition
- parallelism delegated to joblib/sklearn stack
- less domain-specific structure around EEG study outputs

Implication:

- This is better understood as a broad feature-extraction toolbox than as a
  study-integrated EEG descriptor runtime.

## Bottom line on API structure

`coco-pipe` is structurally better aligned with the current project because it
has:

- family-level configuration
- explicit descriptor naming
- explicit failure reporting
- pooled-channel derivation
- corrected spectral logic
- parametric spectral modeling

`mne-features` is structurally stronger as:

- a broad classical feature catalog
- an independent reference implementation source
- a potential sidecar for future classical bivariate features

## Coverage summary

The generated coverage/provenance matrix contains `60` rows.

### Domain counts

- `complexity`: `17`
- `spectral`: `13`
- `parametric`: `11`
- `utility/amplitude`: `12`
- `bivariate`: `5`
- `wavelet`: `2`

### Relationship counts

- `exact_overlap`: `10`
- `approx_overlap`: `6`
- `coco_only`: `23`
- `mne_only`: `21`

### Library implementation counts

- unique `coco-pipe` capabilities covered in the matrix: `39`
- unique `mne-features` capabilities covered in the matrix: `37`

## Provenance summary

### `coco-pipe` source-type counts

- `backend_wrapper`: `15`
- `numpy_scipy_custom`: `3`
- `project_specific_logic`: `12`
- `specparam_wrapper`: `9`

Interpretation:

- `coco-pipe` is a layered system.
- Its main value is not just wrapping third-party functions, but combining
  mature backends with project-specific spectral logic, parametric summaries,
  and workflow-aware behaviors.

### `mne-features` source-type counts

- `handcoded_numpy`: `17`
- `handcoded_scipy`: `6`
- `handcoded_numba`: `2`
- `mne_wrapper`: `10`
- `pywt_wrapper`: `2`

Interpretation:

- `mne-features` is stronger as a classical hand-coded toolbox.
- A large share of its value comes from direct implementations of standard EEG
  features, plus some MNE-backed spectral utilities and PyWavelets-backed
  wavelet/energy functions.

## Overlapping features

### Exact overlaps

These are the overlap rows that can be compared directly with aligned settings:

- `sample_entropy` ↔ `samp_entropy`
  - `coco-pipe`: `antropy.sample_entropy`
  - `mne-features`: `numpy + sklearn.neighbors.KDTree`
- `approx_entropy` ↔ `app_entropy`
  - `coco-pipe`: `antropy.app_entropy`
  - `mne-features`: `numpy + sklearn.neighbors.KDTree`
- `svd_entropy` ↔ `svd_entropy`
  - `coco-pipe`: `antropy.svd_entropy`
  - `mne-features`: `numpy SVD on delay embedding`
- `higuchi_fd` ↔ `higuchi_fd`
  - `coco-pipe`: `antropy.higuchi_fd`
  - `mne-features`: `numba-accelerated custom implementation`
- `katz_fd` ↔ `katz_fd`
  - `coco-pipe`: `antropy.katz_fd`
  - `mne-features`: `custom numpy formula`
- `hjorth_mobility` ↔ `hjorth_mobility`
  - `coco-pipe`: `antropy.hjorth_params`
  - `mne-features`: `custom numpy Hjorth formula`
- `hjorth_complexity` ↔ `hjorth_complexity`
  - `coco-pipe`: `antropy.hjorth_params`
  - `mne-features`: `custom numpy Hjorth formula`
- `zero_crossings` ↔ `zero_crossings`
  - `coco-pipe`: `numpy signbit transitions`
  - `mne-features`: `custom numpy thresholded crossings`
- `kurtosis` ↔ `kurtosis`
  - `coco-pipe`: `scipy.stats.kurtosis`
  - `mne-features`: `scipy.stats.kurtosis`
- `rms` ↔ `rms`
  - `coco-pipe`: `numpy sqrt(mean(x**2))`
  - `mne-features`: `custom numpy formula`

### Approximate overlaps

These are related features, but not strict numeric equivalents:

- `spectral_entropy` ↔ `spect_entropy`
  - aligned conceptually
  - PSD defaults and normalization choices differ
- `hurst_exponent` ↔ `hurst_exp`
  - aligned at the concept level
  - method details differ enough that trend comparison is more appropriate than
    strict equality
- `absolute_power` ↔ `pow_freq_bands`
- `log_absolute_power` ↔ `pow_freq_bands`
- `relative_power` ↔ `pow_freq_bands`
- `ratios` ↔ `pow_freq_bands`

Important caveat:

- `pow_freq_bands` is a flexible spectral utility.
- `coco-pipe` splits spectral outputs into explicit named outputs and exposes
  curated ratio behavior.
- These are therefore useful nearest-neighbor comparisons, not drop-in
  replacements.

## Numeric agreement between overlapping implementations

The numeric agreement script compared aligned overlap features on the same
deterministic synthetic input:

- dataset: `64 epochs x 19 channels x 512 samples`
- sampling rate: `128 Hz`
- comparisons: `17`
  - `10` exact-overlap scalar features
  - `2` approximate-overlap scalar features
  - `5` approximate-overlap band-power rows

### Exact-overlap numeric comparisons

| `coco-pipe` | `mne-features` | Pearson r | MAE | RMSE | Max abs error |
|---|---:|---:|---:|---:|---:|
| `sample_entropy` | `samp_entropy` | `0.999259` | `0.001705` | `0.002269` | `0.009275` |
| `approx_entropy` | `app_entropy` | `0.998602` | `0.001052` | `0.001371` | `0.005475` |
| `svd_entropy` | `svd_entropy` | `0.836548` | `1.653522` | `1.653558` | `1.687481` |
| `higuchi_fd` | `higuchi_fd` | `0.986073` | `0.008334` | `0.009164` | `0.018902` |
| `katz_fd` | `katz_fd` | `1.000000` | `6.12e-08` | `7.10e-08` | `2.38e-07` |
| `hjorth_mobility` | `hjorth_mobility` | `0.999591` | `0.000485` | `0.000812` | `0.004595` |
| `hjorth_complexity` | `hjorth_complexity` | `0.999151` | `0.002557` | `0.003765` | `0.018269` |
| `zero_crossings` | `zero_crossings` | `1.000000` | `0` | `0` | `0` |
| `kurtosis` | `kurtosis` | `1.000000` | `2.994065` | `2.994066` | `2.996925` |
| `rms` | `rms` | `1.000000` | `1.53e-08` | `1.75e-08` | `2.98e-08` |

### Approximate-overlap numeric comparisons

| `coco-pipe` | `mne-features` | Pearson r | MAE | RMSE | Max abs error |
|---|---:|---:|---:|---:|---:|
| `spectral_entropy` | `spect_entropy` | `0.250910` | `0.367897` | `0.419332` | `0.830184` |
| `hurst_exponent` | `hurst_exp` | `0.003929` | `0.499377` | `0.502294` | `1.327432` |
| `absolute_power[delta]` | `pow_freq_bands[band0]` | `0.994357` | `0.578416` | `0.579713` | `0.710346` |
| `absolute_power[theta]` | `pow_freq_bands[band1]` | `0.986207` | `0.140742` | `0.141897` | `0.199219` |
| `absolute_power[alpha]` | `pow_freq_bands[band2]` | `0.989504` | `0.015034` | `0.015647` | `0.035297` |
| `absolute_power[beta]` | `pow_freq_bands[band3]` | `0.995861` | `0.045107` | `0.045681` | `0.079060` |
| `absolute_power[gamma]` | `pow_freq_bands[band4]` | `0.995641` | `0.039989` | `0.040552` | `0.068651` |

### Interpretation of the numeric agreement

What aligns well:

- `sample_entropy` / `samp_entropy`
- `approx_entropy` / `app_entropy`
- `katz_fd`
- `hjorth_mobility`
- `hjorth_complexity`
- `zero_crossings`
- `rms`

These look close enough that `mne-features` is a credible validation oracle for
the corresponding `coco-pipe` implementations.

What aligns reasonably but not perfectly:

- `higuchi_fd`

This is still a good reference pair, but not one to treat as bitwise
interchangeable.

What does **not** look interchangeable despite name overlap:

- `svd_entropy`
  - correlation is only `0.836548`
  - absolute error is large: `1.653522`
- `kurtosis`
  - correlation is effectively `1.0`
  - but the offset is almost exactly `2.994`
  - this strongly suggests a convention mismatch, most likely Fisher vs
    non-Fisher kurtosis
- `spectral_entropy`
  - poor numeric agreement: `r=0.250910`
- `hurst_exponent`
  - essentially no agreement on this setup: `r=0.003929`

What is useful but only as nearest-neighbor comparison:

- band powers through `absolute_power` vs `pow_freq_bands`
  - correlations are high: roughly `0.986` to `0.996`
  - absolute scales differ substantially
  - this is consistent with different PSD and band-integration conventions

Practical consequence:

- `mne-features` is a strong oracle for some classical scalar overlaps, but not
  all same-named features should be treated as numerically equivalent.
- The most important caution flags from the actual value comparison are:
  - `svd_entropy`
  - `kurtosis`
  - `spectral_entropy`
  - `hurst_exponent`

## Non-overlapping capabilities

## High-value `coco-pipe`-only capabilities

These are major reasons not to switch away from `coco-pipe`:

- corrected spectral outputs:
  - `corrected_absolute_power`
  - `corrected_log_absolute_power`
  - `corrected_relative_power`
  - `corrected_ratios`
- parametric aperiodic and peak summaries:
  - `offset`
  - `exponent`
  - `knee`
  - `fit_error`
  - `r_squared`
  - `peak_count`
  - `peak_freq_dom`
  - `peak_power_dom`
  - `peak_bandwidth_dom`
  - `alpha_peak_freq`
  - `alpha_peak_power`
- workflow/runtime capabilities:
  - `pool_channels`
  - `shared_psd_planner`

These are not just missing in `mne-features`; they are central to the current
study design.

## Most useful `mne-features`-only capabilities

The comparison classifies `6` `mne-features` capabilities as good future
`borrow_formula` candidates:

- `skewness`
- `decorr_time`
- `line_length`
- `spect_slope`
- `energy_freq_bands`
- `spect_edge_freq`

Highlights:

- `line_length`
  - source: `handcoded_numpy`
  - backend: `custom numpy line-length formula`
  - usefulness: `high`
  - verdict: strong future native addition candidate
- `spect_edge_freq`
  - source: `mne_wrapper`
  - backend: PSD cumulative energy edge calculation
  - usefulness: `high`
  - verdict: good future native addition candidate
- `spect_slope`
  - source: `handcoded_scipy`
  - backend: PSD + linear regression
  - usefulness: `high`
  - verdict: useful future scalar feature, but should not replace `specparam`
    exponent
- `decorr_time`
  - source: `handcoded_numpy`
  - backend: FFT-based unbiased autocorrelation
  - usefulness: `medium`
  - verdict: interesting but not core for the current study

## Does `mne-features` have a parametric family like `coco-pipe`?

Short answer: no.

`mne-features` does **not** have direct equivalents for the current
`coco-pipe` parametric/specparam-based family:

- `offset`
- `exponent`
- `knee`
- `fit_error`
- `r_squared`
- `peak_count`
- `peak_freq_dom`
- `peak_power_dom`
- `peak_bandwidth_dom`
- `alpha_peak_freq`
- `alpha_peak_power`

What `mne-features` does have are classical spectral utilities, not parametric
spectral model outputs:

- `pow_freq_bands`
- `spect_slope`
- `spect_edge_freq`
- `energy_freq_bands`
- `spect_entropy`

These can be useful future additions or references, but they are not aperiodic
/ periodic decomposition, they do not fit a spectral model, and they do not
produce peak-model diagnostics.

So for the parametric part of the project, `coco-pipe` remains clearly stronger
and there is no realistic case for switching that part to `mne-features`.

## Sidecar-only candidates

The comparison classifies `6` rows as `sidecar_only`. These are better treated
as future separate-family or sidecar features than as immediate additions to the
current descriptor family.

Important examples:

- `phase_lock_val`
  - source: `handcoded_scipy`
  - backend: `scipy.signal.hilbert + custom PLV formula`
  - usefulness: `high`
  - verdict: scientifically useful, but long-term connectivity work should
    likely prefer `mne-connectivity`
- `teager_kaiser_energy`
  - source: `pywt_wrapper`
  - backend: wavelet-assisted Teager-Kaiser energy
  - verdict: interesting, but should be handled under a wavelet-oriented design,
    not folded into current scalar extraction casually

## Validity and stability review

The generated validity/stability review contains `73` unique
`feature_name x library` rows.

## Overlapping features: validity and stability verdict

### Where `coco-pipe` looks stronger

- spectral overlap cases
  - because the implementation is embedded in the same planner and naming model
    as the study runtime
  - because band outputs, corrected outputs, and ratio semantics are explicit
- runtime integration
  - `coco-pipe` descriptors are already packaged around the study’s needs:
    explicit arrays, structured failures, pooled-channel derivation, and
    parametric spectral support
- backend maintainability for classical complexity features
  - many overlap features are delegated to mature backends like `antropy`

### Where `mne-features` is still valuable

- as an independent reference implementation source
- for overlap validation on:
  - sample entropy
  - approximate entropy
  - SVD entropy
  - Higuchi FD
  - Katz FD
  - Hjorth mobility/complexity
  - zero crossings
  - kurtosis
  - RMS
- for classical feature ideas not currently present in `coco-pipe`

### Stability notes

`coco-pipe`

- stronger package-level integration
- lower API mismatch risk for this project
- medium backend risk where `specparam` or `neurokit2` are involved
- stronger naming and failure semantics

`mne-features`

- technically valuable, but with a heavier and older sklearn/joblib-style stack
- more environment-fragile in practice:
  - importing `mne-features` required a writable `HOME` for MNE config
  - process-based parallel paths were less straightforward in this environment
- still very useful as an independent oracle because many implementations are
  hand-coded rather than simple wrappers

## Runtime benchmark

## Executed benchmark case

One representative overlap benchmark was executed through the comparison harness
for both libraries:

- dataset: synthetic deterministic EEG-like signal
- shape: `64 epochs x 19 channels x 512 samples`
- subset: `overlap_combined`
- output size: `323` features for both libraries
- mode: sequential `n_jobs=1`

### Benchmark results

| Library | Python | Features | Cold import (s) | Cold extract (s) | Warm extract (s) | Warm median (5 runs) (s) | Peak RSS (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `coco-pipe` | `3.11.12` | `323` | `0.000147` | `8.164592` | `3.849242` | `3.586859` | `381.719` |
| `mne-features` | `3.14.3` | `323` | `5.156685` | `6.850293` | `6.544633` | `6.085572` | `342.703` |

### Benchmark interpretation

- `coco-pipe` was materially faster in steady-state extraction for this overlap
  workload:
  - `3.586859s` median warm extraction
  - versus `6.085572s` for `mne-features`
- `mne-features` paid a much larger cold import cost:
  - `5.156685s`
  - versus effectively negligible import cost for `coco-pipe.descriptors`
- `mne-features` had slightly lower peak RSS in this one case:
  - `342.703 MB`
  - versus `381.719 MB`

Important caveat:

- these measurements came from different Python environments:
  - `coco-pipe`: Python `3.11.12`
  - `mne-features`: Python `3.14.3`
- they are still decision-useful, but not publication-grade apples-to-apples
  measurements

## Workflow fit for the current project

The current study needs:

- sensor-first extraction
- pooled-channel derivation as a second step
- corrected spectral outputs
- parametric/aperiodic spectral summaries
- explicit descriptor naming
- structured failures
- compatibility with checkpointed shard workflows
- compatibility with subject-level aggregation and merged outputs

### `coco-pipe` fit

`coco-pipe` is a strong fit because it already supports:

- sensor-first descriptor extraction
- pooled-channel derivation through `pool_channels(...)`
- corrected spectra
- aperiodic and peak summaries through `specparam`
- deterministic descriptor names
- result structures that integrate directly with the study scripts

### `mne-features` fit

`mne-features` is a weaker fit as the primary runtime because it does not natively
solve the key project-specific needs:

- no corrected spectral family
- no aperiodic/peak modeling family
- no pooled-channel derivation contract
- no structured failure collection contract
- no project-aligned descriptor naming or workflow semantics

## Recommendation

## Keep native in `coco-pipe`

The comparison classifies `27` rows as `keep_native`.

This is the correct outcome for:

- all corrected spectral outputs
- all parametric/specparam-based outputs
- pooled-channel derivation
- planner/shared-PSD behavior
- project-aligned band logic and ratios

## Use `mne-features` as a validation oracle

The comparison classifies `12` rows as `use_as_reference`.

This is the best role for `mne-features` on the overlap set:

- sample entropy
- approximate entropy
- SVD entropy
- Higuchi FD
- Katz FD
- Hjorth mobility
- Hjorth complexity
- zero crossings
- kurtosis
- RMS
- plus the nearest-neighbor overlap cases around spectral entropy and Hurst

## Borrow later, selectively

The comparison classifies `6` rows as `borrow_formula`.

These are the most useful future candidates inspired by `mne-features`:

- `line_length`
- `spect_edge_freq`
- `spect_slope`
- `skewness`
- `decorr_time`
- `energy_freq_bands`

## Sidecar-only future roles

The comparison classifies `6` rows as `sidecar_only`.

These should stay out of the current descriptor runtime until they get their own
design:

- `phase_lock_val`
- other classical bivariate metrics
- wavelet-oriented energy features like `teager_kaiser_energy`

## Final answer to the switch question

Do not switch the project from `coco-pipe.descriptors` to `mne-features`.

Keep `coco-pipe` as the authoritative descriptor runtime because it is:

- better aligned with the current study architecture
- stronger on corrected and parametric EEG descriptors
- faster on the executed overlap benchmark
- structurally better for pooled channels, failures, and downstream aggregation

Use `mne-features` in three narrower roles:

1. validation oracle for overlapping classical descriptors
2. source of future scalar feature ideas worth adding natively later
3. possible sidecar inspiration for a future separate bivariate/connectivity
   design

## Reproducibility

Comparison harness:

- [scripts/compare_descriptors_vs_mne_features.py](/Users/hamzaabdelhedi/Projects/packages/coco-pipe/scripts/compare_descriptors_vs_mne_features.py)

Regression tests:

- [tests/test_compare_descriptors_vs_mne_features.py](/Users/hamzaabdelhedi/Projects/packages/coco-pipe/tests/test_compare_descriptors_vs_mne_features.py)

Example commands:

```bash
./.venv_coco_pipe/bin/python scripts/compare_descriptors_vs_mne_features.py --skip-benchmarks
```

```bash
./.venv_coco_pipe/bin/python scripts/compare_descriptors_vs_mne_features.py \
  --benchmark-profile quick \
  --real-data-bids-root /tmp/does_not_exist
```
