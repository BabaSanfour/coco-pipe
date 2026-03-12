# Dim Reduction Workflows

## Current State

The dim-reduction stack is centered on the existing `DimReduction` manager and
reducer contracts.

- Optional dependencies are now lazy at import time.
- `DimReduction` caches normalized state on:
  - `metrics_`
  - `metric_records_`
  - `quality_metadata_`
  - `diagnostics_`
  - `interpretation_`
  - `interpretation_records_`
- Plotting is done through `coco_pipe.viz.dim_reduction`, not through manager
  methods on `DimReduction`.
- `DimReduction` does not cache embeddings. Embeddings are returned explicitly
  from `transform()` and `fit_transform()` and must be passed explicitly to
  `score()`, plotting, or report-building paths that need them.
- `coco_pipe.dim_reduction.evaluation.core` is the evaluation authority used by
  `DimReduction.score()`.
- `MethodSelector` is now a post-hoc comparison layer over already-scored
  `DimReduction` objects and exposes tidy metric observations via `to_frame()`.

## Core Interfaces

Use `DimReduction` directly for most workflows:

```python
from coco_pipe.dim_reduction import DimReduction

reducer = DimReduction("PCA", n_components=2, random_state=42)
embedding = reducer.fit_transform(X, y=labels)
scores = reducer.score(embedding, X=X, labels=labels, times=timepoints)
interpretation = reducer.interpret(
    X,
    X_emb=embedding,
    analyses=["correlation"],
    feature_names=feature_names,
)
summary = reducer.get_summary()
```

`DimReduction.get_summary()` returns cached scalar metrics, reducer metadata,
diagnostics, tidy metric records, cached interpretation payloads, cached
interpretation records, and capability flags. It does not carry an embedding
payload.

Evaluation can be narrowed to specific metric families:

```python
from coco_pipe.dim_reduction.config import EvaluationConfig

config = EvaluationConfig(
    metrics=["trustworthiness", "continuity"],
    selection_metric="trustworthiness",
    selection_k=10,
    tie_breakers=["continuity"],
    separation_method="centroid",
)
```

Each reducer is scored directly:

```python
for reducer in reducers:
    embedding = reducer.fit_transform(X, y=labels)
    reducer.score(
        embedding,
        X=X,
        metrics=config.metrics,
        k_values=config.k_range,
        separation_method=config.separation_method,
    )
    reducer.interpret(
        X,
        X_emb=embedding,
        analyses=["correlation"],
        feature_names=feature_names,
    )
```

Then compare the scored reducers:

```python
from coco_pipe.dim_reduction.evaluation import MethodSelector

selector = MethodSelector(reducers).collect()
ranked = selector.rank_methods(
    selection_metric=config.selection_metric,
    selection_k=config.selection_k,
    tie_breakers=config.tie_breakers,
)
best_name = ranked.iloc[0]["method"]
best = selector.reducers[best_name]
```

When trajectory labels are available, `separation_method` is passed through
during `score()` to `trajectory_separation(..., method=...)` for evaluator-level
separation summaries.

Feature interpretation is separate from preservation scoring:

- `score()` evaluates whether the embedding preserves structure
- `interpret()` evaluates which input features appear to drive the embedding

`interpret()` delegates to the pure backend
`coco_pipe.dim_reduction.analysis.interpret_features(...)` and currently
supports:

- `correlation`
- `perturbation`
- `gradient`

## Custom Reducers

`BaseReducer` is a supported extension point and is re-exported from
`coco_pipe.dim_reduction`.

```python
from sklearn.decomposition import PCA

from coco_pipe.dim_reduction import BaseReducer


class CustomPCAReducer(BaseReducer):
    @property
    def capabilities(self):
        caps = super().capabilities
        caps.update({"is_linear": True, "has_components": True})
        return caps

    def fit(self, X, y=None):
        self.model = PCA(n_components=self.n_components, **self.params)
        self.model.fit(X)
        return self

    def transform(self, X):
        return self.model.transform(X)
```

For reducers with nonstandard inputs, override `capabilities` to declare the
expected `input_ndim` and `input_layout`.

If a reducer depends on heavy optional libraries, keep those imports inside
`fit()` / `transform()` paths. The helper
`coco_pipe.utils.import_optional_dependency(...)` exists for built-in reducers
and custom advanced integrations, but it is not the main public entry point.

## Supported Metric Shapes

Metric plotting and reporting now work from tidy observations with these columns:

- `method`
- `metric`
- `value`
- `scope`
- `scope_value`

Optional columns such as `group`, `condition`, `pair`, `subject`, `session`,
`seed`, and `fold` are preserved when present.

## Metric Plot Types

Use `plot_metrics(..., plot_type=...)` or the report comparison helpers.
Embedding visualizations are also external to `DimReduction`; pass the explicit
embedding array to the plotting function you need.

- `grouped_bar`: one scalar per method/metric
- `box` / `boxen`: repeated observations
- `violin`: dense repeated observations
- `raincloud`: violin + box + points
- `strip` / `swarm`: small repeated samples
- `heatmap`: method x metric or method x scope
- `line`: metric sweeps over `k`, time, or windows
- `dumbbell`: direct two-method deltas

Default behavior:

- global scalars -> grouped bars
- repeated observations -> raincloud
- varying `scope_value` -> line
- explicit matrix summaries -> heatmap

## Visualization Entry Points

The dim-reduction viz surface is data-first and explicit. Plotting helpers do
not read manager-owned embedding or context state.

- `plot_embedding(embedding, labels=..., metadata=...)`
- `plot_metrics(metric_records, metric=..., scope=..., method=...)`
- `plot_shepard_diagram(X, embedding, distances=...)`
- `plot_trajectory(trajectories, times=..., labels=..., values=...)`
- `plot_trajectory_metric_series(series, times=..., labels=...)`
- `plot_feature_importance(scores_or_records, analysis=..., method=..., dimension=...)`
- `plot_feature_correlation_heatmap(correlation_payload, method=...)`
- `plot_interpretation(interpretation_payload, analysis=..., method=..., dimension=...)`

`plot_trajectory(...)` and `plot_trajectory_metric_series(...)` require native
trajectory tensors or explicit time-series arrays. They do not reshape flat 2D
embeddings or infer grouping metadata.

## Generic Trajectories

Trajectory scoring is not EEG-specific. Any grouped or ordered embedding can
use trajectory-native metrics when:

- the embedding is already a 3D tensor `(trajectory, time, dim)`

Trajectory reshaping or unstacking must happen upstream. The evaluation module
does not reconstruct 3D trajectories from flat 2D embeddings.

Trajectory outputs include:

- `trajectory_speed_mean`
- `trajectory_speed_peak`
- `trajectory_acceleration_mean`
- `trajectory_acceleration_peak`
- `trajectory_curvature_mean`
- `trajectory_curvature_peak`
- `trajectory_turning_angle_mean`
- `trajectory_turning_angle_peak`
- `trajectory_dispersion_mean`
- `trajectory_dispersion_peak`
- `trajectory_path_length_final`
- `trajectory_displacement_final`
- `trajectory_tortuosity_final`
- pairwise separation AUC / peak summaries when labels exist per trajectory

Detailed timecourses are cached under `diagnostics_`.

`trajectory_dispersion` in the evaluation pipeline is currently the global,
unlabeled dispersion over all trajectories. This is narrower than the lower-level
`geometry.py` primitive, which can also compute label-conditioned dispersion.
Trajectory labels are only used automatically for `trajectory_separation`.

Trajectory metrics are descriptive outputs for plotting and reporting. They are
not used as automatic method-selection metrics by default.

## Reports

`Report.add_reduction()` consumes `get_summary()` when available and accepts an
explicit embedding payload when the section should render an embedding or
trajectory plot.

It can render:

- interactive embeddings when `X_emb` is provided explicitly
- trajectory plots for 3D embeddings
- scalar metric tables and charts
- loss and scree diagnostics
- co-ranking heatmaps
- trajectory metric timecourses
- interpretation plots from `interpretation` / `interpretation_records`

`Report.add_comparison()` accepts tidy metric frames or `MethodSelector`
instances directly.

`from_reductions(...)` follows the same rule: pass `embeddings=[...]` explicitly
when the report should include embedding or trajectory plots.

Shepard plots and comparison/report views reuse cached diagnostics such as
`shepard_distances_` and `coranking_matrix_` when those artifacts already exist.

## End-to-End Execution

Batch execution should use `coco_pipe.io.load_data` plus `DimReduction`
directly. The old `DimReductionPipeline` compatibility wrapper has been
removed.

## Dependency Notes

Heavy optional libraries such as `torch`, `umap`, `meegkit`, and `pydmd` are
loaded inside reducer methods rather than at package import time.

For a complete dim-reduction install, use the umbrella extra:

```bash
pip install coco-pipe[dim-red]
```

Selective extras remain available when you only need part of the reducer stack:

```bash
pip install coco-pipe[dask]
pip install coco-pipe[neighbor]
pip install coco-pipe[parametric-umap]
pip install coco-pipe[ivis]
pip install coco-pipe[topology]
pip install coco-pipe[spatiotemporal]
pip install coco-pipe[eeg]
```

The `neighbor` and `dim-red` extras include `faiss-cpu`, so Pacmap can use
`nn_backend="faiss"` by default on supported platforms.

Base imports that should remain lightweight:

- `import coco_pipe.io`
- `import coco_pipe.report`
- `import coco_pipe.dim_reduction`

## Migration Notes

Legacy local note files under `coco_pipe/dim_reduction/` were consolidated into
this page and can be removed once no longer referenced.
