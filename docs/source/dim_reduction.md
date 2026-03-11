# Dim Reduction Workflows

## Current State

The dim-reduction stack is centered on the existing `DimReduction` manager and
reducer contracts.

- Optional dependencies are now lazy at import time.
- `DimReduction` caches normalized state on:
  - `context_`
  - `metrics_`
  - `quality_metadata_`
  - `diagnostics_`
- `MethodSelector` keeps:
  - per-method wide result tables in `results_`
  - reusable artifacts in `artifacts_`
  - tidy metric observations via `to_frame()`

## Core Interfaces

Use `DimReduction` directly for most workflows:

```python
from coco_pipe.dim_reduction import DimReduction

reducer = DimReduction("PCA", n_components=2, random_state=42)
reducer.set_context(
    labels=labels,
    metadata={"subject": subjects, "condition": conditions},
    sample_ids=sample_ids,
    groups=trial_ids,
    times=timepoints,
)
embedding = reducer.fit_transform(X, y=labels)
scores = reducer.score(X)
summary = reducer.get_summary()
```

Context is stored once and reused by plotting and reporting.

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
`coco_pipe.dim_reduction.reducers.base.import_optional_dependency(...)` exists
for built-in reducers and custom advanced integrations, but it is not the main
public entry point.

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

## Generic Trajectories

Trajectory scoring is not EEG-specific. Any grouped or ordered embedding can
use trajectory-native metrics when either:

- the embedding is already a 3D tensor `(trajectory, time, dim)`, or
- `set_context(groups=..., times=...)` is provided for a 2D embedding

Trajectory outputs include:

- `trajectory_speed_mean`
- `trajectory_speed_peak`
- `trajectory_curvature_mean`
- `trajectory_curvature_peak`
- pairwise separation AUC / peak summaries when labels exist per trajectory

Detailed timecourses are cached under `diagnostics_`.

## Reports

`Report.add_reduction()` now consumes `get_summary()` when available and falls
back to normalized attributes otherwise.

It can render:

- interactive embeddings
- trajectory plots for 3D embeddings
- scalar metric tables and charts
- loss and scree diagnostics
- co-ranking heatmaps
- trajectory metric timecourses

`Report.add_comparison()` accepts tidy metric frames or `MethodSelector`
instances directly.

## End-to-End Execution

For legacy scripts and simple batch execution, `DimReductionPipeline` is now a
thin compatibility wrapper around:

- `coco_pipe.io.load_data`
- `DimReduction`
- optional report generation

It saves `.npz` outputs with at least:

- `reduced`
- `ids`
- `labels`
- `method`
- `metrics_json`

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

The `neighbor` and `dim-red` extras include `faiss-cpu`, so PaCMAP can use
`nn_backend="faiss"` by default on supported platforms.

Base imports that should remain lightweight:

- `import coco_pipe.io`
- `import coco_pipe.report`
- `import coco_pipe.dim_reduction`

## Migration Notes

Legacy local note files under `coco_pipe/dim_reduction/` were consolidated into
this page and can be removed once no longer referenced.
