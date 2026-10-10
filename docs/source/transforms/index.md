# Transforms

`coco_pipe.transforms` contains reusable scikit-learn-compatible data
transformers for subject structure in vectors, token representations, and
temporal trajectories.

```{toctree}
:maxdepth: 2

subject_alignment
temporal_alignment
```

The public registry exposes four methods:

| Registry name | Class | Input | Main operation |
|---|---|---|---|
| `leace` | `LeaceEraser` | `(observation, feature)` | Remove the linearly predictable subject subspace |
| `ea_mean` | `EuclideanAlign` | `(observation, feature)` | Center every subject |
| `ea_coral` | `EuclideanAlign` | `(observation, feature)` | Center and whiten every subject |
| `ra` | `RiemannAlign` | `(window, token, feature)` | Recenter subject covariance matrices and project to tangent space |

Temporal trajectories also support participant-specific PCA plus orthogonal
Procrustes alignment:

| Class | Input | Main operation |
|---|---|---|
| `TemporalProcrustesAlignment` | `(observation, feature, time)` | Rotate each participant's PCA trajectories toward a training template |

See [Subject alignment](subject_alignment.md) for the vector/token methods and
[Temporal trajectory alignment](temporal_alignment.md) for temporal decoding.
