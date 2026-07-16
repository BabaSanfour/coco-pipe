# Transforms

`coco_pipe.transforms` contains reusable scikit-learn-compatible data
transformers. The current transform family removes or normalizes stable subject
structure in frozen embeddings and token representations.

```{toctree}
:maxdepth: 2

subject_alignment
```

The public registry exposes four methods:

| Registry name | Class | Input | Main operation |
|---|---|---|---|
| `leace` | `LeaceEraser` | `(observation, feature)` | Remove the linearly predictable subject subspace |
| `ea_mean` | `EuclideanAlign` | `(observation, feature)` | Center every subject |
| `ea_coral` | `EuclideanAlign` | `(observation, feature)` | Center and whiten every subject |
| `ra` | `RiemannAlign` | `(window, token, feature)` | Recenter subject covariance matrices and project to tangent space |

See [Subject alignment](subject_alignment.md) for fit scope, leakage rules,
configuration, diagnostics, and method-specific assumptions.
