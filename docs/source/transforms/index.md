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
| `leace` | `LeaceEraser` | `(observation, feature)` | Erase the directions in which subjects differ most |
| `ea_mean` | `EuclideanAlign` | `(observation, feature)` | Center every subject |
| `ea_coral` | `EuclideanAlign` | `(observation, feature)` | Center and whiten every subject |
| `ra` | `RiemannAlign` | `(window, token, feature)` | Recenter each subject's token covariances and project to tangent space |

`ea_mean`, `ea_coral` and `ra` recenter every subject, so they remove labels that
are constant within a subject (a diagnosis, age); `leace` is learned on training
subjects and can keep such a label (see the guide).

Each also has a `label` (`Leace`, `EaMean`, `EaCoral`, `Ra`; `EaMeanCh` and
`EaCoralCh` with `channel_axis`), which
{func}`~coco_pipe.io.write_transformed_embeddings` adds to the label of the
embedding files it writes.

Temporal trajectories also support participant-specific PCA plus orthogonal
Procrustes alignment:

| Class | Input | Main operation |
|---|---|---|
| `TemporalProcrustesAlignment` | `(observation, feature, time)` | Rotate each participant's PCA trajectories toward a training template |

See [Subject alignment](subject_alignment.md) for the vector/token methods and
[Temporal trajectory alignment](temporal_alignment.md) for temporal decoding.
