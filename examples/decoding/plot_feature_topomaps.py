# ruff: noqa: E402
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend("Agg")

from coco_pipe.viz.decoding import plot_feature_importance, plot_feature_stability
from tests.fixtures.synthetic_result import (
    make_synthetic_feature_metadata,
    make_synthetic_result,
)

result = make_synthetic_result()
feature_metadata = pd.DataFrame(make_synthetic_feature_metadata())
for fig, ax in [
    plot_feature_importance(result),
    plot_feature_stability(result),
]:
    plt.close(fig)

try:
    from coco_pipe.viz.decoding import plot_decoding_topomap

    coords = feature_metadata.drop_duplicates("Sensor").set_index("Sensor")[["x", "y"]]
    sensor_df = pd.DataFrame(
        {"FeatureName": coords.index, "Importance": range(len(coords))}
    )
    fig, ax = plot_decoding_topomap(sensor_df, value="Importance", coords=coords)
    plt.close(fig)
except ImportError:
    pass
