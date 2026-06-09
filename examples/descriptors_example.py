# ruff: noqa: E402
"""
Minimal descriptors example with explicit NumPy inputs.
"""

from __future__ import annotations

import numpy as np

from coco_pipe.descriptors import DescriptorPipeline
from coco_pipe.io import DataContainer


def main() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(12, 3, 256))
    t = np.linspace(0, 1, 256, endpoint=False)
    X[:, 0, :] += np.sin(2 * np.pi * 10 * t)
    X[:, 1, :] += np.sin(2 * np.pi * 6 * t)
    ids = np.asarray([f"obs-{idx:02d}" for idx in range(12)])

    config = {
        "families": {
            "bands": {
                "enabled": True,
                "outputs": ["absolute_power", "corrected_absolute_power"],
            },
            "parametric": {
                "enabled": True,
                "outputs": ["aperiodic"],
            },
            "complexity": {
                "enabled": True,
                "measures": ["sample_entropy", "hjorth_mobility"],
            },
        },
    }

    channels = ["Fz", "Cz", "Pz"]
    pipe = DescriptorPipeline(config)
    result = pipe.extract(
        X=X,
        ids=ids,
        sfreq=256.0,
        channel_names=channels,
    )

    # Apply pooling after extraction
    result = pipe.pool_channels(result, {"all": channels})

    print("Descriptor matrix shape:", result["X"].shape)
    print("First five names:", result["descriptor_names"][:5])
    print("Failure count:", len(result["failures"]))

    container = DataContainer(
        X=result["X"],
        dims=("obs", "feature"),
        coords={"feature": result["descriptor_names"]},
    )
    grouped = container.aggregate(
        by=["sub-01"] * 6 + ["sub-02"] * 6,
        stats=["mean", "std"],
    )

    print("Grouped descriptor shape:", grouped.X.shape)
    print("Grouped dims:", grouped.dims)
    print("Grouped stats:", grouped.coords["stat"].tolist())


if __name__ == "__main__":
    main()
