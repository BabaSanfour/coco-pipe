from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_visualizations():
    """
    Prevent plots from showing up during tests.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.show = MagicMock()
