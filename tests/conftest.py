import os
import tempfile
from unittest.mock import MagicMock

import pytest

# --- Native threading guard -------------------------------------------------
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@pytest.fixture(scope="session", autouse=True)
def mock_visualizations():
    """
    Prevent plots from showing up during tests.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["figure.max_open_warning"] = 0
    plt.show = MagicMock()


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any figures a test left open.

    Plot helpers return figures to their caller rather than closing them, so
    without this the per-test figures accumulate and matplotlib emits a
    "More than 20 figures have been opened" warning mid-suite.
    """
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture(scope="session", autouse=True)
def _sandbox_runtime_env():
    """Keep third-party cache/config writes inside writable temp dirs."""
    tmp_root = os.path.join(tempfile.gettempdir(), "coco_pipe_test_runtime")
    mpl_dir = os.path.join(tmp_root, "mplconfig")
    mne_dir = os.path.join(tmp_root, "mne")
    os.makedirs(mpl_dir, exist_ok=True)
    os.makedirs(mne_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_dir)
    os.environ.setdefault("MNE_HOME", mne_dir)
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
