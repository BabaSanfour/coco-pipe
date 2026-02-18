"""
Method Selection Core
=====================

Core engine for evaluating and selecting the best dimensionality reduction method
for a given dataset.

Classes
-------
MethodSelector
    Orchestrates the training, embedding, and evaluation of multiple reducers.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from ..config import EvaluationConfig
    from ..core import DimReduction

from .metrics import (
    compute_coranking_matrix,
    compute_mrre,
    continuity,
    lcmc,
    trustworthiness,
)

logger = logging.getLogger(__name__)


class MethodSelector:
    """
    Select the best dimensionality reduction method via quantitative evaluation.

    This class runs multiple `DimReduction` instances on the same data, computes
    rigorous quality metrics (Trustworthiness, Continuity, LCMC, MRRE) across a
    range of neighborhood sizes (scale-space analysis), and provides visualization
    tools to compare them.

    Parameters
    ----------
    reducers : dict or list
        Dictionary mapping names to `DimReduction` instances, or a list of
        `DimReduction` instances.
    n_jobs : int, default=1
        Number of parallel jobs for evaluation. Use -1 for all cores.
    backend : str, optional
        Joblib backend to use (e.g., 'threading', 'multiprocessing', 'loky').
        Defaults to None (loky if n_jobs != 1).

    Attributes
    ----------
    results_ : Dict[str, pd.DataFrame]
        Dictionary mapping reducer names to DataFrames containing metrics vs k.
    embeddings_ : Dict[str, np.ndarray]
        Computed embeddings.

    Examples
    --------
    >>> from coco_pipe.dim_reduction import DimReduction
    >>> from coco_pipe.dim_reduction.evaluation import MethodSelector
    >>> import numpy as np
    >>> X = np.random.rand(100, 50)
    >>> reducers = [DimReduction("PCA"), DimReduction("UMAP")]
    >>> selector = MethodSelector(reducers, data=X)
    >>> selector.run()
    >>> selector.plot(metric='trustworthiness')
    """

    def __init__(
        self,
        reducers: Union[Dict[str, "DimReduction"], List["DimReduction"]],
        n_jobs: int = 1,
        backend: Optional[str] = None,
    ):
        if isinstance(reducers, list):
            self.reducers = {r.name or r.method: r for r in reducers}
        else:
            self.reducers = reducers

        self.n_jobs = n_jobs
        self.backend = backend
        self.data = None
        self.target = None
        self.embeddings_ = {}
        self.results_ = {}
        self.Qs_ = {}

    def run(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        k_range: Union[List[int], np.ndarray, "EvaluationConfig"] = [
            5,
            10,
            20,
            50,
            100,
        ],
    ) -> "MethodSelector":
        """
        Run the evaluation pipeline.

        1. Fits all reducers (if not fitted).
        2. Computes embeddings.
        3. Computes Co-ranking matrix Q for each (efficiently).
        4. Calculates metrics (T, C, LCMC, MRRE) across `k_range`.

        Parameters
        ----------
        X : np.ndarray, optional
            Data to evaluate. Overrides init data.
        y : np.ndarray, optional
            Labels.
        k_range : list of int or EvaluationConfig, default=[5, 10, 20, 50, 100]
            Neighborhood sizes to evaluate. Can also pass an EvaluationConfig object
            directly.

        Returns
        -------
        self : MethodSelector
            Returns self for chaining.
        """
        from ..config import EvaluationConfig

        # Handle Config object
        if isinstance(k_range, EvaluationConfig):
            config = k_range
            k_vals = config.k_range
            # Could also use config.metrics here to filter which metrics to compute
            # For now, we compute all standard ones as per original implementation
        else:
            k_vals = k_range

        self.data = X
        self.target = y

        logger.info(
            f"Evaluating {len(self.reducers)} methods on {self.data.shape[0]} "
            f"samples..."
        )

        from joblib import Parallel, delayed

        # Helper function for parallel execution
        # Must be picklable, so we use a static method or standalone function
        results_list = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
            delayed(_evaluate_single_method)(
                name, reducer, self.data, self.target, k_vals
            )
            for name, reducer in tqdm(self.reducers.items(), desc="Methods")
        )

        # Aggregate results
        for name, emb, Q, df_metrics in results_list:
            self.embeddings_[name] = emb
            self.Qs_[name] = Q
            self.results_[name] = df_metrics

        return self

    def plot(self, metric: str = "trustworthiness", ax=None) -> Any:
        """
        Plot comparison curves (Quality vs Neighborhood Size).

        Parameters
        ----------
        metric : str, default='trustworthiness'
            The metric to plot. Options: 'trustworthiness', 'continuity', 'lcmc',
            'mrre_total'.
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        from ...viz.dim_reduction import plot_comparison

        return plot_comparison(self, metric=metric, ax=ax)


def _evaluate_single_method(
    name: str,
    reducer: "DimReduction",
    data: np.ndarray,
    target: Optional[np.ndarray],
    k_vals: List[int],
) -> Tuple[str, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Evaluate a single reducer. Worker function for parallel execution.
    """
    # Fit/Transform
    if hasattr(reducer, "embedding_") and reducer.embedding_ is not None:
        emb = reducer.embedding_
    else:
        emb = reducer.fit_transform(data, target)

    # Compute Q matrix
    Q = compute_coranking_matrix(data, emb)

    # Calculate Metrics across k
    metrics_list = []
    n_samples = data.shape[0]

    for k in k_vals:
        if k >= n_samples:
            continue

        t_score = trustworthiness(Q, k)
        c_score = continuity(Q, k)
        l_score = lcmc(Q, k)
        mrre_int, mrre_ext = compute_mrre(Q, k)

        metrics_list.append(
            {
                "k": k,
                "trustworthiness": t_score,
                "continuity": c_score,
                "lcmc": l_score,
                "mrre_intrusion": mrre_int,
                "mrre_extrusion": mrre_ext,
                "mrre_total": mrre_int + mrre_ext,
            }
        )

    return name, emb, Q, pd.DataFrame(metrics_list)
