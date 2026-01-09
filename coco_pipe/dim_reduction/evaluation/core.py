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

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Any, Tuple
from tqdm.auto import tqdm
import logging

logger = logging.getLogger(__name__)

from ..core import DimReduction
from .metrics import compute_coranking_matrix, trustworthiness, continuity, lcmc, compute_mrre

class MethodSelector:
    """
    Select the best dimensionality reduction method via quantitative evaluation.

    This class runs multiple `DimReduction` instances on the same data, computes
    rigorous quality metrics (Trustworthiness, Continuity, LCMC, MRRE) across a 
    range of neighborhood sizes (scale-space analysis), and provides visualization 
    tools to compare them.

    Parameters
    ----------
    reducers : List[DimReduction] or Dict[str, DimReduction]
        List or dictionary of configured DimReduction instances.
        If list, names are inferred from the `name` attribute.
    data : np.ndarray, optional
        Data to fit/transform. Can be passed later in `run`.
    target : np.ndarray, optional
        Labels for plotting.
    
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

    def __init__(self, 
                 reducers: Union[List[DimReduction], Dict[str, DimReduction]], 
                 data: Optional[np.ndarray] = None,
                 target: Optional[np.ndarray] = None):
        
        if isinstance(reducers, list):
            self.reducers = {}
            for r in reducers:
                name = getattr(r, 'name', r.__class__.__name__)
                self.reducers[name] = r
        else:
            self.reducers = reducers
            
        self.data = data
        self.target = target
        self.embeddings_ = {}
        self.results_ = {}
        self.Qs_ = {} 

    def run(self, 
            X: Optional[np.ndarray] = None, 
            y: Optional[np.ndarray] = None, 
            k_range: Union[List[int], np.ndarray, "EvaluationConfig"] = [5, 10, 20, 50, 100]) -> "MethodSelector":
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
            Neighborhood sizes to evaluate. Can also pass an EvaluationConfig object directly.

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

        if X is not None:
            self.data = X
        if y is not None:
            self.target = y
            
        if self.data is None:
            raise ValueError("No data provided.")
            
        logger.info(f"Evaluating {len(self.reducers)} methods on {self.data.shape[0]} samples...")
        
        from joblib import Parallel, delayed
        
        # Helper function for parallel execution
        # Must be picklable, so we use a static method or standalone function
        results_list = Parallel(n_jobs=-1)(
            delayed(_evaluate_single_method)(
                name, reducer, self.data, self.target, k_vals
            ) for name, reducer in tqdm(self.reducers.items(), desc="Methods")
        )
        
        # Aggregate results
        for name, emb, Q, df_metrics in results_list:
            self.embeddings_[name] = emb
            self.Qs_[name] = Q
            self.results_[name] = df_metrics
            
        return self

def _evaluate_single_method(name: str, reducer: DimReduction, 
                            data: np.ndarray, target: Optional[np.ndarray], 
                            k_vals: List[int]) -> Tuple[str, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Evaluate a single reducer. Worker function for parallel execution.
    """
    # Fit/Transform
    if hasattr(reducer, 'embedding_') and reducer.embedding_ is not None:
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
        
        metrics_list.append({
            'k': k,
            'trustworthiness': t_score,
            'continuity': c_score,
            'lcmc': l_score,
            'mrre_intrusion': mrre_int,
            'mrre_extrusion': mrre_ext,
            'mrre_total': mrre_int + mrre_ext
        })
    
    return name, emb, Q, pd.DataFrame(metrics_list)

    def plot(self, metric: str = 'trustworthiness', ax=None) -> Any:
        """
        Plot comparison curves (Quality vs Neighborhood Size).
        
        Parameters
        ----------
        metric : str, default='trustworthiness'
            The metric to plot. Options: 'trustworthiness', 'continuity', 'lcmc', 'mrre_total'.
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        from ...viz.dim_reduction import plot_comparison
        return plot_comparison(self, metric=metric, ax=ax)
