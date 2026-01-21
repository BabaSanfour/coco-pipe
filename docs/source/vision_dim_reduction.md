# Strategic Vision: Dimensionality Reduction Module

**A Comprehensive Analysis for Scientific Dimensionality Reduction**

## 1. The Scientific Imperative for Domain-Specific Reduction

The contemporary landscape of scientific data analysis, particularly within the domains of neuroimaging (M/EEG), transcriptomics, and high-dimensional embeddings, stands at a critical inflection point. As data acquisition technologies advance—yielding higher temporal resolutions in electroencephalography (EEG) and massive feature sets in single-cell sequencing—the computational pipelines designed to interpret this data must evolve concomitantly.

The module `coco_pipe.dim_reduction` serves as an **Analytic Engine** that positions `coco_pipe` not only as a general-purpose utility but also a specialized instrument for scientific discovery, this vision outlines a fundamental re-evaluation of its methodological scope. The central thesis is that "generic" manifold learning, while valuable for initial exploration, often fails to capture the intrinsic **physical** and **topological** constraints of scientific data—specifically oscillatory dynamics, causal trial structures, and manifold loops.

We structure our request for a scientific instrument around **Three Tiers of Inquiry**:
1.  **Tier 1**: Exploratory Foundation (Generalist Tools)
2.  **Tier 2**: Domain-Specific Extension (Specialist Tools)
3.  **Tier 3**: Validation Engine (Epistemology)

---

## 2. Tier 1: The Exploratory Foundation (Generalist Tools)

*The Swiss Army Knives of Data Science.*

The current suite of reducers—PCA, t-SNE, UMAP, PaCMAP, TriMap, PHATE—represents the gold standard of general data science. These tools are essential for the initial "scouting" of data territory.

### 2.1 The Limitations of the "Generalist" Paradigm

While statistically sound, these algorithms exhibit specific blind spots when applied to M/EEG data [1]:

*   **Variance Bias (PCA)**: PCA orthogonalizes data based on variance. In EEG, artifactual signals (ocular movements, muscular noise) often possess the highest variance, effectively "hijacking" principal components from subtler, biologically relevant neural oscillations [2]. Furthermore, PCA is time-agnostic; shuffling time points yields identical components, ignoring the dynamic nature of brain states [3].
*   **Manifold Fragmentation (t-SNE/UMAP)**: Non-linear methods excel at separating states but struggle with continuous trajectories. t-SNE often fragments continuous manifolds into discrete clusters, creating the illusion of distinct cell types or brain states where only gradients exist [4].
*   **Physics-Agnostic**: Even PHATE, which preserves global structure via diffusion, does not explicitly model oscillatory physics or supervised experimental designs [5].

---

## 3. Tier 2: The Domain-Specific Extension (Specialist Tools)

*Capturing the Physics of the Data.*

To elevate `coco_pipe`, we broaden the algorithmic repertoire to include methods that incorporate domain knowledge—specifically **Time**, **Task**, and **Topology**.

### 3.1 Spatiotemporal Decomposition: Dynamic Mode Decomposition (DMD)

*   **The Physics**: M/EEG data is defined by oscillation and propagation. DMD shifts from statistical description (variance) to dynamical description (stability and frequency) [6].
*   **Mathematical Basis**: DMD approximates the linear operator $\mathbf{A}$ that maps the state $\mathbf{x}_t$ to $\mathbf{x}_{t+1}$:
    $$
    \mathbf{x}_{k+1} = \mathbf{A}\mathbf{x}_k
    $$
    Using Singular Value Decomposition (SVD) on time-shifted matrices, it extracts eigenvalues $\lambda_j$.
*   **Scientific Utility**:
    *   **Oscillation Frequency**: $\text{Im}(\ln(\lambda_j))$ isolates frequency bands (e.g., Alpha, Gamma) without pre-specified filters.
    *   **Growth/Decay**: $|\lambda_j|$ identifies unstable modes (e.g., seizure onset) [7]. This enables "Dynamical Filtering"—retaining only stable oscillatory modes.

### 3.2 Supervised Enhancement: Task-Related Component Analysis (TRCA)

*   **The Structure**: Scientific experiments are often structured around repetitive tasks. TRCA is a supervised method maximizing **reproducibility** across trials [8].
*   **Mathematical Basis**: TRCA maximizes the Rayleigh quotient of inter-trial covariance $\mathbf{S}$ vs aggregate covariance $\mathbf{Q}$:
    $$
    J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S} \mathbf{w}}{\mathbf{w}^T \mathbf{Q} \mathbf{w}}
    $$
*   **Scientific Utility**: For BCI (SSVEP) and ERP analysis, TRCA significantly outperforms unsupervised ICA/PCA by focusing specifically on the signal locked to the experimental stimulus [9].

### 3.3 Topological Integrity: Topological Autoencoders (TopoAE)

*   **The Shape**: Biological manifolds contain loops (cell cycles) and voids. Standard Autoencoders (MSE loss) often cause "topological collapse."
*   **Mathematical Basis**: TopoAE adds a regularization term based on **Persistent Homology**:
    $$
    \mathcal{L} = \mathcal{L}_{recon} + \lambda \mathcal{L}_{topo}
    $$
    This forces the latent space to preserve the birth/death of topological features (holes, components) found in the input [10].
*   **Scientific Utility**: Critical for developmental trajectories and neural limit cycles, ensuring the "shape" of the data is not lost [11].

### 3.4 Scalable Parametric Mapping: IVIS

*   **The Scale**: Graph-based methods ($O(N^2)$) struggle with massive datasets (Biobank scale).
*   **Mathematical Basis**: IVIS uses Siamese Neural Networks with triplet loss to learn a parametric mapping function $f(x)$ [12].
*   **Scientific Utility**: Scales linearly ($O(N)$), enabling processing of millions of samples. The trained model is reusable, allowing new patient data to be instantaneously projected onto a reference "healthy" manifold [13].

---

## 4. Tier 3: The Validation Engine (Epistemology)

*From "It Looks Good" to "It Is True".*

Visualizations are inherently distorted. `coco_pipe` implements a rigorous metrics module to quantify this distortion via the **Co-ranking Matrix Framework**.

### 4.1 The Co-ranking Matrix ($Q$)

The matrix $Q$ encapsulates rank-order changes between high-dimensional ($\rho_{ij}$) and low-dimensional ($r_{ij}$) neighbors:
$$
Q_{kl} = | \{ (i, j) : \rho_{ij} = k \text{ and } r_{ij} = l \} |
$$
*   **Lower Triangle**: Extrusions (Tearing).
*   **Upper Triangle**: Intrusions (Crushing) [14].

### 4.2 Critical Scientific Metrics

*   **Trustworthiness ($T$)**: Penalizes intrusions ($k > l$). "Can I trust these clusters?"
    $$
    T(k) = 1 - \frac{2}{Nk(2N - 3k - 1)} \sum_{i=1}^N \sum_{j \in U_k(i)} (r_{ij} - k)
    $$
*   **Continuity ($C$)**: Penalizes extrusions ($k < l$). "Did I break the trajectory?"
    $$
    C(k) = 1 - \frac{2}{Nk(2N - 3k - 1)} \sum_{i=1}^N \sum_{j \in V_k(i)} (\rho_{ij} - k)
    $$
*   **LCMC & MRRE**: Normalized scores accounting for chance overlap and rank magnitude errors [15, 16].

### 4.3 Implementation Optimization
To make this viable for large data:
1.  **Tree-Based Search**: KDTree/BallTree for $O(N \log N)$ neighbor finding.
2.  **Vectorization**: Full NumPy vectorization of $Q$ matrix population.
3.  **Benchmarking Class**: `MethodSelector` to generate "Quality Curves" (Metric vs $k$) automatically.

---

## 5. Advanced Inquiry: Dynamics & Attribution

We extend visualization beyond static scatter plots to communicate **Dynamics** and **Causality**.

### 5.1 Visualizing Dynamics: Velocity Embeddings

*   **Concept**: Visualizing the "flow" of neural states [17].
*   **Algorithm**:
    1.  Compute high-D velocity: $\mathbf{v}_i = \mathbf{x}_{i+1} - \mathbf{x}_i$.
    2.  Compute local transition probabilities $P_{ij}$ based on alignment of $\mathbf{v}_i$ with neighbor displacement.
    3.  Project velocity to embedding: $\tilde{\mathbf{v}}_i \approx \sum_j P_{ij} (\mathbf{y}_j - \mathbf{y}_i)$.
*   **Visualization**: Streamlines overlaying the UMAP plot, revealing attractor landscapes and metastable states [18].

### 5.2 Feature Attribution: Opening the Black Box

*   **Concept**: Answering "What drives Axis 1?"
*   **Gradient-Based**: For Parametric UMAP/TopoAE, compute saliency maps $\text{Importance}_{fd} = | \partial z_d / \partial x_f |$.
*   **Perturbation-Based**: For non-parametric methods, shuffle features and measure embedding displacement.
*   **Visualization**: Heatmaps on the embedding showing which features (e.g., "Alpha Power") drive specific clusters [19].

---

## 6. Design and Architectural Philosophy

The module's architecture is designed for robustness and scalability, moving beyond experimental "research code" to a production-grade framework.

### 6.1 Core API and Orchestration
The module strictly adheres to the **scikit-learn API standard**, ensuring distinct familiarity for data scientists.
*   **Uniform Interface**: Every reducer, from simple PCA to complex Topological Autoencoders, inherits from `BaseReducer` and exposes the standard `fit(X)`, `transform(X)`, and `fit_transform(X)` methods.
*   **The Orchestrator**: The `DimReduction` class serves as the high-level manager. It handles method instantiation from config, pipeline execution, and result aggregation.
*   **Benchmarking**: The `MethodSelector` class automates the "tournament" between reducers. It runs multiple methods (or hyperparameter sets) in parallel and ranks them based on the Co-ranking metrics.
*   **Visualization**: Built-in `.plot()` methods on the orchestrator provide production-ready static plots (scatter, velocity streams) instantly, abstracting away matplotlib boilerplate.

### 6.2 Modern Configuration Management
We use a **Hydra + Pydantic** system for configuration [21].
*   **Pydantic Models** define strict schemas for each reducer (e.g., `UMAPConfig`), ensuring fail-fast validation and clear error contracts.
*   **Hydra** manages hierarchical YAML configurations, enabling complex parameter sweeps and ensuring that every experiment is reproducible.

### 6.3 Deep Learning Integration
To bridge the gap between classical ML and Deep Learning, the module integrates **Skorch** [22]. Skorch wraps PyTorch modules (such as the Topological Autoencoder) as scikit-learn compatible estimators. This facilitates a unified API where sophisticated neural networks allow `fit()` and `transform()` usage identical to standard PCA.

### 6.4 Scalability
For global linear algebra operations on datasets exceeding available RAM, the architecture integrates **Dask** [23]. This allows algorithms like Incremental PCA to operate out-of-core on chunked arrays, scaling the analysis to population-level cohorts.

---

## 7. Implementation Roadmap & Status

A phased plan to realize this vision.

| Phase | Focus | Key Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **1** | **Architectural Foundation** | Hydra/Pydantic Refactor; DataLoader Abstraction. | ✅ **Complete** |
| **2** | **The Metrics Engine** | Optimized Co-ranking ($T, C, LCMC$); `MethodSelector`. | ✅ **Complete** |
| **3** | **Scientific Reducers** | DMD (Oscillations); TRCA (Tasks). | ✅ **Complete** |
| **4** | **DL Integration** | Skorch dependency; Parametric UMAP; IVIS; TopoAE. | ✅ **Complete** |
| **5** | **Advanced Viz** | Velocity Streamlines; Feature Attribution. | ✅ **Complete** |
| **6** | **Interactive Utility** | Interactive Dashboards/Plots. | **[TODO]** |

## 8. Conclusion

The evolution of `coco_pipe` shifts its identity from a passive wrapper of sklearn tools to an active, opinionated, and scientifically rigorous platform. By integrating physics-informed methods like DMD and TRCA, it respects the causal nature of M/EEG data. By mandating Co-ranking metrics, it moves the user community toward reproducible validation. This platform is now a definitive instrument for dimensionality reduction in scientific research.

## Works Cited

1. Comparative analysis of dimensionality reduction techniques for EEG-based emotional state classification - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11578865/)
2. Applying dimension reduction to EEG data by Principal Component Analysis reduces the quality of its subsequent Independent Component decomposition - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6650744/)
3. Exploration of Dimension Reduction Methods: Theory and Applications - [PDF](https://hardin47.github.io/st47s-and-d47a/student-work/necdet_canim_2025.pdf)
4. UMAP as Dimensionality Reduction Tool for Molecular Dynamics Simulations of Biomacromolecules - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8356557/)
5. Visualizing Structure and Transitions in High-Dimensional Biological Data - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7073148/)
6. Dynamic Mode Decomposition Based Epileptic Seizure Detection from Scalp EEG - [ResearchGate](https://www.researchgate.net/publication/326218206_Dynamic_Mode_Decomposition_Based_Epileptic_Seizure_Detection_from_Scalp_EEG)
7. Dynamic Mode Decomposition Based Epileptic Seizure Detection from Scalp EEG - [IEEE Xplore](https://ieeexplore.ieee.org/document/8404027)
8. mnakanishi/trca - [GitHub](https://github.com/mnakanishi/trca)
9. A novel hybrid method based on task-related component and canonical correlation analyses (H-TRCCA) - [Frontiers](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1544452/full)
10. Topological Autoencoders - [PDF](http://proceedings.mlr.press/v119/moor20a/moor20a.pdf)
11. Topological Autoencoders - [Semantic Scholar](https://www.semanticscholar.org/paper/Topological-Autoencoders-Moor-Horn/ed69978f1594a4e2b9dccfc950490fa1df817ae8)
12. ivis dimensionality reduction - [Docs](https://bering-ivis.readthedocs.io/)
13. beringresearch/ivis - [GitHub](https://github.com/beringresearch/ivis)
14. Quality assessment of dimensionality reduction: Rank-based criteria - [Neurocomputing](https://doi.org/10.1016/j.neucom.2008.12.017)
15. From High Dimensions to Human Comprehension: Exploring Dimensionality Reduction for Chemical Space Visualization - [ChemRxiv](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/66bb4da5f3f4b05290bccb6e/original/from-high-dimensions-to-human-comprehension-exploring-dimensionality-reduction-for-chemical-space-visualization.pdf)
16. A New Method for Performance Analysis in Nonlinear Dimensionality Reduction - [arXiv](https://arxiv.org/pdf/1711.06252)
17. scvelo.pl.velocity_embedding_stream - [Docs](https://scvelo.readthedocs.io/en/stable/scvelo.pl.velocity_embedding_stream.html)
18. how to plot streamlines - [Stack Overflow](https://stackoverflow.com/questions/8296617/how-to-plot-streamlines-when-i-know-u-and-v-components-of-velocitynumpy-2d-ar)
19. Feature importance of UMAP output - [GitHub](https://github.com/lmcinnes/umap/issues/505)
20. 7 Expert Tips to Build High-Performance Python Data Pipelines - [SoftKraft](https://www.softkraft.co/python-data-pipelines/)
21. Configuration management for model training experiments using Pydantic and Hydra - [TowardsDataScience](https://towardsdatascience.com/configuration-management-for-model-training-experiments-using-pydantic-and-hydra-d14a6ae84c13/)
22. SKORCH: PyTorch Models Trained with a Scikit-Learn Wrapper - [Towards Data Science](https://towardsdatascience.com/skorch-pytorch-models-trained-with-a-scikit-learn-wrapper-62b9a154623e/)
23. How to analyze large datasets with Python - [CodeSignal](https://codesignal.com/blog/how-to-analyze-large-datasets-with-python-key-principles-tips/)
