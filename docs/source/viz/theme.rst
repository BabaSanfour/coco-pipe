.. _viz-theme:

===================================
Theme, Palettes, and Figure Sizing
===================================

``coco_pipe.viz.theme`` is the single source of truth for visual styling across
both backends. Matplotlib rcParams and the Plotly ``coco`` template are derived
from the same palettes and font choices, so changing the theme in one place
updates static and interactive plots consistently.

---

1. Theme Modes
==============

Three rcParam presets target the most common output contexts:

==========  ==================================================================
Mode        Use case
==========  ==================================================================
``paper``   Print figures, tight font sizes (11 pt body, 12 pt titles).
``notebook``  Jupyter / on-screen review at default zoom (13/14 pt).
``poster``  Conference posters and slide decks (16/18 pt).
==========  ==================================================================

All three share the same colorblind-safe palette and remove top/right spines.

.. code-block:: python

   from coco_pipe.viz.theme import coco_theme, set_coco_theme

   # Scoped: safe in tests and notebooks (always restores prior rcParams).
   with coco_theme("paper", colorblind=True):
       fig, ax = plt.subplots()
       ax.plot(x, y)

   # Global: applies for the rest of the process.
   set_coco_theme("notebook")

Prefer the **context manager** in notebooks and tests; reach for
``set_coco_theme`` only at script entry points.

---

2. Palettes
============

Four palette constants are exported from :mod:`coco_pipe.viz.theme` and reused
by every domain plot:

==============================  ==============================================
Constant                        Use
==============================  ==============================================
``SEQUENTIAL`` (``"viridis"``)  Non-negative continuous values (importance,
                                counts, density).
``DIVERGING`` (``"RdBu_r"``)    Signed values centered on zero (signed
                                importance, contrasts, residuals).
``QUALITATIVE`` (``"tab10"``)   Up to ~10 categorical labels.
``QUALITATIVE_COLORBLIND``      Okabe–Ito-style 8-color cycle, the default when
(``"colorblind"``)              ``colorblind=True`` is passed to the theme.
==============================  ==============================================

The ``Plotly`` colorway and color scales are seeded from the same constants in
:mod:`coco_pipe.viz.interactive._utils`, so interactive figures stay aligned
with the matplotlib output.

---

3. Figure Sizing for Papers
============================

``figure_size`` returns inches sized for one-column or two-column manuscripts.
Width defaults match common journal templates (3.5 in single, 7.0 in double);
pass ``width_pt`` to override using a journal-specific value.

.. code-block:: python

   from coco_pipe.viz.theme import coco_theme, figure_size

   with coco_theme("paper"):
       fig, ax = plt.subplots(figsize=figure_size(columns=1, aspect_ratio=0.7))
       fig2, ax2 = plt.subplots(figsize=figure_size(columns=2, aspect_ratio=0.5))

       # Override with a journal-specific column width (in points)
       fig3, ax3 = plt.subplots(figsize=figure_size(columns=1, width_pt=246))

---

4. Saving Figures
==================

``save_figure`` standardizes DPI, background color, and bounding-box mode so
exports look the same regardless of which plot was used.

.. code-block:: python

   from coco_pipe.viz.theme import save_figure

   save_figure(fig, "fig1.pdf")                # 300 dpi, white background, tight bbox
   save_figure(fig, "fig1.png", dpi=600)       # higher resolution for raster output

For interactive Plotly figures, use ``fig.write_html(...)`` /
``fig.write_image(...)`` directly — the ``coco`` template is already applied.

---

5. End-to-End Paper Figure Recipe
==================================

.. code-block:: python

   import matplotlib.pyplot as plt
   from coco_pipe.viz import plot_decoding_scores, plot_confusion_matrix
   from coco_pipe.viz.theme import coco_theme, figure_size, save_figure

   with coco_theme("paper", colorblind=True):
       fig, axes = plt.subplots(
           1, 2, figsize=figure_size(columns=2, aspect_ratio=0.45),
           constrained_layout=True,
       )
       plot_decoding_scores(result, metric="accuracy", ax=axes[0])
       plot_confusion_matrix(result, model="logistic_regression", ax=axes[1])
       save_figure(fig, "figures/fig1.pdf")

---

6. Compatibility Notes
=======================

- ``coco_theme`` only mutates Matplotlib rcParams; it does not touch the Plotly
  template (which is registered eagerly at import time).
- Setting ``colorblind=True`` swaps the matplotlib ``axes.prop_cycle`` to the
  Okabe–Ito colors but leaves continuous colormaps alone — those remain
  perceptually uniform (``viridis``/``RdBu_r``).
- The ``coco`` Plotly template is named :data:`COCO_TEMPLATE` and can be applied
  manually with ``fig.update_layout(template="coco")`` if you build a Plotly
  figure outside the wrappers.
