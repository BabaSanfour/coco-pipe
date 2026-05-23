Decoding Visualization
======================

Static decoding plots live in ``coco_pipe.viz.decoding`` and return
``(Figure, Axes)``. Plot functions accept either an ``ExperimentResult`` or the
tidy DataFrame produced by the corresponding accessor.

Core examples:

.. code-block:: python

   from coco_pipe.viz.decoding import plot_decoding_scores, plot_confusion_matrix

   fig, ax = plot_decoding_scores(result)
   fig, ax = plot_confusion_matrix(result)
