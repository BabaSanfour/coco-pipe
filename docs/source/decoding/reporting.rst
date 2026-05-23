Decoding Reporting
==================

Static HTML reports for decoding results live in
:mod:`coco_pipe.report`. The full guide is at :ref:`report` — see
:ref:`report-section-decoding` for the section catalog and
:ref:`report-example-decoding` for a worked example.

Quickstart:

.. code-block:: python

   from coco_pipe.report import from_experiment_result

   report = from_experiment_result(result, output_path="decoding.html")

Equivalent low-level factory:

.. code-block:: python

   from coco_pipe.report import make_decoding_report

   report = make_decoding_report(result, output_path="decoding.html")
