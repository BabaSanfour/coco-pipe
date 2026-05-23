# Future Work & Roadmap

## `coco_pipe.report.api`
- [ ] Add `compare_experiments([result1, result2, ...])` factory.
  - *Context:* Generates a comparative report from multiple `DecodingResult` objects.
  - *Implementation note:* Reuses `viz.decoding` functions by extracting score DataFrames and prefixing the `"Model"` columns with the experiment name prior to concatenation, preventing the need to rewrite the underlying plotting functions.
- [ ] Add `from_data_quality(container)` factory.
  - *Context:* Automates building a dataset health report using the robust checks in `coco_pipe.report.data_quality`.
- [ ] Add `from_auto(path)` factory.
  - *Context:* Smart dispatch router that automatically delegates to `from_bids`, `from_tabular`, etc., based on file extension or directory contents.
