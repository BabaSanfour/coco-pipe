# ruff: noqa: E402
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend("Agg")

from coco_pipe.report.decoding import make_decoding_report
from tests.fixtures.synthetic_result import (
    make_synthetic_feature_metadata,
    make_synthetic_result,
)

result = make_synthetic_result()
feature_metadata = pd.DataFrame(make_synthetic_feature_metadata())
report = make_decoding_report(
    result,
    feature_metadata=feature_metadata,
    sections=["overview", "performance", "features"],
)
report.save("decoding_report.html")
