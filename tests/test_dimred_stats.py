import pandas as pd

from coco_pipe.dim_reduction.evaluation.stats import (
    grouped_condition_stats,
    paired_condition_stats,
)


def test_paired_condition_stats():
    data = []
    # metric, condition, subject, value
    for sub in range(4):
        data.append(
            {"metric": "speed", "condition": "A", "subject": f"s{sub}", "value": sub}
        )
        data.append(
            {
                "metric": "speed",
                "condition": "B",
                "subject": f"s{sub}",
                "value": sub + 1,
            }
        )
        data.append(
            {"metric": "acc", "condition": "A", "subject": f"s{sub}", "value": sub * 2}
        )
        data.append(
            {
                "metric": "acc",
                "condition": "B",
                "subject": f"s{sub}",
                "value": sub * 2 + 0.5,
            }
        )

    df = pd.DataFrame(data)
    result = paired_condition_stats(df, conditions=["A", "B"])

    assert len(result) == 2
    assert set(result["metric"]) == {"speed", "acc"}
    speed_res = result[result["metric"] == "speed"].iloc[0]
    # A - B = sub - (sub + 1) = -1
    assert speed_res["mean_diff"] == -1.0
    assert "p_fdr" in result.columns


def test_grouped_condition_stats():
    data = []
    for sub in range(4):
        # A1 and A2 will be grouped into G1
        data.append(
            {"metric": "speed", "condition": "A1", "subject": f"s{sub}", "value": sub}
        )
        data.append(
            {
                "metric": "speed",
                "condition": "A2",
                "subject": f"s{sub}",
                "value": sub + 2,
            }
        )
        # B will be G2
        data.append(
            {
                "metric": "speed",
                "condition": "B",
                "subject": f"s{sub}",
                "value": sub + 0.5,
            }
        )

    df = pd.DataFrame(data)
    condition_sets = {
        "set1": {
            "G1": ["A1", "A2"],
            "G2": ["B"],
        }
    }
    result = grouped_condition_stats(df, condition_sets=condition_sets)
    assert len(result) == 1
    # G1 mean is (sub + sub + 2) / 2 = sub + 1
    # G2 mean is sub + 0.5
    # diff = (sub + 1) - (sub + 0.5) = 0.5
    assert result.iloc[0]["mean_diff"] == 0.5
    assert result.iloc[0]["set"] == "set1"
