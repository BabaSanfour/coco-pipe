
import pytest
import numpy as np
import pandas as pd
from coco_pipe.report.quality import check_missingness, check_constant_columns, CheckResult
from coco_pipe.report.core import Section

def test_check_missingness():
    # Only 10% missing (WARN threshold is 1%)
    data = np.random.randn(100)
    data[:10] = np.nan
    
    res = check_missingness(data)
    assert res.status == "WARN"
    assert "High missing data" in res.message
    assert res.metric_value == 0.1

    # 50% missing (FAIL threshold is 20%)
    data[:50] = np.nan
    res = check_missingness(data)
    assert res.status == "FAIL"
    assert "Critical" in res.message

def test_check_constant_columns():
    df = pd.DataFrame({
        "A": [1, 1, 1, 1],
        "B": [1, 2, 3, 4]
    })
    results = check_constant_columns(df)
    assert len(results) == 1
    assert results[0].check_name == "Constant Features"
    assert results[0].status == "WARN"

def test_section_status_update():
    sec = Section("Quality Test")
    assert sec.status == "OK"
    
    # Add WARN
    res_warn = CheckResult("Test", "WARN", "Warning", 5)
    sec.add_finding(res_warn)
    assert sec.status == "WARN"
    
    # Add FAIL (should upgrade)
    res_fail = CheckResult("Test", "FAIL", "Failure", 9)
    sec.add_finding(res_fail)
    assert sec.status == "FAIL"
    
    # Add WARN (should stay FAIL)
    sec.add_finding(res_warn)
    assert sec.status == "FAIL"

def test_findings_serialization():
    sec = Section("Serialization")
    sec.add_finding(CheckResult("Test", "WARN", "Msg", 5))
    
    html = sec.render()
    # Check if finding message is present in HTML (via Jinja loop)
    assert "Msg" in html
    assert "⚠️" in html
