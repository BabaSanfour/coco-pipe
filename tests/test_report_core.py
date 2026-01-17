"""
Tests for coco_pipe.report.core
"""

import pytest

from coco_pipe.report.core import HtmlElement, Report, Section


@pytest.fixture
def tmp_report_file(tmp_path):
    return tmp_path / "test_report.html"


def test_element_rendering():
    el = HtmlElement("<p>Test</p>")
    assert el.render() == "<p>Test</p>"


def test_section_rendering():
    sec = Section(title="My Section", icon="📊")
    sec.add_element("<p>Content</p>")
    html = sec.render()
    assert "My Section" in html
    assert "📊" in html
    assert "<p>Content</p>" in html
    assert "bg-white" in html  # Check for Tailwind class


def test_report_creation_and_save(tmp_report_file):
    rep = Report(title="Unit Test Report")

    # Add simple HTML
    rep.add_element(HtmlElement("<p>Hello World</p>"))

    # Add a section
    sec = Section("Analysis")
    sec.add_element("<b>Bold Content</b>")
    rep.add_section(sec)

    # Add markdown (check fallback or real)
    rep.add_markdown("# Markdown Header\n* Item 1")

    # Save
    rep.save(str(tmp_report_file))

    assert tmp_report_file.exists()
    content = tmp_report_file.read_text()

    # Verify Content
    assert "<!DOCTYPE html>" in content
    assert "Unit Test Report" in content
    assert "Hello World" in content
    assert "Analysis" in content
    assert "Bold Content" in content

    # Verify Markdown rendering (either h1 or pre wrap if fallback)
    # Just checking for "Markdown Header" should be safe
    assert "Markdown Header" in content


def test_fluent_interface_structure():
    rep = Report("Fluency")
    rep.add_element("Start").add_section(Section("Middle")).add_markdown("End")
    assert len(rep.children) == 3
