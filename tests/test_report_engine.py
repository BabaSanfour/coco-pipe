from unittest.mock import MagicMock, patch

import jinja2
import pytest

from coco_pipe.report import _engine


@pytest.fixture(autouse=True)
def reset_env():
    """Reset the cached environment before each test."""
    _engine.get_env.cache_clear()
    yield
    _engine.get_env.cache_clear()


def test_get_env_initialization():
    env = _engine.get_env()
    assert isinstance(env, jinja2.Environment)

    # Check loader
    assert isinstance(env.loader, jinja2.FileSystemLoader)
    assert env.loader.searchpath == [str(_engine.TEMPLATE_DIR)]

    # Check options
    assert env.trim_blocks is True
    assert env.lstrip_blocks is True
    assert env.autoescape is not None


def test_get_env_singleton_cache():
    env1 = _engine.get_env()
    env2 = _engine.get_env()
    # Since we use lru_cache, it should return the exact same object
    assert env1 is env2


@patch("coco_pipe.report._engine.TEMPLATE_DIR")
def test_get_env_missing_dir(mock_template_dir):
    """Test that a missing template directory raises a RuntimeError."""
    mock_template_dir.is_dir.return_value = False

    with pytest.raises(RuntimeError, match="Template directory not found"):
        _engine.get_env()


@patch("coco_pipe.report._engine.get_env")
def test_render_template(mock_get_env):
    mock_env = MagicMock()
    mock_template = MagicMock()
    mock_template.render.return_value = "rendered content"
    mock_env.get_template.return_value = mock_template
    mock_get_env.return_value = mock_env

    result = _engine.render_template("dummy.html", var1="value1", var2=42)

    assert result == "rendered content"
    mock_get_env.assert_called_once_with()
    mock_env.get_template.assert_called_once_with("dummy.html")
    mock_template.render.assert_called_once_with(var1="value1", var2=42)


def test_render_template_not_found():
    """Test rendering a non-existent template raises TemplateNotFound."""
    with pytest.raises(jinja2.exceptions.TemplateNotFound):
        _engine.render_template("non_existent_template_xyz.html")
