import pytest

from coco_pipe.utils import import_optional_dependency


def test_import_optional_dependency():
    with pytest.raises(ImportError, match="demo-lib is required for DemoReducer"):
        import_optional_dependency(
            lambda: (_ for _ in ()).throw(ImportError("missing")),
            feature="DemoReducer",
            dependency="demo-lib",
            install_hint="pip install demo-lib",
        )

    with pytest.raises(
        RuntimeError, match="demo-lib failed to initialize for DemoReducer"
    ):
        import_optional_dependency(
            lambda: (_ for _ in ()).throw(ValueError("boom")),
            feature="DemoReducer",
            dependency="demo-lib",
        )
