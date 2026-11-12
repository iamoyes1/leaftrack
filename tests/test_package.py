"""Smoke tests for the package scaffold."""

import importlib

import pytest

import leaftrack


def test_version_is_semver_string() -> None:
    assert isinstance(leaftrack.__version__, str)
    assert leaftrack.__version__.count(".") == 2


@pytest.mark.parametrize(
    "module",
    ["leaftrack.segment", "leaftrack.track", "leaftrack.metrics", "leaftrack.analytics"],
)
def test_modules_import_and_are_documented(module: str) -> None:
    mod = importlib.import_module(module)
    assert mod.__doc__ and mod.__doc__.strip()
