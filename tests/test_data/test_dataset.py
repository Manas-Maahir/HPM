import pytest


def test_placeholder():
    """Dataset tests require a downloaded face dataset — skipped in CI."""
    pytest.skip("requires face dataset on disk")
