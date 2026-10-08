import pytest

from benchmarks import evaluate


@pytest.fixture
def fast_bootstrap(monkeypatch):
    """Few resamples: these tests check plumbing, not the stability of an interval."""
    monkeypatch.setattr(evaluate, "N_BOOT", 2000)
