"""How a rank leaves the process group: after a barrier on success, without one after an exception (needs PyTorch)."""
import importlib.util
import types

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torch_geometric") is not None
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="needs PyTorch and PyTorch Geometric")


@pytest.fixture
def calls(monkeypatch):
    from nhdp_gnn import distributed

    record = []
    monkeypatch.setattr(distributed, "dist", types.SimpleNamespace(barrier=lambda: record.append("barrier"),
                                                                   destroy_process_group=lambda: record.append("destroy")))
    return record


def test_every_rank_waits_for_the_others_before_it_destroys_the_group(calls, monkeypatch):
    from nhdp_gnn import distributed

    monkeypatch.setattr(distributed, "worker", lambda cfg: calls.append(("work", cfg)))
    distributed.run_and_leave("config")
    assert calls == [("work", "config"), "barrier", "destroy"]


def test_after_an_exception_the_group_is_destroyed_without_waiting(calls, monkeypatch):
    from nhdp_gnn import distributed

    def fails(cfg):
        calls.append("work")
        raise ValueError("a part is too small")

    monkeypatch.setattr(distributed, "worker", fails)
    with pytest.raises(ValueError, match="too small"):
        distributed.run_and_leave("config")
    assert calls == ["work", "destroy"]               # a barrier here could wait for ranks that wait for this one
