"""How a rank leaves a run: barrier, destroy the group, end the process without shutting the interpreter down
(needs PyTorch; the end-to-end behaviour is exercised by the two- and four-rank runs of the other tests)."""
import ast
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torch_geometric") is not None
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="needs PyTorch and PyTorch Geometric")


@pytest.fixture
def calls(monkeypatch):
    """Stand-ins for the process group, the worker and the end of the process; every call is recorded in order."""
    from nhdp_gnn import distributed

    record = []
    fake_dist = types.SimpleNamespace(
        init_process_group=lambda *a, **k: record.append("init"),
        barrier=lambda: record.append("barrier"),
        destroy_process_group=lambda: record.append("destroy"))
    monkeypatch.setattr(distributed, "dist", fake_dist)
    monkeypatch.setattr(distributed, "config_from_dict", lambda d: d)
    monkeypatch.setattr(os, "_exit", lambda status: record.append(("exit", status)))
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


def test_leaving_the_process_flushes_the_streams_and_does_not_shut_the_interpreter_down(calls, monkeypatch):
    from nhdp_gnn import distributed

    class Stream:
        def __init__(self, name):
            self.name = name

        def flush(self):
            calls.append(f"flush {self.name}")

    monkeypatch.setattr(sys, "stdout", Stream("stdout"))
    monkeypatch.setattr(sys, "stderr", Stream("stderr"))
    distributed.leave_process()
    distributed.leave_process(3)
    assert calls == ["flush stdout", "flush stderr", ("exit", 0), "flush stdout", "flush stderr", ("exit", 3)]


def test_a_spawned_rank_ends_its_process_after_leaving_the_group(calls, monkeypatch):
    from nhdp_gnn import distributed

    monkeypatch.setattr(distributed, "worker", lambda cfg: calls.append(("work", cfg)))
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(flush=lambda: calls.append("flush")))
    monkeypatch.setattr(sys, "stderr", types.SimpleNamespace(flush=lambda: calls.append("flush")))
    distributed._local_entry(1, 2, 29500, {"seed": 0})
    assert os.environ["MASTER_ADDR"] == "127.0.0.1" and os.environ["MASTER_PORT"] == "29500"
    assert calls == ["init", ("work", {"seed": 0}), "barrier", "destroy", "flush", "flush", ("exit", 0)]


def test_a_spawned_rank_that_fails_does_not_end_the_process_so_that_the_failure_is_reported(calls, monkeypatch):
    from nhdp_gnn import distributed

    def fails(cfg):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(distributed, "worker", fails)
    with pytest.raises(RuntimeError, match="out of memory"):
        distributed._local_entry(0, 2, 29500, {})
    assert calls == ["init", "destroy"]               # no exit: the exception must reach multiprocessing's error report


def test_python_m_nhdp_gnn_distributed_ends_its_process_after_main():
    source = (Path(__file__).resolve().parents[1] / "nhdp_gnn" / "distributed.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    guard = next(node for node in tree.body if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                 and getattr(node.test.left, "id", None) == "__name__")
    calls = [stmt.value.func.id for stmt in guard.body if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)]
    assert calls == ["main", "leave_process"]
