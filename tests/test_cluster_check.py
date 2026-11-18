"""configs/cluster_check.toml: the config that checks the cluster's plumbing without a dataset.

Without PyTorch the tests cover what can be checked on one machine: that the file is a valid run configuration, that its
graph can be partitioned into 2, 4, 8 and 14 non-empty parts and that stages A and B run on that partition.  With PyTorch
(the GitHub Actions run) the distributed path is run on this machine with 2 and 4 ranks over gloo on the loopback
interface, which says nothing about a network between machines.
"""
import importlib.util
import json
import socket
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nhdp_gnn.config import config_from_dict, parse_args
from nhdp_gnn.data import load_dataset
from nhdp_gnn.pipeline import compute_partition, run_stages

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cluster_check.toml"
HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torch_geometric") is not None


def test_the_file_is_a_run_configuration_that_needs_no_download():
    values = tomllib.loads(CONFIG.read_text())
    cfg = config_from_dict(values)
    assert cfg.dataset == "synthetic" and cfg.synthetic and cfg.regime == "full" and cfg.model == "nhdp"
    assert cfg.epochs <= 20 and cfg.hidden % cfg.heads == 0
    assert cfg.out_dir.startswith("runs/") and cfg.cache_dir.startswith("runs/")
    assert parse_args(["--config", str(CONFIG)]) == cfg                      # the entry point reads it the same way
    assert parse_args(["--config", str(CONFIG), "--depth", "6", "--epochs", "3"]).depth == 6


@pytest.mark.parametrize("world", [2, 4, 8, 14])
def test_every_world_size_gets_non_empty_parts_and_stages_run(world):
    cfg = replace(config_from_dict(tomllib.loads(CONFIG.read_text())), num_parts=world)
    data = load_dataset(cfg.dataset, cfg.root, cfg.synthetic)
    pipeline = cfg.pipeline()
    parts = compute_partition(data, pipeline)
    sizes = np.bincount(parts, minlength=world)
    assert len(sizes) == world and sizes.min() > 0
    out = run_stages(data, pipeline, cache_dir=None)
    assert out.parts.shape == (data.num_nodes,) and set(np.unique(out.parts)) == set(range(world))
    assert out.R is not None and np.all(np.isfinite(out.R))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(not HAS_TORCH, reason="needs PyTorch and PyTorch Geometric")
@pytest.mark.parametrize("world", [2, 4])
def test_the_config_runs_end_to_end_on_this_machine(tmp_path, world):
    from nhdp_gnn.distributed import spawn_local

    cfg = replace(config_from_dict(tomllib.loads(CONFIG.read_text())), threads=1, out_dir=str(tmp_path / "runs"),
                  cache_dir=str(tmp_path / "cache"))
    spawn_local(cfg, world=world, port=_free_port())
    files = list((tmp_path / "runs").glob(f"*-ddp{world}.json"))
    assert len(files) == 1
    res = json.loads(files[0].read_text())
    assert res["world_size"] == world and 0.0 <= res["test"] <= 1.0
    assert res["systems"]["bytes_per_step_payload"] > 0 and res["systems"]["time_per_epoch_s"] > 0
