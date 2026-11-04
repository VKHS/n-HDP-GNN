"""Model, trainer and DDP tests (need torch and torch_geometric; skipped otherwise)."""
import json
import socket
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from nhdp_gnn.data import load_dataset  # noqa: E402
from nhdp_gnn.layers import GatedGATConv  # noqa: E402
from nhdp_gnn.pipeline import build_batches, run_stages  # noqa: E402
from nhdp_gnn.splits import regime_split  # noqa: E402
from nhdp_gnn.train import RunConfig, build_model, parse_args, run  # noqa: E402

SYN = {"num_super": 3, "comm_per_super": 3, "nodes_per_comm": 30, "seed": 0, "class_alignment": 0.9}


def small_cfg(tmp_path, **kw):
    base = dict(dataset="synthetic", synthetic=SYN, regime="full", depth=2, hidden=16, heads=2, epochs=4,
                patience=50, nhdp_K=12, nhdp_T1=4, nhdp_T2=3, nhdp_max_iter=20, threads=1,
                out_dir=str(tmp_path / "runs"), cache_dir=str(tmp_path / "cache"), madgap_sources=64)
    base.update(kw)
    return RunConfig(**base)


def test_gated_attention_zero_gate_isolates_and_unit_gate_is_attention():
    torch.manual_seed(0)
    conv = GatedGATConv(8, heads=2)
    # two triangles {0,1,2} and {3,4,5} joined by edge 2-3, with self loops
    e = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0), (3, 4), (4, 3), (4, 5), (5, 4), (3, 5), (5, 3), (2, 3), (3, 2)]
    e += [(i, i) for i in range(6)]
    ei = torch.tensor(e).t()
    gate = torch.ones(ei.shape[1])
    gate[12:14] = 0.0                                      # gate off the bridge
    x = torch.randn(6, 8)
    out1 = conv(x, ei, gate)
    x2 = x.clone()
    x2[3:] += 5.0                                          # change the other triangle only
    out2 = conv(x2, ei, gate)
    assert torch.allclose(out1[:3], out2[:3], atol=1e-6)
    out3 = conv(x2, ei, torch.ones(ei.shape[1]))
    assert not torch.allclose(out1[:3], out3[:3], atol=1e-4)


@pytest.mark.parametrize("assign", ["nhdp", "louvain", "none"])
@pytest.mark.parametrize("node_op,residual", [("gat", "res"), ("gcn", "initial"), ("gat", "none")])
def test_model_forward_and_all_parameters_get_gradients(tmp_path, assign, node_op, residual):
    cfg = small_cfg(tmp_path, assign=assign, node_op=node_op, residual=residual, depth=3)
    data = load_dataset("synthetic", synthetic=SYN)
    split = regime_split(data.y, "full", 0)
    stages = run_stages(data, cfg.pipeline(), cache_dir=cfg.cache_dir)
    b = build_batches(data, stages, split, gate=cfg.gate)[0]
    model = build_model(cfg, data.x.shape[1], data.num_classes, aux_classes=int(np.unique(stages.louvain).size))
    logits, hidden, aux = model(b)
    assert logits.shape == (data.num_nodes, data.num_classes) and len(hidden) == 4
    loss = logits.logsumexp(1).mean() + (aux.logsumexp(1).mean() if aux is not None else 0)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing


@pytest.mark.parametrize("model_name", ["nhdp", "gcn", "gat", "sgc", "appnp", "arma", "gcnii", "pairnorm",
                                        "dropedge", "graphunet"])
def test_training_smoke_all_models(tmp_path, model_name):
    res = run(small_cfg(tmp_path, model=model_name), verbose=False)
    assert 0.0 <= res["test"] <= 1.0 and res["best_epoch"] >= 1
    saved = json.loads((Path(res["config"]["out_dir"]) / f"{res['key']}.json").read_text())
    assert saved["test"] == res["test"]
    assert {"centroid_gap", "madgap", "nmi", "ari"} <= saved["embedding"].keys()


def test_best_state_is_restored(tmp_path):
    res = run(small_cfg(tmp_path, epochs=12, eval_every=1), verbose=False)
    best = max(c["val"] for c in res["curves"])
    assert res["val"] == pytest.approx(best)


def test_multi_part_training_and_embeddings(tmp_path):
    res = run(small_cfg(tmp_path, num_parts=3, partition="community", save_embeddings=True), verbose=False)
    assert res["stages"]["num_parts"] == 3
    files = res["embedding_files"]
    assert len(files) == 3                                   # encoder output + 2 layers
    emb = np.load(Path(res["config"]["out_dir"]) / files[-1])
    assert emb.shape[0] == 270


def test_cli_parses_types(tmp_path):
    cfg = parse_args(["--dataset", "synthetic", "--depth", "5", "--gate", "false", "--lr", "0.01",
                      "--synthetic", json.dumps(SYN)])
    assert cfg.depth == 5 and cfg.gate is False and cfg.lr == 0.01 and cfg.synthetic == SYN


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_ddp_two_ranks(tmp_path):
    from nhdp_gnn.distributed import spawn_local

    cfg = small_cfg(tmp_path, epochs=3, partition="bfs")
    spawn_local(cfg, world=2, port=_free_port())
    files = list((tmp_path / "runs").glob("*-ddp2.json"))
    assert len(files) == 1
    res = json.loads(files[0].read_text())
    assert res["world_size"] == 2 and 0.0 <= res["test"] <= 1.0
    s = res["systems"]
    assert s["bytes_per_step_payload"] > 0 and s["time_per_epoch_s"] > 0


def test_cora_loads_and_trains(tmp_path):
    try:
        data = load_dataset("cora", root=str(tmp_path / "data"))
    except Exception as exc:                                 # offline runners
        pytest.skip(f"Cora download unavailable: {exc}")
    assert data.num_nodes == 2708 and data.public_split[0].sum() == 140
    res = run(replace(small_cfg(tmp_path), dataset="cora", root=str(tmp_path / "data"), regime="semi",
                      epochs=2, hidden=16), verbose=False)
    assert res["n_train"] == 140 and res["n_test"] == 1000
