"""Model, trainer and DDP tests (need torch and torch_geometric; skipped otherwise)."""
import json
import math
import os
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
    assert torch.equal(out1[:3], out2[:3])                 # a zero gate is an exact cut
    out3 = conv(x2, ei, torch.ones(ei.shape[1]))
    assert not torch.allclose(out1[:3], out3[:3], atol=1e-4)
    out1.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in conv.parameters())


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


ALL_MODELS = ["nhdp", "gcn", "gat", "sgc", "appnp", "arma", "gcnii", "pairnorm", "dropedge", "graphunet"]


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_every_trainable_parameter_receives_a_gradient(tmp_path, model_name):
    """DistributedDataParallel (without find_unused_parameters) needs this for every model."""
    cfg = small_cfg(tmp_path, model=model_name, depth=3)
    data = load_dataset("synthetic", synthetic=SYN)
    split = regime_split(data.y, "full", 0)
    stages = run_stages(data, cfg.pipeline(), cache_dir=cfg.cache_dir)
    b = build_batches(data, stages, split, gate=cfg.gate)[0]
    model = build_model(cfg, data.x.shape[1], data.num_classes)
    model.train()
    logits, _, _ = model(b)
    logits.logsumexp(1).mean().backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


def test_baseline_options_reach_the_layers(tmp_path):
    appnp = build_model(small_cfg(tmp_path, model="appnp", alpha=0.3), 8, 3)
    assert appnp.prop.alpha == pytest.approx(0.3)
    gcnii = build_model(small_cfg(tmp_path, model="gcnii", alpha=0.2, theta=0.7), 8, 3)
    assert gcnii.convs[0].alpha == pytest.approx(0.2)
    assert gcnii.convs[0].beta == pytest.approx(math.log(0.7 / 1 + 1))
    arma = build_model(small_cfg(tmp_path, model="arma", depth=3), 8, 3)
    assert all(c.act is None for c in arma.convs)            # no ReLU on the logits
    assert not any(c.weight.requires_grad for c in arma.convs)
    unet = {d: build_model(small_cfg(tmp_path, model="graphunet", depth=d), 8, 3).propagation_depth
            for d in (2, 4, 8, 10, 20, 30)}
    assert unet == {2: 3, 4: 5, 8: 9, 10: 11, 20: 21, 30: 31}


def test_hidden_must_be_divisible_by_heads(tmp_path):
    with pytest.raises(ValueError, match="divisible"):
        build_model(small_cfg(tmp_path, hidden=10, heads=4, node_op="gcn"), 8, 3)
    with pytest.raises(ValueError, match="divisible"):
        build_model(small_cfg(tmp_path, model="gat", hidden=10, heads=4, depth=3), 8, 3)


def test_focal_loss_with_gamma_zero_is_weighted_cross_entropy():
    from nhdp_gnn.losses import focal_loss

    torch.manual_seed(0)
    logits, target = torch.randn(20, 4), torch.randint(0, 4, (20,))
    w = torch.tensor([1.0, 2.0, 0.5, 3.0])
    ce = torch.nn.functional.cross_entropy(logits, target, weight=w)
    assert torch.allclose(focal_loss(logits, target, 0.0, w), ce, atol=1e-6)


def test_part_statistics_normalises_the_part_and_keeps_running_buffers():
    from nhdp_gnn.train import part_statistics

    bn = torch.nn.BatchNorm1d(3).eval()
    x = torch.randn(10, 3) * 4 + 2
    before = (bn.running_mean.clone(), bn.running_var.clone())
    with torch.no_grad(), part_statistics(bn):
        out = bn(x)
    assert torch.allclose(out.mean(0), torch.zeros(3), atol=1e-5)
    assert torch.equal(bn.running_mean, before[0]) and torch.equal(bn.running_var, before[1])
    assert not bn.training and bn.momentum == 0.1


@pytest.mark.parametrize("name", ["cora", "citeseer", "pubmed", "photo", "computers"])
def test_raw_readers_match_pyg(name):
    """load_dataset reads the raw files itself; the result must equal PyG's processing of the same files."""
    from torch_geometric.datasets import Amazon, Planetoid

    from nhdp_gnn.data import AMAZON, PLANETOID, _row_normalise, undirected

    try:
        ours = load_dataset(name, "data")
    except OSError as exc:
        pytest.skip(f"download unavailable: {exc}")
    if name in PLANETOID:
        ref = Planetoid("data/planetoid", PLANETOID[name])[0]
        for mine, theirs in zip(ours.public_split, (ref.train_mask, ref.val_mask, ref.test_mask)):
            assert np.array_equal(mine, theirs.numpy())
    else:
        ref = Amazon("data/amazon", AMAZON[name])[0]
    assert np.array_equal(ours.x, _row_normalise(ref.x.numpy()))
    assert np.array_equal(ours.y, ref.y.numpy())
    assert np.array_equal(ours.edge_index, undirected(ref.edge_index.numpy(), ours.num_nodes))


class _NotAllowlisted:                                       # module level, so it unpickles by reference
    pass


def test_ogb_compat_loads_legacy_pickles_and_skips_the_prompt(tmp_path, monkeypatch):
    from nhdp_gnn.data import _ogb_compat

    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)
    path = tmp_path / "legacy.pt"
    torch.save({"obj": _NotAllowlisted()}, path)
    with _ogb_compat():
        assert isinstance(torch.load(path)["obj"], _NotAllowlisted)   # no weights_only, as in ogb 1.3.6
        try:
            import ogb.nodeproppred.dataset_pyg as ogb_pyg
        except ImportError:
            ogb_pyg = None
        if ogb_pyg is not None:                               # pytest's stdin is not a terminal
            assert ogb_pyg.decide_download("https://example.invalid/big.zip") is True
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in os.environ


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_training_smoke_all_models(tmp_path, model_name):
    res = run(small_cfg(tmp_path, model=model_name), verbose=False)
    assert 0.0 <= res["test"] <= 1.0 and res["best_epoch"] >= 1
    saved = json.loads((Path(res["config"]["out_dir"]) / f"{res['key']}.json").read_text())
    assert saved["test"] == res["test"]
    assert {"centroid_gap", "madgap", "nmi", "ari"} <= saved["embedding"].keys()
    assert saved["embedding"]["representation"] == "logits" and saved["propagation_depth"] >= 2


def test_best_state_is_restored(tmp_path):
    res = run(small_cfg(tmp_path, epochs=12, eval_every=1), verbose=False)
    best = max(c["val"] for c in res["curves"])
    assert res["val"] == pytest.approx(best)


def test_multi_part_training_and_embeddings(tmp_path):
    res = run(small_cfg(tmp_path, num_parts=3, partition="community", save_embeddings=True), verbose=False)
    assert res["stages"]["num_parts"] == 3
    files = res["embedding_files"]
    assert len(files) == 4 and files[-1].endswith("_logits.npy")   # encoder output, 2 layers, logits
    out = Path(res["config"]["out_dir"])
    assert np.load(out / files[-2]).shape == (270, 16)
    assert np.load(out / files[-1]).shape[0] == 270


def test_sweep_workers_run_the_trainer_and_record_memory(tmp_path):
    from nhdp_gnn import sweep

    todo = [small_cfg(tmp_path, seed=s) for s in (0, 1)]
    counts = sweep.run_pool(todo, workers=2, memory_gb=6.0, poll=0.1)
    assert counts["ok"] == 2
    for c in todo:
        res = json.loads(sweep.result_path(c).read_text())
        assert 0.0 <= res["test"] <= 1.0 and res["peak_rss_mb"] > 50 and res["code_commit"]


def test_time_limit_stops_training_that_cannot_finish(tmp_path):
    from nhdp_gnn import sweep
    from nhdp_gnn.train import run as train_run

    cfg = small_cfg(tmp_path, epochs=400, patience=300, max_train_minutes=1e-5)
    assert cfg.key() == small_cfg(tmp_path, epochs=400, patience=300).key()
    assert sweep.attempt(cfg, train_run) == "time"
    info = json.loads(sweep.failure_path(cfg).read_text())["failure"]
    assert info["reason"] == "time" and info["epochs_run"] == 1 and info["min_remaining_epochs"] == 300
    assert info["min_train_s"] > info["limit_s"]
    # without the limit the same run completes
    assert sweep.attempt(small_cfg(tmp_path, epochs=3), train_run) == "ok"


def test_non_finite_outputs_are_reported_as_diverged(tmp_path, monkeypatch):
    from nhdp_gnn import sweep
    from nhdp_gnn import train as train_mod

    real_predict = train_mod.predict

    def nan_predict(*a, **k):
        logits, layers = real_predict(*a, **k)
        logits[:3] = np.nan
        return logits, layers

    monkeypatch.setattr(train_mod, "predict", nan_predict)
    cfg = small_cfg(tmp_path, epochs=3)
    assert sweep.attempt(cfg, train_mod.run) == "diverged"
    info = json.loads(sweep.failure_path(cfg).read_text())["failure"]
    assert info["nonfinite_nodes"] == 3 and not sweep.result_path(cfg).exists()


def test_cli_parses_types(tmp_path):
    cfg = parse_args(["--dataset", "synthetic", "--depth", "5", "--gate", "false", "--lr", "0.01",
                      "--synthetic", json.dumps(SYN)])
    assert cfg.depth == 5 and cfg.gate is False and cfg.lr == 0.01 and cfg.synthetic == SYN


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("model_name,partition,weighted", [
    ("nhdp", "bfs", False),
    ("nhdp", "community", True),      # Louvain-packed partition on rank 0; class-weighted loss
    ("arma", "bfs", False),           # a baseline through DDP (every parameter must get a gradient)
])
def test_ddp_two_ranks(tmp_path, model_name, partition, weighted):
    from nhdp_gnn.distributed import spawn_local

    cfg = small_cfg(tmp_path, model=model_name, epochs=3, partition=partition, class_weights=weighted)
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
