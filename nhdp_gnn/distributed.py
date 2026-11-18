"""Distributed data-parallel training on a CPU cluster (paper Sect. 10).

One process per worker node, gloo backend, launched with torchrun:

    torchrun --nnodes 14 --nproc-per-node 1 --rdzv-backend c10d --rdzv-endpoint MASTER:29500 \\
        -m nhdp_gnn.distributed --dataset ogbn-products --regime full --depth 10 --out_dir runs/hpc

Data     rank 0 loads (and if needed downloads and processes) the dataset first;
         the other ranks load it after a barrier.
Stage A  rank 0 computes a world_size-way edge cut and broadcasts it; every rank
         runs nested Louvain on its own part.
Stage B  every rank runs n-HDP inference on its part; the global sufficient
         statistics (stick counts, Normal-Gamma moments) are all-reduced once per
         iteration.
Stage C  every rank trains on its part's induced subgraph (cut edges are not
         propagated, node features are local); DistributedDataParallel averages
         gradients.  The loss of each rank is scaled so that the averaged
         gradient equals that of the global (class-weighted, if enabled) mean loss
         over all training nodes.  Dropout masks are seeded per rank.  BatchNorm
         keeps per-rank buffers and evaluation uses each part's own statistics,
         as in single-process multi-part training.

Systems metrics written by rank 0: time per epoch, edge-updates per second
(edges incl. self-loops x depth, summed over ranks, per step), gradient bytes per
step (all-reduce payload and the ring-all-reduce wire estimate 2(p-1)/p x payload),
all-reduce wall time per step (launch to completion; it overlaps backward, so
Comms% is an upper bound of exposed communication), and the partition's
cross-partition edge ratio rho and super-graph edge reduction r_E.
The Louvain auxiliary loss is disabled here because community ids are local to
each part.
"""
from __future__ import annotations

import copy
import os
import socket
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import RunConfig, config_from_dict, parse_args
from .data import load_dataset
from .losses import supervised_loss
from .pipeline import build_batches, compute_partition, run_stages
from .splits import regime_split
from .train import build_model, class_weights, has_batchnorm, part_statistics, score
from .utils import code_version, environment, save_json, set_seed


class CommStats:
    def __init__(self):
        self.bytes, self.seconds, self.calls = 0, 0.0, 0

    def reset(self):
        self.bytes, self.seconds, self.calls = 0, 0.0, 0


def timed_allreduce_hook(state: CommStats, bucket):
    """DDP communication hook: average gradients and record payload size and wall time."""
    t0 = time.perf_counter()
    buf = bucket.buffer()
    state.bytes += buf.numel() * buf.element_size()
    state.calls += 1
    fut = dist.all_reduce(buf.div_(dist.get_world_size()), async_op=True).get_future()

    def done(f):
        state.seconds += time.perf_counter() - t0
        return f.value()[0]

    return fut.then(done)


def np_allreduce(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        t = torch.as_tensor(np.asarray(v, dtype=np.float64))
        dist.all_reduce(t)
        out[k] = t.numpy()
    return out


def _sum(x: float) -> float:
    t = torch.tensor([float(x)], dtype=torch.float64)
    dist.all_reduce(t)
    return float(t.item())


def _global_score(model, batch, y_full: np.ndarray, mask_full: np.ndarray, multilabel: bool) -> float:
    model.eval()
    with torch.no_grad(), part_statistics(model, dist.get_world_size() > 1 and batch.num_nodes > 1):
        logits, _, _ = model(batch)
    ids = batch.node_ids.numpy()
    if not multilabel:
        pred = logits.argmax(1).numpy()
        m = mask_full[ids]
        correct = _sum(float((pred[m] == y_full[ids][m]).sum()))
        total = _sum(float(m.sum()))
        return correct / total if total > 0 else float("nan")
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (ids, logits.numpy()))
    value = [float("nan")]
    if dist.get_rank() == 0:
        full = np.zeros((y_full.shape[0], logits.shape[1]), dtype=np.float32)
        for i, lo in gathered:
            full[i] = lo
        value = [score(full, y_full, mask_full, True)]
    dist.broadcast_object_list(value, src=0)
    return value[0]


def worker(cfg: RunConfig) -> dict | None:
    rank, world = dist.get_rank(), dist.get_world_size()
    set_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    cfg = replace(cfg, num_parts=world, aux_weight=0.0)
    if rank != 0:                                       # rank 0 downloads and processes first
        dist.barrier()
    data = load_dataset(cfg.dataset, cfg.root, cfg.synthetic)
    if rank == 0:
        dist.barrier()
    split = regime_split(data.y, cfg.regime, cfg.seed, full_split=data.official_split,
                         public_split=data.public_split if cfg.regime == "semi" else None,
                         semi_per_class=cfg.semi_per_class, semi_ratio=cfg.semi_ratio, scarce_ratio=cfg.scarce_ratio)

    t0 = time.perf_counter()
    parts = torch.zeros(data.num_nodes, dtype=torch.long)
    if rank == 0:
        parts = torch.as_tensor(compute_partition(data, cfg.pipeline()), dtype=torch.long)
    dist.broadcast(parts, src=0)
    stages = run_stages(data, cfg.pipeline(), cache_dir=None, reduce=np_allreduce, only_part=rank,
                        parts=parts.numpy())
    t_stages = time.perf_counter() - t0
    batch = build_batches(data, stages, split, gate=cfg.gate, parts_to_build=[rank])[0]

    model = build_model(cfg, data.x.shape[1], data.num_classes)
    smallest = torch.tensor([batch.num_nodes], dtype=torch.long)
    dist.all_reduce(smallest, op=dist.ReduceOp.MIN)     # decided on every rank, so all ranks stop together
    if has_batchnorm(model) and int(smallest.item()) < 2:
        raise ValueError(f"a part has {int(smallest.item())} node(s); BatchNorm needs at least 2 "
                         "(use --norm layer or fewer workers)")
    ddp = DDP(model, broadcast_buffers=False)           # broadcasts rank 0's initial parameters
    torch.manual_seed(cfg.seed + rank)                  # independent dropout masks per rank
    comm = CommStats()
    ddp.register_comm_hook(comm, timed_allreduce_hook)
    opt = torch.optim.AdamW(ddp.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    weight = class_weights(data.y, split[0], data.num_classes) if cfg.class_weights and not data.multilabel else None
    y_train = batch.y[batch.train_mask]
    n_train_local = int(batch.train_mask.sum())
    n_train_global = _sum(n_train_local)
    # each rank's loss is a mean over its own training nodes (weighted by class if enabled);
    # rescaling by local/global normaliser x world makes DDP's gradient average the global mean
    norm_local = float(weight[y_train].sum()) if weight is not None else float(n_train_local)
    norm_global = _sum(norm_local)
    edges_global = _sum(batch.edge_index_sl.shape[1])
    y_np, val_mask, test_mask = np.asarray(data.y), split[1], split[2]
    best_val, best_epoch, best_state = -np.inf, 0, None
    step_times, comm_times, comm_bytes, curves = [], [], [], []
    for epoch in range(1, cfg.epochs + 1):
        ddp.train()
        comm.reset()
        t_step = time.perf_counter()
        opt.zero_grad()
        logits, _, _ = ddp(batch)
        if n_train_local > 0:
            loss = supervised_loss(logits[batch.train_mask], y_train, data.multilabel, cfg.loss, cfg.focal_gamma,
                                   weight)
            loss = loss * (norm_local * world / norm_global)
        else:
            loss = logits.sum() * 0.0                   # still joins the gradient all-reduce
        loss.backward()
        opt.step()
        step_times.append(time.perf_counter() - t_step)
        comm_times.append(comm.seconds)
        comm_bytes.append(comm.bytes)
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            val = _global_score(ddp.module, batch, y_np, val_mask, data.multilabel)
            curves.append({"epoch": epoch, "val": val})
            if val > best_val:
                best_val, best_epoch = val, epoch
                best_state = copy.deepcopy(ddp.module.state_dict())
            elif epoch - best_epoch >= cfg.patience:
                break

    if best_state is not None:
        ddp.module.load_state_dict(best_state)
    val = _global_score(ddp.module, batch, y_np, val_mask, data.multilabel)
    test = _global_score(ddp.module, batch, y_np, test_mask, data.multilabel)

    warm = slice(1, None) if len(step_times) > 1 else slice(None)   # drop the first (warm-up) step
    step_mean = _sum(float(np.mean(step_times[warm]))) / world      # synchronous steps: ~equal on all ranks
    comm_mean = _sum(float(np.mean(comm_times[warm]))) / world
    payload = float(np.mean(comm_bytes))
    has_er = "edge_reduction" in stages.diagnostics                 # same on every rank (same assign mode)
    edge_reduction = _sum(stages.diagnostics["edge_reduction"]) / world if has_er else None
    if rank != 0:
        return None
    result = {
        "key": cfg.key() + f"-ddp{world}", "config": asdict(cfg), "metric": data.metric, "world_size": world,
        **code_version(), "environment": environment(),
        "val": val, "test": test, "best_epoch": best_epoch, "epochs_run": len(step_times),
        "propagation_depth": int(getattr(model, "propagation_depth", cfg.depth)),
        "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "n_train": int(n_train_global),
        "systems": {
            "stages_s": t_stages, "time_per_epoch_s": step_mean,
            "edges_per_s": edges_global * cfg.depth / step_mean if step_mean > 0 else float("nan"),
            "allreduce_s_per_step": comm_mean,
            "comms_share": comm_mean / step_mean if step_mean > 0 else float("nan"),
            "bytes_per_step_payload": payload, "bytes_per_step_wire": payload * 2.0 * (world - 1) / world,
            "rho": stages.diagnostics.get("rho"), "edge_reduction": edge_reduction,
        },
        "stages": stages.diagnostics, "curves": curves,
    }
    save_json(Path(cfg.out_dir) / f"{result['key']}.json", result)
    print(f"[ddp x{world}] {cfg.dataset} depth={cfg.depth} test={test:.4f} "
          f"time/epoch={step_mean:.3f}s comms={result['systems']['comms_share']:.2%}")
    return result


def run_and_leave(cfg: RunConfig) -> None:
    """Run ``worker`` in the current process group and leave the group.

    After a successful run every rank waits at a barrier before any rank destroys the group: every rank but 0 returns
    from ``worker`` as soon as its last collective is done, while rank 0 still writes the result.  After an exception
    the barrier is skipped, because the other ranks may be waiting in a collective that this rank will never join.
    A process that gets through this successfully should end with ``leave_process``.
    """
    try:
        worker(cfg)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def leave_process(status: int = 0) -> None:
    """End this process at once, without shutting the interpreter down, after flushing the standard streams.

    A process that returns normally from a run aborts now and then at its very end ("terminate called without an
    active exception", SIGABRT), after the result is written.  Cause, from the core files of such processes: the worker
    threads of the gloo process group are still alive after ``destroy_process_group()``.  The one that has just finished
    a collective releases the tensors of that work, and for a tensor that Python has already dropped this needs the
    GIL.  If the main thread has begun to shut the interpreter down by then, Python ends the thread inside a destructor
    that must not throw, and the C++ runtime calls ``std::terminate``.  ``os._exit`` skips the interpreter shutdown and
    with it the race.  It also skips ``atexit`` handlers and finalizers; the package registers none, and the result is
    written before the barrier.  Measured with ``scripts/check_distributed_exit.py`` (CHANGELOG): 4 ranks, 400 runs of
    each way of leaving; 15 aborts without the barrier, 3 with it, none with it and ``os._exit``.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


def _local_entry(rank: int, world: int, port: int, cfg_dict: dict) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    run_and_leave(config_from_dict(cfg_dict))
    leave_process()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn_local(cfg: RunConfig, world: int = 2, port: int | None = None) -> None:
    """Run ``world`` processes on this machine (tests and single-node scaling runs)."""
    import torch.multiprocessing as mp

    mp.spawn(_local_entry, args=(world, port or _free_port(), asdict(cfg)), nprocs=world, join=True)


def main(argv=None) -> None:
    cfg = parse_args(argv)
    dist.init_process_group(backend=os.environ.get("NHDP_DIST_BACKEND", "gloo"))
    run_and_leave(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
    leave_process()
