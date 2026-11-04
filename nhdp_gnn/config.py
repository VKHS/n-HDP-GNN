"""Run configuration (torch-free): every experiment is one RunConfig.

Fields cover the data and protocol, the model, optimisation, stages A/B and
outputs.  ``key()`` hashes every field that can change a result, so sweeps can
skip runs whose JSON already exists.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, fields

from .nhdp import NHDPConfig
from .pipeline import PipelineConfig
from .utils import config_hash


@dataclass
class RunConfig:
    # data and protocol
    dataset: str = "cora"
    root: str = "data"
    regime: str = "semi"                # full | semi | scarce
    seed: int = 0
    semi_per_class: int = 20
    semi_ratio: float = 0.2
    scarce_ratio: float = 0.05
    synthetic: dict = field(default_factory=dict)
    # model
    model: str = "nhdp"                 # nhdp or a baseline name
    depth: int = 2
    hidden: int = 64
    heads: int = 4
    dropout: float = 0.5
    attn_dropout: float = 0.2
    node_op: str = "gat"                # gat | gcn
    community: bool = True
    global_attn: bool = True
    residual: str = "res"               # res | initial | none
    alpha: float = 0.1                  # initial residual (nhdp, GCNII); teleport probability (APPNP)
    theta: float = 0.5                  # GCNII identity-mapping strength
    norm: str = "batch"                 # batch | layer | none
    # optimisation
    lr: float = 0.005
    weight_decay: float = 5e-4
    epochs: int = 500
    patience: int = 100
    eval_every: int = 1
    loss: str = "ce"                    # ce | focal
    focal_gamma: float = 2.0
    class_weights: bool = False
    aux_weight: float = 0.01            # Louvain auxiliary task (0 disables)
    # stages A and B
    num_parts: int = 1
    partition: str = "auto"
    louvain_resolution: float = 1.0
    louvain_backend: str = "auto"
    assign: str = "nhdp"                # nhdp | louvain | random | none
    gate: bool = True
    topk: int = 4
    feature_dim: int = 32
    feature_hops: int = 0
    nhdp_K: int = 64
    nhdp_T1: int = 16
    nhdp_T2: int = 8
    nhdp_levels: int = 3
    nhdp_gamma: float = 1.0
    nhdp_alpha1: float = 1.0
    nhdp_alpha0: float = 1.0
    nhdp_max_iter: int = 100
    # outputs
    out_dir: str = "runs/default"
    cache_dir: str = "cache"
    save_embeddings: bool = False       # per-layer node embeddings as .npy
    save_curves: bool = True
    diagnostics: bool = True            # MADGap, centroid gap, k-means NMI/ARI
    layer_profile: bool = True          # centroid gap at every layer (graphs up to 200k nodes)
    madgap_sources: int = 1000
    threads: int = 0
    device: str = "cpu"
    tag: str = ""

    _OUTPUT_ONLY = ("root", "out_dir", "cache_dir", "save_embeddings", "save_curves", "threads", "device", "tag")

    def key(self) -> str:
        d = {k: v for k, v in asdict(self).items() if k not in self._OUTPUT_ONLY}
        return config_hash(d)

    def pipeline(self) -> PipelineConfig:
        return PipelineConfig(
            num_parts=self.num_parts, partition=self.partition, louvain_resolution=self.louvain_resolution,
            louvain_backend=self.louvain_backend, assign=self.assign if self.model == "nhdp" else "none",
            gate=self.gate, topk=self.topk, feature_dim=self.feature_dim, feature_hops=self.feature_hops,
            communities=self.model == "nhdp", seed=self.seed,
            nhdp=NHDPConfig(K=self.nhdp_K, T1=self.nhdp_T1, T2=self.nhdp_T2, levels=self.nhdp_levels,
                            gamma=self.nhdp_gamma, alpha1=self.nhdp_alpha1, alpha0=self.nhdp_alpha0,
                            max_iter=self.nhdp_max_iter, seed=self.seed))


def config_from_dict(d: dict) -> RunConfig:
    known = {f.name for f in fields(RunConfig)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return RunConfig(**d)


def _parse_value(text: str, default):
    if isinstance(default, bool):
        if text.lower() in ("1", "true", "yes", "on"):
            return True
        if text.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"not a boolean: {text}")
    if isinstance(default, int):
        return int(text)
    if isinstance(default, float):
        return float(text)
    if isinstance(default, dict):
        return json.loads(text)
    return text


def parse_args(argv=None, description: str = "") -> RunConfig:
    defaults = RunConfig()
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="TOML file with flat RunConfig keys")
    for f in fields(RunConfig):
        parser.add_argument(f"--{f.name}", type=str, default=None)
    args = parser.parse_args(argv)
    values = {}
    if args.config:
        import tomllib

        with open(args.config, "rb") as fh:
            values.update(tomllib.load(fh))
    for f in fields(RunConfig):
        raw = getattr(args, f.name)
        if raw is not None:
            values[f.name] = _parse_value(raw, getattr(defaults, f.name))
    return config_from_dict(values)
