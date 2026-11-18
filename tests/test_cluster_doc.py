"""docs/CLUSTER.md and its diagrams: the example addressing plan is one rule applied consistently everywhere,
nothing in the article-only diagram looks like an address, and the files the page names exist.

The addresses are an example (the article gives none); these tests keep the table, the diagram, the hosts file and
the configuration snippets from drifting apart, and keep the example labelled as one.
"""
import importlib.util
import ipaddress
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "CLUSTER.md").read_text()
SUBNET = ipaddress.ip_network("192.168.100.0/24")
GENERATED = {"docs/cluster/inventory.md", "docs/cluster/inventory.json"}      # written by the collector, on the user's side


def example_table():
    rows = re.findall(r"^\| (worker \d\d|tower[^|]*) \| (\d+|none) \| `([^`]+)` \| `([^`]+)` \|$", DOC, re.M)
    assert rows, "the example addressing table is missing"
    return rows


def code_blocks():
    return re.findall(r"^```[a-z]*\n(.*?)^```", DOC, re.M | re.S)


def test_the_example_table_follows_one_rule():
    rows = example_table()
    assert len(rows) == 15
    master, workers = rows[0], rows[1:]
    assert master[1:] == ("none", "nhdp-master", "192.168.100.10")
    for rank, (_, r, name, address) in enumerate(workers):
        assert (int(r), name, address) == (rank, f"nhdp-w{rank + 1:02d}", f"192.168.100.{11 + rank}")
    addresses = [ipaddress.ip_address(a) for *_, a in rows]
    assert len(set(addresses)) == 15 and all(a in SUBNET and a.is_private for a in addresses)
    assert all(a not in (SUBNET.network_address, SUBNET.broadcast_address) for a in addresses)


def test_the_page_says_the_addresses_are_an_example():
    assert "## 5. A reference addressing plan (example values)" in DOC
    assert "**These are example values, not the addresses of the original cluster.**" in DOC
    assert "gives no HP model numbers, CPU model, memory size, host names or IP" in re.sub(r"\s+", " ", DOC.replace("> ", ""))
    assert "Example addressing plan, not the addresses of the original cluster" in (
        ROOT / "docs" / "diagrams" / "addressing.mmd").read_text()


def test_the_not_stated_table_lists_what_the_article_leaves_out():
    section = DOC.split("## 2. What the article does not state")[1].split("## 3.")[0]
    for item in ("model numbers", "CPU model", "switch", "VLAN", "IP addresses", "Ubuntu release",
                 "Whether the master host also ran a training rank"):
        assert item in section
    assert "not recorded" in section


def test_etc_hosts_and_hosts_file_match_the_table():
    rows = example_table()
    block = next(b for b in code_blocks() if "nhdp-master" in b and "192.168.100.10" in b)
    pairs = [tuple(line.split()) for line in block.strip().splitlines()]
    assert pairs == [(address, name) for _, _, name, address in rows]
    names = [line for line in (ROOT / "configs" / "hosts.example.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    assert names == [name for _, _, name, _ in rows[1:]]


def test_both_tools_read_the_hosts_example_the_same_way():
    spec = importlib.util.spec_from_file_location("cluster_inventory_doc", ROOT / "scripts" / "cluster_inventory.py")
    inv = importlib.util.module_from_spec(spec)
    sys.modules["cluster_inventory_doc"] = inv
    try:
        spec.loader.exec_module(inv)
        workers = inv.read_hosts(ROOT / "configs" / "hosts.example.txt")
    finally:
        sys.modules.pop("cluster_inventory_doc", None)
    assert workers == [f"nhdp-w{i:02d}" for i in range(1, 15)]


def test_the_netplan_example_is_valid_and_inside_the_subnet():
    block = next(b for b in code_blocks() if b.startswith("# /etc/netplan/"))
    assert "dhcp4: false" in block
    address = re.search(r"addresses: \[([0-9./]+)\]", block).group(1)
    assert ipaddress.ip_interface(address).ip == ipaddress.ip_address("192.168.100.11")       # worker 01, rank 0
    assert ipaddress.ip_interface(address).network == SUBNET
    import yaml                                    # in the test extra: a missing module must not skip this check
    parsed = yaml.safe_load(block)
    assert parsed["network"]["ethernets"]["eno1"]["addresses"] == [address]


def test_the_addressing_diagram_matches_the_table():
    src = (ROOT / "docs" / "diagrams" / "addressing.mmd").read_text()
    nodes = re.findall(r"\*\*nhdp-w(\d\d)\*\*\n\.(\d+)\nrank (\d+)`", src)
    assert [(int(n), int(a), int(r)) for n, a, r in nodes] == [(r + 1, 11 + r, r) for r in range(14)]
    assert "192.168.100.10" in src and "192.168.100.0/24" in src and "192.168.100.(11 + rank)" in src


def test_the_article_only_diagram_has_no_addresses():
    src = (ROOT / "docs" / "diagrams" / "cluster.mmd").read_text()
    assert not re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", src)
    assert "example" not in src.lower()


def test_files_the_page_names_exist():
    roots = ("scripts/", "configs/", "tests/", "nhdp_gnn/", "docs/", "examples/")
    named = {t.rstrip("/") for t in re.findall(r"`([A-Za-z0-9_./-]+)`", DOC) if t.startswith(roots) and "*" not in t}
    named -= GENERATED
    missing = sorted(p for p in named if not (ROOT / p).exists())
    assert not missing, f"CLUSTER.md names files that do not exist: {missing}"
    assert {"scripts/launch_cluster.sh", "scripts/cluster_inventory.py", "configs/cluster_check.toml",
            "configs/hosts.example.txt", "nhdp_gnn/distributed.py"} <= named


def test_the_commands_use_options_the_scripts_have():
    inventory = (ROOT / "scripts" / "cluster_inventory.py").read_text()
    for flag in set(re.findall(r"cluster_inventory\.py[^\n]*?(--[a-z-]+)", DOC)) | {"--addresses", "--master-local", "--repo", "--hosts", "--out"}:
        assert f'"{flag}"' in inventory, flag
    launcher = (ROOT / "scripts" / "launch_cluster.sh").read_text()
    for var in ("HOSTFILE", "WORLD_SIZES", "GLOO_SOCKET_IFNAME"):
        assert var in launcher and var in DOC
