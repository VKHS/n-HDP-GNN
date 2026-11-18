"""scripts/launch_cluster.sh: the ssh/torchrun commands it builds, with a stand-in ``ssh`` that only records them.

Nothing here starts a process on another machine; it checks the rank and rendezvous arguments, the world sizes
that are skipped, the pass-through of extra arguments and what happens when the host list is wrong or a rank fails.
"""
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "launch_cluster.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

FAKE_SSH = """#!/usr/bin/env bash
# records "host<TAB>command" and fails for the host named in FAKE_SSH_FAIL
printf '%s\\t%s\\n' "$1" "$2" >> "$FAKE_SSH_LOG"
[[ "$1" == "${FAKE_SSH_FAIL:-}" ]] && exit 7
exit 0
"""


@pytest.fixture
def cluster(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(FAKE_SSH)
    ssh.chmod(ssh.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "ssh.log"

    def launch(hosts_text, *args, world_sizes=None, fail=None, **env):
        log.unlink(missing_ok=True)                       # one record of calls per launch
        hostfile = tmp_path / "hosts.txt"
        hostfile.write_text(hosts_text)
        e = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", HOSTFILE=str(hostfile),
                 FAKE_SSH_LOG=str(log))
        e.pop("WORLD_SIZES", None)
        e.pop("GLOO_SOCKET_IFNAME", None)
        if world_sizes is not None:
            e["WORLD_SIZES"] = world_sizes
        if fail:
            e["FAKE_SSH_FAIL"] = fail
        e.update({k: str(v) for k, v in env.items()})
        proc = subprocess.run(["bash", str(SCRIPT), *args], env=e, cwd=tmp_path, capture_output=True, text=True,
                              timeout=60)
        calls = []
        if log.exists():
            for line in log.read_text().splitlines():
                host, command = line.split("\t", 1)
                calls.append((host, " ".join(command.split())))
        return proc, calls

    return launch


def rank_of(command):
    return int(command.split("--node-rank ")[1].split()[0])


def test_each_world_size_launches_one_rank_per_host_in_file_order(cluster):
    proc, calls = cluster("w01\nw02\nw03\nw04\n", "configs/hpc_products.toml", world_sizes="1 2 4")
    assert proc.returncode == 0, proc.stderr
    assert len(calls) == 1 + 2 + 4
    by_world = {}
    for host, cmd in calls:
        world = int(cmd.split("--nnodes ")[1].split()[0])
        by_world.setdefault(world, []).append((rank_of(cmd), host))
    assert {w: sorted(v) for w, v in by_world.items()} == {
        1: [(0, "w01")], 2: [(0, "w01"), (1, "w02")], 4: [(0, "w01"), (1, "w02"), (2, "w03"), (3, "w04")]}
    assert "=== world size 4 (master w01) ===" in proc.stdout
    assert proc.stdout.strip().endswith("results: python -m nhdp_gnn.report runs/hpc --scaling")


def test_the_first_host_is_the_rendezvous_point_for_every_rank(cluster):
    _, calls = cluster("w01\nw02\nw03\n", "configs/hpc_products.toml", world_sizes="3")
    assert len(calls) == 3
    for _, cmd in calls:
        assert "--master-addr w01 --master-port 29500" in cmd
        assert "--nproc-per-node 1" in cmd and "-m nhdp_gnn.distributed --config 'configs/hpc_products.toml'" in cmd
        assert cmd.startswith("source '/opt/cluster-env/bin/activate' && cd '")


def test_environment_variables_and_extra_arguments_reach_the_nodes(cluster):
    _, calls = cluster("w01\nw02\n", "configs/hpc_products.toml", "--depth", "30", "--model", "gcnii",
                       world_sizes="2", PORT=29777, ENV_ACTIVATE="/srv/venv/bin/activate", REPO="/srv/n-HDP-GNN")
    assert len(calls) == 2
    for _, cmd in calls:
        assert cmd.startswith("source '/srv/venv/bin/activate' && cd '/srv/n-HDP-GNN' && torchrun")
        assert "--master-port 29777" in cmd and cmd.endswith("--config 'configs/hpc_products.toml' --depth 30 --model gcnii")


def test_the_network_interface_is_pinned_only_when_asked(cluster):
    _, calls = cluster("w01\nw02\n", "configs/hpc_products.toml", world_sizes="2", GLOO_SOCKET_IFNAME="eno1")
    assert len(calls) == 2
    for _, cmd in calls:
        assert " && export GLOO_SOCKET_IFNAME='eno1' && cd '" in cmd and cmd.index("export") < cmd.index("torchrun")
    _, calls = cluster("w01\nw02\n", "configs/hpc_products.toml", world_sizes="2")
    assert all("GLOO_SOCKET_IFNAME" not in cmd for _, cmd in calls)


def test_comments_blank_lines_and_windows_line_ends_in_the_host_file(cluster):
    proc, calls = cluster("# workers, in launch order\r\n\r\n  w01  \r\nw02\n\n# spare\nw03\n",
                          "configs/hpc_products.toml", world_sizes="3")
    assert proc.returncode == 0, proc.stderr
    assert sorted(host for host, _ in calls) == ["w01", "w02", "w03"]
    assert all("--master-addr w01 " in cmd for _, cmd in calls)


def test_a_world_size_larger_than_the_host_list_is_skipped_and_said_so(cluster):
    proc, calls = cluster("w01\nw02\nw03\n", "configs/hpc_products.toml", world_sizes="2 14")
    assert proc.returncode == 0
    assert len(calls) == 2
    assert "skip world size 14: only 3 hosts listed" in proc.stderr


def test_nothing_to_run_is_an_error_not_a_success(cluster):
    proc, calls = cluster("w01\n", "configs/hpc_products.toml", world_sizes="2 4")
    assert proc.returncode == 1 and calls == []
    assert "nothing ran" in proc.stderr and "results:" not in proc.stdout


def test_an_empty_or_missing_host_file_is_an_error(cluster, tmp_path):
    proc, calls = cluster("# no hosts\n\n", "configs/hpc_products.toml")
    assert proc.returncode == 2 and calls == [] and "no hosts listed" in proc.stderr
    proc = subprocess.run(["bash", str(SCRIPT), "configs/hpc_products.toml"],
                          env=dict(os.environ, HOSTFILE=str(tmp_path / "absent.txt")), capture_output=True, text=True)
    assert proc.returncode == 2 and "cannot read HOSTFILE" in proc.stderr


def test_the_config_and_the_host_file_are_required(cluster):
    proc = subprocess.run(["bash", str(SCRIPT)], env={**os.environ, "HOSTFILE": "x"}, capture_output=True, text=True)
    assert proc.returncode != 0 and "usage" in proc.stderr
    env = {k: v for k, v in os.environ.items() if k != "HOSTFILE"}
    proc = subprocess.run(["bash", str(SCRIPT), "c.toml"], env=env, capture_output=True, text=True)
    assert proc.returncode != 0 and "HOSTFILE" in proc.stderr


def test_a_failing_rank_stops_the_launcher_with_a_failure_status(cluster):
    proc, calls = cluster("w01\nw02\nw03\n", "configs/hpc_products.toml", world_sizes="2 3", fail="w02")
    assert proc.returncode != 0
    assert "results:" not in proc.stdout
    assert not any("--nnodes 3" in cmd for _, cmd in calls)           # the next world size did not start
