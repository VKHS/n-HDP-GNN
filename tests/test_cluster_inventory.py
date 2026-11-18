"""scripts/cluster_inventory.py: what it reads from a node, what it concludes about 14 of them, and what it writes.

The program that runs on every node is exercised with the output formats of ``lscpu -J``, ``ip -j``, ``lspci -mm``
and /proc as fixtures (the machines in them are made up and say nothing about the article's cluster); the checks
and the report are exercised on fourteen fabricated nodes; the ssh layer is replaced by a function.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inv():
    spec = importlib.util.spec_from_file_location("cluster_inventory", ROOT / "scripts" / "cluster_inventory.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cluster_inventory"] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop("cluster_inventory", None)


# ----------------------------------------------------------------------------- the program that runs on a node
LSCPU = json.dumps({"lscpu": [
    {"field": "Architecture:", "data": "x86_64", "children": [{"field": "CPU op-mode(s):", "data": "32-bit, 64-bit"}]},
    {"field": "CPU(s):", "data": "6"}, {"field": "Model name:", "data": "Test CPU @ 3.00GHz"},
    {"field": "Thread(s) per core:", "data": "1"}, {"field": "Core(s) per socket:", "data": "6"},
    {"field": "Socket(s):", "data": "1"}, {"field": "CPU max MHz:", "data": "4100.0000"}]})
IP_ADDR = json.dumps([
    {"ifindex": 1, "ifname": "lo", "mtu": 65536, "operstate": "UNKNOWN",
     "addr_info": [{"family": "inet", "local": "127.0.0.1", "prefixlen": 8}]},
    {"ifindex": 2, "ifname": "eno1", "mtu": 1500, "operstate": "UP", "address": "aa:bb:cc:dd:ee:ff",
     "addr_info": [{"family": "inet", "local": "192.168.100.11", "prefixlen": 24},
                   {"family": "inet6", "local": "fe80::1", "prefixlen": 64}]},
    {"ifindex": 3, "ifname": "wlp2s0", "mtu": 1500, "operstate": "DOWN", "addr_info": []}])
IP_ROUTE = json.dumps([{"dst": "default", "gateway": "192.168.100.1", "dev": "eno1"}])
LSPCI = ('00:00.0 "Host bridge" "Vendor A" "Host Bridge" -r00 "" ""\n'
         '00:02.0 "VGA compatible controller" "Test Graphics Inc." "Test GT2 [Graphics 000]" -ra1 "" ""\n'
         '00:1f.6 "Ethernet controller" "Vendor B" "Gigabit NIC" -r10 "" ""\n')
FILES = {
    "/etc/os-release": 'NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 24.04 LTS"\nVERSION_ID="24.04"\n',
    "/sys/class/dmi/id/sys_vendor": "Test Vendor", "/sys/class/dmi/id/product_name": "Test Desktop 100",
    "/proc/meminfo": "MemTotal:       16384000 kB\nMemFree:  100 kB\n",
    "/sys/class/net/eno1/speed": "1000", "/sys/class/net/wlp2s0/speed": "-1",
    "/sys/class/net/eno1/mtu": "1500",
    "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor": "performance",
    "/sys/devices/system/cpu/intel_pstate/no_turbo": "0", "/sys/devices/system/cpu/smt/active": "0",
}
ENV_PROBE = json.dumps({"python": "3.12.3", "packages": {"torch": "2.3.0", "torch_geometric": "2.6.0"},
                        "torch": {"threads": 6, "cuda": False, "gloo": True}})


class FakeSocket:
    def __init__(self, resolves_to="192.168.100.11"):
        self.resolves_to = resolves_to

    @staticmethod
    def gethostname():
        return "nhdp-w01"

    def gethostbyname(self, name):
        if self.resolves_to is None:
            raise OSError("Name or service not known")
        return self.resolves_to


def node_program(inv, resolves_to="192.168.100.11"):
    ns: dict = {}
    exec(compile(inv.COLLECT, "collect", "exec"), ns)
    ns["socket"] = FakeSocket(resolves_to)

    def run(cmd, timeout=20):
        key = " ".join(cmd[:3]) if cmd[0] != "git" else " ".join(cmd[3:5])
        if cmd[:2] == ["lscpu", "-J"]:
            return LSCPU
        if cmd[:4] == ["ip", "-j", "-4", "addr"]:
            return IP_ADDR
        if cmd[:4] == ["ip", "-j", "route", "show"]:
            return IP_ROUTE
        if cmd[0] == "lspci":
            return LSPCI
        if cmd[0] == "git" and "rev-parse" in cmd:
            return "0123456789abcdef0123456789abcdef01234567\n"
        if cmd[0] == "git" and "status" in cmd:
            return " M file.py\n"
        if len(cmd) > 1 and cmd[1] == "-c":
            return "warning on the way\n" + ENV_PROBE + "\n"
        return None

    ns["run"] = run
    ns["read"] = lambda path: FILES.get(path)
    return ns


def test_the_node_program_reads_the_tool_formats(inv):
    ns = node_program(inv)
    r = ns["main"]("/definitely/missing/python", "/some/repo")
    assert r["os"] == "Ubuntu 24.04 LTS" and r["hostname"] == "nhdp-w01" and r["hostname_address"] == "192.168.100.11"
    assert r["dmi"]["sys_vendor"] == "Test Vendor" and r["dmi"]["product_name"] == "Test Desktop 100"
    assert r["cpu"]["Model name"] == "Test CPU @ 3.00GHz" and r["cpu"]["CPU(s)"] == "6"
    assert r["cpu"]["Core(s) per socket"] == "6" and r["cpu"]["CPU max MHz"] == "4100.0000"
    assert r["ram_gib"] == 15.62
    assert r["cpu_policy"] == {"scaling_governor": "performance", "intel_no_turbo": "0", "cpufreq_boost": None,
                               "smt_active": "0"}
    eno1 = [n for n in r["interfaces"] if n["name"] == "eno1"][0]
    assert eno1["addresses"] == ["192.168.100.11/24"] and eno1["mtu"] == 1500 and eno1["speed_mbps"] == 1000
    assert all(n["name"] != "lo" for n in r["interfaces"])
    assert [n for n in r["interfaces"] if n["name"] == "wlp2s0"][0]["speed_mbps"] is None   # link down: -1
    assert r["default_route"] == {"interface": "eno1", "gateway": "192.168.100.1"}
    assert r["gpus"] == ["Test Graphics Inc. Test GT2 [Graphics 000]"]                   # not the NIC, not the bridge
    assert r["environment"]["python"] == "3.12.3" and r["environment"]["packages"]["torch"] == "2.3.0"
    assert r["repository"] == {"path_exists": False, "commit": "0123456789abcdef0123456789abcdef01234567",
                               "uncommitted_changes": True}
    assert "mac" not in json.dumps(r).lower() and "aa:bb" not in json.dumps(r)             # no MAC addresses


def test_the_node_program_survives_missing_tools(inv):
    ns = node_program(inv, resolves_to=None)
    ns["run"] = lambda cmd, timeout=20: None
    ns["read"] = lambda path: None
    r = ns["main"]("", "")
    assert r["hostname_address"] is None
    assert r["interfaces"] == [] and r["default_route"] is None and r["gpus"] == [] and r["ram_gib"] is None
    assert r["cpu"]["Model name"] is None and "repository" not in r


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc and /sys")
def test_local_collection_runs_the_same_program(inv, tmp_path):
    code = inv.main(["--local", "--repo", str(ROOT), "--out", str(tmp_path / "inv")])
    assert code == 0
    data = json.loads((tmp_path / "inv.json").read_text())
    host = data["hosts"][0]
    assert host["role"] == "local" and host["kernel"] and host["cpu"]["CPU(s)"] and host["environment"]["python"]
    assert "Cluster inventory" in (tmp_path / "inv.md").read_text()


# ----------------------------------------------------------------------------- fourteen fabricated nodes
def fake_node(i, *, net="192.168.100", speed=1000, torch="2.3.0", commit="a" * 40, address=None,
              governor="performance", hostname_address=None):
    return {"host": f"nhdp-w{i + 1:02d}", "role": "worker", "rank": i, "hostname": f"nhdp-w{i + 1:02d}",
            "hostname_address": hostname_address or (address or f"{net}.{11 + i}/24").split("/")[0],
            "kernel": "6.8.0", "os": "Ubuntu 24.04 LTS", "ram_gib": 15.6,
            "dmi": {"sys_vendor": "Test Vendor", "product_name": "Test Desktop 100"},
            "cpu": {"Model name": "Test CPU", "CPU(s)": "6", "Socket(s)": "1", "Core(s) per socket": "6",
                    "Thread(s) per core": "1", "CPU max MHz": "4100.0000"},
            "cpu_policy": {"scaling_governor": governor, "smt_active": "0"}, "gpus": ["Test Graphics"],
            "interfaces": [{"name": "eno1", "state": "UP", "mtu": 1500, "speed_mbps": speed, "driver": "e1000e",
                            "addresses": [address or f"{net}.{11 + i}/24"]}],
            "default_route": {"interface": "eno1", "gateway": f"{net}.1"},
            "environment": {"python": "3.12.3", "packages": {"torch": torch, "torch_geometric": "2.6.0"}},
            "repository": {"commit": commit, "uncommitted_changes": False}}


def test_checks_on_fourteen_consistent_nodes(inv):
    records = [fake_node(i) for i in range(14)]
    rep = inv.checks(records)
    assert rep["reachable"] == 14 and rep["unreachable"] == [] and rep["same_subnet"] is True
    assert list(rep["primary_networks"]) == ["192.168.100.0/24"] and rep["duplicate_addresses"] == []
    assert rep["differs"] == []
    assert {"model", "cpu", "ram_gib", "kernel", "link_mbps", "torch", "repository_commit"} <= set(rep["same_everywhere"])


def test_checks_find_what_differs_between_nodes(inv):
    records = [fake_node(i) for i in range(14)]
    records[3] = fake_node(3, net="192.168.7")
    records[5] = fake_node(5, speed=100, torch="2.4.1", commit="b" * 40, governor="powersave")
    records[9] = fake_node(9, address="192.168.100.12/24")                      # the address of node 1 (rank 1)
    records[13] = {"host": "nhdp-w14", "role": "worker", "rank": 13, "error": "Connection timed out"}
    rep = inv.checks(records)
    assert rep["reachable"] == 13 and rep["unreachable"] == [{"host": "nhdp-w14", "error": "Connection timed out"}]
    assert rep["same_subnet"] is False and set(rep["primary_networks"]) == {"192.168.100.0/24", "192.168.7.0/24"}
    assert rep["duplicate_addresses"] == [{"address": "192.168.100.12", "hosts": ["nhdp-w02", "nhdp-w10"]}]
    assert {"link_mbps", "torch", "repository_commit", "cpu_governor"} <= set(rep["differs"])
    assert rep["distinct"]["torch"]["2.4.1"] == ["nhdp-w06"] and rep["distinct"]["cpu_governor"]["powersave"] == ["nhdp-w06"]


def test_checks_flag_a_hostname_that_resolves_to_loopback_or_not_at_all(inv):
    records = [fake_node(i) for i in range(4)]
    records[1] = fake_node(1, hostname_address="127.0.1.1")
    records[3]["hostname_address"] = None
    rep = inv.checks(records)
    assert rep["hostname_on_loopback"] == ["nhdp-w02"] and rep["hostname_unresolved"] == ["nhdp-w04"]
    md = inv.render_markdown(records, rep, "cmd", "full")
    assert "resolves to a loopback address on nhdp-w02" in md and "does not resolve on nhdp-w04" in md
    assert "GLOO_SOCKET_IFNAME" in md
    ok = inv.checks([fake_node(i) for i in range(4)])
    assert ok["hostname_on_loopback"] == [] and ok["hostname_unresolved"] == []
    assert "resolves to a non-loopback address" in inv.render_markdown([fake_node(i) for i in range(4)], ok, "cmd", "full")


def test_checks_do_not_claim_a_subnet_without_addresses(inv):
    rep = inv.checks([{"host": "a", "role": "worker", "rank": 0, "interfaces": [], "cpu": {}, "dmi": {},
                       "environment": {"packages": {}}}])
    assert rep["same_subnet"] is None


def test_address_modes_strip_what_they_promise_to(inv):
    records = [fake_node(i) for i in range(3)]
    records[2] = fake_node(2, address="192.168.100.11/24")                       # a duplicate
    rep = inv.checks(records)
    full_rec, full_rep = inv.apply_address_mode(records, rep, "full")
    assert "192.168.100.11/24" in json.dumps(full_rec) and "192.168.100.11" in json.dumps(full_rep)
    sub_rec, sub_rep = inv.apply_address_mode(records, rep, "subnet")
    text = json.dumps([sub_rec, sub_rep])
    assert all(r["hostname_address"] is None for r in sub_rec)
    assert "192.168.100.0/24" in text and "192.168.100.11" not in text and "192.168.100.1\"" not in text
    none_rec, none_rep = inv.apply_address_mode(records, rep, "none")
    assert "192.168.100" not in json.dumps([none_rec, none_rep])
    assert none_rep["duplicate_addresses"][0]["address"] == "omitted"
    assert "192.168.100.11/24" in json.dumps(records)                             # the input is not changed


def test_the_report_has_a_row_per_node_and_names_problems(inv):
    records = [fake_node(i) for i in range(14)]
    records[5] = fake_node(5, speed=100)
    records.append({"host": "tower", "role": "master", "rank": None, "error": "Permission denied (publickey)."})
    rep = inv.checks(records)
    md = inv.render_markdown(records, rep, "python scripts/cluster_inventory.py --hosts hosts.txt", "full")
    assert md.count("| Test Vendor Test Desktop 100 |") == 14 and md.count("| 1 x 6 x 1 | 6 | 4100 | performance |") == 14
    assert "Hosts that did not answer" in md and "Permission denied" in md
    assert "Differs between hosts, link_mbps" in md and "100 (nhdp-w06)" in md
    assert "192.168.100.11/24" in md and "192.168.100.24/24" in md                 # ranks 0 and 13


# ----------------------------------------------------------------------------- the command
def fake_runner(failing=()):
    def run(host, args):
        if host in failing:
            raise RuntimeError("ssh: connect to host failed")
        index = int(host[-2:]) - 1 if host.startswith("nhdp-w") else None
        node = fake_node(index if index is not None else 0)
        node["host"] = host
        return "noise\n" + json.dumps(node) + "\n"
    return run


def test_main_collects_in_hosts_file_order_and_writes_both_files(inv, tmp_path):
    hosts = tmp_path / "hosts.txt"
    hosts.write_text("# the workers, in launch order\nnhdp-w01\n\nnhdp-w02\nnhdp-w03\n")
    code = inv.main(["--hosts", str(hosts), "--out", str(tmp_path / "inv")], runner=fake_runner())
    assert code == 0
    data = json.loads((tmp_path / "inv.json").read_text())
    assert [h["host"] for h in data["hosts"]] == ["nhdp-w01", "nhdp-w02", "nhdp-w03"]
    assert [h["rank"] for h in data["hosts"]] == [0, 1, 2]
    assert data["checks"]["same_subnet"] is True and data["addresses"] == "full"
    assert (tmp_path / "inv.md").read_text().startswith("# Cluster inventory")


def test_main_reports_unreachable_hosts_but_still_writes_and_exits_1(inv, tmp_path, capsys):
    hosts = tmp_path / "hosts.txt"
    hosts.write_text("nhdp-w01\nnhdp-w02\n")
    code = inv.main(["--hosts", str(hosts), "--master", "tower", "--out", str(tmp_path / "inv")],
                    runner=fake_runner(failing={"nhdp-w02"}))
    assert code == 1
    data = json.loads((tmp_path / "inv.json").read_text())
    assert data["hosts"][0]["host"] == "tower" and data["hosts"][0]["role"] == "master"
    assert data["checks"]["unreachable"] == [{"host": "nhdp-w02", "error": "ssh: connect to host failed"}]
    assert "nhdp-w02: ssh: connect to host failed" in capsys.readouterr().err


def test_main_needs_hosts_or_local(inv, tmp_path):
    with pytest.raises(SystemExit):
        inv.main(["--out", str(tmp_path / "inv")])
    empty = tmp_path / "hosts.txt"
    empty.write_text("\n# nothing\n")
    with pytest.raises(SystemExit):
        inv.main(["--hosts", str(empty), "--out", str(tmp_path / "inv")])


def test_ssh_runner_builds_the_command_and_quotes_the_arguments(inv, monkeypatch):
    seen = {}

    def fake_run(cmd, input=None, capture_output=None, text=None, timeout=None):
        seen.update(cmd=cmd, input=input)

        class Done:
            returncode, stdout, stderr = 0, "{}", ""
        return Done()

    monkeypatch.setattr(inv.subprocess, "run", fake_run)
    inv.ssh_runner(["StrictHostKeyChecking=yes"], 30)("nhdp-w01", ["/opt/my env/bin/python", ""])
    cmd = seen["cmd"]
    assert cmd[0] == "ssh" and "BatchMode=yes" in cmd and "StrictHostKeyChecking=yes" in cmd
    assert cmd[-2] == "nhdp-w01" and cmd[-1] == "python3 - '/opt/my env/bin/python' ''"
    assert seen["input"] == inv.COLLECT


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc and /sys")
def test_master_local_collects_this_machine_without_ssh(inv, tmp_path):
    hosts = tmp_path / "hosts.txt"
    hosts.write_text("nhdp-w01\nnhdp-w02\n")
    asked = []

    def workers_only(host, args):
        asked.append(host)
        return fake_runner()(host, args)

    code = inv.main(["--hosts", str(hosts), "--master-local", "--repo", str(ROOT), "--out", str(tmp_path / "inv")],
                    runner=workers_only)
    assert code == 0
    data = json.loads((tmp_path / "inv.json").read_text())
    master = data["hosts"][0]
    assert master["role"] == "master" and master["host"] == inv.socket.gethostname()
    assert master["kernel"] and master["environment"]["python"]               # collected by the real program, here
    assert asked == ["nhdp-w01", "nhdp-w02"]                                   # the runner (ssh) never saw the master
    assert [h["rank"] for h in data["hosts"][1:]] == [0, 1]


def test_master_local_is_not_combined_with_local(inv, tmp_path):
    with pytest.raises(SystemExit):
        inv.main(["--local", "--master-local", "--out", str(tmp_path / "inv")])
