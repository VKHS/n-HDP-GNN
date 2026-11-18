"""Record what the cluster is: hardware, operating system, network addresses and software of every node.

    python scripts/cluster_inventory.py --hosts hosts.txt --master-local --repo /srv/n-HDP-GNN \
        --addresses subnet --out docs/cluster/inventory

The article (Sect. 10) describes the platform -- 14 HP ProDesk worker nodes with a 6-thread CPU each, a tower
host as master and launcher, a 24-port 1 GbE switch, a flat L2 network with static addressing -- but gives no
model numbers, memory sizes or addresses.  This script asks every host for those facts and writes down what
the host reports, so that nothing in docs/CLUSTER.md has to be typed from memory.

    hosts.txt     the worker hosts, one per line, in launch order (the same file as scripts/launch_cluster.sh:
                  the first host is rank 0).  Blank lines and lines that start with # are ignored.
    --master      the master and launcher host, listed first in the tables (optional).
    --master-local  the master is the machine this script runs on: collect it directly, without ssh
                  (--master then only gives the name shown in the tables; default: this machine's hostname).
    --local       collect from this machine only, without ssh (to try the script, or for one node).
    --repo        the repository copy on every node: its git commit and whether it has uncommitted changes.
    --env-python  the Python of the shared environment (default /opt/cluster-env/bin/python): its version and the
                  versions of the packages the runs import.
    --addresses   full (default) | subnet | none: how much of the network addresses is written to the files.
                  Publish only private addresses (10/8, 172.16/12, 192.168/16); a public address of a host that
                  accepts ssh is an invitation.

Each host is asked over ssh (BatchMode, so the password-less login the launcher needs must work) to run a short
Python 3 program that reads /proc, /sys, ``lscpu``, ``ip`` and ``lspci``.  MAC addresses, serial numbers and
UUIDs are not collected.  Output: <out>.json (what each host reported, and the checks) and <out>.md (tables).
A host that cannot be reached is listed with the error; the exit status is 1 if there is one.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import shlex
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PYTHON = "/opt/cluster-env/bin/python"
PACKAGES = ["torch", "torch_geometric", "numpy", "scipy", "networkx", "scikit-learn", "mpi4py", "python-louvain",
            "igraph", "pymetis", "ogb"]

# Runs on every node: ``python3 - ENV_PYTHON REPO`` with this text on stdin.  Python >= 3.8, standard library only.
COLLECT = r'''
import json, os, platform, re, socket, subprocess, sys

def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None

def run(cmd, timeout=20):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True,
                           timeout=timeout)
        return p.stdout if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None

def main(env_python, repo):
    out = {"hostname": socket.gethostname(), "architecture": platform.machine(), "kernel": platform.release()}
    try:
        out["hostname_address"] = socket.gethostbyname(out["hostname"])
    except OSError:
        out["hostname_address"] = None
    osr = read("/etc/os-release") or ""
    m = re.search(r'^PRETTY_NAME="?([^"\n]*)"?', osr, re.M)
    out["os"] = m.group(1) if m else None
    out["dmi"] = {k: read("/sys/class/dmi/id/" + k) for k in
                  ("sys_vendor", "product_name", "product_version", "board_name", "bios_vendor", "bios_version",
                   "bios_date")}
    cpu = {}
    lscpu = run(["lscpu", "-J"])
    if lscpu:
        try:
            for row in json.loads(lscpu)["lscpu"]:
                cpu[row["field"].rstrip(":")] = row["data"]
        except (ValueError, KeyError, TypeError):
            cpu = {}
    if not cpu:
        info = read("/proc/cpuinfo") or ""
        m = re.search(r"^model name\s*:\s*(.*)$", info, re.M)
        cpu = {"Model name": m.group(1) if m else None, "CPU(s)": str(os.cpu_count())}
    keep = ("Model name", "Architecture", "CPU(s)", "Thread(s) per core", "Core(s) per socket", "Socket(s)",
            "CPU max MHz", "CPU min MHz")
    out["cpu"] = {k: cpu.get(k) for k in keep}
    out["cpu_policy"] = {"scaling_governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
                         "intel_no_turbo": read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
                         "cpufreq_boost": read("/sys/devices/system/cpu/cpufreq/boost"),
                         "smt_active": read("/sys/devices/system/cpu/smt/active")}
    meminfo = read("/proc/meminfo") or ""
    m = re.search(r"^MemTotal:\s+(\d+) kB", meminfo, re.M)
    out["ram_gib"] = round(int(m.group(1)) / 1024 / 1024, 2) if m else None
    nets, default = [], None
    ip = run(["ip", "-j", "-4", "addr", "show"])
    if ip:
        try:
            for link in json.loads(ip):
                if link.get("ifname") == "lo":
                    continue
                nets.append({"name": link["ifname"], "state": link.get("operstate"), "mtu": link.get("mtu"),
                             "addresses": ["%s/%s" % (a["local"], a["prefixlen"]) for a in link.get("addr_info", [])
                                           if a.get("family") == "inet"]})
        except (ValueError, KeyError, TypeError):
            nets = []
    if not nets:
        text = run(["ip", "-o", "-4", "addr", "show"]) or ""
        seen = {}
        for line in text.splitlines():
            mm = re.match(r"\d+:\s+(\S+)\s+inet\s+(\S+)", line)
            if mm and mm.group(1) != "lo":
                seen.setdefault(mm.group(1), []).append(mm.group(2))
        nets = [{"name": k, "state": None, "mtu": None, "addresses": v} for k, v in seen.items()]
    for n in nets:
        speed = read("/sys/class/net/%s/speed" % n["name"])
        n["speed_mbps"] = int(speed) if speed and speed.lstrip("-").isdigit() and int(speed) > 0 else None
        link = "/sys/class/net/%s/device/driver" % n["name"]
        n["driver"] = os.path.basename(os.path.realpath(link)) if os.path.islink(link) else None
        if n["mtu"] is None:
            mtu = read("/sys/class/net/%s/mtu" % n["name"])
            n["mtu"] = int(mtu) if mtu and mtu.isdigit() else None
    route = run(["ip", "-j", "route", "show", "default"])
    if route:
        try:
            r = json.loads(route)
            default = {"interface": r[0].get("dev"), "gateway": r[0].get("gateway")} if r else None
        except (ValueError, IndexError, AttributeError):
            default = None
    out["interfaces"], out["default_route"] = nets, default
    gpus = []
    for line in (run(["lspci", "-mm"]) or "").splitlines():
        if re.search(r'"(VGA compatible controller|3D controller|Display controller)"', line):
            parts = re.findall(r'"([^"]*)"', line)
            if len(parts) >= 3:
                gpus.append(parts[1] + " " + parts[2])
    out["gpus"] = gpus
    probe = ("import json, sys\n"
             "from importlib import metadata\n"
             "pk = {}\n"
             "for n in %r:\n"
             "    try:\n        pk[n] = metadata.version(n)\n    except Exception:\n        pk[n] = None\n"
             "t = {}\n"
             "try:\n    import torch\n    t = {'threads': torch.get_num_threads(), 'cuda': bool(torch.cuda.is_available()), "
             "'gloo': bool(torch.distributed.is_gloo_available())}\nexcept Exception as e:\n    t = {'error': str(e)[:120]}\n"
             "print(json.dumps({'python': sys.version.split()[0], 'packages': pk, 'torch': t}))\n") % (__PACKAGES__,)
    py = env_python if env_python and os.path.exists(env_python) else sys.executable
    env = {"python_path": py, "python": None, "packages": {}, "torch": {}}
    text = run([py, "-c", probe], timeout=90)
    if text:
        try:
            env.update(json.loads(text.strip().splitlines()[-1]))
        except (ValueError, IndexError):
            pass
    out["environment"] = env
    if repo:
        head = run(["git", "-C", repo, "rev-parse", "HEAD"])
        dirty = run(["git", "-C", repo, "status", "--porcelain"])
        out["repository"] = {"path_exists": os.path.isdir(repo), "commit": head.strip() if head else None,
                             "uncommitted_changes": bool(dirty.strip()) if dirty is not None else None}
    return out

if __name__ == "__main__":
    args = sys.argv[1:] + ["", ""]
    print(json.dumps(main(args[0], args[1])))
'''.replace("__PACKAGES__", repr(PACKAGES))


# ----------------------------------------------------------------------------- collection
def read_hosts(path: Path) -> list[str]:
    hosts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            hosts.append(line)
    return hosts


def ssh_runner(options: list[str], timeout: float):
    """A function ``(host, args) -> stdout`` that runs COLLECT on ``host`` over ssh."""
    def run(host: str, args: list[str]) -> str:
        remote = "python3 - " + " ".join(shlex.quote(a) for a in args)
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", *[x for o in options for x in ("-o", o)],
               host, remote]
        proc = subprocess.run(cmd, input=COLLECT, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr.strip().splitlines() or [f"ssh exited with status {proc.returncode}"])[-1])
        return proc.stdout
    return run


def local_runner(timeout: float):
    def run(host: str, args: list[str]) -> str:
        proc = subprocess.run([sys.executable, "-", *args], input=COLLECT, capture_output=True, text=True,
                              timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr.strip().splitlines() or ["the collector failed"])[-1])
        return proc.stdout
    return run


def collect_host(host: str, role: str, rank, runner, env_python: str, repo: str) -> dict:
    record = {"host": host, "role": role, "rank": rank}
    try:
        out = runner(host, [env_python, repo])
        record.update(json.loads(out.strip().splitlines()[-1]))
        record.update(host=host, role=role, rank=rank)      # what the launcher assigned, whatever the node reported
    except subprocess.TimeoutExpired:
        record["error"] = "timed out"
    except (RuntimeError, ValueError, IndexError, OSError) as e:
        record["error"] = str(e) or type(e).__name__
    return record


def collect(workers: list[str], master: str | None, runner, env_python: str, repo: str, jobs: int = 8,
            master_runner=None) -> list[dict]:
    """One record per host, the master first.  ``master_runner`` (default: ``runner``) collects the master."""
    plan = ([(master, "master", None, master_runner or runner)] if master else []) + [
        (h, "worker", i, runner) for i, h in enumerate(workers)]
    with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(plan)))) as pool:
        futures = [pool.submit(collect_host, h, role, rank, run, env_python, repo) for h, role, rank, run in plan]
        return [f.result() for f in futures]


# ----------------------------------------------------------------------------- checks
def primary_interface(record: dict) -> dict | None:
    """The interface that carries the default route, else the first one with an IPv4 address."""
    nets = [n for n in record.get("interfaces", []) if n.get("addresses")]
    route = record.get("default_route") or {}
    for n in nets:
        if n["name"] == route.get("interface"):
            return n
    return nets[0] if nets else None


def checks(records: list[dict]) -> dict:
    ok = [r for r in records if "error" not in r]
    out: dict = {"hosts": len(records), "reachable": len(ok),
                 "unreachable": [{"host": r["host"], "error": r["error"]} for r in records if "error" in r]}
    networks: dict[str, list[str]] = {}
    seen: dict[str, list[str]] = {}
    for r in ok:
        for n in r.get("interfaces", []):
            for a in n.get("addresses", []):
                seen.setdefault(a.split("/")[0], []).append(r["host"])
        n = primary_interface(r)
        if n:
            net = str(ipaddress.ip_interface(n["addresses"][0]).network)
            networks.setdefault(net, []).append(r["host"])
    out["primary_networks"] = networks
    out["same_subnet"] = None if not networks else (
        len(networks) == 1 and sum(len(v) for v in networks.values()) == len(ok))
    out["duplicate_addresses"] = [{"address": a, "hosts": sorted(set(h))} for a, h in sorted(seen.items())
                                  if len(set(h)) > 1]
    # gloo binds to the address the node's own hostname resolves to unless GLOO_SOCKET_IFNAME is set
    out["hostname_on_loopback"] = [r["host"] for r in ok if r.get("hostname_address")
                                   and ipaddress.ip_address(r["hostname_address"]).is_loopback]
    out["hostname_unresolved"] = [r["host"] for r in ok if "hostname_address" in r and not r["hostname_address"]]

    def values(get):
        by: dict[str, list[str]] = {}
        for r in ok:
            try:
                v = get(r)
            except (KeyError, TypeError, IndexError):
                v = None
            by.setdefault("not reported" if v is None else str(v), []).append(r["host"])
        return by

    out["distinct"] = {
        "model": values(lambda r: " ".join(x for x in (r["dmi"]["sys_vendor"], r["dmi"]["product_name"]) if x) or None),
        "cpu": values(lambda r: r["cpu"]["Model name"]),
        "logical_cpus": values(lambda r: r["cpu"]["CPU(s)"]),
        "cpu_governor": values(lambda r: r["cpu_policy"]["scaling_governor"]),
        "smt_active": values(lambda r: r["cpu_policy"]["smt_active"]),
        "ram_gib": values(lambda r: r["ram_gib"]),
        "os": values(lambda r: r["os"]),
        "kernel": values(lambda r: r["kernel"]),
        "link_mbps": values(lambda r: primary_interface(r)["speed_mbps"]),
        "mtu": values(lambda r: primary_interface(r)["mtu"]),
        "python": values(lambda r: r["environment"]["python"]),
        "torch": values(lambda r: r["environment"]["packages"]["torch"]),
        "torch_geometric": values(lambda r: r["environment"]["packages"]["torch_geometric"]),
        "repository_commit": values(lambda r: r["repository"]["commit"]) if any("repository" in r for r in ok) else {},
    }
    out["same_everywhere"] = [k for k, v in out["distinct"].items() if len(v) == 1 and "not reported" not in v]
    out["differs"] = [k for k, v in out["distinct"].items() if len(v) > 1]
    return out


def apply_address_mode(records: list[dict], report: dict, mode: str) -> tuple[list[dict], dict]:
    """Strip the network addresses of the records and of the checks down to what ``mode`` allows."""
    if mode == "full":
        return records, report
    records = json.loads(json.dumps(records))
    report = json.loads(json.dumps(report))
    for r in records:
        r["hostname_address"] = None
        for n in r.get("interfaces", []):
            n["addresses"] = [str(ipaddress.ip_interface(a).network) for a in n.get("addresses", [])] \
                if mode == "subnet" else []
        route = r.get("default_route")
        if route:
            route["gateway"] = None
    for d in report["duplicate_addresses"]:
        d["address"] = "omitted"
    if mode == "none":
        report["primary_networks"] = {}
    return records, report


# ----------------------------------------------------------------------------- report
def md_table(header: list[str], rows: list[list]) -> list[str]:
    cell = lambda v: "-" if v in (None, "") else str(v).replace("|", "\\|")
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]


def label(r: dict) -> str:
    return "master" if r["role"] == "master" else f"{r['rank']}"


def render_markdown(records: list[dict], report: dict, command: str, mode: str) -> str:
    ok = [r for r in records if "error" not in r]
    L = ["# Cluster inventory", "",
         f"Written by `{command}` on {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. Every value was reported by "
         "the host itself (`/proc`, `/sys`, `lscpu`, `ip`, `lspci`, `importlib.metadata`, `git`); nothing was typed in. "
         "Rank is the position of the host in the hosts file, as in `scripts/launch_cluster.sh`.", ""]
    if report["unreachable"]:
        L += ["## Hosts that did not answer", ""]
        L += md_table(["host", "error"], [[u["host"], u["error"]] for u in report["unreachable"]]) + [""]
    L += ["## Machines", ""]
    rows = []
    for r in ok:
        cpu, dmi = r["cpu"], r["dmi"]
        shape = [cpu.get(k) for k in ("Socket(s)", "Core(s) per socket", "Thread(s) per core")]
        mhz = cpu.get("CPU max MHz")
        rows.append([label(r), r["host"], " ".join(x for x in (dmi.get("sys_vendor"), dmi.get("product_name")) if x),
                     cpu.get("Model name"), " x ".join(shape) if all(shape) else None,
                     cpu.get("CPU(s)"), f"{float(mhz):.0f}" if mhz else None,
                     (r.get("cpu_policy") or {}).get("scaling_governor"), r.get("ram_gib"),
                     "; ".join(r.get("gpus", [])), r.get("os"), r.get("kernel")])
    L += md_table(["rank", "host", "model", "CPU", "sockets x cores x threads", "logical CPUs", "max MHz", "governor",
                   "RAM visible to the OS (GiB)", "graphics", "OS", "kernel"], rows) + [""]
    L += ["## Network", ""]
    if mode == "none":
        L += ["Addresses are omitted from this file (`--addresses none`).", ""]
    rows = []
    for r in ok:
        for n in r.get("interfaces", []):
            rows.append([label(r), r["host"], n["name"], ", ".join(n["addresses"]) or "-", n.get("state"),
                         n.get("speed_mbps"), n.get("mtu"), n.get("driver"),
                         "yes" if (r.get("default_route") or {}).get("interface") == n["name"] else ""])
    L += md_table(["rank", "host", "interface", "IPv4" if mode == "full" else "IPv4 (subnet only)", "state",
                   "link speed (Mb/s)", "MTU", "driver", "default route"], rows) + [""]
    L += ["## Software", ""]
    rows = []
    for r in ok:
        env, pk = r.get("environment", {}), r.get("environment", {}).get("packages", {})
        repo = r.get("repository") or {}
        commit = (repo.get("commit") or "")[:12] + (" (uncommitted changes)" if repo.get("uncommitted_changes") else "")
        rows.append([label(r), r["host"], env.get("python"), pk.get("torch"), pk.get("torch_geometric"), pk.get("numpy"),
                     pk.get("scipy"), pk.get("networkx"), pk.get("mpi4py"), pk.get("python-louvain"), commit])
    L += md_table(["rank", "host", "Python", "torch", "torch_geometric", "numpy", "scipy", "networkx", "mpi4py",
                   "python-louvain", "repository commit"], rows) + [""]
    L += ["## Checks", ""]
    nets = report["primary_networks"]
    if report["same_subnet"]:
        where = f" in `{next(iter(nets))}`" if nets else ""
        L.append(f"- Network: all {report['reachable']} reachable hosts have their primary address on one subnet"
                 f"{where}. (The same IP subnet is necessary for a flat L2 network, not proof of one.)")
    elif report["same_subnet"] is None:
        L.append("- Network: no host reported an IPv4 address on an interface other than loopback.")
    elif report["reachable"]:
        L.append("- Network: the hosts are NOT all on one subnet" + (": " + "; ".join(
            f"`{net}`: {', '.join(h)}" for net, h in nets.items()) if nets else "") + ".")
    dup = report["duplicate_addresses"]
    L.append("- Duplicate addresses: " + ("none." if not dup else "; ".join(
        f"{d['address']} on {', '.join(d['hosts'])}" for d in dup) + "."))
    if report["hostname_on_loopback"]:
        L.append("- Hostname resolution: the node's own hostname resolves to a loopback address on "
                 + ", ".join(report["hostname_on_loopback"]) + ". gloo then binds to loopback and the ranks cannot reach "
                 "each other, unless `GLOO_SOCKET_IFNAME` names the cluster's interface (docs/CLUSTER.md).")
    if report["hostname_unresolved"]:
        L.append("- Hostname resolution: the node's own hostname does not resolve on "
                 + ", ".join(report["hostname_unresolved"]) + ".")
    if not report["hostname_on_loopback"] and not report["hostname_unresolved"] and report["reachable"]:
        L.append("- Hostname resolution: on every reachable host the node's own hostname resolves to a non-loopback address.")
    if report["same_everywhere"]:
        L.append("- The same on every reachable host: " + ", ".join(report["same_everywhere"]) + ".")
    for key in report["differs"]:
        by = report["distinct"][key]
        L.append(f"- Differs between hosts, {key}: " + "; ".join(f"{k} ({', '.join(h)})" for k, h in by.items()) + ".")
    L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------------------- command
def main(argv=None, runner=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts", help="file with one worker host per line, in launch order")
    ap.add_argument("--master", help="the master and launcher host")
    ap.add_argument("--master-local", action="store_true",
                    help="the master is this machine: collect it without ssh (--master only names it)")
    ap.add_argument("--local", action="store_true", help="collect from this machine only, without ssh")
    ap.add_argument("--repo", default="", help="path of the repository copy on every node")
    ap.add_argument("--env-python", default=DEFAULT_ENV_PYTHON)
    ap.add_argument("--ssh-option", action="append", default=[], help="an ssh -o option, e.g. ConnectTimeout=5")
    ap.add_argument("--timeout", type=float, default=180.0, help="seconds per host")
    ap.add_argument("--addresses", choices=("full", "subnet", "none"), default="full")
    ap.add_argument("--out", default="docs/cluster/inventory", help="output prefix: <out>.json and <out>.md")
    args = ap.parse_args(argv)
    if bool(args.hosts) == bool(args.local):
        ap.error("give --hosts FILE, or --local")
    if args.master_local and args.local:
        ap.error("--master-local goes with --hosts; --local already collects this machine only")

    if args.local:
        run = runner or local_runner(args.timeout)
        records = [collect_host("localhost", "local", 0, run, args.env_python, args.repo)]
    else:
        workers = read_hosts(Path(args.hosts))
        if not workers:
            ap.error(f"{args.hosts} lists no hosts")
        run = runner or ssh_runner(args.ssh_option, args.timeout)
        master = args.master or (socket.gethostname() if args.master_local else None)
        records = collect(workers, master, run, args.env_python, args.repo,
                          master_runner=local_runner(args.timeout) if args.master_local else None)
    report = checks(records)
    command = "python scripts/cluster_inventory.py " + " ".join(
        shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:]))
    published, report = apply_address_mode(records, report, args.addresses)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"command": command, "collected_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "addresses": args.addresses, "hosts": published, "checks": report}
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    out.with_suffix(".md").write_text(render_markdown(published, report, command, args.addresses) + "\n")
    print(f"{report['reachable']} of {report['hosts']} hosts answered; wrote {out.with_suffix('.md')} and "
          f"{out.with_suffix('.json')}")
    for u in report["unreachable"]:
        print(f"  {u['host']}: {u['error']}", file=sys.stderr)
    return 0 if not report["unreachable"] else 1


if __name__ == "__main__":
    sys.exit(main())
