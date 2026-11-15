"""Merge run records into a sweep's results branch and push it, safely under concurrency.

    python scripts/publish_results.py configs/table2_full.toml --records runs/table2_full \\
        [--only ...] [--set ...] [--branch results/table2_full] [--message "..."]

Fetches the branch (if it exists), unpacks its records (runs.jsonl.gz and
failures.jsonl.gz), adds the records in --records (which take precedence), rebuilds
the tables with scripts/make_tables.py and pushes a new commit on top of the branch.
If someone else pushed in between, the push is rejected and everything is redone
from the new branch head, so records are never lost.  The working tree and the
current branch are not touched (the commit is built with a temporary index).

The reproduce workflow runs it at the end of every job, so the results branch grows
as jobs finish.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_tables  # noqa: E402
import unpack_results  # noqa: E402


def git(*args, env=None, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check,
                          env={**os.environ, **(env or {})})


def publish_once(config, name, records, branch, remote, message, only, sets, copy_to=None) -> tuple[bool, str]:
    """One attempt; returns (pushed or already up to date, description).  ``copy_to`` receives the tables."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fetched = git("fetch", "-q", remote, f"+refs/heads/{branch}:refs/remotes/{remote}/{branch}",
                      check=False).returncode == 0
        parent = git("rev-parse", f"refs/remotes/{remote}/{branch}").stdout.strip() if fetched else ""
        run_dirs = [str(r) for r in records]
        if parent:
            archive = tmp / "previous.tar"
            with open(archive, "wb") as fh:
                has_results = subprocess.run(["git", "archive", parent, f"results/{name}"], stdout=fh,
                                             stderr=subprocess.DEVNULL).returncode == 0
            if has_results:
                subprocess.run(["tar", "-xf", str(archive), "-C", str(tmp)], check=True)
                unpack_results.unpack(tmp / "results" / name, tmp / "previous")
                run_dirs.append(str(tmp / "previous"))
        out = tmp / "tree"
        argv = [config, "--out", str(out / "results" / name)]
        for d in run_dirs:
            argv += ["--runs", d]
        if only:
            argv += ["--only", only]
        for s in sets:
            argv += ["--set", s]
        make_tables.main(argv)
        index = {"GIT_INDEX_FILE": str(tmp / "index")}
        git_dir = git("rev-parse", "--absolute-git-dir").stdout.strip()
        subprocess.run(["git", "--git-dir", git_dir, "--work-tree", str(out), "add", "-A", "-f", "."], cwd=out,
                       check=True, capture_output=True, env={**os.environ, **index})
        tree = git("write-tree", env=index).stdout.strip()
        if parent and tree == git("rev-parse", f"{parent}^{{tree}}").stdout.strip():
            text = f"{branch} already up to date ({parent[:12]})"
        else:
            commit = git("commit-tree", tree, *(["-p", parent] if parent else []), "-m", message).stdout.strip()
            push = git("push", "-q", remote, f"{commit}:refs/heads/{branch}", check=False)
            if push.returncode != 0:
                return False, push.stderr.strip()
            text = f"pushed {commit[:12]} to {branch}"
        if copy_to:
            shutil.copytree(out / "results" / name, copy_to, dirs_exist_ok=True)
        return True, text


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--records", action="append", default=[], help="directory of run records (may be repeated)")
    ap.add_argument("--name", default=None, help="sweep name (default: from the sweep file)")
    ap.add_argument("--branch", default=None, help="default: results/<name>")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--message", default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default=None, help="also write the published tables to this directory")
    ap.add_argument("--attempts", type=int, default=12)
    ap.add_argument("--backoff", type=float, default=20.0, help="longest wait between attempts, in seconds")
    args = ap.parse_args(argv)

    import tomllib

    name = args.name or tomllib.loads(Path(args.config).read_text()).get("name", Path(args.config).stem)
    branch = args.branch or f"results/{name}"
    records = [Path(r) for r in args.records if Path(r).exists()]
    message = args.message or f"Results of {name}"
    for attempt in range(1, args.attempts + 1):
        ok, text = publish_once(args.config, name, records, branch, args.remote, message, args.only, args.set,
                                args.out)
        print(f"attempt {attempt}: {text}", flush=True)
        if ok:
            return
        time.sleep(random.uniform(args.backoff / 4, args.backoff) * attempt ** 0.5)
    sys.exit(f"could not push {branch} after {args.attempts} attempts")


if __name__ == "__main__":
    main()
