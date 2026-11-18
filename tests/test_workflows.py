"""The workflow files parse, and what the documents say about them holds.

GitHub reports an invalid workflow only when it tries to run one, and a workflow that does not start is easy to miss,
so every file is parsed here.
"""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
FILES = sorted(WORKFLOWS.glob("*.yml"))


def load(name: str) -> dict:
    data = yaml.safe_load((WORKFLOWS / f"{name}.yml").read_text(encoding="utf-8"))
    data["triggers"] = data.get("on", data.get(True))          # YAML 1.1 reads the key `on` as True
    return data


def steps(data: dict):
    for job in data["jobs"].values():
        yield from job["steps"]


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_each_workflow_parses_and_every_step_does_one_thing(path):
    data = load(path.stem)
    assert data["name"] == path.stem and data["triggers"] and data["jobs"]
    for job_name, job in data["jobs"].items():
        assert job.get("runs-on"), job_name
        for step in job["steps"]:
            assert ("uses" in step) != ("run" in step), f"{job_name}: {step}"


def test_every_action_is_pinned_to_a_major_version():
    for path in FILES:
        for step in steps(load(path.stem)):
            if "uses" in step:
                assert "@v" in step["uses"], f"{path.name}: {step['uses']}"


def test_the_tests_workflow_installs_what_the_figure_and_yaml_tests_need():
    installs = " ".join(s["run"] for s in steps(load("tests")) if "run" in s and "pip install" in s["run"])
    assert "torch_geometric" in installs and ".[test,figures]" in installs


def test_the_smoke_workflow_runs_the_full_level_and_keeps_its_report():
    data = load("smoke")
    runs = " ".join(s["run"] for s in steps(data) if "run" in s)
    assert "--level full" in runs and "scripts/smoke.py" in runs and "GITHUB_STEP_SUMMARY" in runs
    uploads = [s["with"] for s in steps(data) if str(s.get("uses", "")).startswith("actions/upload-artifact")]
    assert [u["name"] for u in uploads] == ["smoke-report"]
    assert "report.md" in uploads[0]["path"] and "report.json" in uploads[0]["path"]
    assert data["triggers"]["push"]["branches"] == ["main", "smoke/**"]
    assert data["triggers"]["workflow_dispatch"]["inputs"]["real"]["type"] == "boolean"
    assert data["permissions"] == {"contents": "read"}


def test_the_demo_workflow_trains_and_publishes_to_the_branch_the_documents_name():
    data = load("demo")
    runs = " ".join(s["run"] for s in steps(data) if "run" in s)
    assert "examples/demo.py" in runs and "--train yes" in runs
    assert "refs/heads/results/demo" in runs and "examples/output" in runs
    uploads = [s["with"] for s in steps(data) if str(s.get("uses", "")).startswith("actions/upload-artifact")]
    assert [u["name"] for u in uploads] == ["demo-output"]
    assert data["triggers"]["push"]["branches"] == ["demo/**"]
    assert data["permissions"] == {"contents": "write"}                  # it pushes the branch
    for doc in ("examples/README.md", "docs/diagrams/verification.mmd"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        assert "results/demo" in text and "demo-output" in text, doc
