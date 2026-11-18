"""docs/diagrams/verification.mmd names workflows, make targets and check counts: they must exist and agree with the code."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAGRAM = ROOT / "docs" / "diagrams" / "verification.mmd"
WORKFLOWS = ROOT / ".github" / "workflows"


def diagram_text() -> str:
    return DIAGRAM.read_text(encoding="utf-8")


def test_the_diagram_shows_every_workflow_and_only_workflows_that_exist():
    named = set(re.findall(r"\b([\w-]+\.yml)\b", diagram_text()))
    assert named == {p.name for p in WORKFLOWS.glob("*.yml")}


def test_make_targets_in_the_diagram_exist():
    targets = set(re.findall(r"\bmake ([a-z][\w-]*)", diagram_text()))
    defined = set(re.findall(r"^([A-Za-z][\w-]*):", (ROOT / "Makefile").read_text(), flags=re.M))
    assert targets == {"test", "smoke", "smoke-full", "demo", "demo-quick"}
    assert targets <= defined


def test_the_check_counts_in_the_diagram_are_those_of_the_smoke_script():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke.py"), "--list"], capture_output=True, text=True,
                         check=True).stdout.splitlines()
    numpy_only = sum("[numpy only]" in line for line in out)
    without_real_data = sum("--real" not in line for line in out)
    text = diagram_text()
    assert int(re.search(r"numpy level: (\d+) checks", text).group(1)) == numpy_only
    assert int(re.search(r"full level: (\d+) checks", text).group(1)) == without_real_data


def test_the_workflows_start_on_the_branches_the_diagram_names():
    text = diagram_text()
    smoke = (WORKFLOWS / "smoke.yml").read_text()
    demo = (WORKFLOWS / "demo.yml").read_text()
    tests = (WORKFLOWS / "tests.yml").read_text()
    assert "smoke/**" in text and 'branches: [main, "smoke/**"]' in smoke
    assert "demo/**" in text and 'branches: ["demo/**"]' in demo
    assert "reproduce/**" in text and "results/**" in text
    assert 'branches-ignore: ["reproduce/**", "results/**"]' in tests
    assert "pull_request" in smoke and "pull_request" in tests and "workflow_dispatch" in smoke
