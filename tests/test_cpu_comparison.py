"""examples/cpu_comparison.json, the script that makes it, and the figures that examples/README.md quotes from it."""
import gzip
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "cpu_comparison.json"
README = ROOT / "examples" / "README.md"
OUTPUT = ROOT / "examples" / "output"


def _load_script():
    spec = importlib.util.spec_from_file_location("compare_demo_runs", ROOT / "scripts" / "compare_demo_runs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cmp = _load_script()


@pytest.fixture(scope="module")
def data():
    return json.loads(DATA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def readme():
    return re.sub(r"\s+", " ", README.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ the file
def test_the_file_holds_three_runs_and_ninety_rows_with_one_value_per_run(data):
    assert len(data["runs"]) == 3 and len({r["run_id"] for r in data["runs"]}) == 3
    for r in data["runs"]:
        assert r["url"] == f"https://github.com/VKHS/n-HDP-GNN/actions/runs/{r['run_id']}"
        assert re.fullmatch(r"[0-9a-f]{40}", r["code_commit"]) and r["cpu_model"]
    assert len(data["rows"]) == 90
    assert len({(r["variant"], r["depth"], r["seed"]) for r in data["rows"]}) == 90
    for r in data["rows"]:
        for field in cmp.FIELDS:
            assert len(r[field]) == 3, (r["variant"], r["depth"], r["seed"], field)
        assert all(0.0 <= a <= 1.0 for a in r["test"] + r["val"])


def test_the_last_run_is_the_committed_output(data):
    doc = json.loads((OUTPUT / "demo.json").read_text(encoding="utf-8"))
    last = data["runs"][-1]
    if (doc.get("ci") or {}).get("run_id") != last["run_id"]:
        pytest.skip("the committed output is from a later run than the last one of the comparison")
    assert doc["environment"]["cpu_model"] == last["cpu_model"]
    with gzip.open(OUTPUT / "runs.jsonl.gz", "rt", encoding="utf-8") as fh:
        committed = {(r["config"]["tag"], r["config"]["depth"], r["config"]["seed"]): r for r in map(json.loads, fh)}
    assert set(committed) == {(r["variant"], r["depth"], r["seed"]) for r in data["rows"]}
    assert all(r["code_commit"] == last["code_commit"] for r in committed.values())
    for row in data["rows"]:
        record = committed[(row["variant"], row["depth"], row["seed"])]
        assert [row[f][-1] for f in cmp.FIELDS] == [record[f] for f in cmp.FIELDS]


# ------------------------------------------------------------------------------------------------ the README
def test_the_readme_quotes_the_runs_and_their_cpu_models(data, readme):
    for r in data["runs"]:
        assert f"[{r['run_id']}]({r['url']})" in readme
        assert r["code_commit"][:7] in readme
    cpus = [r["cpu_model"] for r in data["runs"]]
    assert "AMD EPYC 9V45" in cpus[0] and "6975P-C" in cpus[1] and "AMD EPYC 9V74" in cpus[2]
    for name in ("AMD EPYC 9V45", "Intel Xeon 6975P-C", "AMD EPYC 9V74"):
        assert name in readme


def test_the_two_amd_runs_agree_everywhere_and_the_intel_run_differs_from_both_in_the_same_runs(data):
    for field in cmp.FIELDS:
        assert cmp.differing(data, 0, 2, field) == []
        assert cmp.differing(data, 0, 1, field) == cmp.differing(data, 2, 1, field)


def test_the_readme_figures_follow_from_the_file(data, readme):
    n = len(data["rows"])
    diff = cmp.differing(data, 0, 1)
    assert f"in test accuracy in {len(diff)} of the {n} runs" in readme
    by_depth = {d: sum(1 for r in diff if r["depth"] == d) for d in (2, 8, 32)}
    per_depth = {d: sum(1 for r in data["rows"] if r["depth"] == d) for d in (2, 8, 32)}
    assert f"({by_depth[2]} of {per_depth[2]} at depth 2, {by_depth[8]} of {per_depth[8]} at depth 8, " \
           f"{by_depth[32]} of {per_depth[32]} at depth 32)" in readme
    assert f"and in best epoch in {len(cmp.differing(data, 0, 1, 'best_epoch'))}." in readme

    gaps = sorted(abs(r["test"][0] - r["test"][1]) * 100 for r in diff)
    small, big = gaps[:-3], gaps[-3:]
    assert max(small) <= 4.2 and min(big) >= 23 and max(big) <= 25 and round(max(small), 1) == 4.2
    assert f"Of the {len(diff)}, {len(small)} differ by 4.2 points or less and three by 23 to 25 points" in readme
    largest = {(r["variant"], r["depth"], r["seed"]): r for r in diff if abs(r["test"][0] - r["test"][1]) * 100 > 20}
    assert set(largest) == {("nhdp_gate_off", 32, 0), ("nhdp", 32, 2), ("gat", 8, 4)}
    for key, template in ((("nhdp_gate_off", 32, 0), "n-HDP-GNN with the gate off at depth 32, seed 0 ({:.1f}% on AMD, {:.1f}% on Intel)"),
                          (("nhdp", 32, 2), "n-HDP-GNN at depth 32, seed 2 ({:.1f}%, {:.1f}%)"),
                          (("gat", 8, 4), "GAT at depth 8, seed 4 ({:.1f}%, {:.1f}%)")):
        assert template.format(100 * largest[key]["test"][0], 100 * largest[key]["test"][1]) in readme, key


def test_the_cell_means_in_the_readme_follow_from_the_file(data, readme):
    amd, intel = cmp.cell_means(data, 0), cmp.cell_means(data, 1)
    assert cmp.cell_means(data, 2) == amd                       # the two AMD runs agree, so "AMD" is either of them
    moved = {k: abs(amd[k] - intel[k]) for k in amd if amd[k] != intel[k]}
    assert f"differs in {len(moved)} of the {len(amd)} cells" in readme
    large = sorted(moved, key=lambda k: -moved[k])[:3]
    assert set(large) == {("nhdp", 32), ("nhdp_gate_off", 32), ("gat", 8)}
    assert 4.55 <= min(moved[k] for k in large) and max(moved[k] for k in large) < 5.35
    assert "Three move by 4.6 to 5.3 points" in readme
    for k, text in ((("nhdp", 32), "n-HDP-GNN at depth 32 ({:.1f} on AMD, {:.1f} on Intel)"),
                    (("nhdp_gate_off", 32), "the same with the gate off ({:.1f}, {:.1f})"),
                    (("gat", 8), "GAT at depth 8 ({:.1f}, {:.1f})")):
        assert text.format(amd[k], intel[k]) in readme, k
    small = [v for k, v in moved.items() if k not in large]
    assert len(small) == 6 and 0.1 <= min(small) and max(small) < 0.35 and "six move by 0.1 to 0.3 points" in readme
    unmoved = sorted(k for k in amd if k not in moved)
    assert {v for v, _ in unmoved} >= {"appnp", "gcnii"} and ("gcn", 2) in unmoved and ("gcn", 8) in unmoved \
        and ("gat", 2) in unmoved and len(unmoved) == 9


def test_the_leaders_named_in_the_readme_lead_on_both_cpu_types(data):
    for run in (0, 1, 2):
        mean = cmp.cell_means(data, run)
        for depth in (2, 8):
            assert max((v for v in {k[0] for k in mean}), key=lambda v: mean[(v, depth)]) == "nhdp", (run, depth)
        top_two = sorted({k[0] for k in mean}, key=lambda v: -mean[(v, 32)])[:2]
        assert set(top_two) == {"appnp", "gcnii"}, run
        for lower in ("gat", "gcn"):
            assert mean[("nhdp", 32)] > mean[(lower, 32)] and mean[("nhdp_gate_off", 32)] > mean[(lower, 32)], (run, lower)


# ------------------------------------------------------------------------------------------------ the script
def _write_runs(path: Path, cpu: str, commit: str = "a" * 40, drop: int = 0, accuracy: float = 0.5, depth: int = 2):
    records = [{"config": {"tag": tag, "depth": depth, "seed": seed}, "environment": {"cpu_model": cpu},
                "code_commit": commit, "test": accuracy, "val": 0.6, "best_epoch": 10 + seed}
               for tag in ("nhdp", "gcn") for seed in range(3)][drop:]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_the_script_puts_runs_side_by_side_and_reports_the_differences(tmp_path, capsys):
    one = _write_runs(tmp_path / "one.jsonl.gz", "CPU one", accuracy=0.5)
    two = _write_runs(tmp_path / "two.jsonl.gz", "CPU two", commit="b" * 40, accuracy=0.75)
    out = tmp_path / "comparison.json"
    assert cmp.main(["--out", str(out), f"111={one}", f"222={two}"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [r["cpu_model"] for r in data["runs"]] == ["CPU one", "CPU two"]
    assert [r["code_commit"] for r in data["runs"]] == ["a" * 40, "b" * 40]
    assert len(data["rows"]) == 6 and data["rows"][0] == {"variant": "nhdp", "depth": 2, "seed": 0, "test": [0.5, 0.75],
                                                          "val": [0.6, 0.6], "best_epoch": [10, 10]}
    printed = capsys.readouterr().out
    assert "test       differs in   6 of 6 runs" in printed and "val        differs in   0 of 6 runs" in printed
    assert "25.0 (nhdp, depth 2, seed 0: 50.0 against 75.0)" in printed


@pytest.mark.parametrize("problem", ["two CPU models in one run", "other configurations", "no CPU model", "bad argument"])
def test_the_script_refuses_what_it_cannot_compare(tmp_path, problem):
    good = _write_runs(tmp_path / "good.jsonl.gz", "CPU one")
    if problem == "two CPU models in one run":
        mixed = tmp_path / "mixed.jsonl.gz"
        with gzip.open(good, "rt", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh]
        records[0]["environment"]["cpu_model"] = "CPU two"
        with gzip.open(mixed, "wt", encoding="utf-8") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in records))
        with pytest.raises(SystemExit, match="2 different CPU models"):
            cmp.main([f"1={good}", f"2={mixed}"])
    elif problem == "other configurations":
        other = _write_runs(tmp_path / "other.jsonl.gz", "CPU two", drop=1)
        with pytest.raises(SystemExit, match="other configurations"):
            cmp.main([f"1={good}", f"2={other}"])
    elif problem == "no CPU model":
        bare = _write_runs(tmp_path / "bare.jsonl.gz", None)
        with pytest.raises(SystemExit, match="do not name the CPU model"):
            cmp.main([f"1={bare}", f"2={good}"])
    else:
        with pytest.raises(SystemExit):
            cmp.main([f"x={good}", f"2={good}"])
