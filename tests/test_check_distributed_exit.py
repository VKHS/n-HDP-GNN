"""scripts/check_distributed_exit.py: the arguments it refuses and, with PyTorch, one round of every variant."""
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_distributed_exit.py"
HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torch_geometric") is not None


def _load():
    spec = importlib.util.spec_from_file_location("check_distributed_exit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_script_imports_without_pytorch_and_names_its_variants():
    module = _load()
    assert module.VARIANTS == ("old", "barrier", "exit") and set(module.ENTRIES) == set(module.VARIANTS)


@pytest.mark.parametrize("args, message", [(["--variants", "old,nope"], "unknown variants"),
                                           (["--variants", ""], "unknown variants"),
                                           (["--world", "1"], "--world must be at least 2"),
                                           (["--rounds", "0"], "--rounds at least 1")])
def test_the_script_refuses_what_it_cannot_run(args, message, capsys):
    with pytest.raises(SystemExit) as stop:
        _load().main(args)
    assert stop.value.code == 2 and message in capsys.readouterr().err


@pytest.mark.skipif(not HAS_TORCH, reason="needs PyTorch and PyTorch Geometric")
def test_one_round_of_every_variant_runs_and_is_counted():
    done = subprocess.run([sys.executable, str(SCRIPT), "--world", "2", "--rounds", "1"], cwd=ROOT,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    for variant in ("old", "barrier", "exit"):                  # a rare abort at exit is counted, not an error here
        assert re.search(rf"^  {variant} +[01] of +1 ended without a failure", done.stdout, re.M), done.stdout
