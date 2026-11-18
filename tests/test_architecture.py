"""The module map in docs/diagrams/modules.mmd against the code (no torch needed).

The diagram says: every module of the package is in exactly one box; a layer imports only from the layers
below it, and the boxes of one layer do not import each other; green boxes need PyTorch and blue boxes import
without it.  These tests read the diagram's source and the imports of the modules and check each statement, so
the picture cannot drift from the code.
"""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "nhdp_gnn"
DIAGRAM = ROOT / "docs" / "diagrams" / "modules.mmd"
TORCH = {"torch", "torch_geometric"}


def diagram_modules():
    """module name -> (layer, box id, 'torch' or 'free') for every ``name.py`` named in a box of the diagram."""
    text = DIAGRAM.read_text()
    out = {}
    for layer, body in re.findall(r"subgraph L(\d)\b.*?\n(.*?)\n  end", text, flags=re.S):
        for box, label, kind in re.findall(r'(\w+)\("`(.*?)`"\):::(\w+)', body, flags=re.S):
            for name in re.findall(r"\b([a-z_]+)\.py\b", label):
                assert name not in out, f"{name}.py is in two boxes of the diagram"
                out[name] = (int(layer), box, kind)
    return out


def package_modules():
    return sorted(p.stem for p in PACKAGE.glob("*.py") if p.stem != "__init__")


def imports_of(module: str):
    """(modules of this package imported anywhere in the file, third-party names imported at module level)."""
    tree = ast.parse((PACKAGE / f"{module}.py").read_text())
    local, top = set(), set()
    names = set(package_modules())

    def visit(node):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module:
                local.add(node.module.split(".")[0])
            elif node.level == 1:
                local.update(a.name for a in node.names if a.name in names)
            elif node.module and node.module.split(".")[0] == "nhdp_gnn":
                parts = node.module.split(".")
                if len(parts) > 1:
                    local.add(parts[1])
                else:
                    local.update(a.name for a in node.names if a.name in names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] == "nhdp_gnn" and len(parts) > 1:
                    local.add(parts[1])

    for node in ast.walk(tree):
        visit(node)
    for node in tree.body:                       # imports that run when the module is imported
        stack = [node] if isinstance(node, (ast.Import, ast.ImportFrom)) else \
            [n for n in ast.walk(node) if isinstance(n, (ast.Import, ast.ImportFrom))] \
            if isinstance(node, (ast.Try, ast.If)) else []
        for n in stack:
            if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                top.add(n.module.split(".")[0])
            elif isinstance(n, ast.Import):
                top.update(a.name.split(".")[0] for a in n.names)
    return local & names, top


def test_every_module_is_in_exactly_one_box():
    assert sorted(diagram_modules()) == package_modules()


@pytest.mark.parametrize("module", package_modules())
def test_a_module_imports_only_from_lower_layers(module):
    layers = diagram_modules()
    local, _ = imports_of(module)
    layer, box, _ = layers[module]
    for other in sorted(local):
        other_layer, other_box, _ = layers[other]
        assert other_layer < layer or (other_layer == layer and other_box == box), (
            f"{module}.py (layer {layer}) imports {other}.py (layer {other_layer}, "
            f"{'the same' if other_layer == layer else 'another'} box level)")


@pytest.mark.parametrize("module", package_modules())
def test_boxes_marked_torch_are_the_ones_that_import_torch(module):
    _, _, kind = diagram_modules()[module]
    _, top = imports_of(module)
    assert (kind == "torch") == bool(top & TORCH), (
        f"{module}.py is drawn as '{kind}' but its module-level imports are {sorted(top & TORCH) or 'torch-free'}")


def test_the_torch_free_modules_import_without_torch():
    import subprocess
    import sys

    free = [m for m, (_, _, kind) in diagram_modules().items() if kind == "free"]
    code = ("import sys, importlib\n"
            f"for m in {free!r}:\n    importlib.import_module('nhdp_gnn.' + m)\n"
            "assert 'torch' not in sys.modules, sorted(k for k in sys.modules if k.startswith('torch'))\n")
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                          env={"PYTHONPATH": str(ROOT), "PATH": ""})
    assert proc.returncode == 0, proc.stderr[-1500:]
