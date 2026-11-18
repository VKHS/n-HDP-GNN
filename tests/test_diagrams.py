"""docs/diagrams: every SVG matches its source, and the documents use only diagrams that exist."""
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS = ROOT / "docs" / "diagrams"


def load_script():
    spec = importlib.util.spec_from_file_location("make_diagrams", ROOT / "scripts" / "make_diagrams.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_diagrams"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_every_svg_is_rendered_from_its_current_source():
    mod = load_script()
    try:
        assert mod.problems() == []
    finally:
        sys.modules.pop("make_diagrams", None)


def test_svgs_are_self_contained_text_drawings():
    for svg in DIAGRAMS.glob("*.svg"):
        text = svg.read_text(encoding="utf-8")
        assert "viewBox" in text and "<text" in text, svg.name
        assert "<foreignObject" not in text, f"{svg.name}: labels must be SVG text so every viewer shows them"
        assert "<script" not in text and "@import" not in text, svg.name
        assert not re.search(r'(?:href|src)="https?://', text), f"{svg.name} refers to an external file"


def test_documents_link_only_to_diagrams_that_exist_and_every_diagram_is_used():
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), *sorted((ROOT / "examples").glob("*.md"))]
    used = set()
    for doc in documents:
        for target in re.findall(r"\]\(([^)\s#]*diagrams/[^)\s#]+\.svg)\)", doc.read_text(encoding="utf-8")):
            path = (doc.parent / target).resolve()
            assert path.exists(), f"{doc.relative_to(ROOT)} links to {target}, which does not exist"
            used.add(path.name)
    assert used == {p.name for p in DIAGRAMS.glob("*.svg")}, "a diagram is not shown in any document"
