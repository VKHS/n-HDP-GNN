"""Relative links in the documents point to files that exist, and links to a heading point to a heading that exists."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = sorted({ROOT / "README.md", ROOT / "CHANGELOG.md", *(ROOT / "docs").glob("*.md"),
                    *(ROOT / "examples").glob("*.md"), ROOT / "examples" / "output" / "demo.md", ROOT / "results" / "README.md",
                    ROOT / "artifacts" / "README.md", ROOT / "legacy" / "README.md"})
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)\)|!\[[^\]]*\]\(([^)\s]+)\)")


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"`", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(path: Path) -> set:
    found, in_code = set(), False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        elif not in_code and (m := re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line)):
            found.add(slug(m.group(1)))
    return found


def links(path: Path):
    in_code = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        line = re.sub(r"`[^`]*`", "", line)                     # a link inside inline code is not a link
        for a, b in LINK.findall(line):
            yield a or b


@pytest.mark.parametrize("doc", [d for d in DOCUMENTS if d.exists()], ids=lambda d: str(d.relative_to(ROOT)))
def test_relative_links_resolve(doc):
    broken = []
    for target in links(doc):
        if re.match(r"[a-z][a-z0-9+.-]*:", target):               # http:, https:, mailto:
            continue
        name, _, fragment = target.partition("#")
        path = (doc.parent / name).resolve() if name else doc
        if not path.exists():
            broken.append(f"{target}: no such file")
        elif fragment and path.suffix == ".md" and slug(fragment) not in anchors(path):
            broken.append(f"{target}: no such heading")
    assert not broken, f"{doc.relative_to(ROOT)}: " + "; ".join(broken)
