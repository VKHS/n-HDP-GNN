"""Render docs/diagrams/*.mmd to SVG with the Mermaid CLI and keep a manifest of what was rendered.

    python scripts/make_diagrams.py              # render the diagrams whose source or settings changed
    python scripts/make_diagrams.py --force      # render every diagram
    python scripts/make_diagrams.py --check      # render nothing; exit 1 if an SVG is missing or does not match its source

The Mermaid CLI (``npm install -g @mermaid-js/mermaid-cli``) needs a headless Chromium; on a machine that
runs as root, or in a container, pass ``--puppeteer-config`` with a file holding
``{"args": ["--no-sandbox"]}``.

``docs/diagrams/manifest.json`` records the SHA-256 of every source, of ``mermaid-config.json`` and of every
SVG at the time it was rendered.  The test suite runs ``--check``, so a diagram that was edited without being
rendered again, or an SVG that was edited by hand, fails it.  Hashes are taken over the bytes of the files:
keep them with LF line endings.

The SVGs are drawn with text elements (``htmlLabels`` is off in the settings) in Arial or a metric-compatible
font, so that their boxes fit the text in a browser, in a README preview and in an image viewer alike.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS = ROOT / "docs" / "diagrams"
CONFIG = DIAGRAMS / "mermaid-config.json"
MANIFEST = DIAGRAMS / "manifest.json"
BACKGROUND = "#fcfcfb"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources() -> list[Path]:
    return sorted(DIAGRAMS.glob("*.mmd"))


def read_manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text())
    except (OSError, ValueError):
        return {}


def problems() -> list[str]:
    """What is missing or out of date, as sentences; an empty list means every SVG matches its source."""
    manifest = read_manifest()
    if not manifest:
        return ["docs/diagrams/manifest.json is missing or unreadable; run scripts/make_diagrams.py"]
    out = []
    if manifest.get("config_sha256") != sha256(CONFIG):
        out.append("mermaid-config.json changed since the SVGs were rendered")
    entries = manifest.get("diagrams", {})
    for src in sources():
        name, svg = src.stem, src.with_suffix(".svg")
        entry = entries.get(name)
        if entry is None:
            out.append(f"{src.name} has not been rendered")
        elif entry.get("source_sha256") != sha256(src):
            out.append(f"{src.name} changed since {svg.name} was rendered")
        if not svg.exists():
            out.append(f"{svg.name} is missing")
        elif entry is not None and entry.get("svg_sha256") != sha256(svg):
            out.append(f"{svg.name} does not match the manifest (edited by hand?)")
    for name in entries:
        if not (DIAGRAMS / f"{name}.mmd").exists():
            out.append(f"the manifest lists {name}, which has no .mmd source")
    return out


def render(src: Path, dst: Path, mmdc: str, puppeteer_config: str | None) -> None:
    cmd = [mmdc, "-i", str(src), "-o", str(dst), "-c", str(CONFIG), "-b", BACKGROUND, "--quiet"]
    if puppeteer_config:
        cmd += ["-p", puppeteer_config]
    subprocess.run(cmd, check=True, timeout=240, cwd=ROOT)
    svg = dst.read_text(encoding="utf-8")
    if not svg.lstrip().startswith(("<svg", "<?xml")) or "viewBox" not in svg:
        raise RuntimeError(f"{dst.name} is not an SVG with a viewBox")
    if "<foreignObject" in svg:
        raise RuntimeError(f"{dst.name} contains foreignObject labels; set htmlLabels to false in {CONFIG.name}")


def mermaid_version(mmdc: str) -> str:
    try:
        return subprocess.run([mmdc, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="render nothing; fail if an SVG is stale or missing")
    ap.add_argument("--force", action="store_true", help="render every diagram")
    ap.add_argument("--puppeteer-config", default=os.environ.get("PUPPETEER_CONFIG"),
                    help="JSON file with the Chromium launch arguments (e.g. --no-sandbox)")
    args = ap.parse_args(argv)

    if args.check:
        found = problems()
        for line in found:
            print(line)
        print("diagrams: " + ("up to date" if not found else f"{len(found)} problem(s); run python scripts/make_diagrams.py"))
        return 1 if found else 0

    mmdc = shutil.which("mmdc")
    if mmdc is None:
        print("the Mermaid CLI (mmdc) is not installed: npm install -g @mermaid-js/mermaid-cli", file=sys.stderr)
        return 2
    manifest = read_manifest()
    stale_config = manifest.get("config_sha256") != sha256(CONFIG)
    entries = {} if stale_config or args.force else dict(manifest.get("diagrams", {}))
    todo = [s for s in sources() if args.force or stale_config or s.stem not in entries
            or entries[s.stem].get("source_sha256") != sha256(s) or not s.with_suffix(".svg").exists()]
    for src in todo:
        dst = src.with_suffix(".svg")
        print(f"rendering {src.name} -> {dst.name}")
        render(src, dst, mmdc, args.puppeteer_config)
        entries[src.stem] = {"source_sha256": sha256(src), "svg_sha256": sha256(dst)}
    entries = {k: v for k, v in entries.items() if (DIAGRAMS / f"{k}.mmd").exists()}
    manifest = {"mermaid_cli": mermaid_version(mmdc) if todo else manifest.get("mermaid_cli", ""),
                "config_sha256": sha256(CONFIG), "diagrams": dict(sorted(entries.items()))}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(todo)} rendered, {len(entries) - len(todo)} unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
