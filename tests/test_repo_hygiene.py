"""Guards against committed artifacts that would break clones or leak local paths."""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _readme() -> str:
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_language_count_matches_the_registry():
    """The count drifted twice, and a stale one undersells the tool.

    Both the feature bullet and the comparison table quote a number that
    only LANGUAGES knows.
    """
    from srclight.languages import LANGUAGES

    readme = _readme()
    expected = len(LANGUAGES)

    bullet = re.search(r"\*\*(\d+) languages\*\*", readme)
    assert bullet and int(bullet.group(1)) == expected, (
        f"README feature bullet says {bullet.group(1) if bullet else '?'} "
        f"languages, LANGUAGES has {expected}"
    )
    row = re.search(r"^\| Languages \| (\d+) \|", readme, re.MULTILINE)
    assert row and int(row.group(1)) == expected, (
        f"README comparison table says {row.group(1) if row else '?'} "
        f"languages, LANGUAGES has {expected}"
    )


def test_readme_lists_every_registered_tool():
    """A tool absent from the table is a tool nobody knows to call.

    Nine were missing when this was written, which is also how the count
    above it drifted: nothing tied either to the registry.
    """
    from srclight.server import mcp

    readme = _readme()
    start = readme.index("## MCP Tools (")
    section = readme[start:readme.index("\n## ", start + 10)]

    registered = {t.name for t in asyncio.run(mcp.list_tools())}
    listed = set(re.findall(r"`([a-z_]+)\(", section))

    assert not registered - listed, (
        "tools missing from the README table: "
        + ", ".join(sorted(registered - listed))
    )


def test_readme_tool_count_matches_the_registry():
    """Same drift, on the number an agent reads to know what it can call."""
    from srclight.server import mcp

    readme = _readme()
    expected = len(asyncio.run(mcp.list_tools()))

    heading = re.search(r"^## MCP Tools \((\d+)\)", readme, re.MULTILINE)
    assert heading and int(heading.group(1)) == expected, (
        f"README heading says {heading.group(1) if heading else '?'} tools, "
        f"the registry has {expected}"
    )
    prose = re.search(r"exposes (\d+) MCP tools", readme)
    assert prose and int(prose.group(1)) == expected, (
        f"README prose says {prose.group(1) if prose else '?'} tools, "
        f"the registry has {expected}"
    )
    row = re.search(r"^\| MCP tools \| (\d+) \|", readme, re.MULTILINE)
    assert row and int(row.group(1)) == expected, (
        f"README comparison table says {row.group(1) if row else '?'} tools, "
        f"the registry has {expected}"
    )


@pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
    reason="Not a git checkout (e.g. installed from an sdist tarball)",
)
def test_no_tracked_absolute_path_symlinks():
    """No tracked entry may be a symlink whose target is an absolute path.

    An absolute-path symlink dangles on every clone and in every .zip/.tar.gz
    source archive, and typically leaks the author's local directory layout.
    See issue #21 (CLAUDE.md was committed as a symlink to /mnt/c/Users/...).
    """
    result = subprocess.run(
        ["git", "ls-files", "-s"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    offenders = []
    for line in result.stdout.splitlines():
        mode, sha, _stage_and_path = line.split(" ", 2)
        if mode != "120000":
            continue
        path = _stage_and_path.split("\t", 1)[1]
        target = subprocess.run(
            ["git", "cat-file", "-p", sha],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if target.startswith("/") or (len(target) > 1 and target[1] == ":"):
            offenders.append((path, target))

    assert not offenders, (
        "Tracked absolute-path symlinks found (will dangle on every clone):\n"
        + "\n".join(f"  {p} -> {t}" for p, t in offenders)
    )
