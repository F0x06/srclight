"""Tests for git hook install/uninstall."""

import os
import stat
from pathlib import Path

import pytest

from srclight.cli import (
    _HOOK_MARKER_END,
    _HOOK_MARKER_START,
    _install_hooks_in_repo,
    _uninstall_hooks_in_repo,
    _ensure_srclight_ignored,
    _repo_hook_health,
    hook_install,
    hook_status,
)

STALE_BIN = "/home/nobody/Projects/srclight/.venv/bin/srclight"


@pytest.fixture
def fake_repo(tmp_path):
    """Create a fake git repo with .git/hooks dir."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir()
    return tmp_path


def test_install_creates_both_hooks(fake_repo):
    result = _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    assert "OK" in result
    assert "post-commit" in result
    assert "post-checkout" in result

    pc = fake_repo / ".git" / "hooks" / "post-commit"
    pco = fake_repo / ".git" / "hooks" / "post-checkout"
    assert pc.exists()
    assert pco.exists()

    # Both should be executable
    assert pc.stat().st_mode & stat.S_IXUSR
    assert pco.stat().st_mode & stat.S_IXUSR

    # Both should have shebang
    assert pc.read_text().startswith("#!/bin/sh")
    assert pco.read_text().startswith("#!/bin/sh")

    # Both should have markers
    assert _HOOK_MARKER_START in pc.read_text()
    assert _HOOK_MARKER_START in pco.read_text()

    # post-checkout should check $3 for branch checkout
    assert '"$3" = "1"' in pco.read_text()


def test_install_idempotent(fake_repo):
    _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    result = _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    assert "SKIP" in result
    assert "already installed" in result


def test_install_preserves_existing_hooks(fake_repo):
    # Write existing post-commit hook
    pc = fake_repo / ".git" / "hooks" / "post-commit"
    pc.write_text("#!/bin/sh\necho 'existing hook'\n")
    pc.chmod(0o755)

    result = _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    assert "OK" in result

    content = pc.read_text()
    assert "existing hook" in content
    assert _HOOK_MARKER_START in content


def test_uninstall_removes_both_hooks(fake_repo):
    _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    result = _uninstall_hooks_in_repo(fake_repo)
    assert "OK" in result
    assert "post-commit" in result
    assert "post-checkout" in result

    # Files should be gone (no other content)
    assert not (fake_repo / ".git" / "hooks" / "post-commit").exists()
    assert not (fake_repo / ".git" / "hooks" / "post-checkout").exists()


def test_uninstall_preserves_existing_hooks(fake_repo):
    # Write existing post-commit hook
    pc = fake_repo / ".git" / "hooks" / "post-commit"
    pc.write_text("#!/bin/sh\necho 'existing hook'\n")
    pc.chmod(0o755)

    _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    _uninstall_hooks_in_repo(fake_repo)

    # post-commit should still exist with original content
    assert pc.exists()
    content = pc.read_text()
    assert "existing hook" in content
    assert _HOOK_MARKER_START not in content

    # post-checkout had no prior content, should be deleted
    assert not (fake_repo / ".git" / "hooks" / "post-checkout").exists()


def test_uninstall_noop_when_no_hooks(fake_repo):
    result = _uninstall_hooks_in_repo(fake_repo)
    assert "SKIP" in result
    assert "no srclight hooks" in result


def test_skip_non_git_dir(tmp_path):
    result = _install_hooks_in_repo(tmp_path, "/usr/bin/srclight")
    assert "SKIP" in result
    assert "not a git repo" in result


def test_srclight_dir_created(fake_repo):
    _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    assert (fake_repo / ".srclight").is_dir()


def test_post_checkout_only_on_branch_switch(fake_repo):
    """post-checkout hook should guard on $3=1 and $1!=$2."""
    _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    content = (fake_repo / ".git" / "hooks" / "post-checkout").read_text()
    assert '"$3" = "1"' in content
    assert '"$1" != "$2"' in content


def test_hooks_exit_zero(tmp_path, tmp_path_factory):
    """Hooks must exit 0 (git takes post-checkout's status as its own), binary present or not."""
    import subprocess
    for bin_path in (_exe(tmp_path_factory), STALE_BIN):
        repo = _git_repo(tmp_path / f"repo-{abs(hash(bin_path))}")
        _install_hooks_in_repo(repo, bin_path)
        for name, args in (("post-commit", []), ("post-checkout", ["a", "b", "1"]), ("post-checkout", ["a", "a", "0"])):
            for shell in ("sh", "bash"):
                r = subprocess.run([shell, str(repo / ".git" / "hooks" / name), *args],
                                   cwd=repo, capture_output=True, text=True)
                assert r.returncode == 0, (bin_path, name, args, shell, r.stderr)


def test_hooks_use_flock(fake_repo):
    """Hooks must use flock to prevent concurrent reindex processes."""
    _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    for name in ("post-commit", "post-checkout"):
        content = (fake_repo / ".git" / "hooks" / name).read_text()
        assert "flock -n" in content, f"{name} hook missing 'flock -n'"


def test_install_rewrites_block_pointing_at_old_binary(fake_repo):
    """A moved checkout left hooks guarding on a path that no longer exists.

    The block's `[ -x ]` test fails, the hook exits 0, and auto-reindex is
    silently off. Reinstalling has to replace the block, not skip it because
    the marker is present.
    """
    pc = fake_repo / ".git" / "hooks" / "post-commit"
    pc.write_text("#!/bin/sh\necho 'user hook'\n")
    _install_hooks_in_repo(fake_repo, STALE_BIN)

    result = _install_hooks_in_repo(fake_repo, "/usr/bin/srclight")
    assert "OK" in result

    content = pc.read_text()
    assert STALE_BIN not in content
    assert '"/usr/bin/srclight" index .' in content
    assert content.count(_HOOK_MARKER_START) == 1
    assert content.count(_HOOK_MARKER_END) == 1
    assert "user hook" in content


def test_status_flags_block_whose_binary_is_missing(fake_repo, monkeypatch):
    from click.testing import CliRunner

    _install_hooks_in_repo(fake_repo, STALE_BIN)
    monkeypatch.chdir(fake_repo)
    out = CliRunner().invoke(hook_status, []).output
    assert "STALE" in out
    assert STALE_BIN in out


def test_status_ok_when_binary_exists(fake_repo, monkeypatch, tmp_path_factory):
    from click.testing import CliRunner

    bin_path = tmp_path_factory.mktemp("bin") / "srclight"
    bin_path.write_text("#!/bin/sh\n")
    bin_path.chmod(0o755)
    _install_hooks_in_repo(fake_repo, str(bin_path))
    monkeypatch.chdir(fake_repo)
    out = CliRunner().invoke(hook_status, []).output
    assert "post-commit, post-checkout" in out
    assert "STALE" not in out


def _exe(tmp_path_factory, name="srclight"):
    p = tmp_path_factory.mktemp("bin") / name
    p.write_text("#!/bin/sh\n")
    p.chmod(0o755)
    return str(p)


def test_install_keeps_working_block_for_other_binary(fake_repo, tmp_path_factory):
    """Last-writer-wins would let a uvx or frozen-engine install re-point every hook."""
    first = _exe(tmp_path_factory)
    second = _exe(tmp_path_factory)
    _install_hooks_in_repo(fake_repo, first)
    result = _install_hooks_in_repo(fake_repo, second)
    assert "SKIP" in result
    assert first in (fake_repo / ".git" / "hooks" / "post-commit").read_text()


def test_install_force_rewrites_working_block(fake_repo, tmp_path_factory):
    first = _exe(tmp_path_factory)
    second = _exe(tmp_path_factory)
    _install_hooks_in_repo(fake_repo, first)
    result = _install_hooks_in_repo(fake_repo, second, force=True)
    assert "OK" in result
    content = (fake_repo / ".git" / "hooks" / "post-commit").read_text()
    assert second in content and first not in content


def test_cli_install_refuses_non_executable_binary(fake_repo, monkeypatch):
    """The `python -m srclight.cli` fallback can never pass the hook's [ -x ] test."""
    from click.testing import CliRunner
    from srclight import cli

    monkeypatch.setattr(cli, "_srclight_bin", lambda: "/usr/bin/python3 -m srclight.cli")
    monkeypatch.chdir(fake_repo)
    res = CliRunner().invoke(cli.hook_install, [])
    assert res.exit_code == 1
    assert not (fake_repo / ".git" / "hooks" / "post-commit").exists()


def test_status_reads_target_inside_block_only(fake_repo, monkeypatch, tmp_path_factory):
    from click.testing import CliRunner

    pc = fake_repo / ".git" / "hooks" / "post-commit"
    pc.write_text('#!/bin/sh\n[ -x "/nonexistent/user-tool" ] && /nonexistent/user-tool\n')
    _install_hooks_in_repo(fake_repo, _exe(tmp_path_factory))
    monkeypatch.chdir(fake_repo)
    out = CliRunner().invoke(hook_status, []).output
    assert "STALE" not in out


def test_status_flags_missing_end_marker(fake_repo, monkeypatch):
    from click.testing import CliRunner

    pc = fake_repo / ".git" / "hooks" / "post-commit"
    pc.write_text(f"#!/bin/sh\n{_HOOK_MARKER_START}\necho half a block\n")
    monkeypatch.chdir(fake_repo)
    out = CliRunner().invoke(hook_status, []).output
    assert "BROKEN" in out


def _git_repo(tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def test_install_follows_core_hookspath(tmp_path, tmp_path_factory):
    import subprocess
    repo = _git_repo(tmp_path / "repo")
    custom = tmp_path / "custom-hooks"
    custom.mkdir()
    subprocess.run(["git", "-C", str(repo), "config", "core.hooksPath", str(custom)], check=True)
    result = _install_hooks_in_repo(repo, _exe(tmp_path_factory))
    assert "OK" in result
    assert (custom / "post-commit").exists()
    assert not (repo / ".git" / "hooks" / "post-commit").exists()


def test_install_and_status_report_missing_hookspath(tmp_path, monkeypatch, tmp_path_factory):
    """A hooksPath left at a moved directory means git runs no hooks at all."""
    import subprocess
    from click.testing import CliRunner

    repo = _git_repo(tmp_path / "repo")
    subprocess.run(
        ["git", "-C", str(repo), "config", "core.hooksPath", "/nonexistent/Projects/x/.git/hooks"],
        check=True,
    )
    result = _install_hooks_in_repo(repo, _exe(tmp_path_factory))
    assert "FAIL" in result and "core.hooksPath" in result
    monkeypatch.chdir(repo)
    out = CliRunner().invoke(hook_status, []).output
    assert "STALE" in out and "core.hooksPath" in out


def test_hook_logs_missing_binary(tmp_path):
    """The hook must leave a trace in reindex.log when its binary is gone."""
    import subprocess
    repo = _git_repo(tmp_path / "repo")
    (repo / ".srclight").mkdir()
    _install_hooks_in_repo(repo, STALE_BIN)
    r = subprocess.run(["sh", str(repo / ".git" / "hooks" / "post-commit")],
                       cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0
    log = (repo / ".srclight" / "reindex.log").read_text()
    assert STALE_BIN in log and "not executable" in log


def test_hook_does_nothing_under_git_for_windows(tmp_path, tmp_path_factory):
    """Git for Windows runs hooks in WSL clones under /mnt/c; they must not act there."""
    import subprocess
    repo = _git_repo(tmp_path / "repo")
    (repo / ".srclight").mkdir()
    marker = tmp_path / "ran"
    fake_bin = tmp_path_factory.mktemp("bin") / "srclight"
    fake_bin.write_text(f"#!/bin/sh\ntouch {marker}\n")
    fake_bin.chmod(0o755)
    shim = tmp_path_factory.mktemp("shim")
    (shim / "uname").write_text("#!/bin/sh\necho MINGW64_NT-10.0-26200\n")
    (shim / "uname").chmod(0o755)
    _install_hooks_in_repo(repo, str(fake_bin))
    env = {**os.environ, "PATH": f"{shim}:{os.environ['PATH']}"}
    for name, args in (("post-commit", []), ("post-checkout", ["a", "b", "1"])):
        r = subprocess.run(["sh", str(repo / ".git" / "hooks" / name), *args],
                           cwd=repo, env=env, capture_output=True, text=True)
        assert r.returncode == 0
    assert not marker.exists()
    assert not (repo / ".srclight" / "reindex.log").exists()


def test_hook_still_runs_on_linux(tmp_path, tmp_path_factory):
    """The Windows guard must not stop the hook on Linux."""
    import subprocess
    import time
    repo = _git_repo(tmp_path / "repo")
    marker = tmp_path / "ran"
    fake_bin = tmp_path_factory.mktemp("bin") / "srclight"
    fake_bin.write_text(f"#!/bin/sh\ntouch {marker}\n")
    fake_bin.chmod(0o755)
    _install_hooks_in_repo(repo, str(fake_bin))
    r = subprocess.run(["sh", str(repo / ".git" / "hooks" / "post-commit")],
                       cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0
    deadline = time.monotonic() + 10
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.exists()



# --- post-ship review (council 9286392f) ---------------------------------------------


def test_content_after_block_still_runs(tmp_path, tmp_path_factory):
    """No `exit` in the block: another tool's lines appended after it must run."""
    import subprocess
    repo = _git_repo(tmp_path / "repo")
    _install_hooks_in_repo(repo, _exe(tmp_path_factory))
    pc = repo / ".git" / "hooks" / "post-commit"
    pc.write_text(pc.read_text() + "\necho AFTER-BLOCK\n")
    r = subprocess.run(["sh", str(pc)], cwd=repo, capture_output=True, text=True)
    assert "AFTER-BLOCK" in r.stdout


def test_missing_binary_trace_without_srclight_dir(tmp_path):
    """A fresh clone or a worktree has no .srclight/: the trace must still land, and reach stderr."""
    import shutil
    import subprocess
    repo = _git_repo(tmp_path / "repo")
    _install_hooks_in_repo(repo, STALE_BIN)
    shutil.rmtree(repo / ".srclight")
    r = subprocess.run(["sh", str(repo / ".git" / "hooks" / "post-commit")],
                       cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0
    assert STALE_BIN in (repo / ".srclight" / "reindex.log").read_text()
    assert "not executable" in r.stderr


def test_post_checkout_trace_only_on_branch_switch(tmp_path):
    import shutil
    import subprocess
    repo = _git_repo(tmp_path / "repo")
    _install_hooks_in_repo(repo, STALE_BIN)
    shutil.rmtree(repo / ".srclight")
    r = subprocess.run(["sh", str(repo / ".git" / "hooks" / "post-checkout"), "a", "b", "0"],
                       cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0 and r.stderr == ""
    assert not (repo / ".srclight").exists()


def test_status_and_install_handle_non_executable_hook_file(fake_repo, monkeypatch, tmp_path_factory):
    """git skips a hook without +x; status must say so and install must restore it."""
    from click.testing import CliRunner
    bin_path = _exe(tmp_path_factory)
    _install_hooks_in_repo(fake_repo, bin_path)
    pc = fake_repo / ".git" / "hooks" / "post-commit"
    pc.chmod(0o644)
    healthy, summary = _repo_hook_health(fake_repo)
    assert not healthy and "NOT EXECUTABLE" in summary
    monkeypatch.chdir(fake_repo)
    assert CliRunner().invoke(hook_status, []).exit_code == 1
    result = _install_hooks_in_repo(fake_repo, bin_path)
    assert "OK" in result and "post-commit" in result
    assert os.access(pc, os.X_OK)
    assert _repo_hook_health(fake_repo)[0]


def test_outdated_block_upgraded_in_place_keeping_its_binary(fake_repo, tmp_path_factory):
    """A working hook from an older snippet is upgraded without --force, and keeps its binary."""
    first = _exe(tmp_path_factory)
    other = _exe(tmp_path_factory)
    _install_hooks_in_repo(fake_repo, first)
    pc = fake_repo / ".git" / "hooks" / "post-commit"
    pc.write_text(pc.read_text().replace("# Auto-reindex after commit", "# an older snippet's comment"))
    healthy, summary = _repo_hook_health(fake_repo)
    assert not healthy and "OUTDATED" in summary
    result = _install_hooks_in_repo(fake_repo, other)
    assert "OK" in result
    content = pc.read_text()
    assert first in content and other not in content
    assert "an older snippet" not in content
    assert _repo_hook_health(fake_repo)[0]


def test_status_exit_code_tracks_health(fake_repo, monkeypatch, tmp_path_factory):
    from click.testing import CliRunner
    monkeypatch.chdir(fake_repo)
    _install_hooks_in_repo(fake_repo, _exe(tmp_path_factory))
    assert CliRunner().invoke(hook_status, []).exit_code == 0
    (fake_repo / ".git" / "hooks" / "post-checkout").unlink()
    res = CliRunner().invoke(hook_status, [])
    assert res.exit_code == 1 and "MISSING" in res.output


def test_cli_install_exits_1_when_a_repo_fails(tmp_path, monkeypatch, tmp_path_factory):
    import subprocess
    from click.testing import CliRunner
    from srclight import cli
    repo = _git_repo(tmp_path / "repo")
    subprocess.run(["git", "-C", str(repo), "config", "core.hooksPath", "/nonexistent/hooks"], check=True)
    bin_path = _exe(tmp_path_factory)
    monkeypatch.setattr(cli, "_srclight_bin", lambda: bin_path)
    monkeypatch.chdir(repo)
    res = CliRunner().invoke(hook_install, [])
    assert res.exit_code == 1 and "FAIL" in res.output


def test_install_refuses_committable_hookspath(tmp_path, tmp_path_factory):
    """core.hooksPath=.husky would commit hooks naming a local binary path."""
    import subprocess
    repo = _git_repo(tmp_path / "repo")
    (repo / ".husky").mkdir()
    subprocess.run(["git", "-C", str(repo), "config", "core.hooksPath", ".husky"], check=True)
    result = _install_hooks_in_repo(repo, _exe(tmp_path_factory))
    assert "FAIL" in result and "would be committed" in result
    assert not (repo / ".husky" / "post-commit").exists()


def test_install_allows_ignored_hookspath_in_work_tree(tmp_path, tmp_path_factory):
    import subprocess
    repo = _git_repo(tmp_path / "repo")
    (repo / ".hooks").mkdir()
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "exclude").write_text(".hooks/\n")
    subprocess.run(["git", "-C", str(repo), "config", "core.hooksPath", ".hooks"], check=True)
    result = _install_hooks_in_repo(repo, _exe(tmp_path_factory))
    assert "OK" in result
    assert (repo / ".hooks" / "post-commit").exists()


def test_opt_out_is_respected_by_install_and_status(tmp_path, tmp_path_factory):
    """`git config srclight.hooks false` survives the nightly install."""
    import subprocess
    repo = _git_repo(tmp_path / "repo")
    subprocess.run(["git", "-C", str(repo), "config", "srclight.hooks", "false"], check=True)
    result = _install_hooks_in_repo(repo, _exe(tmp_path_factory))
    assert "SKIP" in result and "disabled" in result
    assert not (repo / ".git" / "hooks" / "post-commit").exists()
    healthy, summary = _repo_hook_health(repo)
    assert healthy and "disabled" in summary


def test_install_ignores_srclight_via_info_exclude_not_gitignore(tmp_path, tmp_path_factory):
    """Tracked files are never edited: third-party clones stayed dirty, CRLF files were rewritten."""
    import subprocess
    repo = _git_repo(tmp_path / "repo")
    (repo / ".gitignore").write_bytes(b"build/\r\n")
    _install_hooks_in_repo(repo, _exe(tmp_path_factory))
    assert (repo / ".gitignore").read_bytes() == b"build/\r\n"
    assert ".srclight/" in (repo / ".git" / "info" / "exclude").read_text()
    (repo / ".srclight" / "index.db").write_text("x")
    st = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True).stdout
    assert ".srclight" not in st


def test_ensure_ignored_leaves_exclude_alone_when_already_ignored(tmp_path):
    repo = _git_repo(tmp_path / "repo")
    (repo / ".gitignore").write_text(".srclight/\n")
    _ensure_srclight_ignored(repo)
    exclude = repo / ".git" / "info" / "exclude"
    assert not exclude.exists() or ".srclight/" not in exclude.read_text()


def test_ensure_ignored_is_idempotent(tmp_path):
    repo = _git_repo(tmp_path / "repo")
    _ensure_srclight_ignored(repo)
    _ensure_srclight_ignored(repo)
    lines = (repo / ".git" / "info" / "exclude").read_text().splitlines()
    assert lines.count(".srclight/") == 1


def test_healthz_hook_problems_lists_unhealthy_repos(tmp_path, monkeypatch, tmp_path_factory):
    from types import SimpleNamespace

    from srclight import web
    from srclight import workspace as ws_mod
    good = _git_repo(tmp_path / "good")
    bad = _git_repo(tmp_path / "bad")
    _install_hooks_in_repo(good, _exe(tmp_path_factory))
    _install_hooks_in_repo(bad, STALE_BIN)
    entries = [SimpleNamespace(name="good", path=str(good)), SimpleNamespace(name="bad", path=str(bad))]
    monkeypatch.setattr(ws_mod.WorkspaceConfig, "load",
                        classmethod(lambda cls, name: SimpleNamespace(get_entries=lambda: entries)))
    monkeypatch.setattr(web, "_hook_health_cache", None)
    problems = web._hook_health_problems("ws")
    assert len(problems) == 1 and problems[0].startswith("bad:") and "STALE" in problems[0]
    assert web._hook_health_problems(None) == []
