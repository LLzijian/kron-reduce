from __future__ import annotations

import json
import subprocess

import pytest

from project_harness import __main__ as cli
from project_harness.git_state import archive_attempt


def _git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, check=True, encoding="utf-8",
    ).stdout.strip()


@pytest.fixture
def git_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "archive-test@example.invalid")
    _git(root, "config", "user.name", "Archive Test")
    _git(root, "config", "commit.gpgSign", "false")
    _git(root, "config", "tag.gpgSign", "false")
    _git(root, "commit", "--allow-empty", "-qm", "Main baseline")
    return root


def test_archive_keeps_rejected_tip_after_branch_deletion(git_repo):
    branch = "codex/rejected-candidate"
    _git(git_repo, "switch", "-c", branch)
    _git(git_repo, "commit", "--allow-empty", "-qm", "Record rejected candidate")
    commit = _git(git_repo, "rev-parse", "HEAD")
    message = "候选方案未通过验证，见归档报告。"

    result = archive_attempt(git_repo, "rejected", message)
    tag = f"attempt/{branch}/{commit}"
    assert result == {
        "tag": tag, "branch": branch, "commit": commit, "outcome": "rejected", "pushed": False,
    }
    _git(git_repo, "switch", "main")
    _git(git_repo, "branch", "-D", branch)

    ref = f"refs/tags/{tag}"
    assert not _git(git_repo, "branch", "--list", branch)
    assert _git(git_repo, "cat-file", "-t", ref) == "tag"
    assert _git(git_repo, "rev-parse", f"{ref}^{{commit}}") == commit
    annotation = _git(git_repo, "cat-file", "-p", ref).partition("\n\n")[2]
    assert all(value in annotation for value in (branch, "rejected", message))


def test_archive_retry_publishes_only_its_tag(git_repo, tmp_path, monkeypatch, capsys):
    remote = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-q", str(remote))
    _git(git_repo, "remote", "add", "origin", str(remote))
    _git(git_repo, "config", "push.followTags", "true")
    _git(git_repo, "tag", "-a", "unrelated", "-m", "Unrelated annotation")
    _git(git_repo, "switch", "-c", "codex/failed")
    outcome, message = "failed", "Runner exited before collecting results."
    first = archive_attempt(git_repo, outcome, message)
    ref = f"refs/tags/{first['tag']}"
    tag_object = _git(git_repo, "rev-parse", ref)
    assert not _git(remote, "for-each-ref", "--format=%(refname)")

    monkeypatch.setattr(cli, "project_root", lambda: git_repo)
    cli.main(["git-archive", "--outcome", outcome, "--message", message, "--push"])
    assert json.loads(capsys.readouterr().out) == {**first, "pushed": True}
    assert _git(remote, "for-each-ref", "--format=%(refname)") == ref
    assert _git(remote, "rev-parse", ref) == tag_object

    with pytest.raises(RuntimeError):
        archive_attempt(git_repo, "completed", "Different outcome.")
    assert _git(git_repo, "rev-parse", ref) == tag_object


def test_archive_refuses_main_and_dirty_attempt(git_repo):
    with pytest.raises(RuntimeError):
        archive_attempt(git_repo, "abandoned", "Main is not an attempt branch.")
    _git(git_repo, "switch", "-c", "codex/uncommitted")
    _git(git_repo, "config", "status.showUntrackedFiles", "no")
    (git_repo / "notes.txt").write_text("Uncommitted notes", encoding="utf-8")
    with pytest.raises(RuntimeError):
        archive_attempt(git_repo, "abandoned", "Uncommitted attempt.")
    assert not _git(git_repo, "tag", "--list")
