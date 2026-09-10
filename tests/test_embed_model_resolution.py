"""Tests for embedding model resolution.

The model can come from three places, in decreasing priority: the --embed
flag, the SRCLIGHT_EMBED_MODEL environment variable, and the model already
stored in the index. The last one is what keeps embeddings alive across the
flag-less reindexes run by the git hooks and the MCP server.
"""

import pytest
from click.testing import CliRunner

from srclight.cli import main
from srclight.db import Database, FileRecord, SymbolRecord
from srclight.embeddings import vector_to_bytes
from srclight.indexer import IndexConfig, Indexer, resolve_embed_model


@pytest.fixture
def db(tmp_path):
    db = Database(tmp_path / "index.db")
    db.open()
    db.initialize()
    yield db
    db.close()


def _seed_embeddings(db: Database, *models: str) -> None:
    """Insert one embedded symbol per given model (repeats allowed)."""
    file_id = db.upsert_file(FileRecord(
        path="src/main.py", content_hash="abc", mtime=1.0,
        language="python", size=10, line_count=2,
    ))
    for i, model in enumerate(models):
        sym_id = db.insert_symbol(SymbolRecord(
            file_id=file_id, kind="function", name=f"f{i}",
            start_line=i + 1, end_line=i + 1,
            content=f"def f{i}(): pass", line_count=1,
        ), "src/main.py")
        db.upsert_embedding(sym_id, model, 3, vector_to_bytes([0.1, 0.2, 0.3]))
    db.commit()


# --- resolve_embed_model ---


def test_explicit_model_wins_over_env_and_index(db, monkeypatch):
    monkeypatch.setenv("SRCLIGHT_EMBED_MODEL", "voyage-code-3")
    _seed_embeddings(db, "qwen3-embedding")

    config = IndexConfig(embed_model="embed-v4.0")

    assert resolve_embed_model(db, config) == "embed-v4.0"


def test_env_var_is_used_when_no_model_is_given(db, monkeypatch):
    monkeypatch.setenv("SRCLIGHT_EMBED_MODEL", "voyage-code-3")

    assert resolve_embed_model(db, IndexConfig()) == "voyage-code-3"


def test_env_var_wins_over_the_model_stored_in_the_index(db, monkeypatch):
    monkeypatch.setenv("SRCLIGHT_EMBED_MODEL", "voyage-code-3")
    _seed_embeddings(db, "qwen3-embedding")

    assert resolve_embed_model(db, IndexConfig()) == "voyage-code-3"


def test_model_stored_in_the_index_is_reused(db, monkeypatch):
    monkeypatch.delenv("SRCLIGHT_EMBED_MODEL", raising=False)
    _seed_embeddings(db, "qwen3-embedding")

    assert resolve_embed_model(db, IndexConfig()) == "qwen3-embedding"


def test_the_dominant_model_wins_when_the_index_holds_several(db, monkeypatch):
    monkeypatch.delenv("SRCLIGHT_EMBED_MODEL", raising=False)
    _seed_embeddings(db, "voyage-code-3", "qwen3-embedding", "qwen3-embedding")

    assert resolve_embed_model(db, IndexConfig()) == "qwen3-embedding"


def test_a_fresh_index_resolves_to_no_model(db, monkeypatch):
    monkeypatch.delenv("SRCLIGHT_EMBED_MODEL", raising=False)

    assert resolve_embed_model(db, IndexConfig()) is None


def test_blank_values_are_ignored(db, monkeypatch):
    monkeypatch.setenv("SRCLIGHT_EMBED_MODEL", "   ")
    _seed_embeddings(db, "qwen3-embedding")

    assert resolve_embed_model(db, IndexConfig(embed_model="  ")) == "qwen3-embedding"


def test_disable_embeddings_overrides_every_source(db, monkeypatch):
    monkeypatch.setenv("SRCLIGHT_EMBED_MODEL", "voyage-code-3")
    _seed_embeddings(db, "qwen3-embedding")

    config = IndexConfig(embed_model="embed-v4.0", disable_embeddings=True)

    assert resolve_embed_model(db, config) is None


# --- CLI wiring ---


@pytest.fixture
def repo(tmp_path):
    """A one-file repository, already indexed once."""
    (tmp_path / "main.py").write_text("def hello():\n    return 1\n")
    result = CliRunner().invoke(main, ["index", str(tmp_path)])
    assert result.exit_code == 0, result.output
    return tmp_path


@pytest.fixture
def embed_calls(monkeypatch):
    """Record the model each index run hands to the embedding builder."""
    calls: list[str] = []

    def fake_build(self, model_spec):
        calls.append(model_spec)
        return 0

    monkeypatch.setattr(Indexer, "_build_embeddings", fake_build)
    return calls


def test_index_reuses_the_model_stored_in_the_index(repo, embed_calls, monkeypatch):
    monkeypatch.delenv("SRCLIGHT_EMBED_MODEL", raising=False)
    db = Database(repo / ".srclight" / "index.db")
    db.open()
    sym_id = db.conn.execute("SELECT id FROM symbols LIMIT 1").fetchone()["id"]
    db.upsert_embedding(sym_id, "qwen3-embedding", 3, vector_to_bytes([0.1, 0.2, 0.3]))
    db.commit()
    db.close()

    result = CliRunner().invoke(main, ["index", str(repo)])

    assert result.exit_code == 0, result.output
    assert embed_calls == ["qwen3-embedding"]


def test_index_reads_the_model_from_the_environment(repo, embed_calls, monkeypatch):
    monkeypatch.setenv("SRCLIGHT_EMBED_MODEL", "voyage-code-3")

    result = CliRunner().invoke(main, ["index", str(repo)])

    assert result.exit_code == 0, result.output
    assert embed_calls == ["voyage-code-3"]


def test_no_embed_skips_the_stored_model(repo, embed_calls, monkeypatch):
    monkeypatch.setenv("SRCLIGHT_EMBED_MODEL", "voyage-code-3")

    result = CliRunner().invoke(main, ["index", str(repo), "--no-embed"])

    assert result.exit_code == 0, result.output
    assert embed_calls == []
