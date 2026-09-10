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


def test_the_recorded_model_beats_the_row_counts(db, monkeypatch):
    """A half-finished switch must not silently revert on the next run.

    embed_symbols returns what it managed to embed when a batch fails, so an
    index can end up mostly old-model, partly new-model. Row counting would
    hand the majority back and re-embed the new rows into the old model,
    paying for the calls twice.
    """
    monkeypatch.delenv("SRCLIGHT_EMBED_MODEL", raising=False)
    _seed_embeddings(db, "qwen3-embedding", "qwen3-embedding", "voyage-code-3")
    db.remember_embedding_model("voyage-code-3")

    assert resolve_embed_model(db, IndexConfig()) == "voyage-code-3"


def test_row_counts_still_answer_for_indexes_built_before_the_record(db, monkeypatch):
    monkeypatch.delenv("SRCLIGHT_EMBED_MODEL", raising=False)
    _seed_embeddings(db, "qwen3-embedding")

    assert db.detect_embedding_model() == "qwen3-embedding"


def test_an_embedding_run_records_the_model_it_used(repo, stub_provider, monkeypatch):
    monkeypatch.delenv("SRCLIGHT_EMBED_MODEL", raising=False)

    result = CliRunner().invoke(main, ["index", str(repo), "--embed", "stub-model"])
    assert result.exit_code == 0, result.output

    db = Database(repo / ".srclight" / "index.db")
    db.open()
    try:
        assert db.detect_embedding_model() == "stub-model"
    finally:
        db.close()


def test_a_failed_run_records_nothing(repo, stub_provider, monkeypatch):
    """A typo'd model must not become the index's remembered choice."""
    monkeypatch.delenv("SRCLIGHT_EMBED_MODEL", raising=False)
    stub_provider["results"] = []  # every batch failed

    result = CliRunner().invoke(main, ["index", str(repo), "--embed", "typo-model"])
    assert result.exit_code == 0, result.output

    db = Database(repo / ".srclight" / "index.db")
    db.open()
    try:
        assert db.detect_embedding_model() is None
    finally:
        db.close()


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
def stub_provider(monkeypatch):
    """Embed for real against a fake provider — no network, no Ollama.

    `results` is what embed_symbols hands back; empty means every batch failed.
    """
    from srclight import embeddings as embeddings_mod

    state: dict = {"results": None}

    class _Stub:
        def __init__(self, spec):
            self.name = spec
            self.dimensions = 3

    def fake_get_provider(model_spec, **kwargs):
        return _Stub(model_spec)

    def fake_embed_symbols(provider, symbols, on_progress=None):
        if state["results"] is not None:
            return state["results"]
        return [(s["id"], vector_to_bytes([0.1, 0.2, 0.3])) for s in symbols]

    monkeypatch.setattr(embeddings_mod, "get_provider", fake_get_provider)
    monkeypatch.setattr(embeddings_mod, "embed_symbols", fake_embed_symbols)
    return state


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


# --- MCP reindex ---


def _call(coro_or_val):
    """Run an async MCP tool the way the rest of the suite does."""
    import asyncio
    return asyncio.run(coro_or_val) if asyncio.iscoroutine(coro_or_val) else coro_or_val


@pytest.fixture
def mcp_repo(repo, monkeypatch):
    """An indexed repo, already embedded once, served over MCP."""
    from srclight import server as server_mod

    # _get_db() walks up from the CWD — never let a test reach the real index.
    monkeypatch.chdir(repo)
    monkeypatch.delenv("SRCLIGHT_EMBED_MODEL", raising=False)

    db_path = repo / ".srclight" / "index.db"
    db = Database(db_path)
    db.open()
    sym_id = db.conn.execute("SELECT id FROM symbols LIMIT 1").fetchone()["id"]
    db.upsert_embedding(sym_id, "qwen3-embedding", 3, vector_to_bytes([0.1, 0.2, 0.3]))
    db.commit()
    db.close()

    server_mod.configure(db_path=db_path, repo_root=repo)
    yield server_mod
    server_mod._close_databases()
    server_mod.configure(db_path=None, repo_root=None)


def test_reindex_embeds_with_the_known_model_by_default(mcp_repo, embed_calls):
    _call(mcp_repo.reindex())

    assert embed_calls == ["qwen3-embedding"]


def test_reindex_skips_embeddings_when_the_agent_asks(mcp_repo, embed_calls):
    _call(mcp_repo.reindex(embed=False))

    assert embed_calls == []


def test_reindex_releases_the_vector_cache_before_indexing(mcp_repo, embed_calls, monkeypatch):
    """The embedding pass rewrites embeddings.npy — which the server may hold mmap'd.

    `os.replace` over a live mapping fails on Windows, _build_embeddings
    swallows it, and the sidecar then keeps a version the bumped
    embedding_cache_version no longer matches: every semantic_search falls
    back to a full SQLite scan for the life of the server. Releasing the
    cache after indexing, as reindex used to, is too late.
    """
    seen = {}
    real_index = Indexer.index

    def spy(self, root, **kwargs):
        seen["cache"] = mcp_repo._vector_cache
        return real_index(self, root, **kwargs)

    monkeypatch.setattr(Indexer, "index", spy)
    mcp_repo._vector_cache = object()  # stand-in for a mapped sidecar

    _call(mcp_repo.reindex())

    assert seen["cache"] is None, "sidecar still mapped while the embedding pass rewrote it"
