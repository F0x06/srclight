"""Tests for embedding model resolution.

The model can come from three places, in decreasing priority: the --embed
flag, the model recorded in the index, and SRCLIGHT_EMBED_MODEL. The middle
one is what keeps embeddings alive across the flag-less reindexes run by the
git hooks and the MCP server; the variable comes last so that exporting it
never switches an index that already embeds.
"""

import pytest
from click.testing import CliRunner

from srclight.cli import main
from srclight.db import Database, FileRecord, SymbolRecord
from srclight.embeddings import vector_to_bytes
from srclight.indexer import IndexConfig, Indexer, resolve_embed_model

from .test_workspace import ws_dir  # noqa: F401  (fixture re-export)


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
    db.remember_embedding_model("qwen3-embedding")

    config = IndexConfig(embed_model="embed-v4.0")

    assert resolve_embed_model(db, config) == "embed-v4.0"


def test_env_var_is_used_when_no_model_is_given(db, monkeypatch):
    monkeypatch.setenv("SRCLIGHT_EMBED_MODEL", "voyage-code-3")

    assert resolve_embed_model(db, IndexConfig()) == "voyage-code-3"


def test_the_stored_model_wins_over_the_env_var(db, monkeypatch):
    """The variable is a default for indexes that have none, not an override.

    Overriding would mean exporting it once — as the README suggests — then
    having the next commit in an unrelated repo silently re-embed every
    symbol it holds, from a detached background hook. Switching an existing
    index stays an explicit --embed.
    """
    monkeypatch.setenv("SRCLIGHT_EMBED_MODEL", "voyage-code-3")
    _seed_embeddings(db, "qwen3-embedding")
    db.remember_embedding_model("qwen3-embedding")

    assert resolve_embed_model(db, IndexConfig()) == "qwen3-embedding"


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
        assert db.detect_embedding_model() == "stub:stub-model"
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
            # Real providers qualify the name they report (OllamaProvider
            # turns "qwen3-embedding" into "ollama:qwen3-embedding"), and it
            # is that name which gets recorded and later re-resolved. A stub
            # echoing the spec back would hide every naming regression.
            self.name = spec if ":" in spec else f"stub:{spec}"
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


def test_switching_the_model_updates_the_record(repo, stub_provider, monkeypatch):
    """A successful switch must overwrite the record, not leave the old one.

    With the record write ignoring conflicts, `--embed model-b` on a model-a
    index succeeds, the record stays model-a, and every flag-less hook run
    after it reverts — re-embedding the new rows back and paying twice.
    """
    monkeypatch.delenv("SRCLIGHT_EMBED_MODEL", raising=False)
    assert CliRunner().invoke(main, ["index", str(repo), "--embed", "model-a"]).exit_code == 0
    assert CliRunner().invoke(main, ["index", str(repo), "--embed", "model-b"]).exit_code == 0

    db = Database(repo / ".srclight" / "index.db")
    db.open()
    try:
        assert db.detect_embedding_model() == "stub:model-b"
    finally:
        db.close()


def test_the_pin_survives_the_index_losing_its_embeddings_mid_run(repo, embed_calls, monkeypatch):
    """The CLI announces a model before the file pass, and must honour it.

    Resolving a second time inside the indexer can disagree: a checkout that
    drops every embedded file cascade-deletes its embeddings during the pass,
    so the later resolve finds nothing and silently skips embedding — after
    the CLI has already printed the model it was going to use.
    """
    monkeypatch.delenv("SRCLIGHT_EMBED_MODEL", raising=False)
    embedded_file = repo / "main.py"

    db = Database(repo / ".srclight" / "index.db")
    db.open()
    sym_id = db.conn.execute("SELECT id FROM symbols LIMIT 1").fetchone()["id"]
    db.upsert_embedding(sym_id, "qwen3-embedding", 3, vector_to_bytes([0.1, 0.2, 0.3]))
    db.commit()
    db.close()

    # The one embedded file disappears: its symbols, and their embeddings,
    # go with it during the run that follows.
    embedded_file.unlink()
    (repo / "replacement.py").write_text("def replacement():\n    return 2\n")

    result = CliRunner().invoke(main, ["index", str(repo)])

    assert result.exit_code == 0, result.output
    assert "Embedding model: qwen3-embedding" in result.output
    assert embed_calls == ["qwen3-embedding"], "announced a model, then embedded with none"


def test_index_reports_the_model_it_actually_uses(repo, embed_calls, monkeypatch):
    """The message and the call must not be able to drift apart."""
    monkeypatch.delenv("SRCLIGHT_EMBED_MODEL", raising=False)
    db = Database(repo / ".srclight" / "index.db")
    db.open()
    db.remember_embedding_model("qwen3-embedding")
    db.commit()
    db.close()

    result = CliRunner().invoke(main, ["index", str(repo)])

    assert result.exit_code == 0, result.output
    assert embed_calls == ["qwen3-embedding"]
    assert "Embedding model: qwen3-embedding (from the existing index)" in result.output


def test_index_reads_the_model_from_the_environment(repo, embed_calls, monkeypatch):
    monkeypatch.setenv("SRCLIGHT_EMBED_MODEL", "voyage-code-3")

    result = CliRunner().invoke(main, ["index", str(repo)])

    assert result.exit_code == 0, result.output
    assert embed_calls == ["voyage-code-3"]


def test_no_embed_skips_the_stored_model(repo, embed_calls, monkeypatch):
    monkeypatch.setenv("SRCLIGHT_EMBED_MODEL", "voyage-code-3")
    db = Database(repo / ".srclight" / "index.db")
    db.open()
    db.remember_embedding_model("qwen3-embedding")
    db.commit()
    db.close()

    result = CliRunner().invoke(main, ["index", str(repo), "--no-embed"])

    assert result.exit_code == 0, result.output
    assert embed_calls == []


def test_forgetting_the_model_stops_later_runs_from_embedding(repo, stub_provider, monkeypatch):
    """Sticky needs an exit, and the hooks' command line is fixed.

    The hooks run a bare `srclight index .`, so --no-embed cannot reach them.
    Without a way to forget, someone who tried `--embed voyage-code-3` once
    keeps calling a metered API on every commit and every branch switch, with
    no supported way out but hand-editing SQLite.
    """
    monkeypatch.delenv("SRCLIGHT_EMBED_MODEL", raising=False)
    assert CliRunner().invoke(main, ["index", str(repo), "--embed", "stub-model"]).exit_code == 0

    result = CliRunner().invoke(main, ["index", str(repo), "--forget-embed-model"])
    assert result.exit_code == 0, result.output

    db = Database(repo / ".srclight" / "index.db")
    db.open()
    try:
        # The rows are still there; they must not resurrect the choice.
        assert db.embedding_stats()["embedded_symbols"] > 0
        assert db.detect_embedding_model() is None
    finally:
        db.close()


def test_a_skipped_embedding_pass_marks_the_sidecar_stale(repo, stub_provider):
    """--no-embed still deletes and re-creates symbols; ids get reused.

    symbol_embeddings cascades on symbol deletion and the cache version is
    bumped only by upsert_embedding, so a --no-embed run that changed symbols
    left a sidecar that still looked valid while describing the previous
    index — and semantic_search served rows for reused ids, i.e. one symbol's
    score attached to another symbol's identity.
    """
    assert CliRunner().invoke(main, ["index", str(repo), "--embed", "stub-model"]).exit_code == 0

    db_path = repo / ".srclight" / "index.db"
    db = Database(db_path)
    db.open()
    before = db.conn.execute(
        "SELECT value FROM schema_info WHERE key='embedding_cache_version'"
    ).fetchone()["value"]
    db.close()

    (repo / "added.py").write_text("def added():\n    return 3\n")
    result = CliRunner().invoke(main, ["index", str(repo), "--no-embed"])
    assert result.exit_code == 0, result.output

    db = Database(db_path)
    db.open()
    try:
        after = db.conn.execute(
            "SELECT value FROM schema_info WHERE key='embedding_cache_version'"
        ).fetchone()["value"]
        assert after != before, "sidecar still looks valid for a changed index"
    finally:
        db.close()


# --- an embedding pass must never cost the index ---


def test_an_unreachable_provider_does_not_lose_the_index(tmp_path, monkeypatch):
    """The whole run used to roll back when the provider was down.

    embed_symbols swallows per-batch failures and returns [], then
    provider.dimensions re-probes the provider and raised out of index(),
    past the single commit that makes the file pass durable. Every
    post-commit hook on a machine with Ollama stopped indexed nothing.
    """
    from srclight import embeddings as embeddings_mod

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("def hello():\n    return 1\n")

    class _DeadProvider:
        name = "ollama:qwen3-embedding"

        @property
        def dimensions(self):
            raise ConnectionError("Cannot reach Ollama at http://localhost:11434")

    def dead_embed_symbols(provider, symbols, on_progress=None):
        return []  # every batch failed, swallowed inside embed_symbols

    monkeypatch.setattr(embeddings_mod, "get_provider", lambda spec, **kw: _DeadProvider())
    monkeypatch.setattr(embeddings_mod, "embed_symbols", dead_embed_symbols)
    monkeypatch.setenv("SRCLIGHT_EMBED_MODEL", "qwen3-embedding")

    result = CliRunner().invoke(main, ["index", str(repo)])
    assert result.exit_code == 0, result.output

    db = Database(repo / ".srclight" / "index.db")
    db.open()
    try:
        assert db.stats()["files"] == 1, "the file pass was rolled back by the embedding failure"
        assert db.get_index_state(str(repo.resolve())) is not None
    finally:
        db.close()


def test_the_file_pass_is_durable_before_embedding_starts(repo, stub_provider, monkeypatch):
    """Embedding can take minutes; it must not hold the write lock.

    index() committed once at the very end, so the implicit transaction
    opened by the first file upsert stayed open across every HTTP call. A
    second writer — the git hook firing while an MCP reindex embeds — hit
    'database is locked' and lost its own run.
    """
    from srclight import embeddings as embeddings_mod

    seen = {}
    (repo / "later.py").write_text("def later():\n    return 2\n")

    def observing_embed_symbols(provider, symbols, on_progress=None):
        other = Database(repo / ".srclight" / "index.db")
        other.open()
        try:
            seen["files"] = other.stats()["files"]
        finally:
            other.close()
        return [(s["id"], vector_to_bytes([0.1, 0.2, 0.3])) for s in symbols]

    monkeypatch.setattr(embeddings_mod, "embed_symbols", observing_embed_symbols)

    result = CliRunner().invoke(main, ["index", str(repo), "--embed", "stub-model"])
    assert result.exit_code == 0, result.output

    assert seen["files"] == 2, "the file pass was still uncommitted while embedding ran"


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
    # test_web.py leaves this set for the rest of the session; reindex must
    # be exercised in single-repo mode whatever ran before us.
    monkeypatch.setattr(server_mod, "_workspace_name", None)

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


def test_the_server_heals_a_sidecar_another_process_could_not_replace(repo, stub_provider):
    """The git hooks embed from a separate process, and cannot fix the sidecar.

    A running server holds embeddings.npy mmap'd for its whole life, and
    Windows refuses os.replace on a mapped file — so the hook's rebuild
    fails, its warning goes to reindex.log, and the sidecar keeps a version
    the bumped embedding_cache_version no longer matches. Nothing else ever
    rebuilds it, so semantic_search fell back to a full SQLite scan forever.
    The server holds the mapping, so the server is who can refresh it.
    """
    from srclight import server as server_mod

    result = CliRunner().invoke(main, ["index", str(repo), "--embed", "stub-model"])
    assert result.exit_code == 0, result.output

    db_path = repo / ".srclight" / "index.db"
    server_mod.configure(db_path=db_path, repo_root=repo)
    try:
        cache = server_mod._get_vector_cache()
        assert cache is not None and cache.is_loaded(), "no sidecar to start from"

        # Another process embeds: rows land in SQLite and the cache version is
        # bumped, but the sidecar on disk stays behind.
        other = Database(db_path)
        other.open()
        try:
            sym_id = other.conn.execute("SELECT id FROM symbols LIMIT 1").fetchone()["id"]
            other.upsert_embedding(sym_id, "stub-model", 3, vector_to_bytes([0.9, 0.9, 0.9]))
            other.commit()
            assert not cache.is_valid(other.conn), "test did not manage to stale the cache"
        finally:
            other.close()

        healed = server_mod._get_vector_cache()

        assert healed is not None
        assert healed.is_valid(server_mod._get_db().conn), "sidecar left stale for the session"
    finally:
        server_mod._close_databases()
        server_mod.configure(db_path=None, repo_root=None)


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


# --- workspace index ---


@pytest.fixture
def workspace(tmp_path, ws_dir):  # noqa: F811
    """A workspace holding one already-indexed project."""
    from srclight.workspace import WorkspaceConfig

    project = tmp_path / "alpha"
    project.mkdir()
    (project / "main.py").write_text("def hello():\n    return 1\n")
    assert CliRunner().invoke(main, ["index", str(project)]).exit_code == 0

    config = WorkspaceConfig(name="embed-ws")
    config.add_project("alpha", str(project))
    config.save()
    return project


def test_workspace_index_reuses_each_project_model(workspace, embed_calls, monkeypatch):
    monkeypatch.delenv("SRCLIGHT_EMBED_MODEL", raising=False)
    db = Database(workspace / ".srclight" / "index.db")
    db.open()
    db.remember_embedding_model("qwen3-embedding")
    db.commit()
    db.close()

    result = CliRunner().invoke(main, ["workspace", "index", "-w", "embed-ws"])

    assert result.exit_code == 0, result.output
    assert embed_calls == ["qwen3-embedding"]


def test_workspace_index_honours_no_embed(workspace, embed_calls, monkeypatch):
    """Skipping was asked for explicitly: on eight projects it is minutes."""
    monkeypatch.delenv("SRCLIGHT_EMBED_MODEL", raising=False)
    db = Database(workspace / ".srclight" / "index.db")
    db.open()
    db.remember_embedding_model("qwen3-embedding")
    db.commit()
    db.close()

    result = CliRunner().invoke(
        main, ["workspace", "index", "-w", "embed-ws", "--embed", "voyage-code-3", "--no-embed"]
    )

    assert result.exit_code == 0, result.output
    assert embed_calls == []
