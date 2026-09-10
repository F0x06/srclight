"""Suite-wide fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _no_ambient_embed_model(monkeypatch):
    """Keep the suite off the network.

    SRCLIGHT_EMBED_MODEL makes an index with no recorded model build
    embeddings, and the README tells users to export it. Around twenty tests
    run a real Indexer against a fresh index, so without this the suite would
    reach for Ollama on a developer machine that followed the docs. Tests
    that want the variable set it themselves.
    """
    monkeypatch.delenv("SRCLIGHT_EMBED_MODEL", raising=False)
