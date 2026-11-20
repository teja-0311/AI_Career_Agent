import pytest
from pathlib import Path
from app.database.persistence import PersistenceService
from app.services.embeddings import EmbeddingsService


def test_thread_isolation_and_user_ownership(tmp_path):
    db_file = tmp_path / "test_memory.db"
    svc = PersistenceService(db_path=db_file)

    # User 1 registers thread_1
    svc.ensure_user_and_thread(user_id="alice", thread_id="thread_1")

    # User 2 tries to access thread_1 -> must raise PermissionError
    with pytest.raises(PermissionError):
        svc.ensure_user_and_thread(user_id="bob", thread_id="thread_1")


def test_cumulative_saved_jobs(tmp_path):
    db_file = tmp_path / "test_memory.db"
    svc = PersistenceService(db_path=db_file)

    assert svc.save_job_for_user(user_id="alice", job_id="job_001") is True
    assert svc.save_job_for_user(user_id="alice", job_id="job_002") is True
    # Duplicate save is handled cleanly
    assert svc.save_job_for_user(user_id="alice", job_id="job_001") is False

    saved = svc.get_saved_jobs("alice")
    assert "job_001" in saved
    assert "job_002" in saved
    assert len(saved) == 2


def test_user_preferences_persistence(tmp_path):
    db_file = tmp_path / "test_memory.db"
    svc = PersistenceService(db_path=db_file)

    svc.update_preferences(
        user_id="alice",
        target_roles=["Backend Engineer"],
        preferred_locations=["Remote"],
        preferred_work_mode="Remote",
    )

    prefs = svc.get_preferences("alice")
    assert prefs["target_roles"] == ["Backend Engineer"]
    assert prefs["preferred_work_mode"] == "Remote"


def test_embeddings_service_backend_selection():
    mock_svc = EmbeddingsService(backend="mock")
    v = mock_svc.embed_query("Python Developer")
    assert len(v) == 768
    assert isinstance(v[0], float)

    # Incompatible backend should fail explicitly
    with pytest.raises(ValueError, match="Unsupported EMBEDDING_BACKEND"):
        EmbeddingsService(backend="invalid_backend")


def test_vector_store_metadata_incompatibility_detection(tmp_path):
    from app.database.vector_store import VectorStoreService
    # Initialize first collection with dimension 768 and mock model
    store1 = VectorStoreService(
        backend="memory",
        chroma_dir=tmp_path / "vectors",
        collection_name="test_meta_coll",
        embedding_dim=768,
        embedding_backend="mock",
        embedding_model="models/text-embedding-004",
    )
    assert store1.count() == 0

    # 1. Dimension mismatch (768 vs 1536) -> ValueError
    with pytest.raises(ValueError, match="Incompatible vector collection detected"):
        VectorStoreService(
            backend="memory",
            chroma_dir=tmp_path / "vectors",
            collection_name="test_meta_coll",
            embedding_dim=1536,
            embedding_backend="mock",
            embedding_model="models/text-embedding-004",
        )

    # 2. Backend mismatch (mock vs gemini) -> ValueError
    with pytest.raises(ValueError, match="Incompatible vector collection detected"):
        VectorStoreService(
            backend="memory",
            chroma_dir=tmp_path / "vectors",
            collection_name="test_meta_coll",
            embedding_dim=768,
            embedding_backend="gemini",
            embedding_model="models/text-embedding-004",
        )

    # 3. Model mismatch (same dimension 768, but model name differs) -> ValueError
    with pytest.raises(ValueError, match="Incompatible vector collection detected"):
        VectorStoreService(
            backend="memory",
            chroma_dir=tmp_path / "vectors",
            collection_name="test_meta_coll",
            embedding_dim=768,
            embedding_backend="mock",
            embedding_model="models/custom-experimental-model-v2",
        )


def test_langgraph_sqlite_checkpointing_and_thread_deletion(tmp_path):
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict, List

    class TurnState(TypedDict, total=False):
        messages: List[str]

    def add_msg(s: TurnState):
        msgs = list(s.get("messages", []))
        msgs.append("turn")
        return {"messages": msgs}

    db_path = tmp_path / "lg_checkpointer.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()

    builder = StateGraph(TurnState)
    builder.add_node("add_msg", add_msg)
    builder.add_edge(START, "add_msg")
    builder.add_edge("add_msg", END)
    graph = builder.compile(checkpointer=saver)

    # Thread 1 turn 1
    r1 = graph.invoke({"messages": ["initial"]}, config={"configurable": {"thread_id": "thread_A"}})
    assert r1["messages"] == ["initial", "turn"]

    # Thread 1 turn 2 (restores thread_A state)
    r2 = graph.invoke({}, config={"configurable": {"thread_id": "thread_A"}})
    assert r2["messages"] == ["initial", "turn", "turn"]

    # Delete thread_A checkpoint
    saver.delete_thread("thread_A")

    # Thread 1 turn 3 (checkpoint was deleted, so does NOT restore previous state)
    r3 = graph.invoke({"messages": ["fresh_start"]}, config={"configurable": {"thread_id": "thread_A"}})
    assert r3["messages"] == ["fresh_start", "turn"]

    # Thread 2 is completely isolated
    r4 = graph.invoke({"messages": ["new_thread"]}, config={"configurable": {"thread_id": "thread_B"}})
    assert r4["messages"] == ["new_thread", "turn"]


def test_production_fails_on_broken_checkpointer(monkeypatch):
    from app.agents.graph import CareerAgentWorkflow
    from unittest.mock import MagicMock
    from app.config import settings

    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "allow_fallback_executor", False)

    mock_conn = MagicMock()
    mock_conn.execute.side_effect = Exception("Disk I/O permission denied on SQLite")

    with pytest.raises(RuntimeError, match="Production startup failure"):
        CareerAgentWorkflow(nodes=MagicMock(), checkpointer_conn=mock_conn)
