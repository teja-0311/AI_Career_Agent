"""Global PyTest configuration and per-test isolation fixtures."""

import uuid
import pytest
from pathlib import Path
from app.config import settings
from app.api import agent, jobs
from app.database.persistence import PersistenceService
from app.database.vector_store import VectorStoreService
from app.agents.nodes import CareerAgentNodes
from app.agents.graph import CareerAgentWorkflow
from app.agents.tools import CareerAgentTools
from app.services.retrieval import RetrievalService
from app.services.ranking import RankingService
from app.services.job_ingestion import JobIngestionService


@pytest.fixture(autouse=True)
def isolate_test_environment(tmp_path, monkeypatch):
    """Ensure every test runs against a brand new temporary database and vector store."""
    unique_id = uuid.uuid4().hex[:8]
    test_db = tmp_path / f"memory_{unique_id}.db"
    test_chroma = tmp_path / f"chroma_{unique_id}"

    monkeypatch.setattr(settings, "sqlite_db_path", test_db)
    monkeypatch.setattr(settings, "chroma_dir", test_chroma)
    monkeypatch.setattr(settings, "vector_store_backend", "memory")
    monkeypatch.setattr(settings, "embedding_backend", "mock")
    monkeypatch.setattr(settings, "resume_parser_backend", "rule_based")

    # Reinitialize singletons for API modules with isolated databases
    fresh_persistence = PersistenceService(db_path=test_db)
    fresh_store = VectorStoreService(backend="memory", chroma_dir=test_chroma, collection_name=f"test_coll_{unique_id}")
    fresh_tools = CareerAgentTools(
        vector_store=fresh_store,
        embeddings_service=agent.embeddings_svc,
        persistence_service=fresh_persistence,
    )
    fresh_retrieval = RetrievalService(fresh_store, agent.embeddings_svc)
    fresh_ranking = RankingService()
    fresh_nodes = CareerAgentNodes(
        resume_parser=agent.resume_parser,
        retrieval_service=fresh_retrieval,
        ranking_service=fresh_ranking,
        persistence_service=fresh_persistence,
        tools=fresh_tools,
        llm_client=agent.llm_client,
    )
    fresh_workflow = CareerAgentWorkflow(fresh_nodes)

    # Patch API modules
    monkeypatch.setattr(agent, "persistence_svc", fresh_persistence)
    monkeypatch.setattr(agent, "vector_store", fresh_store)
    monkeypatch.setattr(agent, "retrieval_svc", fresh_retrieval)
    monkeypatch.setattr(agent, "ranking_svc", fresh_ranking)
    monkeypatch.setattr(agent, "agent_tools", fresh_tools)
    monkeypatch.setattr(agent, "agent_nodes", fresh_nodes)
    monkeypatch.setattr(agent, "agent_workflow", fresh_workflow)

    monkeypatch.setattr(jobs, "persistence_service", fresh_persistence)
    monkeypatch.setattr(jobs, "vector_store", fresh_store)
    monkeypatch.setattr(jobs, "retrieval_service", fresh_retrieval)
    monkeypatch.setattr(jobs, "ranking_service", fresh_ranking)
    monkeypatch.setattr(jobs, "ingestion_service", JobIngestionService(fresh_store, jobs.embeddings_service))

    yield
