import pytest
from pathlib import Path
from app.schemas.job import JobPosting, ExperienceLevel, WorkMode
from app.services.embeddings import EmbeddingsService
from app.database.vector_store import VectorStoreService
from app.services.job_ingestion import JobIngestionService
from app.services.retrieval import RetrievalService


def test_job_ingestion_and_deduplication(tmp_path):
    store = VectorStoreService(chroma_dir=tmp_path / "chroma", collection_name="test_ingest")
    embeddings = EmbeddingsService()
    ingestion = JobIngestionService(store, embeddings)

    jobs = [
        JobPosting(
            job_id="job_dup",
            title="AI Engineer",
            company="TestCo",
            description="LLM and RAG work.",
            required_skills=["Python"],
        ),
        JobPosting(
            job_id="job_dup",  # Duplicate ID
            title="AI Engineer",
            company="TestCo",
            description="LLM and RAG work duplicate.",
            required_skills=["Python"],
        ),
        JobPosting(
            job_id="job_unique",
            title="Frontend Dev",
            company="TestCo",
            description="React work.",
            required_skills=["React"],
        ),
    ]

    res = ingestion.ingest_jobs(jobs)
    assert res.total_received == 3
    assert res.total_indexed == 2
    assert res.duplicates_skipped == 1
    assert store.count() == 2


def test_empty_vector_index(tmp_path):
    store = VectorStoreService(chroma_dir=tmp_path / "chroma_empty", collection_name="test_empty")
    embeddings = EmbeddingsService()
    retrieval = RetrievalService(store, embeddings)

    results = retrieval.retrieve_by_query_text("Data Scientist", top_k=5)
    assert results == []
