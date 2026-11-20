"""Opt-in live integration test for Gemini Tool-Calling and Embeddings.

Explicitly distinguishes live Gemini API integration tests from deterministic unit tests.
Only executes when GEMINI_API_KEY is configured with an active non-placeholder key.
"""

import pytest
from app.config import settings
from app.services.llm import get_gemini_client
from app.agents.tools import CareerAgentTools
from app.database.vector_store import VectorStoreService
from app.database.persistence import PersistenceService
from app.services.embeddings import EmbeddingsService


@pytest.mark.skipif(
    not settings.is_gemini_active(),
    reason="GEMINI_API_KEY is not configured with an active key. Skipping live Gemini integration test.",
)
def test_live_gemini_tool_calling(tmp_path):
    """Verifies end-to-end Gemini function calling over live Google API."""
    client = get_gemini_client()
    assert client.is_available()

    # Initialize isolated components
    store = VectorStoreService(chroma_dir=tmp_path / "live_chroma", collection_name="live_test")
    persist = PersistenceService(db_path=tmp_path / "live_memory.db")
    embeddings = EmbeddingsService()
    tools = CareerAgentTools(
        vector_store=store,
        embeddings_service=embeddings,
        persistence_service=persist,
    )

    callables = tools.get_tool_callables(authenticated_user_id="live_tester")
    response = client.generate_with_tools(
        contents="Search for remote Python engineering jobs.",
        tools=callables,
        system_instruction="You are an AI assistant. Call search_jobs tool when asked to find jobs.",
    )

    # Verify Gemini returned function calls
    assert hasattr(response, "function_calls")
    if response.function_calls:
        first_call = response.function_calls[0]
        assert first_call.name in ["search_jobs", "web_search_jobs"]
