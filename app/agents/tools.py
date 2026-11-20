"""Callable Agent Tools for LangGraph and external invocation with Gemini function calling."""

from typing import Dict, Any, List, Optional
from app.database.vector_store import VectorStoreService
from app.database.persistence import PersistenceService
from app.services.embeddings import EmbeddingsService


class CareerAgentTools:
    """Real callable tools bound to Gemini model and LangGraph agent execution."""

    def __init__(
        self,
        vector_store: VectorStoreService,
        embeddings_service: EmbeddingsService,
        persistence_service: PersistenceService,
        tavily_api_key: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings_service
        self.persistence = persistence_service
        self.tavily_api_key = tavily_api_key

    def search_jobs(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the indexed database for job postings matching the query semantically.
        
        Args:
            query: Keywords, job title, or candidate skills to search for.
            top_k: Number of matching jobs to return (default 5).
        """
        emb = self.embeddings.embed_query(query)
        raw_matches = self.vector_store.query(query_embedding=emb, top_k=top_k)
        results = []
        for m in raw_matches:
            results.append({
                "job_id": m.get("job_id"),
                "title": m.get("metadata", {}).get("title"),
                "company": m.get("metadata", {}).get("company"),
                "location": m.get("metadata", {}).get("location"),
                "work_mode": m.get("metadata", {}).get("work_mode"),
                "required_skills": m.get("metadata", {}).get("required_skills"),
                "similarity": m.get("similarity"),
                "snippet": m.get("document", "")[:250],
            })
        return results

    def get_job_details(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full details of a specific job posting by its unique job_id.
        
        Args:
            job_id: The unique identifier of the job (e.g. 'job_001').
        """
        job = self.vector_store.get_job(job_id)
        if not job:
            return None
        return {
            "job_id": job.get("id") or job_id,
            "document": job.get("document"),
            "metadata": job.get("metadata", {}),
        }

    def save_job(self, user_id: str, job_id: str) -> Dict[str, Any]:
        """Save a job posting to the authenticated user's saved jobs list.
        
        Args:
            user_id: The authenticated user's ID (injected by the system).
            job_id: The unique ID of the job to save.
        """
        job = self.vector_store.get_job(job_id)
        if not job:
            return {
                "success": False,
                "job_id": job_id,
                "error": f"Job ID '{job_id}' does not exist in the indexed catalog.",
            }
        success = self.persistence.save_job_for_user(user_id=user_id, job_id=job_id)
        saved = self.persistence.get_saved_jobs(user_id=user_id)
        return {
            "success": success,
            "job_id": job_id,
            "saved_jobs_count": len(saved),
            "all_saved_job_ids": saved,
        }

    def update_preferences(
        self,
        user_id: str,
        target_roles: Optional[List[str]] = None,
        preferred_locations: Optional[List[str]] = None,
        preferred_work_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update and persist the candidate's job preferences.
        
        Args:
            user_id: The authenticated user's ID (injected by the system).
            target_roles: List of target job titles or roles (e.g. ['Machine Learning Engineer']).
            preferred_locations: List of preferred cities or regions (e.g. ['San Francisco', 'Remote']).
            preferred_work_mode: Desired work mode ('Remote', 'On-site', 'Hybrid').
        """
        self.persistence.update_preferences(
            user_id=user_id,
            target_roles=target_roles,
            preferred_locations=preferred_locations,
            preferred_work_mode=preferred_work_mode,
        )
        return self.persistence.get_preferences(user_id) or {}

    def web_search_jobs(self, query: str) -> List[Dict[str, Any]]:
        """Perform a live web search for external job postings using Tavily search.
        
        Args:
            query: The web search query for open job listings.
        """
        if not self.tavily_api_key or self.tavily_api_key.startswith("your-"):
            return [{"warning": "TAVILY_API_KEY is not configured. Live web search prototype is disabled."}]
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self.tavily_api_key)
            resp = client.search(query=query)
            return resp.get("results", [])
        except Exception as e:
            return [{"error": f"Tavily search failed: {str(e)}"}]

    def get_tool_callables(self, authenticated_user_id: str) -> list:
        """Return bound callables for Gemini model with user_id securely pinned."""
        def search_jobs(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
            """Search indexed jobs matching query."""
            return self.search_jobs(query=query, top_k=top_k)

        def get_job_details(job_id: str) -> Optional[Dict[str, Any]]:
            """Get full details of a specific job by ID."""
            return self.get_job_details(job_id=job_id)

        def save_job(job_id: str) -> Dict[str, Any]:
            """Save a job to the user's saved jobs list."""
            return self.save_job(user_id=authenticated_user_id, job_id=job_id)

        def update_preferences(
            target_roles: Optional[List[str]] = None,
            preferred_locations: Optional[List[str]] = None,
            preferred_work_mode: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Update user job search preferences."""
            return self.update_preferences(
                user_id=authenticated_user_id,
                target_roles=target_roles,
                preferred_locations=preferred_locations,
                preferred_work_mode=preferred_work_mode,
            )

        def web_search_jobs(query: str) -> List[Dict[str, Any]]:
            """Search web for jobs via Tavily."""
            return self.web_search_jobs(query=query)

        return [search_jobs, get_job_details, save_job, update_preferences, web_search_jobs]
