"""Semantic Job Retrieval service using vector search."""

from typing import List, Dict, Any, Optional
from app import config
from app.schemas.resume import CandidateProfile
from app.schemas.job import JobPosting, ExperienceLevel, WorkMode
from app.services.embeddings import EmbeddingsService
from app.database.vector_store import VectorStoreService


class RetrievalService:
    def __init__(self, vector_store: VectorStoreService, embeddings_service: EmbeddingsService):
        self.vector_store = vector_store
        self.embeddings = embeddings_service

    @staticmethod
    def construct_query_string(profile: CandidateProfile) -> str:
        """Create a concentrated semantic search query from structured profile."""
        query_parts = []
        if profile.preferred_roles:
            query_parts.append(f"Target Roles: {', '.join(profile.preferred_roles)}")
        if profile.skills or profile.technologies:
            combined = list(dict.fromkeys(profile.skills + profile.technologies))
            query_parts.append(f"Skills: {', '.join(combined[:12])}")
        if profile.experience_level:
            query_parts.append(f"Level: {profile.experience_level}")
        if profile.work_mode_preference:
            query_parts.append(f"Work Mode: {profile.work_mode_preference}")
        if profile.location_preference:
            query_parts.append(f"Location: {profile.location_preference}")

        if not query_parts and profile.raw_text:
            return profile.raw_text[:300]

        return "\n".join(query_parts) if query_parts else "Software Engineer"

    def retrieve_jobs_for_profile(
        self,
        profile: CandidateProfile,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top K semantically matched jobs for candidate."""
        k = top_k or config.RETRIEVER_K
        query_text = self.construct_query_string(profile)
        query_embedding = self.embeddings.embed_query(query_text)
        return self.vector_store.query(query_embedding=query_embedding, top_k=k)

    def retrieve_by_query_text(
        self,
        query_text: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Direct textual query retrieval."""
        k = top_k or config.RETRIEVER_K
        query_embedding = self.embeddings.embed_query(query_text)
        return self.vector_store.query(query_embedding=query_embedding, top_k=k)
