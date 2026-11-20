"""FastAPI route handlers for job ingestion, search, and recommendation."""

from fastapi import APIRouter, HTTPException, status, Query, Depends
from typing import List, Optional

from app.api.auth import get_current_user, AuthenticatedUser
from app.schemas.job import JobPosting, JobIngestRequest, JobIngestResponse
from app.schemas.resume import CandidateProfile
from app.schemas.recommendation import RecommendationResponse
from app.database.vector_store import VectorStoreService
from app.database.persistence import PersistenceService
from app.services.embeddings import EmbeddingsService
from app.services.job_ingestion import JobIngestionService
from app.services.retrieval import RetrievalService
from app.services.ranking import RankingService

router = APIRouter(prefix="/jobs", tags=["Jobs"])

# Service singletons
vector_store = VectorStoreService()
embeddings_service = EmbeddingsService()
persistence_service = PersistenceService()
ingestion_service = JobIngestionService(vector_store, embeddings_service)
retrieval_service = RetrievalService(vector_store, embeddings_service)
ranking_service = RankingService()


@router.post(
    "/ingest",
    response_model=JobIngestResponse,
    summary="Ingest and index job postings with vector embeddings",
    status_code=status.HTTP_200_OK,
)
async def ingest_jobs(
    request: JobIngestRequest,
    overwrite: bool = Query(False, description="Whether to overwrite existing job IDs"),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Validate, deduplicate, and embed job records (Admin/Protected endpoint)."""
    if not request.jobs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The jobs array cannot be empty.",
        )
    try:
        response = ingestion_service.ingest_jobs(request.jobs, overwrite=overwrite)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest job postings: {str(e)}",
        )


@router.post(
    "/search",
    summary="Perform semantic search over the job index",
    status_code=status.HTTP_200_OK,
)
async def search_jobs(
    query: str = Query(..., min_length=2, description="Job search query string"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results to retrieve"),
):
    """Retrieve top K semantically relevant jobs."""
    try:
        results = retrieval_service.retrieve_by_query_text(query, top_k=top_k)
        return {"query": query, "total_found": len(results), "results": results}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}",
        )


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Compute hybrid-ranked recommendations with full score breakdown and explanations",
    status_code=status.HTTP_200_OK,
)
async def recommend_jobs(
    profile: CandidateProfile,
    top_k: int = Query(5, ge=1, le=20, description="Top N recommendations to return"),
):
    """Full semantic retrieval and 4-way hybrid scoring for a structured candidate profile."""
    try:
        retrieved = retrieval_service.retrieve_jobs_for_profile(profile, top_k=top_k * 2)
        response = ranking_service.rank_jobs(profile, retrieved, top_k=top_k)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation pipeline failed: {str(e)}",
        )


@router.post(
    "/{job_id}/save",
    summary="Save a job to the authenticated user's profile",
    status_code=status.HTTP_200_OK,
)
async def save_job_endpoint(
    job_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Saves a job strictly bound to the authenticated principal's account after validating existence."""
    job = vector_store.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cannot save job: Job ID '{job_id}' does not exist in the index.",
        )
    success = persistence_service.save_job_for_user(user_id=current_user.user_id, job_id=job_id)
    saved = persistence_service.get_saved_jobs(user_id=current_user.user_id)
    return {
        "user_id": current_user.user_id,
        "job_id": job_id,
        "saved": success,
        "total_saved_jobs": len(saved),
    }


@router.get(
    "/saved",
    summary="Retrieve all saved jobs for the authenticated user",
    status_code=status.HTTP_200_OK,
)
async def get_saved_jobs_endpoint(
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Returns saved job IDs strictly belonging to the authenticated principal."""
    saved_ids = persistence_service.get_saved_jobs(user_id=current_user.user_id)
    return {
        "user_id": current_user.user_id,
        "saved_jobs": saved_ids,
        "count": len(saved_ids),
    }
