"""Pydantic schemas for Job Retrieval and Hybrid Ranking Recommendations."""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from app.schemas.job import JobPosting


class ScoreBreakdown(BaseModel):
    semantic_similarity: float = Field(..., ge=0.0, le=1.0, description="Normalized dense cosine similarity (0 to 1)")
    skills_match: float = Field(..., ge=0.0, le=1.0, description="Fraction of required skills satisfied (0 to 1)")
    experience_compatibility: float = Field(..., ge=0.0, le=1.0, description="Experience level alignment score (0 to 1)")
    location_compatibility: float = Field(..., ge=0.0, le=1.0, description="Location and work-mode alignment score (0 to 1)")


class JobRecommendation(BaseModel):
    job: JobPosting
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Weighted composite score between 0.0 and 1.0")
    score_breakdown: ScoreBreakdown
    matching_skills: List[str] = Field(default_factory=list, description="Required and preferred skills matched")
    missing_required_skills: List[str] = Field(default_factory=list, description="Mandatory skills from posting missing on resume")
    matched_preferred_skills: List[str] = Field(default_factory=list, description="Bonus skills matched")
    experience_summary: str = Field(..., description="Explanation of experience suitability")
    location_summary: str = Field(..., description="Explanation of location/remote suitability")
    explanation: str = Field(..., description="Deterministic or LLM-summarized rationale grounded in computed facts")


class RecommendationResponse(BaseModel):
    candidate_summary: str
    total_found: int
    recommendations: List[JobRecommendation]
    weights_used: Dict[str, float]
