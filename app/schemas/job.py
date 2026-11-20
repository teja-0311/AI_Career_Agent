"""Pydantic schemas for Job Postings and Job Ingestion."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class WorkMode(str, Enum):
    REMOTE = "Remote"
    HYBRID = "Hybrid"
    ON_SITE = "On-site"
    NOT_SPECIFIED = "Not specified"


class ExperienceLevel(str, Enum):
    ENTRY_LEVEL = "Entry-level"
    MID_LEVEL = "Mid-level"
    SENIOR = "Senior"
    LEAD = "Lead"
    NOT_SPECIFIED = "Not specified"


class JobPosting(BaseModel):
    job_id: str = Field(..., description="Deterministic unique identifier")
    title: str = Field(..., description="Job role title")
    company: str = Field(..., description="Employer company name")
    description: str = Field(..., description="Full role description and requirements")
    required_skills: List[str] = Field(default_factory=list, description="Mandatory required skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Bonus / preferred skills")
    experience_level: ExperienceLevel = Field(default=ExperienceLevel.NOT_SPECIFIED)
    min_years_experience: Optional[float] = Field(None, description="Minimum years of experience")
    max_years_experience: Optional[float] = Field(None, description="Maximum years of experience")
    location: str = Field(default="Anywhere", description="Geographic location")
    work_mode: WorkMode = Field(default=WorkMode.NOT_SPECIFIED)
    salary_range: Optional[str] = Field(None, description="Salary or compensation details if provided")
    posted_date: Optional[str] = Field(None, description="Posting date string if known")
    source: str = Field(default="sample_dataset", description="Origin source")
    application_url: Optional[str] = Field(None, description="Direct URL to job application")

    def to_searchable_text(self) -> str:
        req = ", ".join(self.required_skills) if self.required_skills else "None specified"
        pref = ", ".join(self.preferred_skills) if self.preferred_skills else "None specified"
        exp_range = f"{self.min_years_experience or 0}-{self.max_years_experience or '+'} years"
        return (
            f"Title: {self.title}\n"
            f"Company: {self.company}\n"
            f"Location: {self.location} ({self.work_mode.value})\n"
            f"Experience: {self.experience_level.value} ({exp_range})\n"
            f"Required Skills: {req}\n"
            f"Preferred Skills: {pref}\n"
            f"Description: {self.description}"
        )


class JobIngestRequest(BaseModel):
    jobs: List[JobPosting]


class JobIngestResponse(BaseModel):
    total_received: int
    total_indexed: int
    duplicates_skipped: int
    failed: int = 0
    message: str
