"""Pydantic schemas for candidate resume and structured candidate profile."""

from typing import List, Optional
from pydantic import BaseModel, Field


class EducationItem(BaseModel):
    institution: Optional[str] = Field(None, description="University, college, or school name")
    degree: Optional[str] = Field(None, description="Degree or certificate title (e.g. B.S., M.S.)")
    field_of_study: Optional[str] = Field(None, description="Major or specialization")
    graduation_year: Optional[str] = Field(None, description="Graduation year (e.g. 2024)")


class ExperienceItem(BaseModel):
    company: Optional[str] = Field(None, description="Organization or company name")
    role: Optional[str] = Field(None, description="Job title / position")
    years: Optional[float] = Field(None, description="Duration in years if stated")
    description: Optional[str] = Field(None, description="Core responsibilities and achievements")
    technologies: List[str] = Field(default_factory=list, description="Technologies applied")


class ProjectItem(BaseModel):
    name: Optional[str] = Field(None, description="Project title")
    description: Optional[str] = Field(None, description="Project summary and impact")
    technologies: List[str] = Field(default_factory=list, description="Tech stack used")


class CandidateProfile(BaseModel):
    name: Optional[str] = Field(None, description="Full candidate name")
    email: Optional[str] = Field(None, description="Contact email address")
    phone: Optional[str] = Field(None, description="Phone number if present")
    
    # Categorized Technical Skills
    technical_skills: List[str] = Field(default_factory=list, description="General technical skills")
    programming_languages: List[str] = Field(default_factory=list, description="Languages (e.g. Python, TypeScript)")
    frameworks: List[str] = Field(default_factory=list, description="Frameworks (e.g. FastAPI, React, LangChain)")
    databases: List[str] = Field(default_factory=list, description="Databases (e.g. PostgreSQL, ChromaDB, Redis)")

    # Legacy alias property for backward compatibility
    skills: List[str] = Field(default_factory=list, description="Combined skills list")
    technologies: List[str] = Field(default_factory=list, description="Combined technologies list")

    education: List[EducationItem] = Field(default_factory=list, description="Educational background")
    work_experience: List[ExperienceItem] = Field(default_factory=list, description="Employment history")
    projects: List[ProjectItem] = Field(default_factory=list, description="Academic or personal projects")

    experience_level: Optional[str] = Field(
        None,
        description="Supported values: Entry-level, Mid-level, Senior, Lead. Null if unknown."
    )
    years_of_experience: Optional[float] = Field(None, description="Total years of experience")
    preferred_job_roles: List[str] = Field(default_factory=list, description="Target job roles")
    preferred_roles: List[str] = Field(default_factory=list, description="Target roles alias")
    preferred_locations: List[str] = Field(default_factory=list, description="Target cities or regions")
    location_preference: Optional[str] = Field(None, description="Single location string preference")
    preferred_work_mode: Optional[str] = Field(None, description="Remote, Hybrid, or On-site")
    work_mode_preference: Optional[str] = Field(None, description="Work mode preference alias")
    
    raw_text: Optional[str] = Field(None, description="Raw extracted text from resume document")

    def model_post_init(self, __context):
        """Synchronize aliases so both new and legacy contracts work seamlessly."""
        all_skills = list(dict.fromkeys(
            self.technical_skills + self.programming_languages + self.frameworks + self.databases + self.skills + self.technologies
        ))
        if not self.skills:
            self.skills = all_skills
        if not self.technologies:
            self.technologies = all_skills
        if not self.technical_skills:
            self.technical_skills = all_skills
        if not self.preferred_roles and self.preferred_job_roles:
            self.preferred_roles = self.preferred_job_roles
        if not self.preferred_job_roles and self.preferred_roles:
            self.preferred_job_roles = self.preferred_roles
        if not self.location_preference and self.preferred_locations:
            self.location_preference = ", ".join(self.preferred_locations)
        if not self.preferred_locations and self.location_preference:
            self.preferred_locations = [self.location_preference]
        if not self.work_mode_preference and self.preferred_work_mode:
            self.work_mode_preference = self.preferred_work_mode
        if not self.preferred_work_mode and self.work_mode_preference:
            self.preferred_work_mode = self.work_mode_preference
