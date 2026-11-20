"""Service for validating, deduplicating, and ingesting job postings into the vector database."""

import csv
import json
import hashlib
from typing import List, Union, Dict, Any, Tuple
from app.schemas.job import JobPosting, ExperienceLevel, WorkMode, JobIngestResponse
from app.services.embeddings import EmbeddingsService
from app.database.vector_store import VectorStoreService


class JobIngestionService:
    def __init__(self, vector_store: VectorStoreService, embeddings_service: EmbeddingsService):
        self.vector_store = vector_store
        self.embeddings = embeddings_service

    @staticmethod
    def generate_deterministic_id(title: str, company: str, location: str) -> str:
        """Create a reproducible hash ID when job_id is missing or needs deduplication."""
        base = f"{title.strip().lower()}|{company.strip().lower()}|{location.strip().lower()}"
        return f"job_{hashlib.md5(base.encode('utf-8')).hexdigest()[:12]}"

    @classmethod
    def normalize_job_dict(cls, data: Dict[str, Any]) -> JobPosting:
        """Sanitize raw dictionary into a valid JobPosting object."""
        title = str(data.get("title", "")).strip()
        company = str(data.get("company", "Unknown")).strip()
        location = str(data.get("location", "Anywhere")).strip()

        job_id = str(data.get("job_id") or "").strip()
        if not job_id:
            job_id = cls.generate_deterministic_id(title, company, location)

        description = str(data.get("description", "")).strip()

        # Handle skill fields that might be lists or comma-delimited strings
        def parse_skills(field_val: Any) -> List[str]:
            if isinstance(field_val, list):
                return [str(s).strip() for s in field_val if str(s).strip()]
            elif isinstance(field_val, str) and field_val.strip():
                return [s.strip() for s in field_val.split(",") if s.strip()]
            return []

        required_skills = parse_skills(data.get("required_skills"))
        preferred_skills = parse_skills(data.get("preferred_skills"))

        # Map experience level
        exp_raw = str(data.get("experience_level", "")).lower()
        exp_level = ExperienceLevel.NOT_SPECIFIED
        if "entry" in exp_raw or "junior" in exp_raw or "intern" in exp_raw:
            exp_level = ExperienceLevel.ENTRY_LEVEL
        elif "mid" in exp_raw or "intermediate" in exp_raw:
            exp_level = ExperienceLevel.MID_LEVEL
        elif "senior" in exp_raw:
            exp_level = ExperienceLevel.SENIOR
        elif "lead" in exp_raw or "principal" in exp_raw or "staff" in exp_raw:
            exp_level = ExperienceLevel.LEAD

        # Map work mode
        wm_raw = str(data.get("work_mode", "")).lower()
        wm = WorkMode.NOT_SPECIFIED
        if "remote" in wm_raw:
            wm = WorkMode.REMOTE
        elif "hybrid" in wm_raw:
            wm = WorkMode.HYBRID
        elif "on-site" in wm_raw or "onsite" in wm_raw:
            wm = WorkMode.ON_SITE

        min_years = data.get("min_years_experience")
        if min_years is not None:
            try:
                min_years = float(min_years)
            except (ValueError, TypeError):
                min_years = None

        return JobPosting(
            job_id=job_id,
            title=title,
            company=company,
            description=description,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            experience_level=exp_level,
            min_years_experience=min_years,
            location=location,
            work_mode=wm,
            source=str(data.get("source", "sample_dataset")),
            application_url=data.get("application_url"),
        )

    def load_from_json(self, json_content: Union[str, bytes]) -> List[JobPosting]:
        """Parse JSON array of job postings."""
        if isinstance(json_content, bytes):
            json_content = json_content.decode("utf-8")
        data = json.loads(json_content)
        if isinstance(data, dict) and "jobs" in data:
            data = data["jobs"]
        if not isinstance(data, list):
            raise ValueError("Expected JSON content to be a list of job objects.")
        return [self.normalize_job_dict(item) for item in data]

    def load_from_csv(self, csv_content: Union[str, bytes]) -> List[JobPosting]:
        """Parse CSV content of job postings."""
        if isinstance(csv_content, bytes):
            csv_content = csv_content.decode("utf-8")
        reader = csv.DictReader(csv_content.splitlines())
        jobs = []
        for row in reader:
            jobs.append(self.normalize_job_dict(row))
        return jobs

    def ingest_jobs(self, jobs: List[JobPosting], overwrite: bool = False) -> JobIngestResponse:
        """Deduplicate, embed, and store job records."""
        if not jobs:
            return JobIngestResponse(
                total_received=0,
                total_indexed=0,
                duplicates_skipped=0,
                failed=0,
                message="No jobs provided to ingest."
            )

        # In-memory deduplication pass on incoming batch
        seen_ids = set()
        unique_jobs = []
        batch_duplicates = 0
        for j in jobs:
            if j.job_id in seen_ids:
                batch_duplicates += 1
                continue
            seen_ids.add(j.job_id)
            unique_jobs.append(j)

        # Generate searchable representations
        texts = [j.to_searchable_text() for j in unique_jobs]
        embeddings = self.embeddings.embed_documents(texts)

        # Store in vector database
        indexed, db_skipped = self.vector_store.add_jobs(
            jobs=unique_jobs,
            embeddings=embeddings,
            overwrite=overwrite,
        )

        total_skipped = batch_duplicates + db_skipped
        return JobIngestResponse(
            total_received=len(jobs),
            total_indexed=indexed,
            duplicates_skipped=total_skipped,
            failed=0,
            message=f"Successfully indexed {indexed} jobs. Skipped {total_skipped} duplicates."
        )
