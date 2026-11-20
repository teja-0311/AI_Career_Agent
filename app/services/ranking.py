"""Ranking Service with alias normalization, structured breakdown, and grounded explanations."""

from typing import List, Dict, Any, Optional, Tuple
from app.config import settings
from app.schemas.resume import CandidateProfile
from app.schemas.job import JobPosting, ExperienceLevel, WorkMode
from app.schemas.recommendation import JobRecommendation, ScoreBreakdown, RecommendationResponse
from app.services.llm import GeminiClient, get_gemini_client

# Skill alias map for normalization
SKILL_ALIASES = {
    "react.js": "react",
    "reactjs": "react",
    "node.js": "node",
    "nodejs": "node",
    "fastapi": "fastapi",
    "fast-api": "fastapi",
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "ts": "typescript",
    "js": "javascript",
    "py": "python",
    "k8s": "kubernetes",
    "docker": "docker",
    "langchain": "langchain",
    "langgraph": "langgraph",
    "chromadb": "chromadb",
    "chroma": "chromadb",
}


class RankingService:
    def __init__(
        self,
        weight_semantic: Optional[float] = None,
        weight_skills: Optional[float] = None,
        weight_experience: Optional[float] = None,
        weight_location: Optional[float] = None,
        llm_client: Optional[GeminiClient] = None,
    ):
        raw_sem = weight_semantic if weight_semantic is not None else settings.weight_semantic
        raw_ski = weight_skills if weight_skills is not None else settings.weight_skills
        raw_exp = weight_experience if weight_experience is not None else settings.weight_experience
        raw_loc = weight_location if weight_location is not None else settings.weight_location

        # Validate: Reject negative component weights
        if any(w < 0 for w in [raw_sem, raw_ski, raw_exp, raw_loc]):
            raise ValueError(
                f"Ranking weights cannot be negative. Got: semantic={raw_sem}, skills={raw_ski}, "
                f"experience={raw_exp}, location={raw_loc}."
            )

        # Validate: Reject total weight <= 0
        total_w = raw_sem + raw_ski + raw_exp + raw_loc
        if total_w <= 0:
            raise ValueError("Sum of ranking weights must be strictly positive.")

        # Normalize weights so sum is 1.0
        self.w_semantic = raw_sem / total_w
        self.w_skills = raw_ski / total_w
        self.w_exp = raw_exp / total_w
        self.w_loc = raw_loc / total_w
        self.llm_client = llm_client or get_gemini_client()

    @staticmethod
    def _normalize_skill(skill: str) -> str:
        s = skill.strip().lower()
        return SKILL_ALIASES.get(s, s)

    @classmethod
    def _normalize_skill_set(cls, skills: List[str]) -> set:
        res = set()
        for s in skills:
            cleaned = cls._normalize_skill(s)
            if cleaned:
                res.add(cleaned)
        return res

    def compute_skill_match(
        self,
        candidate_skills: List[str],
        required_skills: List[str],
        preferred_skills: List[str],
    ) -> Tuple[float, List[str], List[str], List[str]]:
        c_set = self._normalize_skill_set(candidate_skills)

        matched_req = []
        missing_req = []
        for req in required_skills:
            if self._normalize_skill(req) in c_set:
                matched_req.append(req)
            else:
                missing_req.append(req)

        matched_pref = []
        for pref in preferred_skills:
            if self._normalize_skill(pref) in c_set:
                matched_pref.append(pref)

        if not required_skills and not preferred_skills:
            score = 0.5  # Neutral when unspecified
        elif not required_skills:
            score = len(matched_pref) / max(len(preferred_skills), 1)
        else:
            req_ratio = len(matched_req) / len(required_skills)
            pref_ratio = (len(matched_pref) / len(preferred_skills)) if preferred_skills else 1.0
            score = (req_ratio * 0.8) + (pref_ratio * 0.2)

        matched_all = list(dict.fromkeys(matched_req + matched_pref))
        return min(1.0, max(0.0, score)), matched_all, missing_req, matched_pref

    def compute_experience_compatibility(
        self,
        candidate_level: Optional[str],
        candidate_years: Optional[float],
        job_level: ExperienceLevel,
        min_years_required: Optional[float],
        max_years_required: Optional[float],
    ) -> Tuple[float, str]:
        level_map = {
            "entry-level": 1,
            "mid-level": 2,
            "senior": 3,
            "lead": 4,
            "not specified": 0,
        }
        cand_val = level_map.get((candidate_level or "").lower(), 0)
        job_val = level_map.get(job_level.value.lower(), 0)

        score = 0.8
        notes = []

        # Min and max years evaluation
        if candidate_years is not None:
            min_y = min_years_required if (min_years_required is not None and min_years_required >= 0) else None
            max_y = max_years_required if (max_years_required is not None and max_years_required >= 0) else None

            if min_y is not None and candidate_years < min_y:
                ratio = candidate_years / max(min_y, 0.5)
                score = max(0.2, min(0.85, ratio))
                notes.append(f"Candidate has {candidate_years:.1f} yrs (below min {min_y:.1f} yrs).")
            elif max_y is not None and candidate_years > max_y + 3:
                score = 0.75  # Overqualified
                notes.append(f"Candidate has {candidate_years:.1f} yrs (exceeds max target of {max_y:.1f} yrs).")
            else:
                score = 1.0
                notes.append(f"Candidate meets experience requirement ({candidate_years:.1f} yrs).")
        elif cand_val > 0 and job_val > 0:
            diff = cand_val - job_val
            if diff == 0:
                score = 1.0
                notes.append(f"Exact seniority level match ({job_level.value}).")
            elif diff > 0:
                score = 0.9
                notes.append(f"Candidate level ({candidate_level}) meets/exceeds {job_level.value}.")
            elif diff == -1:
                score = 0.6
                notes.append(f"Candidate ({candidate_level}) is 1 level below {job_level.value}.")
            else:
                score = 0.3
                notes.append(f"Significant level difference ({candidate_level} vs {job_level.value}).")
        else:
            score = 0.7
            notes.append("Experience compatibility could not be conclusively determined.")

        return round(score, 4), " ".join(notes)

    def compute_location_compatibility(
        self,
        candidate_loc: Optional[str],
        candidate_work_mode: Optional[str],
        job_loc: str,
        job_work_mode: WorkMode,
    ) -> Tuple[float, str]:
        # Handle cases where candidate preferences are not provided
        if not candidate_loc and not candidate_work_mode:
            return 0.7, "Candidate location and work mode preferences not specified; neutral score applied."

        # Handle cases where job metadata is missing/unspecified
        if (not job_loc or job_loc.lower() == "anywhere") and job_work_mode == WorkMode.NOT_SPECIFIED:
            return 0.6, "Job location and work mode metadata not specified; neutral score applied."

        c_wm = (candidate_work_mode or "").strip().lower()
        j_wm = job_work_mode.value.lower()
        c_loc = (candidate_loc or "").strip().lower()
        j_loc = (job_loc or "").strip().lower()

        # Check for remote compatibility
        if j_wm == "remote":
            if c_wm == "remote":
                return 1.0, "Role and candidate preference are fully Remote."
            elif c_wm in ["on-site", "hybrid"]:
                return 0.85, "Role is fully Remote, which accommodates candidate preference."
            return 0.9, "Role is fully Remote."

        # Candidate insists on Remote, but job is On-site or Hybrid
        if c_wm == "remote":
            if j_wm == "on-site":
                return 0.3, f"Work mode mismatch: Candidate seeks Remote, but role is On-site in {job_loc}."
            if j_wm == "hybrid":
                return 0.5, f"Partial match: Candidate seeks Remote, but role requires Hybrid attendance in {job_loc}."

        # Candidate is On-site or Hybrid or has specific city location
        score = 0.8
        notes = []

        # Check work mode alignment
        if c_wm and j_wm != "not specified":
            if c_wm == j_wm:
                score += 0.1
                notes.append(f"Work mode aligned ({job_work_mode.value}).")
            else:
                score -= 0.1
                notes.append(f"Work mode difference ({c_wm} vs {j_wm}).")

        # Check geographic city alignment
        if c_loc and j_loc and j_loc != "anywhere":
            if c_loc in j_loc or j_loc in c_loc:
                score += 0.1
                notes.append(f"Location aligns ({job_loc}).")
            else:
                score -= 0.4
                notes.append(f"Location mismatch: Candidate in {candidate_loc}, job in {job_loc}.")
        elif not j_loc or j_loc == "anywhere":
            notes.append("Job location is flexible/anywhere.")
        else:
            notes.append(f"Job location is {job_loc}.")

        final_score = round(max(0.1, min(1.0, score)), 4)
        return final_score, " ".join(notes)

    def rank_jobs(
        self,
        profile: CandidateProfile,
        retrieved_items: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> RecommendationResponse:
        recommendations: List[JobRecommendation] = []
        all_candidate_skills = list(dict.fromkeys(
            profile.skills + profile.technologies + profile.technical_skills + profile.programming_languages + profile.frameworks + profile.databases
        ))

        for item in retrieved_items:
            meta = item.get("metadata", {})
            semantic_sim = float(item.get("similarity", 0.5))

            req_skills = [s.strip() for s in meta.get("required_skills", "").split(",") if s.strip()]
            pref_skills = [s.strip() for s in meta.get("preferred_skills", "").split(",") if s.strip()]

            # 1. Skills
            s_score, matched_all, missing_req, matched_pref = self.compute_skill_match(
                candidate_skills=all_candidate_skills,
                required_skills=req_skills,
                preferred_skills=pref_skills,
            )

            # 2. Experience
            exp_lvl = ExperienceLevel(meta.get("experience_level", ExperienceLevel.NOT_SPECIFIED.value))
            min_y = float(meta.get("min_years_experience", -1.0))
            max_y = float(meta.get("max_years_experience", -1.0))
            e_score, exp_summary = self.compute_experience_compatibility(
                candidate_level=profile.experience_level,
                candidate_years=profile.years_of_experience,
                job_level=exp_lvl,
                min_years_required=min_y if min_y >= 0 else None,
                max_years_required=max_y if max_y >= 0 else None,
            )

            # 3. Location / Work Mode
            wm = WorkMode(meta.get("work_mode", WorkMode.NOT_SPECIFIED.value))
            loc_score, loc_summary = self.compute_location_compatibility(
                candidate_loc=profile.location_preference or (profile.preferred_locations[0] if profile.preferred_locations else None),
                candidate_work_mode=profile.work_mode_preference or profile.preferred_work_mode,
                job_loc=meta.get("location", "Anywhere"),
                job_work_mode=wm,
            )

            # Weighted overall score
            overall = (
                (self.w_semantic * semantic_sim)
                + (self.w_skills * s_score)
                + (self.w_exp * e_score)
                + (self.w_loc * loc_score)
            )
            overall = round(min(1.0, max(0.0, overall)), 4)

            breakdown = ScoreBreakdown(
                semantic_similarity=round(semantic_sim, 4),
                skills_match=round(s_score, 4),
                experience_compatibility=round(e_score, 4),
                location_compatibility=round(loc_score, 4),
            )

            job_obj = JobPosting(
                job_id=item["job_id"],
                title=meta.get("title", "Unknown Title"),
                company=meta.get("company", "Unknown Company"),
                description=item.get("document", ""),
                required_skills=req_skills,
                preferred_skills=pref_skills,
                experience_level=exp_lvl,
                location=meta.get("location", "Anywhere"),
                work_mode=wm,
                source=meta.get("source", "sample_dataset"),
                application_url=meta.get("application_url"),
            )

            # Grounded explanation
            expl_parts = [
                f"Overall hybrid match score is {int(overall * 100)}%.",
                f"Matched skills: {', '.join(matched_all[:6]) if matched_all else 'General background'}.",
            ]
            if missing_req:
                expl_parts.append(f"Missing required skills: {', '.join(missing_req)}.")
            else:
                expl_parts.append("Meets all explicitly listed required skills.")
            expl_parts.append(exp_summary)
            expl_parts.append(loc_summary)

            rec = JobRecommendation(
                job=job_obj,
                overall_score=overall,
                score_breakdown=breakdown,
                matching_skills=matched_all,
                missing_required_skills=missing_req,
                matched_preferred_skills=matched_pref,
                experience_summary=exp_summary,
                location_summary=loc_summary,
                explanation=" ".join(expl_parts),
            )
            recommendations.append(rec)

        recommendations.sort(key=lambda r: r.overall_score, reverse=True)
        top_recs = recommendations[:top_k]

        return RecommendationResponse(
            candidate_summary=f"Profile parsed with {len(all_candidate_skills)} skills.",
            total_found=len(recommendations),
            recommendations=top_recs,
            weights_used={
                "semantic": round(self.w_semantic, 2),
                "skills": round(self.w_skills, 2),
                "experience": round(self.w_exp, 2),
                "location": round(self.w_loc, 2),
            },
        )
