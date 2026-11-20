import pytest
from app.schemas.resume import CandidateProfile
from app.schemas.job import JobPosting, ExperienceLevel, WorkMode
from app.services.ranking import RankingService


def test_hybrid_ranking_weights_and_scores():
    ranking_service = RankingService(
        weight_semantic=0.45,
        weight_skills=0.25,
        weight_experience=0.15,
        weight_location=0.15,
    )

    profile = CandidateProfile(
        skills=["Python", "FastAPI", "Docker"],
        technologies=["Python", "FastAPI"],
        experience_level="Mid-level",
        years_of_experience=3.0,
        work_mode_preference="Remote",
    )

    retrieved = [
        {
            "job_id": "job_1",
            "similarity": 0.85,
            "metadata": {
                "title": "Backend Python Engineer",
                "company": "Acme",
                "required_skills": "Python, FastAPI",
                "preferred_skills": "Docker, Kubernetes",
                "experience_level": "Mid-level",
                "work_mode": "Remote",
                "location": "Anywhere",
            },
            "document": "Python engineer role with FastAPI.",
        },
        {
            "job_id": "job_2",
            "similarity": 0.40,
            "metadata": {
                "title": "Ruby on Rails Developer",
                "company": "Legacy Corp",
                "required_skills": "Ruby, Rails",
                "preferred_skills": "AWS",
                "experience_level": "Senior",
                "work_mode": "On-site",
                "location": "Dallas, TX",
            },
            "document": "Legacy Ruby developer.",
        }
    ]

    response = ranking_service.rank_jobs(profile, retrieved, top_k=2)

    assert len(response.recommendations) == 2
    top = response.recommendations[0]
    assert top.job.job_id == "job_1"
    assert top.overall_score > response.recommendations[1].overall_score
    assert "Python" in top.matching_skills
    assert len(top.missing_required_skills) == 0

    second = response.recommendations[1]
    assert "Ruby" in second.missing_required_skills
    assert second.score_breakdown.skills_match < top.score_breakdown.skills_match


def test_missing_skills_identification():
    ranking_service = RankingService()
    score, matched, missing_req, matched_pref = ranking_service.compute_skill_match(
        candidate_skills=["Python", "SQL"],
        required_skills=["Python", "Kubernetes", "Rust"],
        preferred_skills=["Docker"],
    )
    assert "Python" in matched
    assert "Kubernetes" in missing_req
    assert "Rust" in missing_req
    assert len(matched_pref) == 0


def test_ranking_rejects_negative_weights():
    with pytest.raises(ValueError, match="Ranking weights cannot be negative"):
        RankingService(weight_semantic=-0.1, weight_skills=0.5, weight_experience=0.3, weight_location=0.2)


def test_ranking_rejects_zero_total_weight():
    with pytest.raises(ValueError, match="Sum of ranking weights must be strictly positive"):
        RankingService(weight_semantic=0.0, weight_skills=0.0, weight_experience=0.0, weight_location=0.0)


def test_location_and_work_mode_compatibility():
    svc = RankingService()
    # Remote candidate applying to On-site role in Seattle
    score_remote_onsite, notes = svc.compute_location_compatibility(
        candidate_loc="Remote",
        candidate_work_mode="Remote",
        job_loc="Seattle, WA",
        job_work_mode=WorkMode.ON_SITE,
    )
    assert score_remote_onsite <= 0.4
    assert "mismatch" in notes.lower()

    # Remote candidate applying to Remote role
    score_remote_remote, _ = svc.compute_location_compatibility(
        candidate_loc="Remote",
        candidate_work_mode="Remote",
        job_loc="Anywhere",
        job_work_mode=WorkMode.REMOTE,
    )
    assert score_remote_remote == 1.0

    # Missing candidate metadata -> neutral score
    score_neutral, notes_neutral = svc.compute_location_compatibility(
        candidate_loc=None,
        candidate_work_mode=None,
        job_loc="Chicago, IL",
        job_work_mode=WorkMode.HYBRID,
    )
    assert score_neutral == 0.7
    assert "neutral" in notes_neutral.lower()
