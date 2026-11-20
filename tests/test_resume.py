import pytest
from app.schemas.resume import CandidateProfile, EducationItem, ExperienceItem
from app.services.resume_parser import ResumeParserService


def test_resume_parser_rule_based():
    parser = ResumeParserService()
    sample_text = """
    Jane Doe
    Email: jane.doe@example.com
    Experience: Senior Software Engineer with 6 years of experience in Python, FastAPI, and Docker.
    Skills: Python, FastAPI, Docker, PostgreSQL, React
    Education: B.S. in Computer Science, Tech University (2018)
    Interested in remote roles.
    """
    profile = parser.parse(sample_text)

    assert profile.email == "jane.doe@example.com"
    assert "Python" in profile.skills
    assert "FastAPI" in profile.skills
    assert "Docker" in profile.skills
    assert profile.experience_level == "Senior"
    assert profile.years_of_experience == 6.0
    assert profile.work_mode_preference == "Remote"


def test_resume_parser_empty_file():
    parser = ResumeParserService()
    with pytest.raises(ValueError, match="empty"):
        parser.extract_raw_text("resume.pdf", b"")


def test_resume_parser_invalid_format():
    parser = ResumeParserService()
    with pytest.raises(ValueError, match="Unsupported resume format"):
        parser.extract_raw_text("resume.exe", b"binary content")


def test_resume_parser_gemini_mode_no_silent_fallback():
    # If backend is set to gemini and client is unavailable, it must raise RuntimeError explicitly
    parser = ResumeParserService(backend="gemini")
    if not parser.llm_client.is_available():
        with pytest.raises(RuntimeError, match="RESUME_PARSER_BACKEND is set to 'gemini'"):
            parser.parse("Some resume content")


def test_grounded_cover_letter_no_experience_no_projects():
    from app.agents.nodes import CareerAgentNodes
    
    # Candidate with skills but zero experience and zero projects
    cand_profile = {
        "name": "Alex Student",
        "skills": ["Python", "FastAPI"],
        "years_of_experience": 0.0,
        "work_experience": [],
        "projects": [],
    }
    job = {
        "title": "Junior Python Developer",
        "company": "TechStart Inc",
    }
    rec = {
        "matching_skills": ["Python", "FastAPI"],
    }
    letter = CareerAgentNodes._fallback_cover_letter("Alex Student", job, rec, cand_profile)
    
    # Must NOT claim professional experience or past projects
    assert "professional experience" not in letter.lower()
    assert "project work" not in letter.lower()
    assert "Python" in letter
    assert "Alex Student" in letter


def test_cover_letter_fails_without_verifiable_skills():
    from app.agents.nodes import CareerAgentNodes
    from unittest.mock import MagicMock

    nodes = CareerAgentNodes(
        resume_parser=MagicMock(),
        retrieval_service=MagicMock(),
        ranking_service=MagicMock(),
        persistence_service=MagicMock(),
        tools=MagicMock(),
        llm_client=None,
    )
    # State with empty skills and empty recommendations
    state = {
        "parsed_profile": {"name": "Empty Person", "skills": []},
        "ranked_recommendations": [{"job": {"job_id": "j1", "title": "Dev", "company": "Co"}, "matching_skills": []}],
        "target_job_id": "j1",
    }
    result = nodes.cover_letter_node(state)
    assert result["error"] is not None
    assert "no verifiable skills" in result["error"].lower()
    assert result["cover_letter"] is None
