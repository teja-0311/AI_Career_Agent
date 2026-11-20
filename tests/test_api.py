import pytest
import datetime
from fastapi.testclient import TestClient
from app.main import app
from app.api.auth import create_access_token
import jwt

client = TestClient(app)


def auth_headers(user_id: str) -> dict:
    """Helper to generate verified JWT Bearer headers for testing."""
    token = create_access_token(user_id=user_id)
    return {"Authorization": f"Bearer {token}"}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_resume_upload_txt():
    content = b"John Doe\nEmail: john@example.com\nSkills: Python, FastAPI\nExperience: 2 years"
    response = client.post(
        "/resume/upload",
        files={"file": ("resume.txt", content, "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "john@example.com"
    assert "Python" in data["skills"]


def test_resume_upload_invalid_type():
    content = b"binary"
    response = client.post(
        "/resume/upload",
        files={"file": ("resume.exe", content, "application/octet-stream")}
    )
    assert response.status_code == 400


def test_job_ingest_and_recommendation_api():
    jobs_payload = {
        "jobs": [
            {
                "job_id": "api_test_01",
                "title": "Backend Python Engineer",
                "company": "API Corp",
                "description": "Python, FastAPI and SQL backend services.",
                "required_skills": ["Python", "FastAPI"],
                "preferred_skills": ["Docker"],
                "experience_level": "Entry-level",
                "location": "Remote",
                "work_mode": "Remote",
            }
        ]
    }
    # Ingestion requires authenticated principal
    headers = auth_headers("admin_user")
    ingest_res = client.post("/jobs/ingest?overwrite=true", json=jobs_payload, headers=headers)
    assert ingest_res.status_code == 200
    assert ingest_res.json()["total_indexed"] >= 1

    profile_payload = {
        "skills": ["Python", "FastAPI"],
        "technologies": ["Python"],
        "experience_level": "Entry-level",
        "work_mode_preference": "Remote",
    }
    # Recommendation is a public candidate tool
    rec_res = client.post("/jobs/recommend", json=profile_payload)
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert rec_data["total_found"] >= 1
    assert "overall_score" in rec_data["recommendations"][0]


def test_agent_run_workflow():
    headers = auth_headers("alice_user")
    # Seed job explicitly
    client.post(
        "/jobs/ingest?overwrite=true",
        json={
            "jobs": [
                {
                    "job_id": "api_test_01",
                    "title": "Backend Python Engineer",
                    "company": "API Corp",
                    "description": "Python, FastAPI and SQL backend services.",
                    "required_skills": ["Python", "FastAPI"],
                    "preferred_skills": ["Docker"],
                    "experience_level": "Entry-level",
                    "location": "Remote",
                    "work_mode": "Remote",
                }
            ]
        },
        headers=headers,
    )

    payload = {
        "thread_id": "test_thread_api_01",
        "resume_text": "Experienced Python and FastAPI backend developer with 3 years building microservices.",
        "generate_cover_letter": True,
        "target_job_id": "api_test_01",
    }
    res = client.post("/agent/run", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["thread_id"] == "test_thread_api_01"
    assert data["user_id"] == "alice_user"
    assert "parse_profile" in data["step_history"]
    assert "cover_letter" in data["step_history"]
    assert data["cover_letter"] is not None

    # Test thread memory retrieval
    mem_res = client.get("/agent/memory/test_thread_api_01", headers=headers)
    assert mem_res.status_code == 200
    mem_data = mem_res.json()
    assert mem_data["thread_id"] == "test_thread_api_01"
    assert mem_data["user_id"] == "alice_user"


def test_auth_jwt_validation_and_tampering():
    # 1. Missing token on protected endpoint
    res_no_token = client.post("/agent/run", json={"thread_id": "t1", "resume_text": "sample resume"})
    assert res_no_token.status_code == 401
    assert "Authentication required" in res_no_token.json()["detail"]

    # 2. Forged signature
    fake_token = jwt.encode(
        {"sub": "attacker", "exp": int((datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)).timestamp())},
        "wrong_secret_key_12345678901234567890",
        algorithm="HS256"
    )
    res_fake = client.post("/agent/run", json={"thread_id": "t1", "resume_text": "sample resume"}, headers={"Authorization": f"Bearer {fake_token}"})
    assert res_fake.status_code == 401
    assert "signature" in res_fake.json()["detail"].lower()

    # 3. Expired token
    expired_token = create_access_token(user_id="alice", expires_delta=datetime.timedelta(seconds=-10))
    res_expired = client.post("/agent/run", json={"thread_id": "t1", "resume_text": "sample resume"}, headers={"Authorization": f"Bearer {expired_token}"})
    assert res_expired.status_code == 401
    assert "expired" in res_expired.json()["detail"].lower()


def test_cross_user_thread_authorization():
    alice_headers = auth_headers("alice_verified")
    bob_headers = auth_headers("bob_verified")

    # Alice creates a thread
    res = client.post(
        "/agent/run",
        json={"thread_id": "alice_secret_thread", "resume_text": "Experienced Python engineer with 5 years experience."},
        headers=alice_headers,
    )
    assert res.status_code == 200
    assert res.json()["user_id"] == "alice_verified"

    # Bob attempts to read Alice's thread -> Must return 403 Forbidden
    bob_read = client.get("/agent/memory/alice_secret_thread", headers=bob_headers)
    assert bob_read.status_code == 403
    assert "does not belong" in bob_read.json()["detail"].lower()

    # Bob attempts to delete Alice's thread -> Must return 403 Forbidden
    bob_del = client.delete("/agent/memory/alice_secret_thread", headers=bob_headers)
    assert bob_del.status_code == 403


def test_saved_jobs_validation_and_isolation():
    alice_headers = auth_headers("alice_user")
    bob_headers = auth_headers("bob_user")

    # Seed job_ai_001
    client.post(
        "/jobs/ingest?overwrite=true",
        json={
            "jobs": [
                {
                    "job_id": "job_ai_001",
                    "title": "AI Engineer",
                    "company": "DeepCo",
                    "description": "Python LLM work.",
                    "required_skills": ["Python"],
                }
            ]
        },
        headers=alice_headers,
    )

    # Attempt to save non-existent job ID -> Must return 404 Not Found
    res_404 = client.post("/jobs/non_existent_job_999/save", headers=alice_headers)
    assert res_404.status_code == 404
    assert "does not exist" in res_404.json()["detail"].lower()

    # Alice saves valid job
    save_res = client.post("/jobs/job_ai_001/save", headers=alice_headers)
    assert save_res.status_code == 200
    assert save_res.json()["user_id"] == "alice_user"

    # Alice verifies saved jobs
    alice_saved = client.get("/jobs/saved", headers=alice_headers).json()
    assert "job_ai_001" in alice_saved["saved_jobs"]

    # Bob verifies saved jobs -> isolated, empty
    bob_saved = client.get("/jobs/saved", headers=bob_headers).json()
    assert "job_ai_001" not in bob_saved["saved_jobs"]


def test_checkpoint_cleanup():
    headers = auth_headers("charlie_user")
    thread_id = "thread_to_be_cleared"

    # Turn 1: execute agent chat turn which stores conversation messages and LangGraph checkpoint
    res = client.post(
        "/agent/chat",
        json={"thread_id": thread_id, "message": "Hello agent, what is your name?", "max_tool_calls": 3},
        headers=headers,
    )
    assert res.status_code == 200

    # Memory exists
    mem_before = client.get(f"/agent/memory/{thread_id}", headers=headers)
    assert mem_before.status_code == 200
    assert len(mem_before.json()["recent_messages"]) > 0

    # Delete memory and checkpoints
    del_res = client.delete(f"/agent/memory/{thread_id}", headers=headers)
    assert del_res.status_code == 200
    assert "cleared successfully" in del_res.json()["message"]

    # Memory is now empty
    mem_after = client.get(f"/agent/memory/{thread_id}", headers=headers)
    assert mem_after.status_code == 200
    assert len(mem_after.json()["recent_messages"]) == 0


def test_agent_interactive_chat_tool_calling_and_limits():
    headers = auth_headers("david_user")

    # API boundary validation: max_tool_calls > 5 must return 422
    invalid_res = client.post(
        "/agent/chat",
        json={"thread_id": "chat_t1", "message": "Search jobs", "max_tool_calls": 10},
        headers=headers,
    )
    assert invalid_res.status_code == 422

    # API boundary validation: max_tool_calls < 1 must return 422
    invalid_res2 = client.post(
        "/agent/chat",
        json={"thread_id": "chat_t1", "message": "Search jobs", "max_tool_calls": 0},
        headers=headers,
    )
    assert invalid_res2.status_code == 422

    # Valid turn 1: User asks agent to search jobs (triggers tool calling)
    chat_res = client.post(
        "/agent/chat",
        json={"thread_id": "chat_t1", "message": "Search for Python backend jobs", "max_tool_calls": 3},
        headers=headers,
    )
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert chat_data["tool_call_count"] >= 1
    assert "tool_execution" in chat_data["step_history"]

    # Valid turn 2: Direct answer without tools
    direct_res = client.post(
        "/agent/chat",
        json={"thread_id": "chat_t1", "message": "What is your role?", "max_tool_calls": 3},
        headers=headers,
    )
    assert direct_res.status_code == 200
    direct_data = direct_res.json()
    assert direct_data["tool_call_count"] == 0
    assert "agent_answered" in direct_data["step_history"]
