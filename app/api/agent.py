"""FastAPI route handlers for executing the LangGraph Career Agent workflow."""

from fastapi import APIRouter, HTTPException, Depends, Header, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from app.config import settings
from app.api.auth import get_current_user, AuthenticatedUser
from app.agents.state import CareerAgentState
from app.agents.nodes import CareerAgentNodes
from app.agents.graph import CareerAgentWorkflow
from app.agents.tools import CareerAgentTools
from app.database.vector_store import VectorStoreService
from app.database.persistence import PersistenceService
from app.services.embeddings import EmbeddingsService
from app.services.resume_parser import ResumeParserService
from app.services.retrieval import RetrievalService
from app.services.ranking import RankingService
from app.services.llm import get_gemini_client

router = APIRouter(prefix="/agent", tags=["Agent"])

# Singletons with dependency injection
persistence_svc = PersistenceService()
vector_store = VectorStoreService()
embeddings_svc = EmbeddingsService()
llm_client = get_gemini_client()
resume_parser = ResumeParserService(llm_client=llm_client)
retrieval_svc = RetrievalService(vector_store, embeddings_svc)
ranking_svc = RankingService(llm_client=llm_client)

agent_tools = CareerAgentTools(
    vector_store=vector_store,
    embeddings_service=embeddings_svc,
    persistence_service=persistence_svc,
    tavily_api_key=settings.tavily_api_key,
)

agent_nodes = CareerAgentNodes(
    resume_parser=resume_parser,
    retrieval_service=retrieval_svc,
    ranking_service=ranking_svc,
    persistence_service=persistence_svc,
    tools=agent_tools,
    llm_client=llm_client,
)
agent_workflow = CareerAgentWorkflow(agent_nodes)


class AgentRunRequest(BaseModel):
    thread_id: str = Field(..., description="Unique conversation or session thread ID")
    resume_text: str = Field(..., min_length=10, description="Raw text of candidate's resume")
    generate_cover_letter: bool = Field(False, description="Whether to craft tailored cover letter")
    target_job_id: Optional[str] = Field(None, description="Specific job ID to target for cover letter")


class AgentRunResponse(BaseModel):
    thread_id: str
    user_id: str
    parsed_profile: Optional[Dict[str, Any]]
    search_query: Optional[str]
    ranked_recommendations: List[Dict[str, Any]]
    cover_letter: Optional[str]
    step_history: List[str]
    error: Optional[str] = None


class AgentChatRequest(BaseModel):
    thread_id: str = Field(..., description="Unique thread ID for conversational memory")
    message: str = Field(..., min_length=1, description="Candidate chat message or tool request")
    max_tool_calls: int = Field(5, ge=1, le=5, description="Maximum allowed tool calls (1 to 5)")


class AgentChatResponse(BaseModel):
    thread_id: str
    user_id: str
    response: str
    tool_call_count: int
    step_history: List[str]
    messages: List[Dict[str, Any]]


@router.post(
    "/run",
    response_model=AgentRunResponse,
    summary="Execute the end-to-end Career Agent LangGraph workflow",
    status_code=status.HTTP_200_OK,
)
async def run_agent(
    request: AgentRunRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Executes stateful LangGraph career workflow with memory checkpointing."""
    user_id = current_user.user_id
    try:
        input_state: CareerAgentState = {
            "thread_id": request.thread_id,
            "user_id": user_id,
            "resume_text": request.resume_text,
            "generate_cover_letter": request.generate_cover_letter,
            "target_job_id": request.target_job_id,
            "step_history": [],
        }

        output_state = agent_workflow.run(input_state)

        # Log conversation turn to memory
        if output_state.get("ranked_recommendations"):
            top_rec = output_state["ranked_recommendations"][0]
            persistence_svc.record_message(
                thread_id=request.thread_id,
                user_id=user_id,
                role="assistant",
                content=f"Recommended {len(output_state['ranked_recommendations'])} jobs. Top match: {top_rec['job']['title']} at {top_rec['job']['company']}."
            )

        return AgentRunResponse(
            thread_id=request.thread_id,
            user_id=user_id,
            parsed_profile=output_state.get("parsed_profile"),
            search_query=output_state.get("search_query"),
            ranked_recommendations=output_state.get("ranked_recommendations", []),
            cover_letter=output_state.get("cover_letter"),
            step_history=output_state.get("step_history", []),
            error=output_state.get("error"),
        )
    except PermissionError as pe:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(pe))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution encountered an unhandled error: {str(e)}",
        )


@router.post(
    "/chat",
    response_model=AgentChatResponse,
    summary="Interactive tool-calling conversational agent turn",
    status_code=status.HTTP_200_OK,
)
async def chat_agent(
    request: AgentChatRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Executes interactive agent decision and tool execution loop."""
    try:
        input_state: CareerAgentState = {
            "thread_id": request.thread_id,
            "user_id": current_user.user_id,
            "user_message": request.message,
            "max_tool_calls": request.max_tool_calls,
            "tool_call_count": 0,
            "step_history": [],
        }
        out = agent_workflow.run_chat(input_state)
        return AgentChatResponse(
            thread_id=request.thread_id,
            user_id=current_user.user_id,
            response=out.get("final_response") or "Agent completed turn.",
            tool_call_count=out.get("tool_call_count", 0),
            step_history=out.get("step_history", []),
            messages=out.get("messages", []),
        )
    except PermissionError as pe:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(pe))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent chat turn failed: {str(e)}",
        )


@router.get(
    "/memory/{thread_id}",
    summary="Retrieve persistent preferences and memory for a thread",
    status_code=status.HTTP_200_OK,
)
async def get_agent_memory(
    thread_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Retrieve isolated thread memory, preferences, and recent conversation turns."""
    try:
        prefs = persistence_svc.get_preferences(current_user.user_id)
        history = persistence_svc.get_conversation_history(thread_id=thread_id, user_id=current_user.user_id)
        saved = persistence_svc.get_saved_jobs(current_user.user_id)
        return {
            "thread_id": thread_id,
            "user_id": current_user.user_id,
            "preferences": prefs,
            "saved_jobs": saved,
            "recent_messages": history,
        }
    except PermissionError as pe:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(pe))


@router.delete(
    "/memory/{thread_id}",
    summary="Clear thread memory and LangGraph checkpoints",
    status_code=status.HTTP_200_OK,
)
async def clear_agent_memory(
    thread_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    """Delete isolated thread memory and LangGraph checkpoint state."""
    try:
        # Enforce thread ownership and clear SQLite messages
        persistence_svc.clear_thread(thread_id=thread_id, user_id=current_user.user_id)
        # Purge LangGraph checkpointer state for thread
        agent_workflow.delete_thread_checkpoint(thread_id=thread_id)
        return {"message": f"Memory and checkpoint state for thread '{thread_id}' cleared successfully."}
    except PermissionError as pe:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(pe))
