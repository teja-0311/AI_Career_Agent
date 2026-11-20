"""Typed State definition for LangGraph Career Agent workflow with tools and memory."""

from typing import TypedDict, List, Optional, Dict, Any


class CareerAgentState(TypedDict, total=False):
    thread_id: str
    user_id: str
    resume_text: Optional[str]
    user_message: Optional[str]
    messages: List[Dict[str, Any]]
    parsed_profile: Optional[Dict[str, Any]]
    search_query: Optional[str]
    retrieved_raw: List[Dict[str, Any]]
    ranked_recommendations: List[Dict[str, Any]]
    generate_cover_letter: bool
    target_job_id: Optional[str]
    cover_letter: Optional[str]
    saved_jobs: List[str]
    tool_call_count: int
    max_tool_calls: int
    tool_calls_requested: List[Dict[str, Any]]
    final_response: Optional[str]
    error: Optional[str]
    step_history: List[str]
