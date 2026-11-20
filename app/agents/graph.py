"""Career Agent Graph assembly with real tool calling, SQLite checkpointing, and conditional routing."""

import sqlite3
from typing import Optional, Dict, Any
from app.config import settings
from app.agents.state import CareerAgentState
from app.agents.nodes import CareerAgentNodes


def should_retrieve_after_parsing(state: CareerAgentState) -> str:
    """Conditional edge deciding whether parsing was successful."""
    if state.get("error"):
        return "end"
    if not state.get("parsed_profile"):
        return "end"
    return "retrieve"


def should_generate_cover_letter(state: CareerAgentState) -> str:
    """Conditional edge determining whether cover letter generation was requested."""
    if state.get("error"):
        return "end"
    if state.get("generate_cover_letter", False):
        return "cover"
    return "end"


def route_agent_action(state: CareerAgentState) -> str:
    """Conditional edge routing tool calls or ending turn."""
    reqs = state.get("tool_calls_requested", [])
    count = state.get("tool_call_count", 0)
    max_c = state.get("max_tool_calls", 5)

    if reqs and count < max_c:
        return "tools"
    return "end"


class CareerAgentWorkflow:
    def __init__(self, nodes: CareerAgentNodes, checkpointer_conn: Optional[sqlite3.Connection] = None):
        self.nodes = nodes
        self.conn = checkpointer_conn or sqlite3.connect(str(settings.sqlite_db_path), check_same_thread=False)
        self.saver = None
        self._compiled_pipeline = None
        self._compiled_interactive = None
        self._init_graphs()

    def _init_graphs(self):
        """Build LangGraph StateGraphs with SQLite checkpointer."""
        import logging
        try:
            from langgraph.graph import StateGraph, START, END
            from langgraph.checkpoint.sqlite import SqliteSaver

            self.saver = SqliteSaver(self.conn)
            self.saver.setup()

            # 1. Pipeline Graph (Resume Parsing -> Retrieval -> Deterministic Hybrid Ranking -> Grounded Cover Letter)
            pipe_builder = StateGraph(CareerAgentState)
            pipe_builder.add_node("parse_profile", self.nodes.parse_profile_node)
            pipe_builder.add_node("retrieve_jobs", self.nodes.retrieve_jobs_node)
            pipe_builder.add_node("rank_jobs", self.nodes.rank_jobs_node)
            pipe_builder.add_node("cover_letter", self.nodes.cover_letter_node)

            pipe_builder.add_edge(START, "parse_profile")
            pipe_builder.add_conditional_edges(
                "parse_profile",
                should_retrieve_after_parsing,
                {"retrieve": "retrieve_jobs", "end": END},
            )
            pipe_builder.add_edge("retrieve_jobs", "rank_jobs")
            pipe_builder.add_conditional_edges(
                "rank_jobs",
                should_generate_cover_letter,
                {"cover": "cover_letter", "end": END},
            )
            pipe_builder.add_edge("cover_letter", END)
            self._compiled_pipeline = pipe_builder.compile(checkpointer=self.saver)

            # 2. Interactive Tool-Calling Agent Graph (Reasoning -> Conditional Tools Execution Loop -> Return)
            agent_builder = StateGraph(CareerAgentState)
            agent_builder.add_node("agent_decision", self.nodes.agent_decision_node)
            agent_builder.add_node("tool_execution", self.nodes.tool_execution_node)

            agent_builder.add_edge(START, "agent_decision")
            agent_builder.add_conditional_edges(
                "agent_decision",
                route_agent_action,
                {"tools": "tool_execution", "end": END},
            )
            agent_builder.add_edge("tool_execution", "agent_decision")
            self._compiled_interactive = agent_builder.compile(checkpointer=self.saver)

        except Exception as e:
            logging.error(f"LangGraph initialization failed: {str(e)}")
            if settings.environment == "production" and not settings.allow_fallback_executor:
                raise RuntimeError(
                    f"Production startup failure: LangGraph SQLite checkpointer failed to initialize: {str(e)}. "
                    "Ensure langgraph, langgraph-checkpoint-sqlite, and valid storage permissions are present, "
                    "or set ENVIRONMENT=development / ALLOW_FALLBACK_EXECUTOR=true."
                )
            self._compiled_pipeline = None
            self._compiled_interactive = None
            self.saver = None

    def delete_thread_checkpoint(self, thread_id: str):
        """Purges LangGraph checkpoint state for the specified thread."""
        if self.saver is not None and hasattr(self.saver, "delete_thread"):
            try:
                self.saver.delete_thread(thread_id)
            except Exception:
                pass
        # Clean from SQLite checkpointer tables directly to ensure total cleanup
        try:
            with self.conn:
                self.conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
                self.conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        except Exception:
            pass

    def run(self, input_state: CareerAgentState) -> CareerAgentState:
        """Execute career agent pipeline with thread checkpointing."""
        thread_id = input_state.get("thread_id", "default_thread")
        config = {"configurable": {"thread_id": thread_id}}

        if self._compiled_pipeline is not None:
            return self._compiled_pipeline.invoke(input_state, config=config)

        # Fallback executor
        state = dict(input_state)
        state["step_history"] = list(state.get("step_history", []))

        res = self.nodes.parse_profile_node(state)
        state.update(res)

        if should_retrieve_after_parsing(state) == "end":
            return state

        res = self.nodes.retrieve_jobs_node(state)
        state.update(res)

        res = self.nodes.rank_jobs_node(state)
        state.update(res)

        if should_generate_cover_letter(state) == "cover":
            res = self.nodes.cover_letter_node(state)
            state.update(res)

        return state

    def run_chat(self, input_state: CareerAgentState) -> CareerAgentState:
        """Execute conversational tool-calling agent with thread checkpointing and loop control."""
        thread_id = input_state.get("thread_id", "default_thread")
        config = {"configurable": {"thread_id": thread_id}}

        if self._compiled_interactive is not None:
            return self._compiled_interactive.invoke(input_state, config=config)

        # Fallback tool loop
        state = dict(input_state)
        state["step_history"] = list(state.get("step_history", []))
        max_loops = state.get("max_tool_calls", 5)

        for _ in range(max_loops):
            decision = self.nodes.agent_decision_node(state)
            state.update(decision)
            if route_agent_action(state) != "tools":
                break
            exec_res = self.nodes.tool_execution_node(state)
            state.update(exec_res)

        return state
