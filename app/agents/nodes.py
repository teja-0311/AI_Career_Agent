"""LangGraph nodes for Career Agent with persistence and dependency-injected LLM."""

from typing import Dict, Any, List, Optional
from app.agents.state import CareerAgentState
from app.schemas.resume import CandidateProfile
from app.services.resume_parser import ResumeParserService
from app.services.retrieval import RetrievalService
from app.services.ranking import RankingService
from app.database.persistence import PersistenceService
from app.agents.tools import CareerAgentTools
from app.services.llm import GeminiClient, get_gemini_client


class CareerAgentNodes:
    def __init__(
        self,
        resume_parser: ResumeParserService,
        retrieval_service: RetrievalService,
        ranking_service: RankingService,
        persistence_service: PersistenceService,
        tools: CareerAgentTools,
        llm_client: Optional[GeminiClient] = None,
    ):
        self.parser = resume_parser
        self.retrieval = retrieval_service
        self.ranking = ranking_service
        self.persistence = persistence_service
        self.tools = tools
        self.llm_client = llm_client or get_gemini_client()

    def parse_profile_node(self, state: CareerAgentState) -> Dict[str, Any]:
        """Node 1: Extract candidate profile and apply stored user preferences."""
        raw_text = state.get("resume_text", "").strip()
        history = list(state.get("step_history", []))
        history.append("parse_profile")

        if not raw_text:
            return {
                "error": "Resume text is empty. Cannot extract candidate profile.",
                "step_history": history,
            }

        try:
            profile: CandidateProfile = self.parser.parse(raw_text)

            user_id = state.get("user_id", "default_user")
            thread_id = state.get("thread_id", "default_thread")
            self.persistence.ensure_user_and_thread(user_id, thread_id)

            # Load persistent user preferences
            stored_prefs = self.persistence.get_preferences(user_id)
            if stored_prefs:
                if stored_prefs.get("target_roles") and not profile.preferred_job_roles:
                    profile.preferred_job_roles = stored_prefs["target_roles"]
                if stored_prefs.get("preferred_locations") and not profile.preferred_locations:
                    profile.preferred_locations = stored_prefs["preferred_locations"]
                if stored_prefs.get("preferred_work_mode") and not profile.preferred_work_mode:
                    profile.preferred_work_mode = stored_prefs["preferred_work_mode"]

            # Save updated preferences back to persistence
            self.persistence.update_preferences(
                user_id=user_id,
                target_roles=profile.preferred_job_roles,
                preferred_locations=profile.preferred_locations,
                preferred_work_mode=profile.preferred_work_mode,
            )

            return {
                "parsed_profile": profile.model_dump(),
                "step_history": history,
            }
        except Exception as e:
            return {
                "error": f"Profile parsing failed: {str(e)}",
                "step_history": history,
            }

    def retrieve_jobs_node(self, state: CareerAgentState) -> Dict[str, Any]:
        """Node 2: Semantic retrieval from vector store."""
        history = list(state.get("step_history", []))
        history.append("retrieve_jobs")

        profile_dict = state.get("parsed_profile")
        if not profile_dict:
            return {"error": "Missing profile for retrieval", "step_history": history}

        profile = CandidateProfile(**profile_dict)
        query_str = self.retrieval.construct_query_string(profile)
        retrieved = self.retrieval.retrieve_jobs_for_profile(profile, top_k=10)

        return {
            "search_query": query_str,
            "retrieved_raw": retrieved,
            "step_history": history,
        }

    def rank_jobs_node(self, state: CareerAgentState) -> Dict[str, Any]:
        """Node 3: Deterministic 4-way hybrid scoring and grounded explanations."""
        history = list(state.get("step_history", []))
        history.append("rank_jobs")

        profile_dict = state.get("parsed_profile")
        retrieved = state.get("retrieved_raw", [])
        if not profile_dict:
            return {"error": "Missing profile for ranking", "step_history": history}

        profile = CandidateProfile(**profile_dict)
        rec_response = self.ranking.rank_jobs(profile=profile, retrieved_items=retrieved, top_k=5)
        serialized_recs = [r.model_dump() for r in rec_response.recommendations]

        return {
            "ranked_recommendations": serialized_recs,
            "step_history": history,
        }

    def cover_letter_node(self, state: CareerAgentState) -> Dict[str, Any]:
        """Node 4: Grounded cover letter generation. Requires target job ID and verifiable skills."""
        history = list(state.get("step_history", []))
        history.append("cover_letter")

        recs = state.get("ranked_recommendations", [])
        target_job_id = state.get("target_job_id")

        target_rec = None
        if target_job_id:
            for r in recs:
                if r["job"]["job_id"] == target_job_id:
                    target_rec = r
                    break
            # If not in top recs, query vector store directly by ID
            if not target_rec:
                job_data = self.tools.get_job_details(target_job_id)
                if not job_data:
                    return {
                        "error": f"Target job ID '{target_job_id}' not found in job index.",
                        "cover_letter": None,
                        "step_history": history,
                    }
                target_rec = {
                    "job": {
                        "job_id": target_job_id,
                        "title": job_data["metadata"].get("title", "Target Role"),
                        "company": job_data["metadata"].get("company", "Target Company"),
                        "description": job_data.get("document", ""),
                    },
                    "matching_skills": [],
                }
        else:
            if recs:
                target_rec = recs[0]
            else:
                return {
                    "error": "No jobs available to generate a targeted cover letter.",
                    "cover_letter": None,
                    "step_history": history,
                }

        job = target_rec["job"]
        profile = state.get("parsed_profile", {})
        cand_name = profile.get("name") or "Candidate"
        matching_skills = target_rec.get("matching_skills", []) or profile.get("skills", [])

        # Grounding check: verify that candidate has actual skills to ground the letter
        if not matching_skills:
            return {
                "error": "Candidate profile contains no verifiable skills to ground a targeted cover letter.",
                "cover_letter": None,
                "step_history": history + ["cover_letter_insufficient_data"],
            }

        skills = ", ".join(matching_skills[:5])
        prompt = (
            f"Write a concise, professional, and personalized cover letter for {cand_name} applying to '{job['title']}' at '{job['company']}'.\n"
            f"Candidate Skills: {skills}\n"
            f"Candidate Experience Level: {profile.get('experience_level', 'Not specified')}\n"
            f"Job Summary:\n{job.get('description', '')[:400]}\n\n"
            "Guidelines:\n"
            "- Ground all statements strictly in the candidate's actual skills.\n"
            "- Never fabricate past employers, degrees, or certifications.\n"
            "- Keep length under 250 words."
        )

        letter = None
        if self.llm_client and self.llm_client.is_available():
            try:
                letter = self.llm_client.generate_text(prompt=prompt)
            except Exception as e:
                # Do not silently swallow LLM errors
                return {
                    "error": f"LLM generation failed: {str(e)}",
                    "cover_letter": None,
                    "step_history": history + ["cover_letter_llm_failure"],
                }

        if not letter:
            letter = self._fallback_cover_letter(cand_name, job, target_rec, profile)

        return {
            "cover_letter": letter,
            "step_history": history,
        }

    @staticmethod
    def _fallback_cover_letter(cand_name: str, job: Dict[str, Any], rec: Dict[str, Any], profile: Dict[str, Any]) -> str:
        matching = ", ".join(rec.get("matching_skills", [])[:5]) or ", ".join(profile.get("skills", [])[:5])
        
        # Ground experience claims strictly in profile data
        years_exp = float(profile.get("years_of_experience") or 0.0)
        has_work_exp = bool(profile.get("work_experience")) or years_exp > 0.0
        has_projects = bool(profile.get("projects"))

        if has_work_exp:
            body_para = (
                f"Drawing from my {years_exp:.1f} years of professional experience and technical background in {matching}, "
                f"I focus on delivering clean, reliable, and scalable solutions that meet engineering objectives."
            )
        elif has_projects:
            body_para = (
                f"Through hands-on project work emphasizing {matching}, "
                f"I have developed practical capabilities in designing and maintaining clean, functional software."
            )
        else:
            body_para = (
                f"With a dedicated technical foundation in {matching}, "
                f"I am eager to contribute my technical training, problem-solving mindset, and dedication to your engineering team."
            )

        return (
            f"Dear Hiring Team at {job.get('company', 'the organization')},\n\n"
            f"I am writing to express my enthusiastic interest in the {job.get('title', 'open position')} role. "
            f"My technical background aligns well with your requirements in {matching}.\n\n"
            f"{body_para}\n\n"
            f"Thank you for your time and consideration. I look forward to the possibility of discussing how my skills align with your goals.\n\n"
            f"Sincerely,\n{cand_name}"
        )

    def agent_decision_node(self, state: CareerAgentState) -> Dict[str, Any]:
        """Agent Reasoning Node: Analyzes context, invokes LLM with tools, or answers directly."""
        history = list(state.get("step_history", []))
        history.append("agent_decision")

        user_id = state.get("user_id", "default_user")
        thread_id = state.get("thread_id", "default_thread")
        self.persistence.ensure_user_and_thread(user_id, thread_id)

        call_count = state.get("tool_call_count", 0)
        max_calls = state.get("max_tool_calls", 5)

        # Loop prevention
        if call_count >= max_calls:
            msg = f"Maximum tool call limit of {max_calls} reached. Halting tool loop."
            return {
                "final_response": msg,
                "tool_calls_requested": [],
                "step_history": history + ["max_loops_reached"],
            }

        user_msg = state.get("user_message", "")
        messages = list(state.get("messages", []))
        if user_msg and (not messages or messages[-1].get("content") != user_msg):
            messages.append({"role": "user", "content": user_msg})
            self.persistence.record_message(thread_id, user_id, "user", user_msg)

        # Context loading: User preferences and past memory
        prefs = self.persistence.get_preferences(user_id) or {}
        context_str = f"User Preferences: {prefs}. Conversation History Length: {len(messages)}."

        # If LLM is available and active
        if self.llm_client and self.llm_client.is_available():
            try:
                tools = self.tools.get_tool_callables(authenticated_user_id=user_id)
                prompt_content = f"{context_str}\n\nUser Message: {user_msg}\nRecent Messages: {messages[-4:]}"
                response = self.llm_client.generate_with_tools(
                    contents=prompt_content,
                    tools=tools,
                    system_instruction=(
                        "You are a helpful AI Career Agent. Use provided tools to search jobs, fetch details, "
                        "save jobs, and update candidate preferences. If no tools are required, respond directly."
                    ),
                )
                if hasattr(response, "function_calls") and response.function_calls:
                    reqs = []
                    for fc in response.function_calls:
                        reqs.append({
                            "name": fc.name,
                            "args": dict(fc.args) if hasattr(fc, "args") else {},
                        })
                    return {
                        "messages": messages,
                        "tool_calls_requested": reqs,
                        "step_history": history + ["agent_tool_call_requested"],
                    }
                else:
                    ans = getattr(response, "text", "") or "I reviewed your career query."
                    self.persistence.record_message(thread_id, user_id, "assistant", ans)
                    messages.append({"role": "assistant", "content": ans})
                    return {
                        "messages": messages,
                        "final_response": ans,
                        "tool_calls_requested": [],
                        "step_history": history + ["agent_answered"],
                    }
            except Exception as e:
                # LLM failure should not crash agent
                err_ans = f"Error during agent reasoning: {str(e)}"
                return {
                    "error": err_ans,
                    "final_response": err_ans,
                    "tool_calls_requested": [],
                    "step_history": history + ["agent_reasoning_error"],
                }

        # Offline / deterministic fallback for mock testing
        lower_msg = user_msg.lower()
        if "search" in lower_msg or "find" in lower_msg:
            return {
                "messages": messages,
                "tool_calls_requested": [{"name": "search_jobs", "args": {"query": user_msg, "top_k": 3}}],
                "step_history": history + ["agent_tool_call_requested"],
            }
        elif "save" in lower_msg and "job" in lower_msg:
            # Extract job_id or default to mock
            import re
            m = re.search(r"job_[\w\d]+", user_msg)
            j_id = m.group(0) if m else "api_test_01"
            return {
                "messages": messages,
                "tool_calls_requested": [{"name": "save_job", "args": {"job_id": j_id}}],
                "step_history": history + ["agent_tool_call_requested"],
            }
        elif "preference" in lower_msg:
            return {
                "messages": messages,
                "tool_calls_requested": [{
                    "name": "update_preferences",
                    "args": {"preferred_work_mode": "Remote", "target_roles": ["Python Developer"]},
                }],
                "step_history": history + ["agent_tool_call_requested"],
            }
        else:
            ans = "I am your Career Agent. I can help search jobs, save jobs, and update your preferences."
            self.persistence.record_message(thread_id, user_id, "assistant", ans)
            messages.append({"role": "assistant", "content": ans})
            return {
                "messages": messages,
                "final_response": ans,
                "tool_calls_requested": [],
                "step_history": history + ["agent_answered"],
            }

    def tool_execution_node(self, state: CareerAgentState) -> Dict[str, Any]:
        """Tool Execution Node: Executes requested tools with authenticated context and strict budget limit."""
        history = list(state.get("step_history", []))
        history.append("tool_execution")

        reqs = list(state.get("tool_calls_requested", []))
        user_id = state.get("user_id", "default_user")
        call_count = state.get("tool_call_count", 0)
        max_calls = state.get("max_tool_calls", 5)
        messages = list(state.get("messages", []))

        outputs = []
        for call in reqs:
            # Enforce max_tool_calls strictly before executing each tool
            if call_count >= max_calls:
                history.append("tool_budget_exhausted")
                break

            name = call.get("name")
            args = call.get("args", {})
            result = None

            if name == "search_jobs":
                result = self.tools.search_jobs(
                    query=args.get("query", ""),
                    top_k=int(args.get("top_k", 5)),
                )
            elif name == "get_job_details":
                result = self.tools.get_job_details(job_id=args.get("job_id", ""))
            elif name == "save_job":
                result = self.tools.save_job(user_id=user_id, job_id=args.get("job_id", ""))
            elif name == "update_preferences":
                result = self.tools.update_preferences(
                    user_id=user_id,
                    target_roles=args.get("target_roles"),
                    preferred_locations=args.get("preferred_locations"),
                    preferred_work_mode=args.get("preferred_work_mode"),
                )
            elif name == "web_search_jobs":
                result = self.tools.web_search_jobs(query=args.get("query", ""))
            else:
                result = {"error": f"Unknown tool: '{name}'"}

            call_count += 1
            outputs.append({"tool": name, "result": result})
            messages.append({
                "role": "tool",
                "name": name,
                "content": result,
            })

        # Generate summary of executed tools for final_response
        summary_lines = [f"Executed {len(outputs)} tool(s) (total calls used: {call_count}/{max_calls}):"]
        for o in outputs:
            summary_lines.append(f"- {o['tool']}: {str(o['result'])[:150]}...")

        return {
            "messages": messages,
            "tool_call_count": call_count,
            "tool_calls_requested": [],
            "final_response": "\n".join(summary_lines),
            "step_history": history,
        }
