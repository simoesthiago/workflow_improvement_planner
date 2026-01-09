"""Workflow Improvement Planner - Streamlit application.

This module provides the main Streamlit UI and LangGraph workflow orchestration
for the Workflow Improvement Planner. It coordinates:
- Case management (sidebar)
- Assessment wizard (form entry)
- Solution Designer agent execution
- Prototype Builder agent execution  
- Export pack generation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import streamlit as st
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

import utils
from config import (
    INDUSTRY_RESEARCH_MAX_QUERIES,
    INDUSTRY_RESEARCH_RESULTS_PER_QUERY,
    LLM_TEMPERATURE,
    MAX_REVISION_CYCLES,
    PROTOTYPE_BUILDER_MAX_ITERATIONS,
    PROTOTYPE_BUILDER_MAX_SEARCH_CALLS,
    SOLUTION_DESIGNER_MAX_ITERATIONS,
    SOLUTION_DESIGNER_MAX_SEARCH_CALLS,
    VALIDATOR_MAX_ITERATIONS,
    VALIDATOR_MAX_SEARCH_CALLS,
    Config,
    get_config,
    load_config,
)
from prompts import (
    CONSISTENCY_VALIDATOR_PROMPT,
    PROTOTYPE_BUILDER_PROMPT,
    REVISION_FEEDBACK_TEMPLATE,
    SOLUTION_DESIGNER_PROMPT,
)
from schemas import PlannerState

if TYPE_CHECKING:
    pass  # Additional type imports if needed


@dataclass
class RunMetadata:
    """Structured metadata for agent runs."""

    type: str
    started_at: str
    ended_at: str
    web_search_calls: int
    sources_used_ids: list[str]
    status: str
    error: str | None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "web_search_calls": self.web_search_calls,
            "sources_used_ids": self.sources_used_ids,
            "status": self.status,
            "error": self.error,
        }


def _get_llm() -> ChatOpenAI:
    config = get_config()
    return ChatOpenAI(model=config.openai_model, temperature=LLM_TEMPERATURE)


@dataclass
class SearchState:
    """Mutable state for tracking web search calls during agent execution."""

    count: int = 0
    source_ids: list[str] | None = None

    def __post_init__(self) -> None:
        if self.source_ids is None:
            self.source_ids = []


def _make_tools(
    case_id: str,
    max_search_calls: int,
    max_results: int,
    recency_days: int,
    max_chars: int,
) -> tuple[list[Callable], SearchState]:
    """Create tools for agent execution with search call tracking."""
    search_state = SearchState()

    @tool
    def read_case_context() -> dict:
        """Read assessment, context_text, warnings, and attachments for this case."""
        return utils.read_case_context(case_id, max_chars=max_chars)

    @tool
    def web_search(query: str) -> dict:
        """Search the web (Tavily). Use sparingly."""
        if search_state.count >= max_search_calls:
            return {"error": f"web_search limit reached ({max_search_calls})"}
        search_state.count += 1
        try:
            res = utils.web_search(
                case_id,
                query=query,
                max_results=max_results,
                recency_days=recency_days,
            )
            search_state.source_ids.extend(res.get("source_ids", []))
            return res
        except Exception as exc:
            return {"error": f"Web search failed: {exc}"}

    return [read_case_context, web_search], search_state


def _build_agent(prompt_text: str, tools: list, max_iterations: int) -> AgentExecutor:
    """Build an AgentExecutor with the given prompt and tools."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_text),
            ("user", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(_get_llm(), tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=False, max_iterations=max_iterations
    )


def _run_agent(
    agent_type: str,
    case_id: str,
    prompt_text: str,
    user_input: str,
    max_search_calls: int,
    max_iterations: int,
) -> tuple[str | None, dict]:
    """Execute an agent and return (output_text, run_metadata).

    This is the core agent execution logic shared by Solution Designer
    and Prototype Builder.
    """
    config = get_config()
    start_time = utils.utc_now_iso()

    tools, search_state = _make_tools(
        case_id,
        max_search_calls=max_search_calls,
        max_results=config.web_search_max_results,
        recency_days=config.web_search_recency_days,
        max_chars=config.max_context_chars,
    )

    agent = _build_agent(prompt_text, tools=tools, max_iterations=max_iterations)

    try:
        result = agent.invoke({"input": user_input})
        output_text = result.get("output") or ""
        metadata = RunMetadata(
            type=agent_type,
            started_at=start_time,
            ended_at=utils.utc_now_iso(),
            web_search_calls=search_state.count,
            sources_used_ids=search_state.source_ids or [],
            status="success",
            error=None,
        )
        return output_text, metadata.to_dict()
    except Exception as exc:
        metadata = RunMetadata(
            type=agent_type,
            started_at=start_time,
            ended_at=utils.utc_now_iso(),
            web_search_calls=search_state.count,
            sources_used_ids=search_state.source_ids or [],
            status="error",
            error=str(exc),
        )
        return None, metadata.to_dict()


def run_solution_designer(case_id: str) -> tuple[str | None, dict]:
    """Run the Solution Designer agent."""
    user_input = (
        "Generate the Solution Designer report. "
        "Always call read_case_context first. Use web_search only if needed."
    )
    return _run_agent(
        agent_type="solution_designer",
        case_id=case_id,
        prompt_text=SOLUTION_DESIGNER_PROMPT,
        user_input=user_input,
        max_search_calls=SOLUTION_DESIGNER_MAX_SEARCH_CALLS,
        max_iterations=SOLUTION_DESIGNER_MAX_ITERATIONS,
    )


def run_prototype_builder(
    case_id: str, solution_md: str | None, industry_research: str | None = None
) -> tuple[str | None, dict]:
    """Run the Prototype Builder agent."""
    research_context = ""
    if industry_research:
        research_context = f"\n\n## Industry Research Context:\n{industry_research}\n\n"

    user_input = (
        "Generate the Prototype Builder blueprint. Always call read_case_context first. "
        "Use web_search only if needed."
        f"{research_context}"
        f"Incorporate the latest Solution output:\n\n"
        f"{solution_md or '(no solution output available)'}"
    )
    return _run_agent(
        agent_type="prototype_builder",
        case_id=case_id,
        prompt_text=PROTOTYPE_BUILDER_PROMPT,
        user_input=user_input,
        max_search_calls=PROTOTYPE_BUILDER_MAX_SEARCH_CALLS,
        max_iterations=PROTOTYPE_BUILDER_MAX_ITERATIONS,
    )


def run_consistency_validator(
    case_id: str, solution_md: str, prototype_md: str
) -> tuple[dict, dict]:
    """Run the Consistency Validator agent.

    Returns:
        Tuple of (validation_result dict, run_metadata dict)
    """
    config = get_config()
    start_time = utils.utc_now_iso()

    tools, search_state = _make_tools(
        case_id,
        max_search_calls=VALIDATOR_MAX_SEARCH_CALLS,
        max_results=config.web_search_max_results,
        recency_days=config.web_search_recency_days,
        max_chars=config.max_context_chars,
    )

    agent = _build_agent(
        CONSISTENCY_VALIDATOR_PROMPT,
        tools=tools,
        max_iterations=VALIDATOR_MAX_ITERATIONS,
    )

    user_input = (
        "Review the following Solution Designer output and Prototype Builder output. "
        "Validate their consistency and alignment.\n\n"
        "## Solution Designer Output:\n"
        f"{solution_md}\n\n"
        "## Prototype Builder Output:\n"
        f"{prototype_md}\n\n"
        "Produce your validation report now."
    )

    try:
        result = agent.invoke({"input": user_input})
        output_text = result.get("output") or ""

        # Parse the validation output
        validation_result = _parse_validation_output(output_text)

        metadata = RunMetadata(
            type="consistency_validator",
            started_at=start_time,
            ended_at=utils.utc_now_iso(),
            web_search_calls=search_state.count,
            sources_used_ids=search_state.source_ids or [],
            status="success",
            error=None,
        )
        validation_result["run_id"] = None  # Will be set when saved

        return validation_result, metadata.to_dict()

    except Exception as exc:
        metadata = RunMetadata(
            type="consistency_validator",
            started_at=start_time,
            ended_at=utils.utc_now_iso(),
            web_search_calls=search_state.count,
            sources_used_ids=search_state.source_ids or [],
            status="error",
            error=str(exc),
        )
        # Return a default validation result on error
        return {
            "is_valid": True,  # Don't block on validator errors
            "needs_revision": False,
            "revision_target": None,
            "issues": [],
            "summary": f"Validation failed with error: {exc}. Proceeding without validation.",
            "run_id": None,
        }, metadata.to_dict()


def _parse_validation_output(output_text: str) -> dict:
    """Parse validator output to extract structured validation result."""
    # Look for verdict indicators
    output_lower = output_text.lower()

    # Determine if revision is needed
    needs_revision = "revision needed" in output_lower or "🚫" in output_text
    is_valid = not needs_revision

    # Determine revision target
    revision_target = None
    if needs_revision:
        if "both" in output_lower:
            revision_target = "both"
        elif "prototype" in output_lower and "solution" not in output_lower:
            revision_target = "prototype"
        elif "solution" in output_lower:
            revision_target = "solution"
        else:
            revision_target = "both"  # Default to both if unclear

    # Count issues by severity
    issues = []
    lines = output_text.split("\n")
    for line in lines:
        if "✅" in line:
            issues.append({
                "severity": "success",
                "category": "alignment",
                "description": line.strip(),
                "recommendation": None,
            })
        elif "⚠️" in line:
            issues.append({
                "severity": "warning",
                "category": "concern",
                "description": line.strip(),
                "recommendation": None,
            })
        elif "🚫" in line:
            issues.append({
                "severity": "critical",
                "category": "critical",
                "description": line.strip(),
                "recommendation": None,
            })

    return {
        "is_valid": is_valid,
        "needs_revision": needs_revision,
        "revision_target": revision_target,
        "issues": issues,
        "summary": output_text,
        "run_id": None,
    }


def run_industry_research(case_id: str, assessment: dict) -> dict | None:
    """Run industry research and return results."""
    config = get_config()

    if not config.has_tavily_key:
        return None

    try:
        result = utils.conduct_industry_research(
            case_id=case_id,
            assessment=assessment,
            max_queries=INDUSTRY_RESEARCH_MAX_QUERIES,
            results_per_query=INDUSTRY_RESEARCH_RESULTS_PER_QUERY,
            recency_days=config.web_search_recency_days,
        )
        utils.save_industry_research(case_id, result)
        return result
    except Exception as exc:
        utils.log_event(
            case_id,
            {
                "event": "industry_research_failed",
                "error": str(exc),
                "timestamp": utils.utc_now_iso(),
            },
        )
        return None


def run_solution_designer_with_research(
    case_id: str, industry_research: str | None = None
) -> tuple[str | None, dict]:
    """Run Solution Designer with optional industry research context."""
    research_context = ""
    if industry_research:
        research_context = f"\n\n## Industry Research Context:\n{industry_research}\n\n"

    user_input = (
        "Generate the Solution Designer report. "
        "Always call read_case_context first. "
        f"{research_context}"
        "Use web_search only if you need additional information not covered in the research."
    )
    return _run_agent(
        agent_type="solution_designer",
        case_id=case_id,
        prompt_text=SOLUTION_DESIGNER_PROMPT,
        user_input=user_input,
        max_search_calls=SOLUTION_DESIGNER_MAX_SEARCH_CALLS,
        max_iterations=SOLUTION_DESIGNER_MAX_ITERATIONS,
    )


# ---- LangGraph workflow ----


def load_case_context_node(state: PlannerState) -> dict:
    """Load case context from disk into workflow state."""
    config = get_config()
    ctx = utils.read_case_context(state.case_id, max_chars=config.max_context_chars)
    return {
        "assessment": ctx["assessment"],
        "context_text": ctx["context_text"],
        "context_warnings": ctx["context_warnings"],
    }


def run_industry_research_node(state: PlannerState) -> dict:
    """Conduct industry research based on assessment."""
    result = run_industry_research(state.case_id, state.assessment)
    if result:
        return {
            "industry_research": result.get("research_context"),
            "industry_research_source_ids": result.get("source_ids", []),
        }
    return {"industry_research": None, "industry_research_source_ids": []}


def run_solution_node(state: PlannerState) -> dict:
    """Execute Solution Designer agent and persist results."""
    md, meta = run_solution_designer_with_research(
        state.case_id,
        industry_research=state.industry_research,
    )
    run_id = utils.save_run_meta(state.case_id, meta)
    external_sources = utils.load_external_sources(state.case_id)
    if md:
        utils.save_output_md(state.case_id, "solution_designer", md)
    return {"solution_md": md, "run_id_solution": run_id, "external_sources": external_sources}


def run_prototype_node(state: PlannerState) -> dict:
    """Execute Prototype Builder agent and persist results."""
    solution_md = state.solution_md or utils.load_latest_output(state.case_id, "solution_designer")
    md, meta = run_prototype_builder(
        state.case_id,
        solution_md=solution_md,
        industry_research=state.industry_research,
    )
    run_id = utils.save_run_meta(state.case_id, meta)
    external_sources = utils.load_external_sources(state.case_id)
    if md:
        utils.save_output_md(state.case_id, "prototype_builder", md)
    return {"prototype_md": md, "run_id_prototype": run_id, "external_sources": external_sources}


def run_validator_node(state: PlannerState) -> dict:
    """Execute Consistency Validator and determine if revision is needed."""
    solution_md = state.solution_md or utils.load_latest_output(state.case_id, "solution_designer")
    prototype_md = state.prototype_md or utils.load_latest_output(state.case_id, "prototype_builder")

    if not solution_md or not prototype_md:
        return {
            "validation_result": {
                "is_valid": False,
                "needs_revision": False,
                "summary": "Cannot validate: missing solution or prototype output.",
                "issues": [],
                "revision_target": None,
                "run_id": None,
            }
        }

    validation_result, meta = run_consistency_validator(
        state.case_id, solution_md, prototype_md
    )
    run_id = utils.save_run_meta(state.case_id, meta)
    validation_result["run_id"] = run_id

    # Save validation result
    utils.save_output_md(
        state.case_id,
        "validation_report",
        validation_result.get("summary", "No summary available"),
    )

    return {
        "validation_result": validation_result,
        "run_id_validator": run_id,
    }


def should_revise(state: PlannerState) -> str:
    """Determine next step based on validation result."""
    if state.needs_revision:
        return "revise"
    return "end"


def run_revision_node(state: PlannerState) -> dict:
    """Run revision cycle based on validator feedback."""
    validation_result = state.validation_result or {}
    revision_target = validation_result.get("revision_target", "both")
    feedback = validation_result.get("summary", "Please address alignment issues.")

    updates = {"revision_count": state.revision_count + 1}

    # Build revision feedback
    if revision_target in ("solution", "both"):
        focus = "- Address strategic gaps\n- Ensure recommendations are implementable"
        revision_prompt = REVISION_FEEDBACK_TEMPLATE.format(
            validator_feedback=feedback,
            focus_areas=focus,
        )
        md, meta = run_solution_designer_with_research(
            state.case_id,
            industry_research=f"{state.industry_research or ''}\n\n{revision_prompt}",
        )
        if md:
            utils.save_output_md(state.case_id, "solution_designer", md)
            updates["solution_md"] = md
        utils.save_run_meta(state.case_id, meta)

    if revision_target in ("prototype", "both"):
        focus = "- Align technical approach with solution recommendations\n- Address feasibility concerns"
        revision_prompt = REVISION_FEEDBACK_TEMPLATE.format(
            validator_feedback=feedback,
            focus_areas=focus,
        )
        solution_md = updates.get("solution_md") or state.solution_md
        md, meta = run_prototype_builder(
            state.case_id,
            solution_md=solution_md,
            industry_research=f"{state.industry_research or ''}\n\n{revision_prompt}",
        )
        if md:
            utils.save_output_md(state.case_id, "prototype_builder", md)
            updates["prototype_md"] = md
        utils.save_run_meta(state.case_id, meta)

    return updates


def build_graph() -> StateGraph:
    """Build the enhanced LangGraph workflow.

    Flow: Assessment → Research → Solution → Prototype → Validate → [Revise] → End
    """
    graph = StateGraph(PlannerState)

    # Add nodes
    graph.add_node("load_case_context", load_case_context_node)
    graph.add_node("industry_research", run_industry_research_node)
    graph.add_node("solution", run_solution_node)
    graph.add_node("prototype", run_prototype_node)
    graph.add_node("validate", run_validator_node)
    graph.add_node("revise", run_revision_node)

    # Define edges
    graph.add_edge(START, "load_case_context")
    graph.add_edge("load_case_context", "industry_research")
    graph.add_edge("industry_research", "solution")
    graph.add_edge("solution", "prototype")
    graph.add_edge("prototype", "validate")

    # Conditional edge: validate → revise or end
    graph.add_conditional_edges(
        "validate",
        should_revise,
        {"revise": "revise", "end": END},
    )

    # After revision, go back to validate (but revision_count prevents infinite loop)
    graph.add_edge("revise", "validate")

    return graph


def build_simple_graph() -> StateGraph:
    """Build the original simple workflow (without research/validation).

    For backwards compatibility or when running individual agents.
    """
    graph = StateGraph(PlannerState)
    graph.add_node("load_case_context", load_case_context_node)
    graph.add_node("solution", run_solution_node)
    graph.add_node("prototype", run_prototype_node)
    graph.add_edge(START, "load_case_context")
    graph.add_edge("load_case_context", "solution")
    graph.add_edge("solution", "prototype")
    graph.add_edge("prototype", END)
    return graph


def sidebar_case_manager() -> str | None:
    st.header("Case Manager")

    try:
        cases = utils.list_cases()
    except Exception as exc:
        st.error(f"Failed to load cases: {exc}")
        return None

    cases_by_id = {c.get("case_id"): c for c in cases if isinstance(c, dict)}
    case_id_options = [""] + [c["case_id"] for c in cases if "case_id" in c]

    if "selected_case_id" not in st.session_state:
        st.session_state["selected_case_id"] = ""

    selected_case_id = st.selectbox(
        "Open case",
        options=case_id_options,
        index=case_id_options.index(st.session_state["selected_case_id"])
        if st.session_state["selected_case_id"] in case_id_options
        else 0,
        format_func=lambda cid: "-- Select a case --"
        if cid == ""
        else f"{cases_by_id.get(cid, {}).get('name', cid)} [{cid}]",
    )
    st.session_state["selected_case_id"] = selected_case_id

    with st.expander("New case", expanded=False):
        with st.form("new_case_form", clear_on_submit=True):
            name = st.text_input("Name*", placeholder="e.g., Vendor onboarding")
            area = st.text_input("Area", placeholder="e.g., Procurement")
            tags_raw = st.text_input("Tags (comma-separated)", placeholder="e.g., bpm, automation")
            description = st.text_area("Description (optional)")
            submitted = st.form_submit_button("Create case")

        if submitted:
            try:
                tags = [t.strip() for t in tags_raw.split(",")] if tags_raw else []
                meta = utils.create_case(
                    name=name,
                    area=area,
                    tags=tags,
                    description=description if description.strip() else None,
                )
            except Exception as exc:
                st.error(f"Failed to create case: {exc}")
            else:
                st.success(f"Created case: {meta['case_id']}")
                st.session_state["selected_case_id"] = meta["case_id"]
                st.rerun()

    if selected_case_id:
        try:
            meta = utils.load_case_meta(selected_case_id)
            status = utils.case_status(selected_case_id)
        except Exception as exc:
            st.error(f"Failed to load case details: {exc}")
        else:
            st.caption(f"case_id: {meta.get('case_id')}")
            st.caption(f"created_at: {meta.get('created_at')}")
            st.caption(f"updated_at: {meta.get('updated_at')}")
            st.caption(f"area: {meta.get('area')}")
            st.caption(f"tags: {', '.join(meta.get('tags') or [])}")

            st.divider()
            st.subheader("Status")
            st.write(
                {
                    "has_assessment": status["has_assessment"],
                    "has_solution_output": status["has_solution_output"],
                    "has_prototype_output": status["has_prototype_output"],
                }
            )

            with st.expander("Danger zone", expanded=False):
                confirm = st.checkbox(
                    "I understand this will delete the case folder from disk.",
                    value=False,
                )
                if st.button(
                    "Delete case",
                    type="primary",
                    disabled=not confirm,
                ):
                    try:
                        utils.delete_case(selected_case_id)
                    except Exception as exc:
                        st.error(f"Failed to delete case: {exc}")
                    else:
                        st.success("Case deleted.")
                        st.session_state["selected_case_id"] = ""
                        st.rerun()

    return selected_case_id if selected_case_id else None


def _render_environment_status() -> None:
    """Render environment configuration status in sidebar."""
    config = get_config()
    st.header("Environment")
    st.write(
        {
            "OPENAI_API_KEY": "set" if config.has_openai_key else "missing",
            "TAVILY_API_KEY": "set" if config.has_tavily_key else "missing",
            "OPENAI_MODEL": config.openai_model,
            "MAX_CONTEXT_CHARS": config.max_context_chars,
            "WEB_SEARCH_RECENCY_DAYS": config.web_search_recency_days,
            "WEB_SEARCH_MAX_RESULTS": config.web_search_max_results,
        }
    )


def _build_export_files(
    solution_text: str | None,
    prototype_text: str | None,
    validation_text: str | None = None,
) -> dict[str, str]:
    """Build the export files from agent outputs."""
    sol = solution_text or "Solution Designer output not available."
    proto = prototype_text or "Prototype Builder output not available."

    executive = "# Executive Brief\n\n" + sol
    technical = "# Technical Blueprint\n\n" + proto

    # Build implementation plan with validation summary if available
    impl_sections = [
        "## Phases",
        "- Phase 1: Stand up the fixed workflow (LangGraph) and case manager.",
        "- Phase 2: Assessment ingestion and context building.",
        "- Phase 3: Agents with web_search guardrails and reproducible outputs.",
        "- Phase 4: Export pack and release hardening.",
        "",
        "## Risks and mitigations",
        "- LLM variability: use fixed prompts/headings and low temperature.",
        "- Context limits: enforce MAX_CONTEXT_CHARS and warn on truncation.",
        "- Web search availability: log failures and allow offline runs.",
    ]

    if validation_text:
        impl_sections.extend([
            "",
            "## Quality Validation",
            validation_text,
        ])

    implementation = "# Implementation Plan\n\n" + "\n".join(impl_sections)

    return {
        "Executive_Brief.md": executive,
        "Technical_Blueprint.md": technical,
        "Implementation_Plan.md": implementation,
    }


# ---- Streamlit UI Tab Renderers ----


def _render_assessment_tab(case_id: str, config: Config) -> None:
    """Render the Assessment Wizard tab content."""

    meta = utils.load_case_meta(case_id)
    existing_assessment = utils.load_assessment(case_id)
    defaults = utils.flatten_assessment_for_display(existing_assessment)

    st.caption("Fill the assessment and save. This will also rebuild context.txt.")
    st.caption(f"Last saved (updated_at): {meta.get('updated_at', '-')}")

    with st.form("assessment_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            process_name = st.text_input(
                "Process name / area", value=defaults["process_name"]
            )
            process_objective = st.text_input(
                "Process objective", value=defaults["process_objective"]
            )
            current_workflow = st.text_area(
                "Current workflow description (main steps)",
                value=defaults["current_workflow"],
                height=120,
            )
            actors = st.text_input("Involved actors/teams", value=defaults["actors"])
            systems = st.text_input(
                'Systems used today (accept "I don\'t know")',
                value=defaults["systems"],
            )
        with col2:
            volume_frequency = st.text_input(
                "Volume/frequency", value=defaults["volume_frequency"]
            )
            slas = st.text_input("SLAs/lead time", value=defaults["slas"])
            bottlenecks = st.text_area(
                "Main bottlenecks/pain points",
                value=defaults["bottlenecks"],
                height=120,
            )
            risks_compliance = st.text_area(
                "Risks/compliance (if any)",
                value=defaults["risks_compliance"],
                height=120,
            )
            desired_outcome_kpis = st.text_area(
                "Desired outcome and KPIs (or estimates)",
                value=defaults["desired_outcome_kpis"],
                height=120,
            )

        submitted = st.form_submit_button("Save assessment")

    if submitted:
        payload = utils.normalize_assessment_input(
            {
                "process_name": process_name,
                "process_objective": process_objective,
                "current_workflow": current_workflow,
                "actors": actors,
                "systems": systems,
                "volume_frequency": volume_frequency,
                "slas": slas,
                "bottlenecks": bottlenecks,
                "risks_compliance": risks_compliance,
                "desired_outcome_kpis": desired_outcome_kpis,
            }
        )
        try:
            utils.save_assessment(case_id, payload)
            utils.save_assessment_snapshot(case_id, payload)
            ctx = utils.rebuild_context_txt(case_id, max_chars=config.max_context_chars)
        except Exception as exc:
            st.error(f"Failed to save assessment: {exc}")
        else:
            st.success("Assessment saved and context.txt rebuilt.")
            if ctx["warnings"]:
                st.warning("\n".join(ctx["warnings"]))

    # Attachments section
    st.divider()
    st.subheader("Attachments")
    uploaded_files = st.file_uploader(
        "Upload attachments (PDF/TXT/MD/PNG/JPG)",
        type=["pdf", "txt", "md", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        all_warnings: list[str] = []
        for f in uploaded_files:
            try:
                saved_path = utils.save_attachment(case_id, f)
                text, warns = utils.extract_text_from_attachment(str(saved_path))
                utils.save_attachment_text(case_id, Path(saved_path).name, text)
                all_warnings.extend(warns)
            except Exception as exc:
                all_warnings.append(f"{f.name}: failed to process ({exc})")
        ctx = utils.rebuild_context_txt(case_id, max_chars=config.max_context_chars)
        all_warnings.extend(ctx["warnings"])
        st.success("Attachments saved and context.txt rebuilt.")
        if all_warnings:
            st.warning("\n".join(all_warnings))

    attachments = utils.list_attachments(case_id)
    if attachments:
        st.write("Existing attachments:")
        for item in attachments:
            st.write(
                f"- {item['name']} ({item['size_bytes']} bytes, "
                f"modified {item['modified_at']})"
            )
    else:
        st.info("No attachments uploaded yet.")

    # Context section
    st.divider()
    st.subheader("Context")
    ctx_path = Path(utils.case_dir(case_id)) / "context.txt"
    if ctx_path.exists():
        context_text = ctx_path.read_text(encoding="utf-8", errors="ignore")
        st.caption(
            f"context.txt size: {len(context_text)} chars "
            f"(max {config.max_context_chars})"
        )
        st.text_area("context.txt (read-only)", value=context_text, height=200)
    else:
        st.info("context.txt not generated yet.")


def _render_solution_tab(case_id: str, config: Config) -> None:
    """Render the Solution Designer tab content."""
    # Industry Research Section
    st.subheader("📚 Industry Research")
    existing_research = utils.load_industry_research(case_id)

    if existing_research:
        with st.expander(
            f"✅ Research completed ({existing_research.get('total_sources_found', 0)} sources)",
            expanded=False,
        ):
            st.markdown(existing_research.get("research_context", "No content"))
            st.caption(f"Queries: {', '.join(existing_research.get('queries_executed', []))}")
    else:
        st.info("No industry research yet. Run research before Solution Designer for better results.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "🔍 Run Industry Research",
            disabled=not config.has_tavily_key,
            help="Searches for relevant tools, best practices, and compliance info",
        ):
            assessment = utils.load_assessment(case_id)
            if not assessment:
                st.warning("Please complete the assessment first.")
            else:
                with st.spinner("Conducting industry research (3-4 searches)..."):
                    result = run_industry_research(case_id, assessment)
                    if result:
                        st.success(f"Research complete! Found {result.get('total_sources_found', 0)} sources.")
                        st.rerun()
                    else:
                        st.warning("Research completed with no results. Check Tavily API key.")

    st.divider()

    # Solution Designer Section
    st.subheader("💡 Solution Designer Output")
    latest_solution = utils.load_latest_output(case_id, "solution_designer")
    if latest_solution:
        st.markdown(latest_solution)
    else:
        st.info("No solution output yet.")

    if not config.has_openai_key:
        st.error("OPENAI_API_KEY not set. Add it to your .env to run.")

    with col2:
        if st.button(
            "▶️ Run Solution Designer",
            type="primary",
            disabled=not config.has_openai_key,
        ):
            research_context = None
            if existing_research:
                research_context = existing_research.get("research_context")

            with st.spinner("Running Solution Designer..."):
                md, meta = run_solution_designer_with_research(case_id, research_context)
                run_id = utils.save_run_meta(case_id, meta)
                if md:
                    utils.save_output_md(case_id, "solution_designer", md)
                    st.success(f"Solution Designer complete (run_id={run_id}).")
                    st.rerun()
                else:
                    st.error(f"Solution Designer failed: {meta.get('error')}")


def _render_prototype_tab(case_id: str, config: Config) -> None:
    """Render the Prototype Builder tab content."""
    latest_prototype = utils.load_latest_output(case_id, "prototype_builder")

    st.subheader("🏗️ Prototype Builder Output")
    if latest_prototype:
        st.markdown(latest_prototype)
    else:
        st.info("No prototype output yet. Run Solution Designer first.")

    if not config.has_openai_key:
        st.error("OPENAI_API_KEY not set. Add it to your .env to run.")

    solution_md = utils.load_latest_output(case_id, "solution_designer")
    if not solution_md:
        st.warning("⚠️ Run Solution Designer first for best results.")

    if st.button(
        "▶️ Run Prototype Builder",
        type="primary",
        disabled=not config.has_openai_key,
    ):
        existing_research = utils.load_industry_research(case_id)
        research_context = existing_research.get("research_context") if existing_research else None

        with st.spinner("Running Prototype Builder..."):
            md, meta = run_prototype_builder(
                case_id,
                solution_md=solution_md,
                industry_research=research_context,
            )
            run_id = utils.save_run_meta(case_id, meta)
            if md:
                utils.save_output_md(case_id, "prototype_builder", md)
                st.success(f"Prototype Builder complete (run_id={run_id}).")
                st.rerun()
            else:
                st.error(f"Prototype Builder failed: {meta.get('error')}")

    # Validation Section
    st.divider()
    st.subheader("✅ Consistency Validation")

    validation_report = utils.load_latest_output(case_id, "validation_report")

    if validation_report:
        # Parse for display
        if "APPROVED" in validation_report.upper() or "🚫" not in validation_report:
            st.success("Validation Status: **APPROVED** ✅")
        else:
            st.warning("Validation Status: **REVISION RECOMMENDED** ⚠️")

        with st.expander("View Validation Report", expanded=False):
            st.markdown(validation_report)
    else:
        st.info("No validation report yet. Run validation after both agents complete.")

    if st.button(
        "🔍 Run Consistency Validator",
        disabled=not (config.has_openai_key and solution_md and latest_prototype),
        help="Checks alignment between Solution and Prototype outputs",
    ):
        with st.spinner("Validating consistency..."):
            validation_result, meta = run_consistency_validator(
                case_id, solution_md, latest_prototype
            )
            run_id = utils.save_run_meta(case_id, meta)
            utils.save_output_md(
                case_id,
                "validation_report",
                validation_result.get("summary", "No summary"),
            )
            st.success(f"Validation complete (run_id={run_id}).")
            st.rerun()


def _render_export_tab(case_id: str) -> None:
    """Render the Export Pack tab content."""
    solution_md = utils.load_latest_output(case_id, "solution_designer")
    prototype_md = utils.load_latest_output(case_id, "prototype_builder")
    validation_md = utils.load_latest_output(case_id, "validation_report")

    # Status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        if solution_md:
            st.success("✅ Solution Designer")
        else:
            st.warning("⏳ Solution Designer")
    with col2:
        if prototype_md:
            st.success("✅ Prototype Builder")
        else:
            st.warning("⏳ Prototype Builder")
    with col3:
        if validation_md:
            st.success("✅ Validation Complete")
        else:
            st.info("ℹ️ Validation (optional)")

    if not solution_md or not prototype_md:
        st.warning("Run Solution Designer and Prototype Builder before exporting.")

    if st.button(
        "📦 Generate Export Pack",
        type="primary",
        disabled=not (solution_md and prototype_md),
    ):
        ts = utils.utc_now_iso().replace(":", "-")
        files = _build_export_files(solution_md, prototype_md, validation_md)
        export_dir = utils.write_export_pack(case_id, ts, files)
        zip_path = utils.make_export_zip(export_dir)
        st.success(f"Export generated: {export_dir}")
        with open(zip_path, "rb") as f:
            st.download_button(
                "⬇️ Download Export ZIP",
                data=f.read(),
                file_name=Path(zip_path).name,
                mime="application/zip",
            )

    exports = utils.list_exports(case_id)
    if exports:
        st.write("Previous exports:")
        for item in exports:
            st.markdown(f"- {item['export_ts']}")
            for md_file in item["files"]:
                st.markdown(f"  - {Path(md_file).name}")
            if item.get("zip"):
                with open(item["zip"], "rb") as f:
                    st.download_button(
                        f"Download {Path(item['zip']).name}",
                        data=f.read(),
                        file_name=Path(item["zip"]).name,
                        mime="application/zip",
                        key=f"zip-{item['export_ts']}",
                    )
    else:
        st.info("No exports yet.")


def _render_workflow_section(case_id: str, config: Config) -> None:
    """Render the full workflow execution section."""
    st.caption(
        "Enhanced workflow: Assessment → Research → Solution → Prototype → Validation → Export"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "🚀 Run Full Enhanced Workflow",
            type="primary",
            disabled=not config.has_openai_key,
            help="Runs research, both agents, and validation automatically",
        ):
            progress = st.progress(0, text="Starting workflow...")

            with st.spinner("Running enhanced workflow..."):
                graph = build_graph()
                app = graph.compile()
                state = PlannerState(case_id=case_id, max_revisions=MAX_REVISION_CYCLES)

                progress.progress(10, text="Loading context...")
                result = app.invoke(state)
                progress.progress(100, text="Workflow complete!")

                if result.get("prototype_md") or result.get("solution_md"):
                    st.success("✅ Workflow completed successfully!")

                    # Show summary
                    validation = result.get("validation_result", {})
                    if validation:
                        issues = validation.get("issues", [])
                        success_count = sum(1 for i in issues if i.get("severity") == "success")
                        warning_count = sum(1 for i in issues if i.get("severity") == "warning")
                        critical_count = sum(1 for i in issues if i.get("severity") == "critical")
                        st.info(
                            f"Validation: ✅ {success_count} aligned, "
                            f"⚠️ {warning_count} concerns, "
                            f"🚫 {critical_count} critical"
                        )

                    with st.expander("Solution Designer Output", expanded=False):
                        st.markdown(result.get("solution_md", "No output"))

                    with st.expander("Prototype Builder Output", expanded=False):
                        st.markdown(result.get("prototype_md", "No output"))

                    if result.get("industry_research"):
                        with st.expander("Industry Research", expanded=False):
                            st.markdown(result.get("industry_research"))
                else:
                    st.error(f"Workflow failed. Errors: {result.get('errors')}")

    with col2:
        if st.button(
            "⚡ Run Simple Workflow (No Research/Validation)",
            type="secondary",
            disabled=not config.has_openai_key,
            help="Original workflow without research and validation",
        ):
            with st.spinner("Running simple workflow..."):
                graph = build_simple_graph()
                app = graph.compile()
                state = PlannerState(case_id=case_id)
                result = app.invoke(state)
                if result.get("prototype_md") or result.get("solution_md"):
                    st.success("Workflow completed.")
                    st.rerun()
                else:
                    st.error(f"Workflow failed. Errors: {result.get('errors')}")


def main() -> None:
    st.set_page_config(page_title="Workflow Improvement Planner", layout="wide")

    config = load_config()
    utils.ensure_index()

    st.title("🔄 Workflow Improvement Planner")
    st.caption(
        "Enhanced agentic workflow: Assessment → Industry Research → "
        "Solution Designer → Prototype Builder → Consistency Validation → Export"
    )

    with st.sidebar:
        selected_case_id = sidebar_case_manager()

        st.divider()
        _render_environment_status()

    assessment_tab, solution_tab, prototype_tab, export_tab = st.tabs(
        ["📝 Assessment", "💡 Solution Designer", "🏗️ Prototype Builder", "📦 Export"]
    )

    with assessment_tab:
        st.subheader("Assessment Wizard")
        if not selected_case_id:
            st.info("Select or create a case in the sidebar to begin.")
        else:
            _render_assessment_tab(selected_case_id, config)

    with solution_tab:
        st.subheader("Solution Designer (Agent 1)")
        if not selected_case_id:
            st.info("Select or create a case in the sidebar to begin.")
        else:
            _render_solution_tab(selected_case_id, config)

    with prototype_tab:
        st.subheader("Prototype Builder (Agent 2)")
        if not selected_case_id:
            st.info("Select or create a case in the sidebar to begin.")
        else:
            _render_prototype_tab(selected_case_id, config)

    with export_tab:
        st.subheader("Export Pack")
        if not selected_case_id:
            st.info("Select or create a case in the sidebar to begin.")
        else:
            _render_export_tab(selected_case_id)

    st.divider()
    st.subheader("Run full workflow (LangGraph)")
    if not selected_case_id:
        st.info("Select or create a case to run the workflow.")
    else:
        _render_workflow_section(selected_case_id, config)


if __name__ == "__main__":
    main()
