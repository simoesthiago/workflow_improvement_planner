from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, Field


# ---- TypedDicts for structured dictionaries ----


class CaseMeta(TypedDict):
    """Metadata for a case stored in case_meta.json."""

    case_id: str
    name: str
    area: str
    tags: list[str]
    description: str | None
    created_at: str
    updated_at: str


class CaseStatus(TypedDict):
    """Status flags for a case."""

    has_assessment: bool
    has_solution_output: bool
    has_prototype_output: bool


class AttachmentInfo(TypedDict):
    """Information about an uploaded attachment."""

    name: str
    path: str
    size_bytes: int
    modified_at: str


class ExternalSource(TypedDict):
    """External source from web search."""

    source_id: str
    query: str
    title: str | None
    url: str | None
    snippet: str | None
    retrieved_at: str
    used_in_run: str


class ContextResult(TypedDict):
    """Result from rebuilding context.txt."""

    context_text: str
    warnings: list[str]
    attachments_meta: list[AttachmentInfo]


class ExportInfo(TypedDict):
    """Information about an export pack."""

    export_ts: str
    dir: str
    zip: str | None
    files: list[str]


class WebSearchResult(TypedDict):
    """Result from a web search call."""

    title: str | None
    url: str | None
    snippet: str | None
    retrieved_at: str


class WebSearchResponse(TypedDict):
    """Response from the web_search function."""

    results: list[WebSearchResult]
    source_ids: list[str]
    retrieved_at: str


class IndustryResearchResult(TypedDict):
    """Result from industry research function."""

    research_context: str
    source_ids: list[str]
    queries_executed: list[str]
    total_sources_found: int


class ValidationIssue(TypedDict):
    """A single validation issue found by the Consistency Validator."""

    severity: str  # "success", "warning", "critical"
    category: str  # e.g., "alignment", "feasibility", "completeness"
    description: str
    recommendation: str | None


class ValidationResult(TypedDict):
    """Result from the Consistency Validator agent."""

    is_valid: bool
    needs_revision: bool
    revision_target: str | None  # "solution", "prototype", "both", or None
    issues: list[ValidationIssue]
    summary: str
    run_id: str | None


class AssessmentData(TypedDict, total=False):
    """Assessment data structure with 9 sections and ~30 fields."""

    # Section 1: Workflow Identification
    workflow_name: str
    owning_department: str
    primary_people_involved: str

    # Section 2: Current State
    current_process_steps: str
    trigger: str
    frequency: str
    time_consumed: str

    # Section 3: Challenges
    primary_challenges: list[str]  # Multi-select checkboxes
    definition_of_success: str
    ideal_outcome: str

    # Section 4: Inputs & Outputs
    workflow_inputs: str
    workflow_outputs: str
    required_output_format: str

    # Section 5: Requirements & Limitations
    quality_standards: str
    existing_templates: str
    current_tools_systems: str
    time_constraints: str
    budget_constraints: str

    # Section 6: Technical Details
    file_types: list[str]  # Multi-select checkboxes
    other_file_types: str
    data_sources: str
    privacy_security: str

    # Section 7: Decision Points
    decision_points: str
    decision_criteria: str

    # Section 8: Success Metrics
    success_metrics: list[str]  # Multi-select checkboxes

    # Section 9: Additional Context
    additional_context: str
    questions_ai_assistance: str
    sample_data_availability: str


# ---- Pydantic models for LangGraph state ----


class PlannerState(BaseModel):
    """State model for the LangGraph workflow."""

    case_id: str

    # Assessment and context
    assessment: dict = Field(default_factory=dict)
    context_text: str = ""
    context_warnings: list[str] = Field(default_factory=list)

    # Industry research (new)
    industry_research: str | None = None
    industry_research_source_ids: list[str] = Field(default_factory=list)

    # Agent outputs
    solution_md: str | None = None
    prototype_md: str | None = None

    # Validation results (new)
    validation_result: dict | None = None  # ValidationResult as dict
    revision_count: int = 0
    max_revisions: int = 1

    # Source tracking
    external_sources: list[ExternalSource] = Field(default_factory=list)

    # Run tracking
    run_id_solution: str | None = None
    run_id_prototype: str | None = None
    run_id_validator: str | None = None
    export_ts: str | None = None

    # Error handling
    errors: list[str] = Field(default_factory=list)

    def append_error(self, msg: str) -> None:
        """Append an error message to the errors list."""
        self.errors.append(msg)

    @property
    def needs_revision(self) -> bool:
        """Check if validation flagged critical issues requiring revision."""
        if not self.validation_result:
            return False
        return (
            self.validation_result.get("needs_revision", False)
            and self.revision_count < self.max_revisions
        )

    @property
    def revision_target(self) -> str | None:
        """Get which agent(s) need to revise."""
        if not self.validation_result:
            return None
        return self.validation_result.get("revision_target")
