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


# ---- Pydantic models for LangGraph state ----


class PlannerState(BaseModel):
    """State model for the LangGraph workflow."""

    case_id: str

    assessment: dict = Field(default_factory=dict)
    context_text: str = ""
    context_warnings: list[str] = Field(default_factory=list)

    solution_md: str | None = None
    prototype_md: str | None = None

    external_sources: list[ExternalSource] = Field(default_factory=list)

    run_id_solution: str | None = None
    run_id_prototype: str | None = None
    export_ts: str | None = None

    errors: list[str] = Field(default_factory=list)

    def append_error(self, msg: str) -> None:
        """Append an error message to the errors list."""
        self.errors.append(msg)
