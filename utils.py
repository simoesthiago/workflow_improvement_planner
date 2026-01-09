"""Utility functions for case management, persistence, and external integrations.

This module provides:
- Case CRUD operations (create, read, update, delete)
- Assessment persistence and versioning
- Attachment handling and text extraction
- Context building for LLM consumption
- External source (web search) management
- Export pack generation
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from schemas import (
        AttachmentInfo,
        CaseMeta,
        CaseStatus,
        ContextResult,
        ExportInfo,
        ExternalSource,
        IndustryResearchResult,
        WebSearchResponse,
        WebSearchResult,
    )

DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "index.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    tmp_path.write_text(content, encoding=encoding, newline="\n")
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_index() -> None:
    ensure_data_dir()
    if INDEX_PATH.exists():
        return
    atomic_write_json(INDEX_PATH, {"cases": []})


def case_dir(case_id: str) -> Path:
    return DATA_DIR / case_id


def _ensure_case_dirs(base_dir: Path) -> None:
    (base_dir / "assessment_versions").mkdir(parents=True, exist_ok=True)
    (base_dir / "attachments").mkdir(parents=True, exist_ok=True)
    (base_dir / "attachments_text").mkdir(parents=True, exist_ok=True)
    (base_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (base_dir / "runs").mkdir(parents=True, exist_ok=True)
    (base_dir / "sources").mkdir(parents=True, exist_ok=True)
    (base_dir / "exports").mkdir(parents=True, exist_ok=True)
    (base_dir / "logs").mkdir(parents=True, exist_ok=True)


def list_cases() -> list[CaseMeta]:
    """List all cases, sorted by most recently updated first."""
    ensure_index()
    index = read_json(INDEX_PATH)
    cases = index.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("data/index.json is invalid (expected key 'cases' to be a list).")
    return sorted(
        cases,
        key=lambda c: (c.get("updated_at") or c.get("created_at") or ""),
        reverse=True,
    )


def _write_index(cases: list[dict]) -> None:
    atomic_write_json(INDEX_PATH, {"cases": cases})


def _generate_case_id() -> str:
    date_part = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    suffix = secrets.token_hex(2)  # 4 hex chars
    return f"case_{date_part}_{suffix}"


def create_case(name: str, area: str, tags: list[str], description: str | None) -> CaseMeta:
    ensure_index()

    if not name.strip():
        raise ValueError("Case name is required.")

    now = utc_now_iso()
    case_id = _generate_case_id()
    for _ in range(5):
        if not case_dir(case_id).exists():
            break
        case_id = _generate_case_id()
    else:
        raise RuntimeError("Failed to generate a unique case_id.")

    meta = {
        "case_id": case_id,
        "name": name.strip(),
        "area": area.strip(),
        "tags": [t.strip() for t in tags if t.strip()],
        "description": description.strip() if description and description.strip() else None,
        "created_at": now,
        "updated_at": now,
    }

    base_dir = case_dir(case_id)
    base_dir.mkdir(parents=True, exist_ok=False)
    _ensure_case_dirs(base_dir)
    atomic_write_json(base_dir / "case_meta.json", meta)

    cases = list_cases()
    cases.append(meta)
    _write_index(cases)

    return meta


def load_case_meta(case_id: str) -> CaseMeta:
    """Load case metadata from disk."""
    path = case_dir(case_id) / "case_meta.json"
    if not path.exists():
        raise FileNotFoundError(f"case_meta.json not found for case_id={case_id}")
    meta = read_json(path)
    if not isinstance(meta, dict):
        raise ValueError(f"case_meta.json is invalid for case_id={case_id}")
    return meta


def update_case_updated_at(case_id: str) -> None:
    ensure_index()
    now = utc_now_iso()

    meta_path = case_dir(case_id) / "case_meta.json"
    meta = read_json(meta_path)
    if not isinstance(meta, dict):
        raise ValueError(f"case_meta.json is invalid for case_id={case_id}")
    meta["updated_at"] = now
    atomic_write_json(meta_path, meta)

    cases = list_cases()
    updated = False
    for c in cases:
        if c.get("case_id") == case_id:
            c["updated_at"] = now
            updated = True
            break

    if updated:
        _write_index(cases)


def delete_case(case_id: str) -> None:
    ensure_index()
    target_dir = case_dir(case_id)
    if target_dir.exists():
        shutil.rmtree(target_dir)

    cases = [c for c in list_cases() if c.get("case_id") != case_id]
    _write_index(cases)


def case_status(case_id: str) -> CaseStatus:
    """Get status flags for a case."""
    base_dir = case_dir(case_id)
    return {
        "has_assessment": (base_dir / "assessment.json").exists(),
        "has_solution_output": (base_dir / "outputs" / "solution_designer.md").exists(),
        "has_prototype_output": (base_dir / "outputs" / "prototype_builder.md").exists(),
    }


# ---- Assessment persistence ----

# Multi-select field options
PRIMARY_CHALLENGES_OPTIONS = [
    "Time-intensive",
    "Manual or repetitive tasks",
    "Scattered information",
    "Quality or consistency issues",
    "Communication delays",
    "Scalability limitations",
    "Knowledge dependencies",
    "Other",
]

FILE_TYPES_OPTIONS = [
    "Text documents",
    "Email",
    "Images/media",
    "Spreadsheets",
    "Surveys",
    "Presentations",
    "PDFs",
    "Database",
    "Web content",
]

SUCCESS_METRICS_OPTIONS = [
    "Time saved",
    "Quality improvement",
    "Consistency",
    "Staff satisfaction",
    "Capacity gain",
    "Error reduction",
    "Turnaround time",
    "Other",
]

ASSESSMENT_FIELDS_ORDER: tuple[str, ...] = (
    # Section 1: Workflow Identification
    "workflow_name",
    "owning_department",
    "primary_people_involved",
    # Section 2: Current State
    "current_process_steps",
    "trigger",
    "frequency",
    "time_consumed",
    # Section 3: Challenges
    "primary_challenges",
    "definition_of_success",
    "ideal_outcome",
    # Section 4: Inputs & Outputs
    "workflow_inputs",
    "workflow_outputs",
    "required_output_format",
    # Section 5: Requirements & Limitations
    "quality_standards",
    "existing_templates",
    "current_tools_systems",
    "time_constraints",
    "budget_constraints",
    # Section 6: Technical Details
    "file_types",
    "other_file_types",
    "data_sources",
    "privacy_security",
    # Section 7: Decision Points
    "decision_points",
    "decision_criteria",
    # Section 8: Success Metrics
    "success_metrics",
    # Section 9: Additional Context
    "additional_context",
    "questions_ai_assistance",
    "sample_data_availability",
)


def save_assessment(case_id: str, payload: dict) -> None:
    """Persist assessment and bump updated_at."""
    base_dir = case_dir(case_id)
    atomic_write_json(base_dir / "assessment.json", payload)
    update_case_updated_at(case_id)


def save_assessment_snapshot(case_id: str, payload: dict) -> str:
    """Versioned snapshot for history."""
    ts = utc_now_iso().replace(":", "-")
    filename = f"{ts}.json"
    base_dir = case_dir(case_id)
    snapshot_path = base_dir / "assessment_versions" / filename
    atomic_write_json(snapshot_path, payload)
    return filename


def load_assessment(case_id: str) -> dict | None:
    path = case_dir(case_id) / "assessment.json"
    if not path.exists():
        return None
    data = read_json(path)
    return data if isinstance(data, dict) else None


# ---- Attachments and context ----


def _sanitize_name(name: str) -> str:
    return Path(name).name


def _unique_attachment_name(base_dir: Path, name: str) -> str:
    target = Path(name).name
    candidate = target
    stem, suffix = Path(target).stem, Path(target).suffix
    counter = 1
    while (base_dir / candidate).exists():
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def save_attachment(case_id: str, uploaded_file) -> Path:
    """Save uploaded file bytes into attachments/ and return the saved path."""
    base_dir = case_dir(case_id)
    attachments_dir = base_dir / "attachments"
    attachments_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _unique_attachment_name(attachments_dir, _sanitize_name(uploaded_file.name))
    target = attachments_dir / safe_name
    # Streamlit UploadedFile supports getbuffer/read; fall back to .getvalue
    if hasattr(uploaded_file, "getbuffer"):
        data = uploaded_file.getbuffer()
    elif hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
    else:
        raise ValueError("Unsupported uploaded file object.")
    target.write_bytes(bytes(data))
    return target


def save_attachment_text(case_id: str, attachment_name: str, text: str) -> Path:
    base_dir = case_dir(case_id)
    out_path = base_dir / "attachments_text" / f"{Path(attachment_name).name}.txt"
    atomic_write_text(out_path, text)
    return out_path


def list_attachments(case_id: str) -> list[AttachmentInfo]:
    """List all attachments for a case."""
    base_dir = case_dir(case_id)
    attachments_dir = base_dir / "attachments"
    if not attachments_dir.exists():
        return []
    items: list[AttachmentInfo] = []
    for path in sorted(attachments_dir.iterdir()):
        if path.is_file():
            stat = path.stat()
            items.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
    return items


def extract_text_from_attachment(file_path: str) -> tuple[str, list[str]]:
    """Extract text best-effort and return (text, warnings)."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    warnings: list[str] = []

    try:
        if suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                warnings.append("pypdf not installed; cannot extract PDF text.")
                return "", warnings

            reader = PdfReader(path)
            parts: list[str] = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                parts.append(page_text)
            text = "\n".join(parts)
            if not text.strip():
                warnings.append("PDF text extraction returned empty content (scanned PDF?).")
        elif suffix in {".png", ".jpg", ".jpeg"}:
            warnings.append("No OCR enabled for images; content not extracted.")
            text = ""
        else:
            warnings.append(f"Unsupported attachment type: {suffix}")
            text = ""
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Failed to extract text: {exc}")
        text = ""

    return text, warnings


def rebuild_context_txt(case_id: str, max_chars: int = 60000) -> ContextResult:
    """Build context.txt from assessment + extracted attachment texts."""
    base_dir = case_dir(case_id)
    context_path = base_dir / "context.txt"
    warnings: list[str] = []
    parts: list[str] = []

    assessment = load_assessment(case_id) or {}
    if assessment:
        parts.append("== Assessment ==")
        for key in ASSESSMENT_FIELDS_ORDER:
            if key in assessment:
                value = assessment[key]
                if isinstance(value, list):
                    # Multi-select: join with commas
                    formatted_value = ", ".join(value) if value else "(none selected)"
                    parts.append(f"{key}: {formatted_value}")
                elif value:
                    parts.append(f"{key}: {value}")
        for key, value in assessment.items():
            if key not in ASSESSMENT_FIELDS_ORDER:
                if isinstance(value, list):
                    formatted_value = ", ".join(str(v) for v in value)
                    parts.append(f"{key}: {formatted_value}")
                else:
                    parts.append(f"{key}: {value}")
    else:
        warnings.append("No assessment found; context uses attachments only.")

    attachments_text_dir = base_dir / "attachments_text"
    attachments_texts: list[tuple[str, str]] = []
    if attachments_text_dir.exists():
        for text_path in sorted(attachments_text_dir.iterdir()):
            if text_path.is_file():
                attachments_texts.append(
                    (text_path.name, text_path.read_text(encoding="utf-8", errors="ignore"))
                )

    if attachments_texts:
        parts.append("== Attachments (extracted text) ==")
        for name, text in attachments_texts:
            parts.append(f"[{name}]")
            parts.append(text)
    else:
        warnings.append("No attachment text available.")

    combined = "\n\n".join(parts).strip()
    if len(combined) > max_chars:
        warnings.append(f"context.txt truncated to {max_chars} characters.")
        combined = combined[:max_chars]

    atomic_write_text(context_path, combined)

    attachments_meta = list_attachments(case_id)
    return {"context_text": combined, "warnings": warnings, "attachments_meta": attachments_meta}


# ---- Helpers for assessment form ----


def normalize_assessment_input(form_data: dict) -> dict:
    """Strip whitespace and keep only known keys. Handle both string and list fields."""
    cleaned: dict = {}
    for key, value in form_data.items():
        if key in ASSESSMENT_FIELDS_ORDER:
            if isinstance(value, list):
                # Multi-select fields: filter empty strings, keep list
                filtered = [v.strip() for v in value if isinstance(v, str) and v.strip()]
                if filtered:
                    cleaned[key] = filtered
            elif isinstance(value, str):
                # Text fields: strip whitespace
                stripped = value.strip()
                if stripped:
                    cleaned[key] = stripped
    return cleaned


def flatten_assessment_for_display(assessment: dict | None) -> dict:
    """Return all fields with defaults: empty strings for text, empty lists for multi-select."""
    defaults = {}
    for key in ASSESSMENT_FIELDS_ORDER:
        if key in ["primary_challenges", "file_types", "success_metrics"]:
            defaults[key] = []  # Multi-select fields default to empty list
        else:
            defaults[key] = ""  # Text fields default to empty string

    if assessment:
        for key, value in assessment.items():
            if key in defaults:
                defaults[key] = value

    return defaults


# ---- Sources and logging ----


def _sources_path(case_id: str) -> Path:
    return case_dir(case_id) / "sources" / "external_sources.json"


def _ensure_sources_file(case_id: str) -> None:
    path = _sources_path(case_id)
    if not path.exists():
        atomic_write_json(path, [])


def load_external_sources(case_id: str) -> list[ExternalSource]:
    """Load external sources for a case."""
    path = _sources_path(case_id)
    if not path.exists():
        return []
    data = read_json(path)
    return data if isinstance(data, list) else []


def append_external_sources(
    case_id: str, run_id: str, query: str, results: list[dict]
) -> list[str]:
    """Persist web search results and return generated source_ids."""
    _ensure_sources_file(case_id)
    existing = load_external_sources(case_id)
    next_idx = len(existing) + 1
    source_ids: list[str] = []
    timestamp = utc_now_iso()
    for res in results:
        source_id = f"src_{next_idx:03d}"
        next_idx += 1
        source_ids.append(source_id)
        existing.append(
            {
                "source_id": source_id,
                "query": query,
                "title": res.get("title"),
                "url": res.get("url"),
                "snippet": res.get("snippet"),
                "retrieved_at": res.get("retrieved_at", timestamp),
                "used_in_run": run_id,
            }
        )
    atomic_write_json(_sources_path(case_id), existing)
    return source_ids


def log_event(case_id: str, event: dict) -> None:
    """Append JSONL event to logs/events.jsonl."""
    base_dir = case_dir(case_id)
    base_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False)
    with (logs_dir / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---- Outputs and runs ----


def save_output_md(case_id: str, name: str, content: str) -> Path:
    base_dir = case_dir(case_id)
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    path = outputs_dir / f"{name}.md"
    atomic_write_text(path, content)
    update_case_updated_at(case_id)
    return path


def load_latest_output(case_id: str, name: str) -> str | None:
    path = case_dir(case_id) / "outputs" / f"{name}.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")


def save_run_meta(case_id: str, meta: dict) -> str:
    base_dir = case_dir(case_id)
    runs_dir = base_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = meta.get("run_id") or f"{meta.get('type', 'run')}_{utc_now_iso().replace(':', '-')}"
    meta = {**meta, "run_id": run_id}
    atomic_write_json(runs_dir / f"{run_id}.json", meta)
    return run_id


# ---- Export helpers (Phase 4 will flesh out templates) ----


def write_export_pack(case_id: str, export_ts: str, files: dict[str, str]) -> str:
    base_dir = case_dir(case_id)
    export_dir = base_dir / "exports" / export_ts
    export_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        atomic_write_text(export_dir / name, content)
    return str(export_dir)


def make_export_zip(export_dir: str) -> str:
    export_path = Path(export_dir)
    zip_path = export_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in export_path.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(export_path))
    return str(zip_path)


def list_exports(case_id: str) -> list[ExportInfo]:
    """List all exports for a case, most recent first."""
    exports_dir = case_dir(case_id) / "exports"
    if not exports_dir.exists():
        return []
    items: list[ExportInfo] = []
    for path in sorted(exports_dir.iterdir(), reverse=True):
        if path.is_dir():
            zip_path = path.with_suffix(".zip")
            items.append(
                {
                    "export_ts": path.name,
                    "dir": str(path),
                    "zip": str(zip_path) if zip_path.exists() else None,
                    "files": sorted([str(p) for p in path.glob("*.md")]),
                }
            )
    return items


# ---- Tools for agents / workflow ----


def read_case_context(case_id: str, max_chars: int = 60000) -> dict:
    """Return assessment, attachments, context_text, and context_warnings."""
    rebuild = not (case_dir(case_id) / "context.txt").exists()
    if rebuild:
        ctx = rebuild_context_txt(case_id, max_chars=max_chars)
        context_text = ctx["context_text"]
        context_warnings = ctx["warnings"]
        attachments_meta = ctx["attachments_meta"]
    else:
        context_text = (case_dir(case_id) / "context.txt").read_text(
            encoding="utf-8", errors="ignore"
        )
        context_warnings = []
        attachments_meta = list_attachments(case_id)
    assessment = load_assessment(case_id) or {}
    return {
        "assessment": assessment,
        "context_text": context_text,
        "context_warnings": context_warnings,
        "attachments_meta": attachments_meta,
    }


def web_search(
    case_id: str, query: str, max_results: int = 5, recency_days: int = 365
) -> "WebSearchResponse":
    """Perform a Tavily search and persist sources."""
    from tavily import TavilyClient  # lazy import

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    raw = client.search(query=query, max_results=max_results, days=recency_days)
    results: list[dict] = []
    for item in raw.get("results", []):
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("content"),
                "retrieved_at": utc_now_iso(),
            }
        )

    run_id = f"websearch_{utc_now_iso().replace(':', '-')}"
    source_ids = append_external_sources(
        case_id, run_id=run_id, query=query, results=results
    )
    log_event(
        case_id,
        {
            "event": "web_search",
            "case_id": case_id,
            "query": query,
            "results_count": len(results),
            "source_ids": source_ids,
            "run_id": run_id,
            "timestamp": utc_now_iso(),
        },
    )
    return {"results": results, "source_ids": source_ids, "retrieved_at": utc_now_iso()}


# ---- Industry Research ----


def conduct_industry_research(
    case_id: str,
    assessment: dict,
    max_queries: int = 4,
    results_per_query: int = 3,
    recency_days: int = 365,
) -> "IndustryResearchResult":
    """Conduct targeted industry research based on assessment data.

    Performs multiple web searches to gather:
    - Industry-specific automation tools
    - Solutions for stated pain points
    - Compliance/regulatory requirements
    - Best practices for the process type

    Args:
        case_id: Case identifier for source persistence
        assessment: Assessment data dictionary
        max_queries: Maximum number of search queries
        results_per_query: Results per query
        recency_days: How recent results should be

    Returns:
        IndustryResearchResult with formatted context and source IDs
    """
    # Extract key information from assessment
    process_name = assessment.get("process_name", "business process")
    process_area = assessment.get("area", "")
    bottlenecks = assessment.get("bottlenecks", "")
    systems = assessment.get("systems", "")
    risks_compliance = assessment.get("risks_compliance", "")

    # Build targeted queries based on assessment
    queries = _build_research_queries(
        process_name=process_name,
        process_area=process_area,
        bottlenecks=bottlenecks,
        systems=systems,
        risks_compliance=risks_compliance,
        max_queries=max_queries,
    )

    all_source_ids: list[str] = []
    research_sections: list[str] = []

    for query in queries:
        try:
            result = web_search(
                case_id=case_id,
                query=query,
                max_results=results_per_query,
                recency_days=recency_days,
            )
            all_source_ids.extend(result.get("source_ids", []))

            # Format results for this query
            if result.get("results"):
                section = f"### Research: {query}\n"
                for i, res in enumerate(result["results"], 1):
                    src_id = result["source_ids"][i - 1] if i <= len(result["source_ids"]) else "unknown"
                    title = res.get("title", "Untitled")
                    snippet = res.get("snippet", "")[:200]
                    section += f"- **[WEB:{src_id}]** {title}\n  {snippet}...\n"
                research_sections.append(section)

        except Exception as exc:
            log_event(
                case_id,
                {
                    "event": "industry_research_error",
                    "query": query,
                    "error": str(exc),
                    "timestamp": utc_now_iso(),
                },
            )
            # Continue with other queries even if one fails
            continue

    # Compile research context
    if research_sections:
        research_context = (
            "# Industry Research Summary\n\n"
            f"The following research was gathered for **{process_name}**:\n\n"
            + "\n".join(research_sections)
        )
    else:
        research_context = (
            "# Industry Research Summary\n\n"
            "No research results were retrieved. Proceeding with assessment data only."
        )

    # Log the research completion
    log_event(
        case_id,
        {
            "event": "industry_research_complete",
            "case_id": case_id,
            "queries_executed": queries,
            "total_sources": len(all_source_ids),
            "timestamp": utc_now_iso(),
        },
    )

    return {
        "research_context": research_context,
        "source_ids": all_source_ids,
        "queries_executed": queries,
        "total_sources_found": len(all_source_ids),
    }


def _build_research_queries(
    process_name: str,
    process_area: str,
    bottlenecks: str,
    systems: str,
    risks_compliance: str,
    max_queries: int,
) -> list[str]:
    """Build targeted research queries from assessment data."""
    queries = []

    # Query 1: Industry-specific automation tools
    area_context = f" {process_area}" if process_area else ""
    queries.append(f"{process_name}{area_context} workflow automation tools best practices 2024")

    # Query 2: Pain point solutions (if bottlenecks described)
    if bottlenecks and len(bottlenecks.strip()) > 10:
        # Extract key pain point terms
        pain_keywords = bottlenecks[:100].replace("\n", " ")
        queries.append(f"solve {pain_keywords} automation solutions")

    # Query 3: System integrations (if systems mentioned)
    if systems and systems.lower() not in ["i don't know", "unknown", "none", ""]:
        queries.append(f"{systems} integration automation API capabilities")

    # Query 4: Compliance/regulatory (if mentioned)
    if risks_compliance and len(risks_compliance.strip()) > 10:
        queries.append(f"{process_name} compliance regulatory requirements automation")

    # Ensure we don't exceed max queries
    return queries[:max_queries]


def save_industry_research(case_id: str, research_result: dict) -> Path:
    """Persist industry research results to disk."""
    base_dir = case_dir(case_id)
    research_path = base_dir / "industry_research.json"
    atomic_write_json(research_path, research_result)
    return research_path


def load_industry_research(case_id: str) -> dict | None:
    """Load previously saved industry research."""
    path = case_dir(case_id) / "industry_research.json"
    if not path.exists():
        return None
    return read_json(path)
