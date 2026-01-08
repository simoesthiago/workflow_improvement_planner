# Roadmap — Workflow Improvement Planner (0% → 100% “ready to use”)

This roadmap turns `docs/PRD_workflow_improvement_planner.md` into an implementation plan that can take the app from an empty repo to a usable local Streamlit application.

## Scope (what “100% done” means)
The app is considered **100% complete** when a user can:
- Create/open/delete **cases** persisted on disk (filesystem + JSON).
- Complete an **Assessment Wizard** with optional attachments; the app builds `context.txt` with limits/warnings.
- Run the full workflow end-to-end (LangGraph):
  - `load_case_context` → `solution_designer_agent` → `prototype_builder_agent` → `export_pack_builder`
- Re-run **Solution** and **Prototype** independently and see the latest outputs.
- Use `web_search` (Tavily) with per-run limits; save external sources to the case.
- Generate an **Export Pack** (3 Markdown files) + downloadable ZIP; keep export history per case.
- Start from a clean machine following `README.md` and run with `streamlit run workflow_planner.py`.

Out of scope (by design): advanced RAG, real enterprise integrations, auth/RBAC, cloud deploy.

## Assumptions
- 1 sprint = **1 week** (adjust as needed).
- Python app managed via **Poetry**.
- LLM: OpenAI via API (`OPENAI_API_KEY`), web search: Tavily (`TAVILY_API_KEY`).
- Data is local-only in `data/` and **gitignored**.

## Target repo structure (end state)
```
workflow-improvement-planner/
  workflow_planner.py          # Streamlit UI + LangGraph execution
  schemas.py                   # Pydantic models + PlannerState
  prompts.py                   # prompt templates (Solution / Prototype)
  utils.py                     # filesystem persistence, ingestion, web_search, export, logging
  pyproject.toml
  poetry.lock
  README.md
  .env.example
  .gitignore
  docs/
    PRD_workflow_improvement_planner.md
    roadmap_app_implementation.md
  data/                        # gitignored (created at runtime)
```

---

## Phase 1 — Foundations (Sprints 1–2)
Goal: create a runnable project skeleton with a stable local storage layer.

### Sprint 1 — Repo bootstrap + Streamlit skeleton
**Deliverables**
- `pyproject.toml` (Poetry) with baseline dependencies.
- `.env.example`, `.gitignore`, `README.md` (getting started).
- `workflow_planner.py` launches Streamlit and renders the planned tabs (placeholders).

**Work**
- [x] Initialize Poetry project (pin Python version).
- [x] Add deps: `streamlit`, `langchain-openai`, `langgraph`, `python-dotenv`, `tavily-python`, plus PDF extractor (e.g., `pypdf`).
- [x] Add `.env.example` with `OPENAI_API_KEY`, `OPENAI_MODEL`, `TAVILY_API_KEY`, and optional limits.
- [x] Add `.gitignore` (at minimum: `data/`, `.env`, `.venv/`, `__pycache__/`).
- [x] Implement config loader (dotenv + environment validation).
- [x] Create Streamlit app skeleton:
  - Sidebar: case selector placeholder + buttons (disabled)
  - Tabs: Assessment, Solution, Prototype, Export, History/Sources (optional)

**Acceptance criteria**
- `poetry install` succeeds.
- `streamlit run workflow_planner.py` opens the app without errors (even without API keys).

### Sprint 2 — Filesystem case store + sidebar CRUD
**Deliverables**
- `utils.py` implements case CRUD and creates per-case folder structure.
- Streamlit sidebar can create/open/delete cases and shows `case_id` + timestamps.

**Work**
- [x] Implement case store in `utils.py`:
  - `list_cases`, `create_case`, `load_case_meta`, `delete_case`, `update_case_updated_at`
  - `data/index.json` as the global index
  - per-case folder: `attachments/`, `attachments_text/`, `outputs/`, `runs/`, `sources/`, `exports/`, `logs/`
- [x] Add safe file writes (atomic write pattern for JSON).
- [x] Wire Streamlit sidebar:
  - Create case (name/area/tags/description)
  - Open/select case
  - Delete with confirmation
  - Show “case status” (has assessment? has outputs?)

**Acceptance criteria**
- Cases persist across app restarts.
- Delete removes the case folder and removes it from `data/index.json`.

---

## Phase 2 — Assessment & Context Ingestion (Sprints 3–4)
Goal: reliably capture the assessment + attachments and build deterministic context text.

### Sprint 3 — Assessment Wizard + versioning
**Deliverables**
- Assessment form saves/loads correctly per case.
- Snapshot versioning on save.

**Work**
- [x] Implement form fields per PRD (minimum set).
- [x] Implement assessment persistence in `utils.py`:
  - `save_assessment`, `load_assessment`, `save_assessment_snapshot`
- [x] Streamlit: “Save Assessment” writes `assessment.json` + snapshot.
- [x] Display last saved time and a small “assessment completeness” indicator (optional).

**Acceptance criteria**
- Reopening a case restores the form values from disk.
- Saving creates `assessment_versions/<timestamp>.json`.

### Sprint 4 — Attachments, text extraction, and `context.txt`
**Deliverables**
- Upload attachments to case folder.
- Best-effort text extraction for TXT/MD/PDF and warnings for unsupported/failed extraction.
- Deterministic `context.txt` rebuild with max length + warnings.

**Work**
- [x] Implement `save_attachment` and store attachment metadata (filename, size, uploaded_at).
- [x] Implement `extract_text_from_attachment`:
  - TXT/MD: read as text
  - PDF: best-effort extraction
  - PNG/JPG: store file; include warning (OCR is optional and can be a later enhancement)
- [x] Implement `rebuild_context_txt(case_id, max_chars)`:
  - assessment summary (structured → text)
  - concatenated extracted texts
  - truncation strategy + `context_warnings`
- [x] Streamlit UI:
  - attachments list + extraction status
  - show `context_warnings` clearly
  - “Rebuild context” button (optional; or rebuild on save)

**Acceptance criteria**
- `context.txt` exists after saving assessment and/or attachments.
- Failures do not break the app; warnings are visible to the user.

---

## Phase 3 — Tools, Agents, and Workflow (Sprints 5–6)
Goal: implement tool calling + deterministic orchestration (standalone agents and full workflow).

### Sprint 5 — Logging, web search, sources, and run metadata
**Deliverables**
- `logs/events.jsonl` populated with structured events.
- Tavily-backed `web_search` with per-run limits and persisted sources.
- Run metadata saved for Solution and Prototype runs.

**Work**
- [x] Implement `log_event(case_id, event)` (append JSONL).
- [x] Implement sources persistence:
  - `append_external_sources`, `load_external_sources`
  - store in `sources/external_sources.json` with stable `source_id`
- [x] Implement `web_search` wrapper (Tavily):
  - enforce max calls per run (Solution=2, Prototype=3)
  - enforce `max_results` and `recency_days`
  - log events and persist sources
- [x] Implement run meta persistence:
  - `save_run_meta` to `runs/<type>_<ts>.json`
  - include `web_search_calls`, `sources_used_ids`, `status`, `error`

**Acceptance criteria**
- A `web_search` call creates both an event log entry and a new entry in `sources/external_sources.json`.
- A run can be marked success/failure with an error message stored on disk.

### Sprint 6 — Agents (Solution/Prototype) + LangGraph end-to-end run
**Deliverables**
- Solution Designer and Prototype Builder can run independently and save outputs.
- Full LangGraph workflow runs end-to-end and produces outputs + export timestamp.

**Work**
- [x] Implement `prompts.py` templates with required fixed headings and “External sources” section.
- [x] Implement LangChain agent setup (ChatOpenAI + AgentExecutor):
  - tools exposed: `read_case_context`, `web_search`
  - iteration limits and low temperature
- [x] Implement `read_case_context` tool (returns assessment + attachments + `context_text` + warnings).
- [x] Implement “Run Solution Designer” UI:
  - executes agent
  - saves `outputs/solution_designer.md`
  - saves run meta
- [x] Implement “Run Prototype Builder” UI:
  - includes `solution_md` as input context
  - saves `outputs/prototype_builder.md`
  - saves run meta
- [x] Implement LangGraph:
  - `schemas.py` PlannerState
  - nodes: `load_case_context`, `run_solution_designer`, `run_prototype_builder`, `build_export_pack` (export node can be stubbed to only set `export_ts` until Phase 4)
  - “Run Workflow” button runs the compiled graph and renders results

**Acceptance criteria**
- Outputs are saved and visible in the UI tabs.
- A workflow run can be repeated and produces a new run meta entry.
- External sources are consistently referenced and listed when web_search is used.

---

## Phase 4 — Export Pack + Release Readiness (Sprints 7–8)
Goal: produce final artifacts, make the app stable for real use, and document it.

### Sprint 7 — Export Pack (3 Markdown files) + ZIP + history
**Deliverables**
- Deterministic export pack generator.
- ZIP download in Streamlit.
- Export history list per case.

**Work**
- [x] Implement export pack writer:
  - `write_export_pack(case_id, export_ts, files)` → `exports/<ts>/...`
  - `make_export_zip(export_dir)` → bytes for `st.download_button`
- [x] Create deterministic templates:
  - `Executive_Brief.md` (from Solution)
  - `Technical_Blueprint.md` (from Prototype)
  - `Implementation_Plan.md` (consolidated: phases, risks, KPIs)
- [x] Streamlit “Export Pack” tab:
  - generate export from latest outputs
  - list older exports and allow download

**Acceptance criteria**
- Export directory contains exactly the 3 expected `.md` files.
- ZIP download works and matches the export folder contents.

### Sprint 8 — Polish, robustness, docs, and “ready to use” checklist
**Deliverables**
- UX polish and robust error handling.
- Documentation complete for a clean setup.
- 1–2 sample cases for demo (optional but recommended).

**Work**
- [x] UX:
  - spinners/progress during runs
  - clear errors when keys are missing or calls fail
  - disable buttons when prerequisites are missing (no case selected, no assessment, etc.)
- [x] Guardrails:
  - enforce `MAX_CONTEXT_CHARS`
  - handle large attachments gracefully
  - protect against partial writes (atomic JSON writes everywhere)
- [x] Docs:
  - README: install, configure `.env`, run app, how data is stored, limitations
  - add screenshots placeholders (or instructions to capture them)
- [ ] QA:
  - add minimal automated tests for `utils.py` (CRUD + JSON writes + context rebuild)
  - manual smoke-test checklist (create case → assessment → run workflow → export ZIP)

**Acceptance criteria**
- A new developer can follow README and run end-to-end without guesswork.
- Manual smoke test passes on a clean environment with valid API keys.

---

## Post-100% backlog (optional enhancements)
- OCR for PNG/JPG and scanned PDFs (opt-in).
- Advanced RAG (chunking/reranking) and eval harness.
- Multi-user, auth, and remote deployment.
- Rich run history UI (diff outputs, compare runs, more observability).
