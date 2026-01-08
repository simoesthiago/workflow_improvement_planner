# PRD — Workflow Improvement Planner (Streamlit + LangChain + LangGraph)

## 0) TL;DR (what this project is)
A **local** **Streamlit** app (runs on your computer) where the user creates **cases** (projects) and completes an **Assessment Wizard** (form) with optional attachments (PDF/TXT/MD/PNG/JPG).
After submitting the assessment, the system runs a **fixed, chained workflow**:

**Assessment → Solution Designer (Agent 1) → Prototype Builder (Agent 2) → Export Pack**

The MVP focuses on convincingly demonstrating:
- **Workflow-driven agentic AI** (a chain of agents in deterministic steps)
- **Tool calling** (`web_search` + context reading + export writing)
- **Web research with sources** (links/titles/snippets saved in the case)
- **Final artifacts** (Markdown + ZIP) ready to read

> Important: “run locally” here means the **app and data** stay local, but the **LLM is called via API** (OpenAI) using the user's key.

---

## 1) Goals, non-goals, and principles

### 1.1 MVP goals
1) Create/open/delete **cases** persisted locally.
2) Capture a structured assessment + attachments and generate a **simple textual context** (no advanced RAG).
3) Automatically run a chain of 2 agents:
   - **Solution Designer**: macro/managerial view, opportunities, approach recommendation, KPIs, risks.
   - **Prototype Builder**: technical view, architecture, tools/specs, minimal governance, phased plan.
4) Allow both agents to use **web_search** (with limits) and record external sources in the case.
5) Generate an **Export Pack** with 3 Markdown files and a ZIP folder.

### 1.2 Non-goals (out of scope for the MVP)
- Advanced RAG (sophisticated chunking, reranking, complex eval harness).
- Real integrations with corporate systems (only “spec/mock”).
- Login, multi-user, cloud deploy, real RBAC.
- Advanced OCR for scanned PDFs (best-effort text extraction only).
- Enterprise observability (OpenTelemetry, dashboards, etc.).

### 1.3 Design principles (keep it simple and portfolio-ready)
- **Few files, high signal**: a simple repo that is still well-architected.
- **Explicit workflow**: deterministic chaining (LangGraph).
- **Agentic only where it adds value**: tool calling and synthesis in the Solution/Prototype steps.
- **Transparent persistence**: filesystem + JSON, easy to inspect and version.
- **Reproducibility**: outputs and sources are stored per run.

---

## 2) End-to-end user flow (from zero to ZIP)

### 2.1 Main flow
1) **New Case** (sidebar) → set name/area/tags.
2) **Assessment Wizard**: fill fields + upload optional attachments → **Save Assessment**.
   Note: the assessment wizard is inspired by this Claude artifact: https://claude.ai/public/artifacts/77610fec-9e07-4abb-a204-46df4dbdc988
3) **Run Workflow**:
   - Node A: `load_case_context`
   - Node B: `solution_designer_agent`
   - Node C: `prototype_builder_agent`
4) Review outputs in tabs (Solution/Prototype).
5) **Generate Export Pack** → save 3 `.md` files and a ZIP.
6) Download the ZIP and `.md` files.

### 2.2 Alternative flows (MVP)
- Re-run Solution Designer (improve results without changing the assessment).
- Re-run Prototype Builder (when Solution changes).
- If attachment extraction fails → proceed with the assessment only, with a warning in the output.

---

## 3) Architecture decisions (recommended)

### 3.1 Orchestration: LangGraph + AgentExecutor (per stage)
**Final recommendation (decision):**
- **LangGraph** orchestrates the fixed workflow (4 nodes).
- Each “agentic” node uses **LangChain AgentExecutor** with minimal tools.

**Why (one sentence):** LangGraph makes the pipeline explicit and auditable; AgentExecutor provides real tool calling without reinventing the loop.

### 3.2 LLM: OpenAI via API (runs “remote”), app runs locally
- Use `ChatOpenAI` (langchain-openai).
- Model configurable via `.env` (e.g., `gpt-4o-mini`).
- Key via `OPENAI_API_KEY`.

### 3.3 Persistence: filesystem + JSON
- Simpler than SQLite for the MVP.
- Easier for Cursor to understand and for you to debug.

### 3.4 Web search: Tavily (primary) with limits
- Requires `TAVILY_API_KEY` (document in README).
- Guardrails: max N searches per execution + max results.

---

## 4) Streamlit screens (navigation and components)

### 4.1 Sidebar (always visible)
- **Cases**: selectbox with name + area.
- Buttons:
  - `New Case`
  - `Open Case`
  - `Delete Case` (with confirmation)
- Shows: `case_id`, last update, status (has assessment? has outputs?).

### 4.2 Main area (tabs)
Implement with `st.tabs` (simple):
1) **Assessment Wizard**
2) **Solution Designer**
3) **Prototype Builder**
4) **Export Pack**
5) (Optional) **History & Sources** (or use expanders in tabs 2/3/4)

---

## 5) Functional requirements (MVP)

### 5.1 Case Manager
- **RF-C1** Create a case with: name, description (optional)
- **RF-C2** List cases and open an existing case.
- **RF-C3** Delete a case (remove folder and remove from index).
- **RF-C4** Persist `case_meta.json` and update `updated_at` on operations.

**Acceptance criteria**
- Cases persist after restarting the app.
- Deleting a case removes it from the index and from disk.

### 5.2 Assessment Wizard (Review: https://claude.ai/public/artifacts/77610fec-9e07-4abb-a204-46df4dbdc988)
Minimum fields:
- Process name / area
- Process objective
- Current workflow description (main steps)
- Involved actors/teams
- Systems used today (accept “I don't know”)
- Volume/frequency
- SLAs/lead time
- Main bottlenecks/pain points
- Risks/compliance (if any)
- Desired outcome and KPIs (or estimates)

Uploads:
- PDF, TXT, MD, PNG, JPG

- **RF-A1** Save the assessment to `assessment.json`.
- **RF-A2** Version snapshots in `assessment_versions/<timestamp>.json`.
- **RF-A3** Save attachments in `attachments/` and extract text to `attachments_text/`.
- **RF-A4** Generate `context.txt` (concat) with limits and warnings.

**Acceptance criteria**
- When reopening a case, the form loads previous values.
- Uploaded files are listed and extracted text is accessible.

### 5.3 Solution Designer (Agent 1)
**Input**: assessment + `context.txt` + (optional) sources via web search.
**Output** (Markdown):
- Bottleneck diagnosis
- Opportunities (quick wins vs structural)
- Recommendation: fixed workflow with AI vs agentic AI vs hybrid (with rationale)
- Risks and dependencies
- Suggested KPIs and baseline
- Next steps
- **External sources** (if used): links/titles/snippets

- **RF-S1** “Run Solution Designer” button generates `outputs/solution_designer.md`.
- **RF-S2** Save run metadata to `runs/solution_<ts>.json`.
- **RF-S3** If web search is used, save results to `sources/external_sources.json` with `used_in_run`.

**Acceptance criteria**
- Always includes an “External sources” section (even if “None used”).
- Runs are re-executable and versioned.

### 5.4 Prototype Builder (Agent 2)
**Input**: assessment + `context.txt` + Solution Designer output + web search.
**Output** (Markdown):
- Proposed architecture (simple, end-to-end)
- Components and responsibilities (app, integrations, AI)
- List of tools/functions (mock/spec) an agent would call (expected inputs/outputs)
- Minimal governance (approvals, logs, conceptual access)
- Phased plan (MVP → improvements)
- Technical risks and mitigations
- Assumptions and validations (POC)
- External sources (if used)

- **RF-P1** “Run Prototype Builder” button generates `outputs/prototype_builder.md`.
- **RF-P2** Save run metadata to `runs/prototype_<ts>.json`.
- **RF-P3** Record external sources used in the same sources repository.

### 5.5 Export Pack
Files (required):
1) `Executive_Brief.md` (based on Solution)
2) `Technical_Blueprint.md` (based on Prototype)
3) `Implementation_Plan.md` (consolidated)

- **RF-E1** “Generate Export Pack” button saves files to `exports/<ts>/`.
- **RF-E2** Generate ZIP for download (`export_pack_<ts>.zip`).
- **RF-E3** List older exports and allow download.

---

## 6) Tools (LangChain Tools) — minimum viable

> All tools must log an event to `logs/events.jsonl`.

### 6.1 Tool: `read_case_context(case_id)`
Returns:
- `assessment` (dict)
- list of attachments and paths
- `context_text` (string)
- `context_warnings` (list)

Usage: always called at the start of agents.

### 6.2 Tool: `web_search(case_id, query, max_results=5, recency_days=365)`
Returns:
- `results: [{title, url, snippet}]`
- `retrieved_at`

Rules:
- Solution: max 2 calls per execution
- Prototype: max 3 calls per execution

Persistence:
- append to `sources/external_sources.json`, with `query`, `retrieved_at`, `used_in_run`.

### 6.3 Helper (app-side, not exposed to agents): `write_export_pack(case_id, export_ts, files: dict[str, str])`
Writes the files to the case folder and returns paths.

### 6.4 Optional helper: `summarize_attachment(case_id, filename)`
Use only if `context_text` exceeds the limit and you want to reduce it.

---

## 7) Prompts (templates) — direct and “didactic”

### 7.1 Prompt — Solution Designer (base)
- Instruction: generate a macro-level, practical report with fixed sections.
- Condition: use web_search only if needed; always list external sources.
- Tone: executive/operational, no hype.

### 7.2 Prompt — Prototype Builder (base)
- Instruction: generate an implementable blueprint.
- Condition: when uncertain, mark as an assumption and suggest validation via POC.
- Tone: technical, implementation-oriented.

### 7.3 Prompt — Web query helper (optional)
- Given the assessment, suggest 2–3 objective queries for web_search (when needed).
- This matches the Perplexity-clone pattern (LLM generates queries → search).

---

## 8) Data model (JSON files)

### 8.1 `data/index.json`
```json
{
  "cases": [
    {
      "case_id": "case_2025-12-24_ab12",
      "name": "Vendor onboarding",
      "area": "Procurement",
      "tags": ["bpm", "automation"],
      "created_at": "2025-12-24T10:00:00",
      "updated_at": "2025-12-24T10:30:00"
    }
  ]
}
```

### 8.2 `case_meta.json`
Case metadata, same as the index (local source of truth for the case).

### 8.3 `assessment.json`
```json
{
  "process_name": "...",
  "process_objective": "...",
  "current_workflow": "...",
  "actors": "...",
  "systems": "...",
  "volume_frequency": "...",
  "slas": "...",
  "bottlenecks": "...",
  "risks_compliance": "...",
  "desired_outcome_kpis": "..."
}
```

### 8.4 `runs/solution_<ts>.json` and `runs/prototype_<ts>.json`
```json
{
  "run_id": "solution_2025-12-24T10-05-00",
  "type": "solution_designer",
  "started_at": "...",
  "ended_at": "...",
  "web_search_calls": 2,
  "sources_used_ids": ["src_001", "src_002"],
  "status": "success",
  "error": null
}
```

### 8.5 `sources/external_sources.json`
```json
[
  {
    "source_id": "src_001",
    "query": "ServiceNow ticket automation best practices",
    "title": "…",
    "url": "https://…",
    "snippet": "…",
    "retrieved_at": "2025-12-24T10:05:10",
    "used_in_run": "solution_2025-12-24T10-05-00"
  }
]
```

---

## 9) Non-functional requirements (MVP)

### 9.1 Privacy and data
- All case content stays locally in `data/`.
- Content sent to the LLM (OpenAI) is the prompt + context (assessment + attachments).
- Do not store API keys on disk; read from `.env`.

### 9.2 Limits (avoid cost/errors)
- `MAX_CONTEXT_CHARS` (e.g., 30k–60k) for `context.txt`.
- web_search limits per run (2 for Solution, 3 for Prototype).
- Max attachments per case (e.g., 10) and max size (e.g., 20MB).

### 9.3 Robustness
- If PDF extraction fails: record a warning and proceed with the assessment.
- If web_search fails: proceed without external sources and record the error in the run.

---

## 10) Repo architecture (minimal, Perplexity-clone style)

Keep 4 main files (high signal):

- `workflow_planner.py` — Streamlit entrypoint + LangGraph graph + execution
- `schemas.py` — Pydantic models + LangGraph state
- `prompts.py` — prompts and templates
- `utils.py` — persistence, ingestion, web_search, export, logging

---

## 11) Workflow diagram (LangGraph)

```mermaid
flowchart LR
  A[START] --> B[load_case_context]
  B --> C[solution_designer_agent]
  C --> D[prototype_builder_agent]
  D --> E[export_pack_builder]
  E --> F[END]
```

**Notes:**
- `solution_designer_agent` and `prototype_builder_agent` use AgentExecutor and tools.
- `export_pack_builder` is deterministic (templates + outputs).

---

## 12) Implementation plan (7–10 days)

**Day 1 — Setup and Case Store**
- `data/` structure, `index.json`, CRUD for cases (utils).
- Streamlit sidebar: list/new/open/delete.

**Day 2 — Assessment Wizard**
- Form + save + versioned snapshot.
- Upload attachments and listing.

**Day 3 — Simple attachment extraction**
- TXT/MD/PDF (+ optional DOCX).
- Generate `context.txt` with limits and warnings.

**Day 4 — Tools + logging**
- Implement LangChain tools and `logs/events.jsonl`.
- Implement `web_search` (Tavily).

**Day 5 — Solution Designer (AgentExecutor)**
- Prompt template + execution.
- Persist `solution_designer.md` + run meta + sources.

**Day 6 — Prototype Builder (AgentExecutor)**
- Prompt template + execution.
- Persist `prototype_builder.md` + run meta + sources.

**Day 7 — LangGraph (fixed workflow)**
- Create a StateGraph with 4 nodes.
- “Run Workflow” button executes the sequence and saves outputs.

**Day 8 — Export Pack + ZIP**
- Generate 3 MDs + ZIP.
- List older exports.

**Day 9 — Polish (UX and README)**
- Improve status messages, errors, spinners.
- README with demo script and screenshots.

**Day 10 — Optional**
- “Refine question” per agent (incremental chat) with output versioning.

---

## 13) Risks and mitigation (short)

1) **Context too large** → truncate + (optional) summarize large attachments.
2) **Scanned PDFs** → declare limitation; suggest TXT/MD.
3) **Variable web results** → save date/snippet/URL and clearly mark as “external reference”.
4) **API cost** → limits on context and web calls; configurable model.
5) **Inconsistent outputs** → fixed headings and “required format” in prompts.

---

## 14) Suggested repository structure (tree)

```
workflow-improvement-planner/
  workflow_planner.py
  schemas.py
  prompts.py
  utils.py
  pyproject.toml
  poetry.lock
  README.md
  .env.example
  .gitignore
  data/                 # gitignored
```

---

## 15) Tech stack (recommended) and setup

### 15.1 Stack (MVP)
- **UI**: Streamlit
- **Orchestration**: LangGraph (StateGraph)
- **Agents**: LangChain (AgentExecutor + tool calling)
- **LLM**: OpenAI via API (langchain-openai)
- **Web search**: Tavily (tavily-python)
- **Config**: python-dotenv (.env)
- **Persistence**: filesystem + JSON (no DB)

### 15.2 Environment variables (.env)
Create a `.env` file (do not commit) with:
- `OPENAI_API_KEY=...`
- `OPENAI_MODEL=gpt-4o-mini` (or another)
- `TAVILY_API_KEY=...` (required for web search in the MVP)
- (optional) `MAX_CONTEXT_CHARS=60000`
- (optional) `WEB_SEARCH_RECENCY_DAYS=365`
- (optional) `WEB_SEARCH_MAX_RESULTS=5`

Include `.env.example` in the repo with empty keys and comments.

### 15.3 Minimal dependencies (via Poetry)
You can start from your current `pyproject.toml` and remove what you don't use:
- keep: `streamlit`, `langchain-openai`, `langgraph`, `python-dotenv`, `tavily-python`
- remove (MVP): `openperplex`, `langchain-ollama`, `tinydb` (optional)

---

## 16) Code contracts (so Cursor can implement without ambiguity)

### 16.1 Key functions in `utils.py` (suggested signatures)

#### Case store (filesystem + JSON)
- `list_cases() -> list[dict]`
- `create_case(name: str, area: str, tags: list[str], description: str | None) -> dict`
- `load_case_meta(case_id: str) -> dict`
- `delete_case(case_id: str) -> None`
- `update_case_updated_at(case_id: str) -> None`

#### Assessment
- `save_assessment(case_id: str, payload: dict) -> None`
- `save_assessment_snapshot(case_id: str, payload: dict) -> str  # returns snapshot filename`
- `load_assessment(case_id: str) -> dict | None`

#### Attachments + simple extraction
- `save_attachment(case_id: str, uploaded_file) -> str  # returns file path`
- `extract_text_from_attachment(file_path: str) -> tuple[str, list[str]]  # text, warnings`
- `rebuild_context_txt(case_id: str, max_chars: int) -> dict  # {context_text, warnings, attachments_meta}`

#### Sources (web)
- `append_external_sources(case_id: str, run_id: str, query: str, results: list[dict]) -> list[str]  # returns source_ids`
- `load_external_sources(case_id: str) -> list[dict]`

#### Outputs and runs
- `save_output_md(case_id: str, name: str, content: str) -> str  # returns path`
- `save_run_meta(case_id: str, meta: dict) -> str  # returns run_id`
- `load_latest_output(case_id: str, name: str) -> str | None`

#### Export
- `write_export_pack(case_id: str, export_ts: str, files: dict[str, str]) -> str  # returns export_dir`
- `make_export_zip(export_dir: str) -> bytes | str  # bytes for download or path`

#### Logging
- `log_event(case_id: str, event: dict) -> None  # appends to logs/events.jsonl`

> Note: keeping these signatures simple helps Cursor implement quickly and avoids “accidental architecture”.

---

### 16.2 LangGraph state (`schemas.py`)
Define a minimal Pydantic state (similar to your `ReportState`):

- `case_id: str`
- `assessment: dict`
- `context_text: str`
- `context_warnings: list[str]`
- `solution_md: str | None`
- `prototype_md: str | None`
- `external_sources: list[dict]` (accumulated during a run)
- `run_id_solution: str | None`
- `run_id_prototype: str | None`
- `export_ts: str | None`
- `errors: list[str]`

Implementation tip: `external_sources` can use `Annotated[List[...], operator.add]` to accumulate in the graph (same as your clone).

---

### 16.3 LangGraph nodes (`workflow_planner.py`)
Contracts (input → partial state output):

1) `load_case_context(state) -> dict`
- read assessment + `context.txt` (rebuild if needed)
- return `assessment`, `context_text`, `context_warnings`

2) `run_solution_designer(state) -> dict`
- call Solution AgentExecutor
- save `outputs/solution_designer.md`
- save `runs/solution_<ts>.json`
- return `solution_md` + `run_id_solution`

3) `run_prototype_builder(state) -> dict`
- call Prototype AgentExecutor (includes `solution_md`)
- save `outputs/prototype_builder.md`
- save `runs/prototype_<ts>.json`
- return `prototype_md` + `run_id_prototype`

4) `build_export_pack(state) -> dict`
- assemble 3 MDs (templates) and write to `exports/<ts>/`
- return `export_ts`

---

## 17) Agent configuration (LangChain) — stable minimum

### 17.1 Tools available to agents
To keep control and simplicity, expose only:
- `read_case_context`
- `web_search`

And keep **export outside the agent** (deterministic), to avoid “creative” file writing.

### 17.2 Recommended limits (avoid loops and cost)
- AgentExecutor `max_iterations`: 4 (Solution) and 6 (Prototype)
- Web search limit:
  - Solution: 2 calls
  - Prototype: 3 calls
- Temperature: 0.2 (more deterministic)

### 17.3 Format for external sources in text
Standardize this so Cursor can implement easily and you can demo it in your portfolio:

- In the body text, when using web sources:
  - “(...) per reference [WEB:src_001].”
- In “External sources”, list:
  - `- [WEB:src_001] Title — URL — short snippet (retrieved on YYYY-MM-DD)`

---

## 18) Code skeleton (pseudocode) — so Cursor can “see the movie”

### 18.1 Graph construction (LangGraph)
```python
graph = StateGraph(PlannerState)
graph.add_node("load_case_context", load_case_context)
graph.add_node("solution", run_solution_designer)
graph.add_node("prototype", run_prototype_builder)
graph.add_node("export", build_export_pack)

graph.add_edge(START, "load_case_context")
graph.add_edge("load_case_context", "solution")
graph.add_edge("solution", "prototype")
graph.add_edge("prototype", "export")
graph.add_edge("export", END)

app = graph.compile()
```

### 18.2 Streamlit execution (summary)
- user selects a case
- saves assessment + attachments
- clicks “Run Workflow”
- calls `app.invoke({"case_id": case_id})`
- renders `solution_md`, `prototype_md`, and the export list

---

## 19) Quality criteria (for Cursor to validate during implementation)

### 19.1 Solution Designer output checklist
- Fixed headings (8 sections)
- Quick wins vs structural opportunities
- Recommendation workflow vs agentic (with rationale)
- Suggested KPIs
- “External sources” section (or “None used”)

### 19.2 Prototype Builder output checklist
- Simple, end-to-end architecture
- Tools/functions list (mock) with expected inputs/outputs
- Minimal governance (approval/log/conceptual access)
- Phased plan + technical risks
- “Assumptions and validations (POC)”
- “External sources” section (or “None used”)

---

## 20) Portfolio notes (what makes this repo convincing)
- Include 1–2 **synthetic cases** (no real data) as examples in `sample_cases/` (optional).
- Take screenshots of the 4 tabs and include them in README.
- In logs (`events.jsonl`), show at least:
  - one `web_search` call
  - one context read
- In README, make it explicit: “No advanced RAG by design; focus on workflow-driven agentic + tool calling.”
