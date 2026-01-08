# Workflow Improvement Planner

A **local** Streamlit app that helps you capture a workflow/process assessment, run a fixed AI workflow (Solution Designer -> Prototype Builder), and export final Markdown artifacts + a ZIP.

## What's in this repo
- `docs/PRD_workflow_improvement_planner.md`: product requirements (English).
- `docs/roadmap_app_implementation.md`: phased roadmap (0% -> 100%).
- `workflow_planner.py`: Streamlit app entrypoint.
- `utils.py`: filesystem persistence, ingestion, web search, export helpers.

## Quickstart
This project runs locally and calls OpenAI + Tavily via API keys.

1) Copy `.env.example` to `.env` and fill `OPENAI_API_KEY`, `TAVILY_API_KEY` (set `OPENAI_MODEL` if you want).
2) Install deps (Poetry recommended):
   - `poetry install`
3) Run:
   - `poetry run streamlit run workflow_planner.py`

Data lives under `data/` (gitignored). Cases are local and versioned on disk; exports are generated as 3 Markdown files + a ZIP per case.
