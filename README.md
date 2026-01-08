# Workflow Improvement Planner

A **local** Streamlit app that helps you capture a workflow/process assessment, run a fixed AI workflow (Solution Designer → Prototype Builder), and export final Markdown artifacts + a ZIP.

## What’s in this repo (so far)
- `docs/PRD_workflow_improvement_planner.md`: product requirements (English).
- `docs/roadmap_app_implementation.md`: phased roadmap (0% → 100%).
- `workflow_planner.py`, `schemas.py`, `utils.py`, `prompts.py`: implementation entrypoints (initial skeleton).

## Quickstart (later)
This project is designed to run locally. It will require:
- `OPENAI_API_KEY` (LLM via API)
- `TAVILY_API_KEY` (web search)

Create a `.env` file based on `.env.example` and run:
- `streamlit run workflow_planner.py`

> Note: `data/` is local-only and is gitignored by design.
