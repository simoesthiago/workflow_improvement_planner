# Workflow Improvement Planner

A **local** Streamlit app that helps you capture a workflow/process assessment, run a fixed AI workflow (Solution Designer -> Prototype Builder), and export final Markdown artifacts + a ZIP.

## What's in this repo
- `docs/PRD_workflow_improvement_planner.md`: product requirements (English).
- `docs/roadmap_app_implementation.md`: phased roadmap (0% -> 100%).
- `workflow_planner.py`: Streamlit app entrypoint (Phase 1: case manager + skeleton tabs).
- `utils.py`: filesystem case store utilities (Phase 1).

## Quickstart
1) Create a `.env` file based on `.env.example`.
2) Install dependencies (Poetry recommended):
   - `poetry install`
3) Run:
   - `poetry run streamlit run workflow_planner.py`

> Note: `data/` is local-only and is gitignored by design.
