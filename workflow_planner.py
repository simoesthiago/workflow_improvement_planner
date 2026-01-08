import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Workflow Improvement Planner", layout="wide")

    st.title("Workflow Improvement Planner")
    st.caption("Local Streamlit app • Fixed workflow: Assessment → Solution → Prototype → Export")

    with st.sidebar:
        st.header("Case Manager")
        st.info("Case CRUD is planned (see docs/roadmap_app_implementation.md).")

        st.divider()
        st.header("Environment")
        st.code(
            "\n".join(
                [
                    "Required:",
                    "- OPENAI_API_KEY",
                    "- TAVILY_API_KEY",
                    "",
                    "Optional:",
                    "- OPENAI_MODEL",
                    "- MAX_CONTEXT_CHARS",
                    "- WEB_SEARCH_RECENCY_DAYS",
                    "- WEB_SEARCH_MAX_RESULTS",
                ]
            ),
            language="text",
        )

    assessment_tab, solution_tab, prototype_tab, export_tab = st.tabs(
        ["Assessment Wizard", "Solution Designer", "Prototype Builder", "Export Pack"]
    )

    with assessment_tab:
        st.subheader("Assessment Wizard")
        st.warning("Not implemented yet. Follow the roadmap in docs/roadmap_app_implementation.md.")

    with solution_tab:
        st.subheader("Solution Designer (Agent 1)")
        st.warning("Not implemented yet. Follow the roadmap in docs/roadmap_app_implementation.md.")

    with prototype_tab:
        st.subheader("Prototype Builder (Agent 2)")
        st.warning("Not implemented yet. Follow the roadmap in docs/roadmap_app_implementation.md.")

    with export_tab:
        st.subheader("Export Pack")
        st.warning("Not implemented yet. Follow the roadmap in docs/roadmap_app_implementation.md.")


if __name__ == "__main__":
    main()
