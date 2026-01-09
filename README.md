# Workflow Improvement Planner

> **A local, agentic AI workspace for analyzing business processes and generating technical blueprints.**

The **Workflow Improvement Planner** is a Streamlit application that guides you through assessing a business process, analyzing bottlenecks with AI agents, and generating a complete implementation roadmap. It runs locally on your machine, keeping your data private while leveraging OpenAI and Tavily for intelligence and research.

---

## 🚀 Features

- **📂 Local Case Management**: Create, manage, and persist cases on your filesystem. No external database required.
- **📝 Assessment Wizard**: Structured intake form for processes (pain points, actors, systems, SLAs) with attachment support (PDF/TXT/MD).
- **🔍 Automatic Industry Research**: Pre-gathers relevant tools, best practices, and compliance requirements before analysis.
- **🤖 Dual-Agent Workflow**:
  - **Solution Designer**: Analyzes the assessment to identify bottlenecks, opportunities, and strategic recommendations.
  - **Prototype Builder**: Generates a technical architectural blueprint based on the solution design.
- **✅ Consistency Validator**: QA agent that checks alignment between Solution and Prototype outputs, flagging issues.
- **🔄 Smart Revision**: Automatic revision cycle when critical misalignments are detected.
- **🌐 Smart Research**: Agents can browse the web (via Tavily) to find relevant integrations, tools, and best practices.
- **📦 Export Pack**: One-click generation of a downloadable ZIP containing an Executive Brief, Technical Blueprint, Implementation Plan, and Validation Report.

---

## 🛠️ Quick Start

### Prerequisites
- Python 3.10+
- [Poetry](https://python-poetry.org/) (recommended) or pip
- OpenAI API Key
- Tavily API Key (for web research)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/workflow-improvement-planner.git
   cd workflow-improvement-planner
   ```

2. **Install dependencies**
   ```bash
   poetry install
   ```

3. **Configure environment**
   Copy the example file and add your keys:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```ini
   OPENAI_API_KEY=sk-...
   TAVILY_API_KEY=tvly-...
   OPENAI_MODEL=gpt-4o-mini
   ```

4. **Run the app**
   ```bash
   poetry run streamlit run workflow_planner.py
   ```

---

## 📖 Enhanced Workflow

The application now features an intelligent, multi-stage workflow:

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Assessment │ ──▶ │ Industry Research│ ──▶ │ Solution Designer│
└─────────────┘     └──────────────────┘     └──────────────────┘
                                                      │
                                                      ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Export    │ ◀── │    Validator     │ ◀── │ Prototype Builder│
└─────────────┘     └──────────────────┘     └──────────────────┘
                           │
                           ▼ (if critical issues)
                    ┌──────────────────┐
                    │     Revision     │
                    └──────────────────┘
```

### Step-by-Step

1. **Create a Case**: Open the sidebar, enter a name (e.g., "Vendor Onboarding"), area, and tags.
2. **Fill Assessment**: Go to the **Assessment** tab. Describe the current process, upload documentation, and save.
3. **Run Industry Research** (automatic or manual): The system searches for relevant tools, best practices, and compliance requirements.
4. **Run Solution Designer**: The agent analyzes your inputs with research context and produces a strategic report.
5. **Run Prototype Builder**: Creates a technical spec aligned with the solution recommendations.
6. **Consistency Validation**: Automatically checks for alignment issues between Solution and Prototype.
7. **Export**: Download the full report bundle including validation results.

---

## 🏗️ Architecture

The application follows a clean, modular structure powered by **LangGraph** for orchestration and **Streamlit** for the UI.

```mermaid
flowchart LR
    A[Assessment] --> B[Industry Research]
    B --> C[Solution Designer]
    C --> D[Prototype Builder]
    D --> E{Validator}
    E -->|Approved| F[Export Pack]
    E -->|Critical Issues| G[Revision]
    G --> E
    B -.->|Tavily| H[Web Search]
    C -.->|Tavily| H
    D -.->|Tavily| H
    E -.->|Tavily| H
```

### File Structure
```text
.
├── workflow_planner.py  # Main entrypoint & UI components
├── config.py           # Centralized configuration & environment vars
├── schemas.py          # Pydantic models & TypedDicts
├── utils.py            # Persistence, file I/O, research, & helper functions
├── prompts.py          # LLM system prompts for all agents
├── data/               # Local storage (gitignored)
└── docs/               # Documentation & PRDs
```

### Agents

| Agent | Purpose | Max Searches |
|-------|---------|--------------|
| **Industry Research** | Gathers tools, best practices, compliance info | 4 queries |
| **Solution Designer** | Strategic analysis and recommendations | 2 calls |
| **Prototype Builder** | Technical blueprint generation | 3 calls |
| **Consistency Validator** | QA alignment check | 2 calls |

---

## ⚙️ Configuration

Control the application behavior via `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Required for agents. | - |
| `TAVILY_API_KEY` | Required for web search and research. | - |
| `OPENAI_MODEL` | LLM model to use. | `gpt-4o-mini` |
| `MAX_CONTEXT_CHARS` | Limit for input context size. | `60000` |
| `WEB_SEARCH_MAX_RESULTS` | Max links per search query. | `5` |
| `WEB_SEARCH_RECENCY_DAYS` | How recent search results should be. | `365` |

---

## 🔒 Data Privacy

- **Local First**: All case data, assessments, and outputs are stored in the `data/` directory on your local machine.
- **No Cloud Storage**: We do not send your data to any proprietary cloud database.
- **LLM Privacy**: Data is sent to OpenAI/Tavily only for processing. Refer to their respective privacy policies regarding API usage.

---

## ✅ Validation Report

The Consistency Validator checks:

- **Strategic-Technical Alignment**: Does the prototype implement the solution's recommendations?
- **Scope Consistency**: Are both documents addressing the same problems?
- **Feasibility**: Is the technical approach realistic given constraints?
- **Completeness**: Are there gaps in coverage?
- **Risk Acknowledgment**: Are solution risks addressed in prototype mitigations?
- **Timeline Realism**: Do phases match the urgency/complexity?

Results are marked as:
- ✅ **Aligned**: No issues
- ⚠️ **Minor Concern**: Noted but not blocking
- 🚫 **Critical Issue**: Must be addressed

---

## 🤝 Contributing

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

Distributed under the MIT License.
