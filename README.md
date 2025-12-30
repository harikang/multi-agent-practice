# 📚 Multi-Agent Research Paper Analysis System

A sophisticated multi-agent system built with the **Strands** framework, designed to help researchers analyze, implement, and explore research papers comprehensively.

## 🚀 Overview

This system coordinates four specialized AI agents to handle different aspects of research:
1.  **Summary Agent**: Digests complex papers into clear summaries, fetching **real architecture figures** from Ar5iv.
2.  **Code Agent**: Implements paper methodologies in Python (PyTorch/TensorFlow).
3.  **Idea Agent**: Generates novel follow-up research directions.
4.  **Figure Generation Agent**: Visualizes architectures conceptually and provides PPT drawing guides.
5.  **Related Paper Agent**: Searches arXiv and maintains a **local knowledge base**.

## 🏗️ Architecture

The system uses a Hub-and-Spoke model where a central **Orchestrator** (powered by Claude 3.5 Sonnet on Bedrock) routes user queries to the most appropriate specialized agent.

### 🤖 Agents

| Agent | File | Responsibilities |
|-------|------|------------------|
| **Research Orchestrator** | `research_orchestrator.py` | Central brain. Routes tasks to specialized agents based on user intent. |
| **Summary Paper Agent** | `summary_paper_agent.py` | Summarizes papers. **Features:** Auto-downloads paper figures from Ar5iv and displays them inline. Uses `http_request` for full-text access. |
| **Code Implementation Agent** | `from_scratch_agent.py` | Writes production-ready Python code to implement algorithms from papers. |
| **Idea Generation Agent** | `idea_paper.py` | Brainstorms 3-5 novel follow-up research ideas based on a paper. |
| **Figure Generation Agent** | `figure_generation_agent.py` | Generates conceptual diagrams and provides step-by-step PPT creation guides. |
| **Related Paper & Memory Agent** | `related_paper_agent.py` | Searches arXiv for related work and stores/retrieves knowledge using a **local Faiss Vector DB**. |

### 🧠 Local Memory & Persistence

- **Local Vector Store**: `related_paper_agent` uses `faiss-cpu` to store paper embeddings locally in `local_memory_db/`.
- **Figure Downloads**: `summary_paper_agent` automatically downloads extracted figures to `downloaded_figures/` for reliable display.

## 💻 Installation

1.  **Prerequisites**: Python 3.9+
2.  **Virtual Environment**:
    ```bash
    python -m venv harivenv
    source harivenv/bin/activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Key deps: `strands`, `streamlit`, `faiss-cpu`, `sentence-transformers`, `beautifulsoup4`, `requests`)*

## 🛠️ Usage

### 1. Web Interface (Recommended)

Run the polished Streamlit Chatbot UI:

```bash
streamlit run streamlit_app.py
```

- **Features**:
  - 💬 Interactive chat with multi-agent routing
  - 🖼️ **Inline Image Preview**: Instantly view extracted figures or generated diagrams
  - 🧩 Visual badges for active agents

### 2. CLI Mode

Run the system directly in your terminal:

```bash
python research_orchestrator.py
```

## 📂 File Structure

- **`streamlit_app.py`**: The web application entry point.
- **`research_orchestrator.py`**: Main agent definition & prompt.
- **`teacher_assistant.py`**: Alternative orchestrator entry point (Legacy).
- **`summary_paper_agent.py`**: Summary & Figure Extraction logic.
- **`figure_generation_agent.py`**: Image generation logic.
- **`from_scratch_agent.py`**: Code generation logic.
- **`related_paper_agent.py`**: ArXiv search & Local Memory logic.
- **`downloaded_figures/`**: Directory where extracted paper figures are saved.
- **`local_memory_db/`**: Directory where Faiss index is persisted.

---
*Powered by Strands Framework & AWS Bedrock*
