# 📚 Multi-Agent Research Paper Analysis System

A sophisticated multi-agent system built with the **Strands** framework, designed to help researchers analyze, implement, and explore research papers comprehensively.

## 🚀 Overview

This system coordinates specialized AI agents to handle different aspects of research, from summarization to code implementation and connecting related works.

## 🏗️ Architecture & Agent Hierarchy

The system operates on a **Hub-and-Spoke** architecture. The **Research Orchestrator** acts as the central brain, routing user intents to the appropriate sub-agents.

### 🧠 **Research Orchestrator** (`research_orchestrator.py`)
- **Role**: Central controller. Analyzes user queries and dispatches tasks.
- **Tools**: Manages the following sub-agents:

    ### 1. 📝 **Paper Summary Agent** (`summary_paper_agent.py`)
    - **Role**: Summarizes papers and extracts architecture figures.
    - **Tools**:
        - `file_read`, `file_write`, `editor`: Manage local paper files and summary drafts.
        - `http_request`: Fetch paper contents or figures from the web (e.g., Ar5iv).
        - `shell`: Advanced command execution for file handling.

    ### 2. 💻 **Code Implementation Agent** (`from_scratch_agent.py`)
    - **Role**: Implements paper algorithms in Python.
    - **Tools**:
        - `python_repl`: **Executes Python code** to verify implementation logic.
        - `file_read`, `file_write`, `editor`: writes source code to files.

    ### 3. 💡 **Idea Generation Agent** (`idea_paper.py`)
    - **Role**: Brainstorms follow-up research ideas.
    - **Tools**: *Pure LLM* (Relies on internal model knowledge).

    ### 4. 🔍 **Related Paper Agent** (`related_paper_agent.py`)
    - **Role**: Searches arXiv and maintains a persistent Knowledge Base.
    - **Tools**:
        - `shell`: Executes `curl` or CLI commands to query the arXiv API.
        - `http_request`: Sends HTTP GET requests to API endpoints.
        - **`local_memory_tool` (CUSTOM 🛠️)**:
            - **Purpose**: Local RAG (Retrieval-Augmented Generation) system.
            - **Backend**: Uses **SentenceTransformer** for embeddings and **Faiss** for vector storage.
            - **Actions**:
                - `store`: Vectorize and save information.
                - `retrieve`: Semantic search for relevant paper context.

    ### 5. 🎨 **Figure Generation Agent** (`figure_generation_agent.py`)
    - **Role**: Visualizes architectures conceptually.
    - **Tools**:
        - `generate_image`: DALL-E/ImageGen tool to create visual drafts.

## 🧠 Local Memory System

The `related_paper_agent` features a fully local memory system to ensure privacy and persistence:
- **Vector Store**: `faiss-cpu`
- **Location**: `local_memory_db/`
- **Embeddings**: `all-MiniLM-L6-v2` (via `sentence-transformers`)

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

## 🛠️ Usage

### 1. Web Interface (Streamlit)
```bash
streamlit run streamlit_app.py
```

### 2. CLI Mode
```bash
python research_orchestrator.py
```

---
*Powered by Strands Framework & AWS Bedrock*
