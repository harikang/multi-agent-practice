# 📚 Multi-Agent Research Paper Analysis System

A sophisticated multi-agent system built with the **Strands** framework, designed to help researchers analyze, implement, and explore research papers comprehensively.

## 🚀 Overview

This system coordinates four specialized AI agents to handle different aspects of research:
1.  **Summary Agent**: Digests complex papers into clear summaries.
2.  **Code Agent**: Implements paper methodologies in Python (PyTorch/TensorFlow).
3.  **Idea Agent**: Generates novel follow-up research directions.
4.  **Related Paper Agent**: Searches arXiv and maintains a **local knowledge base**.

## 🏗️ Architecture

The system uses a Hub-and-Spoke model where a central **Orchestrator** routes user queries to the most appropriate specialized agent.

### 🤖 Agents

| Agent | File | Responsibilities |
|-------|------|------------------|
| **Research Orchestrator** | `research_orchestrator.py` | Central brain. Analyzes user intent and routes tasks to specialized agents. |
| **Summary Paper Agent** | `summary_paper_agent.py` | Summarizes papers, extracts key innovations and results. |
| **Code Implementation Agent** | `from_scratch_agent.py` | Writes production-ready Python code to implement algorithms from papers. |
| **Idea Generation Agent** | `idea_paper.py` | Brainstorms 3-5 novel follow-up research ideas based on a paper. |
| **Related Paper & Memory Agent** | `related_paper_agent.py` | Searches arXiv for related work and stores/retrieves knowledge using a **local Faiss Vector DB**. |

### � Local Memory System

The `related_paper_agent` features a fully local memory system:
- **Vector Store**: `faiss-cpu`
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Storage**: `local_memory_db/` (Persisted locally)
- **Privacy**: No external API calls (e.g., OpenAI) required for memory operations.

## 💻 Installation

1.  **Prerequisites**: Python 3.9+
2.  **Install Dependencies**:

    ```bash
    pip install strands streamlit faiss-cpu sentence-transformers numpy torch
    ```
    *(Note: `ollama` and `beautifulsoup4` may be required depending on specific tool usage)*

## 🛠️ Usage

### 1. Web Interface (Recommended)

Run the polished Streamlit Chatbot UI:

```bash
streamlit run streamlit_app.py
```

- Provides interactive chat interface
- Visual status indicators for active agents
- Easy-to-use example buttons

### 2. CLI Mode

Run the system directly in your terminal:

```bash
python research_orchestrator.py
```

## � File Structure

- **`streamlit_app.py`**: The web application entry point.
- **`research_orchestrator.py`** (aka `teacher_assistant.py`): Main agent definition.
- **`related_paper_agent.py`**: Contains `arxiv_search_agent` and `LocalMemory` class.
- **`summary_paper_agent.py`**: Contains `summary_paper_agent`.
- **`from_scratch_agent.py`**: Contains `code_implementation_agent`.
- **`idea_paper.py`**: Contains `idea_generation_agent`.
- **`verify_local_memory.py`**: Script to test the local Faiss memory system.
- **`TRANSFORMER_README.md`**: Documentation for the Transformer implementation files (`transformer.py`, etc.) also present in this directory.

---
*Powered by Strands Framework*
