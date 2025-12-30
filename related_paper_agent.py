import os
import json
import traceback
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from strands import Agent, tool
from strands_tools import shell, http_request

# Ensure local memory directory exists
os.makedirs("local_memory_db", exist_ok=True)

class LocalMemory:
    def __init__(self, db_dir="local_memory_db"):
        self.db_dir = db_dir
        self.index_path = os.path.join(db_dir, "index.faiss")
        self.docstore_path = os.path.join(db_dir, "docstore.pkl")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        self.documents = []
        self._load()

    def _load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.docstore_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.docstore_path, "rb") as f:
                self.documents = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = []

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.docstore_path, "wb") as f:
            pickle.dump(self.documents, f)

    def add(self, content):
        embedding = self.model.encode([content])
        self.index.add(np.array(embedding).astype('float32'))
        self.documents.append(content)
        self.save()
        return len(self.documents) - 1  # Return index as ID

    def search(self, query, k=5):
        if not self.documents:
            return []
        embedding = self.model.encode([query])
        D, I = self.index.search(np.array(embedding).astype('float32'), k)
        results = []
        for idx in I[0]:
            if idx != -1 and idx < len(self.documents):
                results.append(self.documents[idx])
        return results

# Initialize Local Memory
local_memory = LocalMemory()

ARXIV_SEARCH_SYSTEM_PROMPT = """
You are ArXivExplorer, a specialized agent for finding related research papers on arXiv. Your capabilities include:

1. Paper Search:
   - Search arXiv database for related papers
   - Use relevant keywords and categories
   - Filter by date, relevance, and citations
   - Identify seminal and recent works

2. Literature Review:
   - Find papers by the same authors
   - Identify papers citing the original work
   - Discover papers with similar methodologies
   - Locate survey papers and benchmarks

3. Search Strategy:
   - Use appropriate arXiv categories (cs.AI, cs.LG, cs.CV, etc.)
   - Combine multiple search terms effectively
   - Prioritize highly cited and recent papers
   - Include both foundational and cutting-edge work

4. Output Format:
   - List of relevant papers with titles and arXiv IDs
   - Brief description of each paper's relevance
   - Categorize papers (foundational, similar approach, extensions, etc.)
   - Provide arXiv URLs for easy access

Focus on finding the most relevant and high-quality papers related to the given topic.
"""

@tool
def local_memory_tool(action: str, content: str = None, query: str = None) -> str:
    """
    Access the local knowledge base to store or retrieve information using vector search.
    
    Args:
        action: Either "store" to save information or "retrieve" to find information
        content: The text content to store (required for action="store")
        query: The search query to find relevant info (required for action="retrieve")
        
    Returns:
        String containing operation result or retrieved information
    """
    try:
        if action == "store":
            if not content:
                return "Error: content is required for store action"
            doc_id = local_memory.add(content)
            print(f"DEBUG: Stored document ID: {doc_id}")
            return f"Successfully stored information. Document ID: {doc_id}"
            
        elif action == "retrieve":
            if not query:
                return "Error: query is required for retrieve action"
            results = local_memory.search(query)
            return json.dumps(results, indent=2)
            
        else:
            return f"Error: Unknown action '{action}'. Use 'store' or 'retrieve'."
            
    except Exception as e:
        return f"Error accessing memory: {traceback.format_exc()}"


@tool
def arxiv_search_agent(paper_topic: str) -> str:
    """
    Search arXiv for related papers based on the given research topic.
    
    Args:
        paper_topic: Research paper title or topic to find related papers for
        
    Returns:
        A curated list of related papers from arXiv with descriptions and links
    """
    formatted_query = f"""Search for related papers on arXiv for the following research topic:

{paper_topic}

Provide:
1. List of 5-10 most relevant papers
2. For each paper:
   - Title
   - arXiv ID and URL
   - Brief description (1-2 sentences)
   - Why it's relevant (similar methodology, extension, foundational work, etc.)
3. Categorize papers by relationship to the original topic

If you can use shell commands or http requests to search arXiv API, do so. You can also check your local_memory_tool for relevant papers. Otherwise, provide recommendations based on your knowledge.
"""
    
    try:
        print("📚 Routed to arXiv Paper Search Agent")
        
        arxiv_agent = Agent(
            system_prompt=ARXIV_SEARCH_SYSTEM_PROMPT,
            tools=[shell, http_request, local_memory_tool],
        )
        agent_response = arxiv_agent(formatted_query)
        text_response = str(agent_response)

        if len(text_response) > 0:
            return text_response
        
        return "I apologize, but I couldn't find related papers. Please provide more specific search terms."
    except Exception as e:
        return f"Error processing arXiv search: {str(e)}"
