import os
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from strands import Agent, tool
from strands_tools import file_read, file_write, editor, http_request, shell

SUMMARY_AGENT_SYSTEM_PROMPT = """
You are ResearchSummarizer, a specialized agent for reading and summarizing research papers. Your goal is to provide comprehensive, accurate, and readable summaries.

Your summary should include:
1. Core Problem & Motivation
2. Key Methodology/Approach
3. Main Results & Contributions
4. Limitations & Future Work

Structure the output in clear Markdown with headers. 
If a local image path is provided in the context, mention it in the text (e.g., "See the Model Architecture figure below") but do not try to embed it with markdown syntax `![...](path)` as it may not render in all browsers. The system will display the image automatically.
"""

def search_arxiv_id(query):
    """Search arXiv API for the most relevant paper ID."""
    try:
        # Sort by relevance to get the exact match title first
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=1&sortBy=relevance&sortOrder=descending"
        response = requests.get(url)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            # Namespace for atom
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entry = root.find('atom:entry', ns)
            if entry:
                id_text = entry.find('atom:id', ns).text
                # Extract ID from http://arxiv.org/abs/1706.03762v5 -> 1706.03762
                return id_text.split('/abs/')[-1].split('v')[0]
    except Exception as e:
        print(f"Error searching arXiv: {e}")
    return None

def download_figure(url, arxiv_id):
    """Download figure to a local directory."""
    try:
        save_dir = os.path.join(os.getcwd(), "downloaded_figures")
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a filename from the URL or ID
        filename = f"{arxiv_id}_overview.png"
        filepath = os.path.join(save_dir, filename)
        
        # Download
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return filepath
    except Exception as e:
        print(f"Error downloading figure: {e}")
    return None

def extract_overview_figure(arxiv_id):
    """Extract and download the most likely overview figure from Ar5iv HTML."""
    if not arxiv_id:
        return None
        
    url = f"https://ar5iv.org/html/{arxiv_id}"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        figures = soup.find_all('figure')
        
        score_map = {}
        for fig in figures:
            caption = fig.find('figcaption')
            caption_text = caption.get_text().lower() if caption else ""
            img = fig.find('img')
            if not img or not img.get('src'):
                continue
                
            img_src = img.get('src')
            if img_src.startswith('/'):
                 full_img_url = f"https://ar5iv.org{img_src}"
            else:
                 full_img_url = f"https://ar5iv.org/html/{arxiv_id}/{img_src}"

            score = 0
            keywords = ['overview', 'architecture', 'model', 'framework', 'schematic', 'figure 1', 'fig. 1']
            for word in keywords:
                if word in caption_text:
                    score += 1
            if "figure 1" in caption_text or "fig. 1" in caption_text:
                score += 2
                
            if score > 0:
                score_map[full_img_url] = score

        if score_map:
            best_url = max(score_map, key=score_map.get)
            print(f"Found best figure URL: {best_url}")
            return download_figure(best_url, arxiv_id)
            
    except Exception as e:
        print(f"Error extracting figure: {e}")
    return None

@tool
def summary_paper_agent(paper_topic: str) -> str:
    """
    Summarize a research paper given its topic or title.
    Automatically finds and includes the paper's architecture figure if available.
    
    Args:
        paper_topic: The title or topic of the research paper.
        
    Returns:
        A formatted markdown summary, including the paper's architecture diagram if found.
    """
    formatted_query = f"""Please summarize the research paper related to: "{paper_topic}".
    
Provide a structured summary focusing on the problem, methodology, and results.
If you need to read the full paper, you can use http_request to fetch it from arXiv or ar5iv.
"""
    
    try:
        print("📝 Routed to Summary Paper Agent")
        
        # 1. Try to find the architecture figure first
        arxiv_id = search_arxiv_id(paper_topic)
        local_figure_path = extract_overview_figure(arxiv_id)
        
        context_msg = ""
        if local_figure_path:
            print(f"🖼️  Downloaded architecture figure to: {local_figure_path}")
            # Explicitly tell the LLM to use this image path reference
            context_msg = f"\n\n[IMPORTANT] I have downloaded the official model architecture figure for this paper to: {local_figure_path}. \nPlease mention this in your text, for example: 'The architecture is shown in the figure below: {local_figure_path}'."
            formatted_query += context_msg
        else:
            print("⚠️  Could not find architecture figure automatically.")

        summarizer_agent = Agent(
            system_prompt=SUMMARY_AGENT_SYSTEM_PROMPT,
            tools=[file_read, file_write, editor, http_request, shell],
        )
        agent_response = summarizer_agent(formatted_query)
        return str(agent_response)
        
    except Exception as e:
        return f"Error processing summary: {str(e)}"
