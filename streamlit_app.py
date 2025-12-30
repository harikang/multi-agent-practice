#!/usr/bin/env python3
"""
Streamlit Chatbot Interface for Research Paper Analysis Multi-Agent System
"""

import streamlit as st
import re
import os
from strands import Agent
from summary_paper_agent import summary_paper_agent
from from_scratch_agent import code_implementation_agent
from idea_paper import idea_generation_agent
from related_paper_agent import arxiv_search_agent
from figure_generation_agent import figure_generation_agent
from strands.models import BedrockModel

# Create a BedrockModel
bedrock_model = BedrockModel(
    model_id="anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-west-1",
    temperature=0.9,
)

# Page configuration
st.set_page_config(
    page_title="Research Paper Analysis Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    .badge-summary { background-color: #D1FAE5; color: #065F46; }
    .badge-code { background-color: #E9D5FF; color: #6B21A8; }
    .badge-idea { background-color: #FED7AA; color: #9A3412; }
    .badge-arxiv { background-color: #FEE2E2; color: #991B1B; }
    .badge-figure { background-color: #BFDBFE; color: #1E40AF; }
</style>
""", unsafe_allow_html=True)

# Initialize the Research Orchestrator
RESEARCH_ORCHESTRATOR_PROMPT = """
You are ResearchOrchestrator, a sophisticated research assistant designed to coordinate comprehensive paper analysis across multiple specialized agents. Your role is to:

1. Analyze incoming research queries and determine which specialized agent(s) to utilize:
   - Paper Summary Agent: For summarizing research papers, extracting key findings
   - Code Implementation Agent: For implementing paper methodologies in Python
   - Idea Generation Agent: For generating novel follow-up research ideas
   - arXiv Search Agent: For finding related papers on arXiv
   - Figure Generation Agent: For visualizing architectures and concepts

2. Key Responsibilities:
   - Accurately understand the user's research needs
   - Route requests to the appropriate specialized agent(s)
   - Coordinate complex multi-step research workflows
   - Provide comprehensive analysis when multiple agents are needed

3. Decision Protocol:
   - If user wants a paper summary → Paper Summary Agent
   - If user wants Python implementation → Code Implementation Agent
   - If user wants follow-up research ideas → Idea Generation Agent
   - If user wants to find related papers → arXiv Search Agent
   - If user wants to visualize/draw architecture → Figure Generation Agent
   - For comprehensive analysis, coordinate multiple agents sequentially

Always provide thorough, well-organized research assistance.
"""


@st.cache_resource
def get_research_orchestrator():
    """Initialize and cache the research orchestrator agent"""
    return Agent(
        system_prompt=RESEARCH_ORCHESTRATOR_PROMPT,
        model=bedrock_model,
        callback_handler=None,
        tools=[
            summary_paper_agent,
            code_implementation_agent,
            idea_generation_agent,
            arxiv_search_agent,
            figure_generation_agent
        ],
    )


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = get_research_orchestrator()


def extract_images_from_text(text):
    """Extract image paths from text (both absolute paths and markdown links)"""
    images = []
    
    # Pattern 1: Absolute paths in text (e.g. /path/to/image.png)
    # Matches paths strictly ending with image extensions, optionally quoted
    path_pattern = r"['\"]?(\/[^'\"\n]+\.(?:png|jpg|jpeg|webp))['\"]?"
    for match in re.finditer(path_pattern, text):
        path = match.group(1)
        if os.path.exists(path):
            images.append(path)
            
    # Pattern 2: Markdown image links ![alt](/path/to/image.png)
    md_pattern = r"!\[.*?\]\((.*?)\)"
    for match in re.finditer(md_pattern, text):
        path = match.group(1)
        # Handle relative paths or file protocols if needed, but focus on existing files
        if path.startswith('file://'):
            path = path.replace('file://', '')
        
        if os.path.exists(path):
            images.append(path)
            
    # Remove duplicates while preserving order
    return list(dict.fromkeys(images))


def display_chat_message(role, content):
    """Display a chat message with appropriate styling and images"""
    with st.chat_message(role):
        st.markdown(content)
        
        # Check for images in the content and display them
        images = extract_images_from_text(content)
        for img_path in images:
            try:
                st.image(img_path, caption="Generated Visualization", use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")


def main():
    """Main Streamlit application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📚 Research Paper Analysis Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered multi-agent system for comprehensive paper analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Agent Capabilities")
        st.markdown("""
        <div>
            <span class="agent-badge badge-summary">📝 Paper Summary</span>
            <span class="agent-badge badge-code">💻 Code Implementation</span>
            <span class="agent-badge badge-idea">💡 Research Ideas</span>
            <span class="agent-badge badge-arxiv">🔍 arXiv Search</span>
            <span class="agent-badge badge-figure">🎨 Figure Gen</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 💡 Example Queries")
        
        example_queries = {
            "📝 Summarize Paper": "Summarize the Attention is All You Need paper",
            "💻 Implement Code": "Implement a simple Transformer attention mechanism in Python",
            "💡 Generate Ideas": "Generate follow-up research ideas for BERT model",
            "🔍 Find Papers": "Find related papers about Vision Transformers on arXiv",
            "🎨 Draw Figure": "Draw the architecture of a Transformer model",
            "🎯 Full Analysis": "Provide a comprehensive analysis of ResNet paper"
        }
        
        for label, query in example_queries.items():
            if st.button(label, use_container_width=True):
                st.session_state.example_query = query
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This multi-agent system helps researchers by:
        - 📖 Summarizing research papers
        - ⚡ Implementing methodologies in Python
        - 🚀 Generating novel research ideas
        - 📚 Finding related work on arXiv
        """)
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])
    
    # Handle example query button clicks
    if "example_query" in st.session_state:
        user_input = st.session_state.example_query
        del st.session_state.example_query
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        display_chat_message("user", user_input)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your query and routing to appropriate agents..."):
                try:
                    response = st.session_state.orchestrator(user_input)
                    assistant_response = str(response)
                    st.markdown(assistant_response)
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                except Exception as e:
                    error_message = f"❌ An error occurred: {str(e)}\n\nPlease try rephrasing your question."
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about research papers..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your query and routing to appropriate agents..."):
                try:
                    response = st.session_state.orchestrator(prompt)
                    assistant_response = str(response)
                    st.markdown(assistant_response)
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                except Exception as e:
                    error_message = f"❌ An error occurred: {str(e)}\n\nPlease try rephrasing your question."
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #64748B; font-size: 0.875rem;">'
        'Powered by Strands Multi-Agent Framework | Built with Streamlit'
        '</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
