#!/usr/bin/env python3
"""
Example usage script for the Research Paper Analysis Multi-Agent System
"""

from strands import Agent
from summary_paper_agent import summary_paper_agent
from from_scratch_agent import code_implementation_agent
from idea_paper import idea_generation_agent
from related_paper_agent import arxiv_search_agent


def example_usage():
    """Demonstrate the multi-agent system with example queries"""
    
    print("=" * 80)
    print("📚 Research Paper Analysis Multi-Agent System - Example Usage")
    print("=" * 80)
    
    # Create the orchestrator
    RESEARCH_ORCHESTRATOR_PROMPT = """
You are ResearchOrchestrator, a sophisticated research assistant designed to coordinate 
comprehensive paper analysis across multiple specialized agents.
"""
    
    research_orchestrator = Agent(
        system_prompt=RESEARCH_ORCHESTRATOR_PROMPT,
        callback_handler=None,
        tools=[
            summary_paper_agent,
            code_implementation_agent,
            idea_generation_agent,
            arxiv_search_agent
        ],
    )
    
    # Example 1: Paper Summary
    print("\n" + "=" * 80)
    print("Example 1: Summarizing 'Attention is All You Need' paper")
    print("=" * 80)
    query1 = "Summarize the 'Attention is All You Need' paper"
    print(f"\nQuery: {query1}\n")
    response1 = research_orchestrator(query1)
    print(response1)
    
    # Example 2: Code Implementation
    print("\n\n" + "=" * 80)
    print("Example 2: Implementing a simple Transformer attention mechanism")
    print("=" * 80)
    query2 = "Implement a simplified self-attention mechanism from Transformer in Python"
    print(f"\nQuery: {query2}\n")
    response2 = research_orchestrator(query2)
    print(response2)
    
    # Example 3: Idea Generation
    print("\n\n" + "=" * 80)
    print("Example 3: Generating follow-up research ideas for BERT")
    print("=" * 80)
    query3 = "Generate follow-up research ideas for BERT model"
    print(f"\nQuery: {query3}\n")
    response3 = research_orchestrator(query3)
    print(response3)
    
    # Example 4: arXiv Search
    print("\n\n" + "=" * 80)
    print("Example 4: Finding related papers on Vision Transformers")
    print("=" * 80)
    query4 = "Find related papers about Vision Transformers on arXiv"
    print(f"\nQuery: {query4}\n")
    response4 = research_orchestrator(query4)
    print(response4)
    
    print("\n\n" + "=" * 80)
    print("✅ Example usage complete!")
    print("=" * 80)


if __name__ == "__main__":
    example_usage()
