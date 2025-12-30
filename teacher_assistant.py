#!/usr/bin/env python3
"""
# � Research Paper Analysis Orchestrator

A specialized Strands agent orchestrator for comprehensive research paper analysis.
Coordinates multiple specialized agents to analyze, implement, and explore research papers.

## What This System Does

This multi-agent system helps researchers by:
- Summarizing research papers
- Implementing papers in Python code
- Generating follow-up research ideas
- Finding related papers on arXiv
"""

from strands import Agent
from strands.models import BedrockModel
from summary_paper_agent import summary_paper_agent
from from_scratch_agent import code_implementation_agent
from idea_paper import idea_generation_agent
from related_paper_agent import arxiv_search_agent


RESEARCH_ORCHESTRATOR_PROMPT = """
You are ResearchOrchestrator, a sophisticated research assistant designed to coordinate comprehensive paper analysis across multiple specialized agents. Your role is to:

1. Analyze incoming research queries and determine which specialized agent(s) to utilize:
   - Paper Summary Agent: For summarizing research papers, extracting key findings
   - Code Implementation Agent: For implementing paper methodologies in Python
   - Idea Generation Agent: For generating novel follow-up research ideas
   - arXiv Search Agent: For finding related papers on arXiv

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
   - For comprehensive analysis, coordinate multiple agents sequentially

4. Workflow Examples:
   - "Analyze this paper" → Use Summary Agent
   - "Implement this paper in Python" → Use Code Implementation Agent
   - "What are some follow-up ideas for this paper?" → Use Idea Generation Agent
   - "Find related papers on arXiv" → Use arXiv Search Agent
   - "Full research analysis" → Use all agents in sequence

Always provide thorough, well-organized research assistance.
"""

bedrock_model = BedrockModel(
    model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    region_name="us-west-2",
    temperature=0.9,
)

# Create the orchestrator agent with all specialized agents as tools
research_orchestrator = Agent(
    system_prompt=RESEARCH_ORCHESTRATOR_PROMPT,
    model=bedrock_model,
    callback_handler=None,
    tools=[
        summary_paper_agent,
        code_implementation_agent,
        idea_generation_agent,
        arxiv_search_agent
    ],
)


# Example usage
if __name__ == "__main__":
    print("\n� Research Paper Analysis Multi-Agent System �\n")
    print("=" * 60)
    print("Available capabilities:")
    print("  1. 📝 Summarize research papers")
    print("  2. 💻 Implement papers in Python")
    print("  3. 💡 Generate follow-up research ideas")
    print("  4. 🔍 Find related papers on arXiv")
    print("=" * 60)
    print("\nType 'exit' to quit.\n")

    # Interactive loop
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() == "exit":
                print("\n👋 Goodbye! Happy researching!")
                break

            print("\n" + "=" * 60)
            response = research_orchestrator(user_input)
            
            # Extract and print the response
            content = str(response)
            print(content)
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Execution interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Please try asking a different question.")
