from strands import Agent, tool

IDEA_GENERATION_SYSTEM_PROMPT = """
You are ResearchIdeator, a specialized agent for generating novel research ideas based on existing papers. Your capabilities include:

1. Idea Generation:
   - Identify limitations and gaps in current research
   - Propose novel extensions and improvements
   - Suggest interdisciplinary applications
   - Generate testable hypotheses

2. Research Strategy:
   - Analyze current state-of-the-art approaches
   - Identify unexplored directions
   - Propose incremental and radical improvements
   - Consider practical feasibility

3. Innovation Areas:
   - Methodological improvements
   - Novel application domains
   - Addressing current limitations
   - Combining multiple approaches
   - Scale and efficiency improvements

4. Output Format:
   - Multiple concrete research ideas (3-5 ideas)
   - Clear motivation for each idea
   - Expected contributions and impact
   - Potential challenges and solutions
   - Suggested experiments or validation methods

Focus on generating creative yet feasible research directions that build upon the given paper.
"""


@tool
def idea_generation_agent(paper_topic: str) -> str:
    """
    Generate novel research ideas for follow-up papers based on the given paper topic.
    
    Args:
        paper_topic: Research paper title or topic to generate follow-up ideas for
        
    Returns:
        A list of novel research ideas with motivations and potential impact
    """
    formatted_query = f"""Based on the following research paper topic, generate novel follow-up research ideas:

{paper_topic}

For each idea, provide:
1. Idea title and brief description
2. Motivation (what gap/limitation it addresses)
3. Expected contributions and novelty
4. Potential challenges
5. Suggested approach or methodology

Generate 3-5 concrete, feasible research ideas.
"""
    
    try:
        print("💡 Routed to Research Idea Generation Agent")
        
        idea_agent = Agent(
            system_prompt=IDEA_GENERATION_SYSTEM_PROMPT,
            tools=[],
        )
        agent_response = idea_agent(formatted_query)
        text_response = str(agent_response)

        if len(text_response) > 0:
            return text_response
        
        return "I apologize, but I couldn't generate research ideas. Please provide more details about the paper."
    except Exception as e:
        return f"Error processing idea generation: {str(e)}"
