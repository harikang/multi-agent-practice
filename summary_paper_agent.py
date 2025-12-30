from strands import Agent, tool
from strands_tools import file_read, file_write, editor

SUMMARY_AGENT_SYSTEM_PROMPT = """
You are PaperSummarizer, a specialized research paper summarization assistant. Your capabilities include:

1. Paper Analysis:
   - Extract key contributions and findings
   - Identify methodology and approach
   - Highlight experimental results
   - Note limitations and future work

2. Summarization Skills:
   - Create concise executive summaries
   - Break down complex concepts into understandable parts
   - Identify paper structure (Abstract, Introduction, Methods, Results, Conclusion)
   - Extract key citations and related work

3. Output Format:
   - Provide structured summaries with clear sections
   - Use bullet points for key findings
   - Highlight novel contributions
   - Mention dataset/benchmark used if applicable

Focus on being comprehensive yet concise, capturing the essence of the research paper.
"""



@tool
def summary_paper_agent(paper_topic: str) -> str:
    """
    Analyze and summarize a research paper based on the given topic or content.
    
    Args:
        paper_topic: Research paper title, topic, or content to summarize
        
    Returns:
        A structured summary of the research paper with key findings and contributions
    """
    formatted_query = f"""Please provide a comprehensive summary of the following research paper topic/content:

{paper_topic}

Include:
1. Main objective and research question
2. Key methodology and approach
3. Main findings and contributions
4. Practical implications
5. Limitations (if mentioned)
"""
    
    try:
        print("🔍 Routed to Paper Summarization Agent")
        
        summarizer_agent = Agent(
            system_prompt=SUMMARY_AGENT_SYSTEM_PROMPT,
            tools=[file_read, file_write, editor],
        )
        agent_response = summarizer_agent(formatted_query)
        text_response = str(agent_response)

        if len(text_response) > 0:

            
            return text_response
        
        return "I apologize, but I couldn't summarize this paper. Please provide more details about the paper."
    except Exception as e:
        return f"Error processing paper summarization: {str(e)}"
