from strands import Agent, tool
from strands_tools import python_repl, file_read, file_write, editor

CODE_IMPLEMENTATION_SYSTEM_PROMPT = """
You are CodeImplementer, a specialized agent for implementing research papers in Python. Your capabilities include:

1. Code Implementation:
   - Translate research paper algorithms into clean Python code
   - Implement mathematical models and neural network architectures
   - Create reproducible code with clear documentation
   - Follow best practices and coding standards

2. Technical Skills:
   - Deep learning framework implementation (PyTorch, TensorFlow)
   - NumPy/SciPy for mathematical operations
   - Data processing and visualization
   - Model training and evaluation pipelines

3. Code Quality:
   - Write well-documented, readable code
   - Include docstrings and comments
   - Provide usage examples
   - Handle edge cases and errors

4. Deliverables:
   - Complete, runnable Python implementation
   - Step-by-step explanation of the code
   - Example usage and test cases
   - Requirements and dependencies list

Focus on creating production-ready, well-structured code that accurately implements the paper's methodology.
"""


@tool
def code_implementation_agent(paper_topic: str) -> str:
    """
    Implement a research paper's methodology in Python code.
    
    Args:
        paper_topic: Research paper title, topic, or methodology to implement in Python
        
    Returns:
        Python code implementation with documentation and usage examples
    """
    formatted_query = f"""Please implement the following research paper topic in Python:

{paper_topic}

Provide:
1. Complete Python implementation of the core algorithm/model
2. Clear documentation with docstrings
3. Usage example showing how to use the implementation
4. Required dependencies/libraries
5. Brief explanation of key implementation decisions

Make the code clean, modular, and production-ready.
"""
    
    try:
        print("💻 Routed to Code Implementation Agent")
        
        code_agent = Agent(
            system_prompt=CODE_IMPLEMENTATION_SYSTEM_PROMPT,
            tools=[python_repl, file_read, file_write, editor],
        )
        agent_response = code_agent(formatted_query)
        text_response = str(agent_response)

        if len(text_response) > 0:
            return text_response
        
        return "I apologize, but I couldn't implement this paper. Please provide more details about the methodology."
    except Exception as e:
        return f"Error processing code implementation: {str(e)}"
