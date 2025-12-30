from strands import Agent, tool
from strands_tools import generate_image

FIGURE_GENERATION_SYSTEM_PROMPT = """
You are FigureArchitect, a specialized agent for visualizing software architectures and research paper concepts. Your capabilities include:

1. Visual Generation:
   - Create conceptual diagrams of system architectures
   - Visualize data flows and model structures (e.g., Transformers, Multi-Agent Systems)
   - Generate "temp" figures to visualize abstract ideas

2. Presentation Guide:
   - Provide step-by-step instructions to recreate the generated figure in PowerPoint (PPT)
   - Suggest shapes, colors, and layout strategies for professional presentation
   - Explain how to represent relationships (arrows, grouping, layers)

How to work:
1. FIRST, use the `generate_image` tool to create a visual representation of the requested architecture or concept. Use a descriptive prompt for the image generation.
2. AFTER generating the image, provide a detailed text response containing:
   - The path to the generated image.
   - A "PPT Recreation Guide": Numbered steps on how to draw this professionally in PowerPoint using standard shapes (rectangles, cylinders, arrows).

Focus on clarity and professional aesthetics in your instructions.
"""

@tool
def figure_generation_agent(architecture_description: str) -> str:
    """
    Generate a visual figure for a software architecture or concept, and provide PPT drawing instructions.
    
    Args:
        architecture_description: Description of the architecture, system, or concept to visualize.
        
    Returns:
        Path to the generated image and text instructions for PPT recreation.
    """
    formatted_query = f"""Visualize the following architecture/concept and provide PPT instructions:

{architecture_description}
"""
    
    try:
        print("🎨 Routed to Figure Generation Agent")
        
        figure_agent = Agent(
            system_prompt=FIGURE_GENERATION_SYSTEM_PROMPT,
            tools=[generate_image],
        )
        agent_response = figure_agent(formatted_query)
        return str(agent_response)
        
    except Exception as e:
        return f"Error creating figure: {str(e)}"
