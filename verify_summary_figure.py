import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from summary_paper_agent import summary_paper_agent

print("📝 Testing Summary Paper Agent with Figure Extraction...")

# Test Query
test_query = "Attention Is All You Need"

print(f"\nQuery: {test_query}")
print("-" * 50)

try:
    result = summary_paper_agent(test_query)
    print("\n✅ Agent Response Snippet:")
    print(result[:500] + "...") # Print first 500 chars
    
    # Check for local path
    if "downloaded_figures" in result and ".png" in result:
        print("\n✅ Verification Successful: Local figure path found in response.")
    else:
        print("\n❌ Verification Failed: Local figure path NOT found.")
        # Debug: check if functions work
        from summary_paper_agent import search_arxiv_id, extract_overview_figure
        aid = search_arxiv_id(test_query)
        print(f"Debug - ArXiv ID: {aid}")
        if aid:
            img_path = extract_overview_figure(aid)
            print(f"Debug - Extracted Image Path: {img_path}")
        sys.exit(1)
        
except Exception as e:
    print(f"\n❌ Verification Failed with Exception: {e}")
    sys.exit(1)
