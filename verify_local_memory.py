import sys
import os

# Add current directory to path to import related_paper_agent
sys.path.append(os.getcwd())

from related_paper_agent import local_memory_tool

print("🧬 Testing Local Memory Tool with Direct Faiss Backend...")

# Test 1: Store Information
print("\n1. Testing Store Action...")
try:
    store_result = local_memory_tool(
        action="store", 
        content="The transformer model was introduced in the paper 'Attention Is All You Need' in 2017."
    )
    print(f"Store Result: {store_result}")
except Exception as e:
    print(f"❌ Store failed: {e}")
    sys.exit(1)

# Test 2: Retrieve Information
print("\n2. Testing Retrieve Action...")
try:
    retrieve_result = local_memory_tool(
        action="retrieve", 
        query="transformer"
    )
    print(f"Retrieve Result:\n{retrieve_result}")
    
    if "Attention Is All You Need" in str(retrieve_result):
        print("\n✅ Verification Successful: Content retrieved correctly!")
    else:
        print("\n❌ Verification Failed: Content not found in retrieval results.")
        
except Exception as e:
    print(f"❌ Retrieve failed: {e}")
    sys.exit(1)
