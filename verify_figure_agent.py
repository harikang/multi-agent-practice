import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from figure_generation_agent import figure_generation_agent

print("🎨 Testing Figure Generation Agent...")

# Test Query
test_query = "A simple client-server architecture with a load balancer."

print(f"\nQuery: {test_query}")
print("-" * 50)

try:
    result = figure_generation_agent(test_query)
    print("\n✅ Agent Response:")
    print(result)
    
    if "Error" in result:
        print("\n❌ Verification Failed: Agent returned an error.")
        sys.exit(1)
    else:
        print("\n✅ Verification Successful: Image path and instructions generated.")
        
except Exception as e:
    print(f"\n❌ Verification Failed with Exception: {e}")
    sys.exit(1)
