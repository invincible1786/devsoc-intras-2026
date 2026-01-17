"""
Quick test for the API fixes
"""
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

from src.services.chat_agent import GoTMoEEngine


async def quick_test():
    """Quick test of a single query."""
    
    print("Initializing engine...")
    engine = GoTMoEEngine()
    
    query = "Who founded TFPS?"
    print(f"\nQuery: {query}")
    print("Processing...\n")
    
    result = await engine.process_query(query)
    
    print(f"Answer: {result['answer']}")
    print(f"\nConfidence: {result['confidence']:.2f}")
    print(f"Sources: {result['sources']}")
    print(f"Reasoning Path: {len(result['reasoning_path'])} steps")
    print(f"\nReasoning Details:")
    for i, step in enumerate(result['reasoning_path']):
        print(f"  {i+1}. {step['question']}")
        print(f"     Verified: {step['verified']}, Score: {step['score']}/10")
    
    print(f"\nGraph Stats: {result['graph_stats']}")
    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(quick_test())
