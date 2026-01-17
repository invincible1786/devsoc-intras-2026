"""
Test the new GoT + MoE implementation in chat_agent
"""

import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.services.chat_agent import GoTMoEEngine


async def test_queries():
    """Test the GoT + MoE engine with sample queries."""
    
    # Initialize engine
    logger.info("Initializing GoTMoEEngine...")
    engine = GoTMoEEngine()
    
    # Test queries
    test_queries = [
        "Who is the Vice President of Technology Film and Photography Society?",
        "When was TFPS founded?",
        "List the current General Secretaries of TSG for 2025",
    ]
    
    for query in test_queries:
        logger.info(f"\n{'='*80}")
        logger.info(f"Query: {query}")
        logger.info(f"{'='*80}\n")
        
        result = await engine.process_query(query)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Final Answer:\n{result['answer']}")
        logger.info(f"\nConfidence: {result['confidence']:.2f}")
        logger.info(f"Sources: {result['sources']}")
        logger.info(f"Graph Stats: {result['graph_stats']}")
        logger.info(f"{'='*80}\n")


def main():
    """Run the test."""
    asyncio.run(test_queries())


if __name__ == "__main__":
    main()
