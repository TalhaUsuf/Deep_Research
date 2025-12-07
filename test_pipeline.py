"""Test script for the local LLM pipeline."""

import asyncio
import os
import warnings

# Set environment variables
os.environ["TAVILY_API_KEY"] = "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"

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.research_agent_full import deep_researcher_builder


async def main():
    """Run a test query through the research pipeline."""
    print("=" * 60)
    print("Testing Local LLM Pipeline")
    print("=" * 60)

    # Print configuration
    from deep_research.config import Config
    print(f"\nConfiguration:")
    print(f"  LLM Base URL: {Config.LLM_BASE_URL}")
    print(f"  LLM Model: {Config.LLM_MODEL}")
    print(f"  Embedding Base URL: {Config.EMBEDDING_BASE_URL}")
    print(f"  Embedding Model: {Config.EMBEDDING_MODEL}")
    print()

    # Compile the agent with checkpointer
    checkpointer = InMemorySaver()
    full_agent = deep_researcher_builder.compile(checkpointer=checkpointer)

    # Test query
    query = "What are the main LLM quantization methods? Compare GPTQ, AWQ, and GGUF formats."

    print(f"Query: {query}")
    print("=" * 60)
    print("\nRunning research pipeline...\n")

    # Run the agent
    thread = {"configurable": {"thread_id": "test-1", "recursion_limit": 50}}

    try:
        result = await full_agent.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config=thread
        )

        print("\n" + "=" * 60)
        print("FINAL REPORT")
        print("=" * 60)

        if "final_report" in result:
            print(result["final_report"])
        else:
            print("No final report generated.")
            print("Available keys:", list(result.keys()))

    except Exception as e:
        print(f"\nError occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
