"""Test script for the local LLM pipeline with streaming and graph visualization."""

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


def save_graph_as_png(graph, filename: str = "workflow_graph.png"):
    """Save the LangGraph workflow as a PNG image.

    Args:
        graph: Compiled LangGraph graph
        filename: Output filename for the PNG
    """
    try:
        # Get the graph and draw as PNG using mermaid
        png_data = graph.get_graph().draw_mermaid_png()

        with open(filename, "wb") as f:
            f.write(png_data)

        print(f"Graph saved as '{filename}'")
        return True
    except Exception as e:
        print(f"Could not save graph as PNG: {e}")
        print("Falling back to Mermaid text representation...")
        try:
            mermaid_text = graph.get_graph().draw_mermaid()
            mermaid_filename = filename.replace(".png", ".mmd")
            with open(mermaid_filename, "w") as f:
                f.write(mermaid_text)
            print(f"Mermaid diagram saved as '{mermaid_filename}'")
        except Exception as e2:
            print(f"Could not save Mermaid diagram: {e2}")
        return False


async def run_with_streaming(graph, query: str, thread_config: dict):
    """Run the research pipeline with streaming output.

    Args:
        graph: Compiled LangGraph graph
        query: The research query
        thread_config: Thread configuration for checkpointing
    """
    print("\n" + "=" * 60)
    print("STREAMING WORKFLOW EXECUTION")
    print("=" * 60 + "\n")

    final_result = None

    # Use astream to get updates as the workflow executes
    async for event in graph.astream(
        {"messages": [HumanMessage(content=query)]},
        config=thread_config,
        stream_mode="updates"
    ):
        # event is a dict with node name as key and output as value
        for node_name, node_output in event.items():
            print(f"\n{'─' * 40}")
            print(f"📍 Node: {node_name}")
            print(f"{'─' * 40}")

            # Print relevant information from the node output
            if isinstance(node_output, dict):
                # Check for messages
                if "messages" in node_output:
                    messages = node_output["messages"]
                    if messages:
                        last_msg = messages[-1] if isinstance(messages, list) else messages
                        if hasattr(last_msg, "content"):
                            content = last_msg.content
                            # Truncate long content for display
                            if len(content) > 500:
                                print(f"   Content: {content[:500]}...")
                            else:
                                print(f"   Content: {content}")

                # Check for research brief
                if "research_brief" in node_output and node_output["research_brief"]:
                    brief = node_output["research_brief"]
                    print(f"   Research Brief: {brief[:300]}..." if len(brief) > 300 else f"   Research Brief: {brief}")

                # Check for draft report
                if "draft_report" in node_output and node_output["draft_report"]:
                    print(f"   Draft Report Generated: {len(node_output['draft_report'])} characters")

                # Check for research findings
                if "research_findings" in node_output and node_output["research_findings"]:
                    findings = node_output["research_findings"]
                    print(f"   Research Findings: {len(findings)} findings collected")

                # Check for final report
                if "final_report" in node_output and node_output["final_report"]:
                    print(f"   ✅ Final Report Generated: {len(node_output['final_report'])} characters")
                    final_result = node_output

            print()

    return final_result


async def main():
    """Run a test query through the research pipeline with streaming."""
    print("=" * 60)
    print("Testing Local LLM Pipeline (Streaming Mode)")
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

    # Save the graph as PNG
    print("Saving workflow graph...")
    save_graph_as_png(full_agent, "workflow_graph.png")
    print()

    # Test query
    query = "What are the main LLM quantization methods? Compare GPTQ, AWQ, and GGUF formats."

    print(f"Query: {query}")
    print("=" * 60)

    # Run the agent with streaming
    thread = {"configurable": {"thread_id": "test-1", "recursion_limit": 50}}

    try:
        result = await run_with_streaming(full_agent, query, thread)

        print("\n" + "=" * 60)
        print("FINAL REPORT")
        print("=" * 60)

        if result and "final_report" in result:
            print(result["final_report"])
        else:
            # Fallback: get the final state
            final_state = await full_agent.aget_state(thread)
            if final_state and final_state.values and "final_report" in final_state.values:
                print(final_state.values["final_report"])
            else:
                print("No final report generated.")
                if final_state and final_state.values:
                    print("Available keys:", list(final_state.values.keys()))

    except Exception as e:
        print(f"\nError occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
