"""Test script for the local LLM pipeline with streaming, graph visualization, and interrupt handling."""

import asyncio
import os
import warnings
from datetime import datetime

# Set environment variables
os.environ["TAVILY_API_KEY"] = "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"

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

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


async def check_for_interrupt(graph, thread_config: dict) -> dict | None:
    """Check if the graph is in an interrupted state and return interrupt info.

    Args:
        graph: Compiled LangGraph graph
        thread_config: Thread configuration for checkpointing

    Returns:
        Dictionary with interrupt info if interrupted, None otherwise
    """
    state = await graph.aget_state(thread_config)

    if state and state.tasks:
        for task in state.tasks:
            if hasattr(task, 'interrupts') and task.interrupts:
                for interrupt_info in task.interrupts:
                    if hasattr(interrupt_info, 'value'):
                        return interrupt_info.value
    return None


async def run_with_streaming_and_interrupt(graph, query: str, thread_config: dict, auto_responses: dict = None):
    """Run the research pipeline with streaming output and interrupt handling.

    Args:
        graph: Compiled LangGraph graph
        query: The research query
        thread_config: Thread configuration for checkpointing
        auto_responses: Optional dict mapping interrupt types to automatic responses for testing

    Returns:
        Final result from the workflow
    """
    print("\n" + "=" * 60)
    print("STREAMING WORKFLOW EXECUTION (with Interrupt Support)")
    print("=" * 60 + "\n")

    final_result = None
    current_input = {"messages": [HumanMessage(content=query)]}

    while True:
        # Use astream to get updates as the workflow executes
        async for event in graph.astream(
            current_input,
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

        # Check if graph is interrupted
        interrupt_info = await check_for_interrupt(graph, thread_config)

        if interrupt_info:
            print("\n" + "=" * 60)
            print("⏸️  WORKFLOW INTERRUPTED - CLARIFICATION NEEDED")
            print("=" * 60)

            if isinstance(interrupt_info, dict):
                if "question" in interrupt_info:
                    print(f"\n🤔 Question: {interrupt_info['question']}")
                if "verification" in interrupt_info:
                    print(f"📋 Verification: {interrupt_info['verification']}")

                interrupt_type = interrupt_info.get("type", "unknown")

                # Check if we have an auto-response for this interrupt type
                if auto_responses and interrupt_type in auto_responses:
                    user_response = auto_responses[interrupt_type]
                    print(f"\n🤖 Auto-response (for testing): {user_response}")
                else:
                    # In interactive mode, get user input
                    print("\nPlease provide your response to continue:")
                    user_response = input(">>> ").strip()

                print(f"\n✅ Resuming workflow with response: {user_response[:100]}...")

                # Resume the workflow with user's response
                current_input = Command(resume=user_response)
            else:
                print(f"\nInterrupt info: {interrupt_info}")
                print("Unable to handle this interrupt type automatically.")
                break
        else:
            # No interrupt, workflow completed
            break

    return final_result


def save_report_to_file(report: str, query: str, filename: str = None) -> str:
    """Save the final report to a markdown file.

    Args:
        report: The final report content
        query: The original research query
        filename: Optional custom filename (auto-generated if not provided)

    Returns:
        The filename where the report was saved
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_report_{timestamp}.md"

    # Create report with metadata header
    full_report = f"""---
query: "{query}"
generated_at: {datetime.now().isoformat()}
---

{report}
"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_report)

    return filename


async def main():
    """Run a test query through the research pipeline with streaming and interrupt handling."""
    print("=" * 60)
    print("Testing Local LLM Pipeline (Streaming Mode with Interrupts)")
    print("=" * 60)

    # Print configuration
    from deep_research.config import Config
    print(f"\nConfiguration:")
    print(f"  LLM Base URL: {Config.LLM_BASE_URL}")
    print(f"  LLM Model: {Config.LLM_MODEL}")
    print(f"  LLM Context Length: {Config.LLM_CONTEXT_LENGTH:,} tokens")
    print(f"  Max Tokens (Default): {Config.MAX_TOKENS_DEFAULT:,}")
    print(f"  Max Tokens (Writer): {Config.MAX_TOKENS_WRITER:,}")
    print(f"  Embedding Base URL: {Config.EMBEDDING_BASE_URL}")
    print(f"  Embedding Model: {Config.EMBEDDING_MODEL}")
    print()

    # Compile the agent with checkpointer (required for interrupt support)
    checkpointer = InMemorySaver()
    full_agent = deep_researcher_builder.compile(checkpointer=checkpointer)

    # Save the graph as PNG
    print("Saving workflow graph...")
    save_graph_as_png(full_agent, "workflow_graph.png")
    print()

    # Test query - intentionally vague to trigger clarification
    # Change to a more specific query to skip clarification
    query = "What are the main LLM quantization methods? Compare GPTQ, AWQ, and GGUF formats."

    print(f"Query: {query}")
    print("=" * 60)

    # Auto-responses for testing (optional)
    # Comment out to enable interactive mode
    auto_responses = {
        "clarification_needed": "I want a comprehensive comparison covering accuracy, speed, memory usage, and ease of use. Focus on practical deployment scenarios."
    }

    # Run the agent with streaming and interrupt handling
    thread = {"configurable": {"thread_id": "test-1", "recursion_limit": 50}}

    try:
        result = await run_with_streaming_and_interrupt(
            full_agent,
            query,
            thread,
            auto_responses=auto_responses  # Remove this line for interactive mode
        )

        print("\n" + "=" * 60)
        print("FINAL REPORT")
        print("=" * 60)

        final_report = None

        if result and "final_report" in result:
            final_report = result["final_report"]
        else:
            # Fallback: get the final state
            final_state = await full_agent.aget_state(thread)
            if final_state and final_state.values and "final_report" in final_state.values:
                final_report = final_state.values["final_report"]

        if final_report:
            print(final_report)

            # Save the report to a file
            print("\n" + "=" * 60)
            print("SAVING REPORT")
            print("=" * 60)
            report_filename = save_report_to_file(final_report, query)
            print(f"✅ Report saved to: {report_filename}")
        else:
            print("No final report generated.")
            if 'final_state' in dir() and final_state and final_state.values:
                print("Available keys:", list(final_state.values.keys()))

    except Exception as e:
        print(f"\nError occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
