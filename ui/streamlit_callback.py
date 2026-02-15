# ABOUTME: Custom LangChain callback handler for Streamlit status updates
# ABOUTME: Shows live thinking steps, tool execution, and progress during agent invocation

from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class StatusCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler to update Streamlit status container
    during agent execution with live, detailed thinking steps.
    """

    def __init__(self, status_container):
        self.status_container = status_container
        self.step_count = 0
        self.current_tool = None

    def _add_step(self, emoji: str, description: str) -> None:
        """Add a new step to the status container."""
        self.step_count += 1
        self.status_container.write(
            f"**{emoji} Step {self.step_count}:** {description}"
        )

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        # Get model name if available
        model_name = (serialized or {}).get("kwargs", {}).get("model", "Gemini")
        if isinstance(model_name, str) and "/" in model_name:
            model_name = model_name.split("/")[-1]

        self.status_container.update(label="🤔 Thinking...", state="running")
        self._add_step("🧠", f"Generating response with {model_name}...")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM finishes generating."""
        # Check if response contains tool calls
        if response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg and hasattr(msg, "tool_calls") and msg.tool_calls:
                        # LLM decided to use tools, don't mark as complete yet
                        return
        self.status_container.write("   ✓ Response ready")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        tool_name = serialized.get("name", "tool")
        self.current_tool = tool_name

        # Format tool name nicely
        display_name = tool_name.replace("_", " ").title()

        self.status_container.update(
            label=f"🔧 Using {display_name}...", state="running"
        )
        self._add_step("🔧", f"Calling `{tool_name}`")

        # Show input if it's meaningful and not too long
        if input_str and len(input_str) < 150:
            try:
                # Try to parse as JSON for better display
                import json

                parsed = json.loads(input_str)
                if isinstance(parsed, dict):
                    params = ", ".join(f"{k}={v}" for k, v in list(parsed.items())[:3])
                    self.status_container.write(f"   → Parameters: {params}")
            except (json.JSONDecodeError, TypeError):
                pass
            except Exception as e:
                # Catch other unexpected errors but don't crash
                pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        self.status_container.write(f"   ✓ `{self.current_tool}` completed")

        # Show brief output preview if available
        if output and len(output) > 0:
            preview = output[:100].replace("\n", " ")
            if len(output) > 100:
                preview += "..."
            self.status_container.write(f"   → Result: {preview}")

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> Any:
        """Run when tool errors."""
        self.status_container.write(f"   ❌ Tool error: {str(error)[:100]}")

    def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
        """Run on agent action - when agent decides to use a tool."""
        if hasattr(action, "tool"):
            display_name = action.tool.replace("_", " ").title()
            self.status_container.update(
                label=f"🎯 Planning: {display_name}", state="running"
            )

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts."""
        # Only show for the main agent chain, not sub-chains
        chain_name = (serialized or {}).get("name", "")
        if self.step_count == 0 and "agent" in chain_name.lower():
            self.status_container.update(label="🚀 Starting...", state="running")
            self._add_step("📋", "Analyzing your request...")

    def on_retriever_start(
        self, serialized: Dict[str, Any], query: str, **kwargs: Any
    ) -> None:
        """Run when retriever starts - for RAG operations."""
        self._add_step("🔍", "Searching knowledge base...")

    def on_retriever_end(self, documents: List[Any], **kwargs: Any) -> None:
        """Run when retriever ends."""
        doc_count = len(documents) if documents else 0
        self.status_container.write(f"   ✓ Found {doc_count} relevant documents")
