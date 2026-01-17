from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class StatusCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler to update Streamlit status container
    during agent execution.
    """

    def __init__(self, status_container):
        self.status_container = status_container

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        # self.status_container.write("🤔 Heathcliff is thinking...")
        pass

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""
        tool_name = serialized.get("name", "Unknown Tool")
        self.status_container.markdown(f"**🔧 Executing:** `{tool_name}`")
        if input_str:
            self.status_container.code(input_str)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        self.status_container.markdown(f"**✅ Output:**")
        self.status_container.code(output)

    def on_tool_error(
        self, error: BaseException, **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        self.status_container.markdown(f"**❌ Tool Error:**")
        self.status_container.error(str(error))

    def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

