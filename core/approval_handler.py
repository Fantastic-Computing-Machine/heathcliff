# ABOUTME: Custom HumanApprovalCallbackHandler for Streamlit UI integration
# ABOUTME: Allows users to approve, modify, or reject sensitive tool executions

from typing import Any, Dict, Optional
from uuid import UUID

from langchain_community.callbacks.human import HumanApprovalCallbackHandler

from logger import logger

# Tools that require human approval before execution
SENSITIVE_TOOLS = {
    "send_email",
    "GmailSendMessage",  # LangChain Gmail tool
    "GmailCreateDraft",
    "create_event",
    "GoogleCalendarCreateTool",  # LangChain Calendar create
    "update_event",
    "GoogleCalendarUpdateTool",  # LangChain Calendar update
    "cancel_event",
    "GoogleCalendarDeleteTool",  # LangChain Calendar delete
    "send_to_telegram",
}


class StreamlitApprovalHandler(HumanApprovalCallbackHandler):
    """
    Custom approval handler for Streamlit applications.

    Extends LangChain's HumanApprovalCallbackHandler to work with Streamlit's
    session state instead of blocking CLI input. When a sensitive tool is about
    to be executed, it stores the request in session state and raises an exception
    to pause execution until the user responds via the Streamlit UI.

    Usage:
        # In agent initialization
        callbacks = [StreamlitApprovalHandler(streamlit_session_state=st.session_state)]

        # In Streamlit UI (before agent invocation)
        if st.session_state.get("pending_approval"):
            # Show approval UI with Approve/Modify/Reject buttons
            ...
    """

    def __init__(self, streamlit_session_state: Optional[Dict[str, Any]] = None):
        """
        Initialize the Streamlit approval handler.

        Args:
            streamlit_session_state: Reference to Streamlit's session_state object.
                                    If None, approval will be disabled (useful for testing).
        """
        self.session_state = streamlit_session_state

        # Initialize parent with our custom functions
        super().__init__(
            should_check=self._should_check_tool,
            approve=self._request_approval,
        )

    def _should_check_tool(self, serialized: Dict[str, Any]) -> bool:
        """
        Determine if a tool requires approval.

        Args:
            serialized: Tool metadata dictionary containing tool name and other info

        Returns:
            True if the tool requires approval, False otherwise
        """
        # Extract tool name from serialized dict
        tool_name = serialized.get("name", "")

        # Also check id field (some tools use this)
        if not tool_name:
            tool_name = serialized.get("id", ["unknown"])
            if isinstance(tool_name, list):
                tool_name = tool_name[-1] if tool_name else "unknown"

        should_check = tool_name in SENSITIVE_TOOLS

        if should_check:
            logger.info(f"🔒 Tool '{tool_name}' requires human approval")

        return should_check

    def _request_approval(self, input_str: str) -> bool:
        """
        Request approval from user via Streamlit session state.

        Args:
            input_str: String representation of tool inputs

        Returns:
            False to pause execution (HumanRejectedException), True to proceed
        """
        if self.session_state is None:
            logger.warning(
                "Approval requested but no session_state provided - auto-approving"
            )
            return True

        # Check existing approval
        if "pending_approval" in self.session_state:
            approval = self.session_state["pending_approval"]
            
            # Check if this matches the current request
            # We compare input, but ideally we'd have a unique ID per request
            if approval.get("tool_input") == input_str:
                status = approval.get("status")
                
                if status == "approved":
                    logger.info("✅ consuming approved request")
                    # Consume the approval so it doesn't persist
                    del self.session_state["pending_approval"]
                    return True
                
                if status == "rejected":
                    logger.info("❌ rejecting request per user decision")
                    # We leave it in session state so UI can show "rejected" feedback if needed
                    # or we can clear it here too.
                    # For now, let's clear it to reset state
                    del self.session_state["pending_approval"]
                    return False

        # New request or different request
        self.session_state["pending_approval"] = {
            "tool_input": input_str,
            "status": "pending",
            "response": None,
        }

        logger.info(f"Approval request stored in session state: {input_str[:100]}...")
        return False


def is_approval_pending(session_state: Dict[str, Any]) -> bool:
    """
    Check if there's a pending approval request.

    Args:
        session_state: Streamlit session state

    Returns:
        True if an approval request is pending
    """
    approval = session_state.get("pending_approval")
    return approval is not None and approval.get("status") == "pending"


def approve_request(
    session_state: Dict[str, Any], modified_input: Optional[str] = None
):
    """
    Approve a pending tool execution request.

    Args:
        session_state: Streamlit session state
        modified_input: Optional modified input string (if user edited the args)
    """
    if "pending_approval" in session_state:
        session_state["pending_approval"]["status"] = "approved"
        if modified_input:
            session_state["pending_approval"]["tool_input"] = modified_input
        logger.info("✅ Tool execution approved by user")


def reject_request(session_state: Dict[str, Any]):
    """
    Reject a pending tool execution request.

    Args:
        session_state: Streamlit session state
    """
    if "pending_approval" in session_state:
        session_state["pending_approval"]["status"] = "rejected"
        logger.info("❌ Tool execution rejected by user")


def clear_approval(session_state: Dict[str, Any]):
    """
    Clear the approval request from session state.
    
    DEPRECATED: Consumption is now handled in _request_approval.
    Kept for backward compatibility.
    """
    if "pending_approval" in session_state:
        del session_state["pending_approval"]
