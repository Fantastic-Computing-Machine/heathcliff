# ABOUTME: Simplified agent orchestrator using LangChain's unified agent framework
# ABOUTME: Replaces custom LangGraph StateGraph with built-in agent framework
import sys

sys.path.append(".")

import os
import uuid
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from middlewares.built_in import BuiltInMiddlewares

from config import Config
from instructions.prompts import build_system_prompt, build_user_context
from logger import logger
from tools import get_all_tools
from utils.langfuse_client import (
    get_langfuse_callback_handler,
    log_langfuse_interaction,
)

os.environ["MEM0_TELEMETRY"] = "False"


class HeathcliffAgent:
    """
    Main agent orchestrator using LangChain's create_agent.

    This is the "master-class" for Heathcliff - a single entry point that
    auto-initializes memory, tools, and prompts. Use HeathcliffAgent.create()
    for the simplest instantiation, or pass custom components as needed.

    Example:
        >>> agent = HeathcliffAgent.create()
        >>> response = agent.ask("What's the weather?")
    """

    @classmethod
    def create(cls) -> "HeathcliffAgent":
        """
        Factory method for simple one-liner instantiation.

        Creates an agent with all defaults (auto-initializes MemoryManager and tools).

        Returns:
            HeathcliffAgent: Fully initialized agent ready to use.
        """
        return cls()

    def __init__(self, memory_manager=None) -> None:
        """
        Initialize the Heathcliff agent.

        Args:
            memory_manager: Optional MemoryManager instance. If not provided,
                            a new one will be auto-initialized.
        """
        from core.memory_manager import MemoryManager

        # Auto-initialize memory if not provided
        self.memory_manager = (
            memory_manager if memory_manager is not None else MemoryManager()
        )

        # Keep original LangChain Tool objects for structured calling
        # NOTE: LangChain's create_agent expects iterable BaseTool objects, not a
        # mapping. Passing a dict of callables caused AttributeError: 'function'
        # object has no attribute 'name'.
        self._original_langchain_tools = list(get_all_tools())
        logger.debug(
            "Prepared %d tools for the agent.", len(self._original_langchain_tools)
        )

        self.max_iterations = Config.MAX_ITERATIONS
        self.callbacks: List[Any] = []

        # Initialize Langfuse callback
        langfuse_handler = get_langfuse_callback_handler()
        if langfuse_handler:
            self.callbacks.append(langfuse_handler)
            logger.info("Langfuse callback handler enabled")
        else:
            logger.info(
                "Langfuse callback handler unavailable; falling back to manual trace events only"
            )

        # Initialize middleware stack (will be created with LLM after initialization)
        self.middleware_stack: List[Any] = []

        # Initialize Gemini LLM
        llm_kwargs: Dict[str, Any] = {
            "model": Config.MODEL,
            "google_api_key": Config.GEMINI_API_KEY,
            "temperature": Config.TEMPERATURE,
            "max_output_tokens": Config.MAX_TOKENS,
            "top_p": Config.TOP_P,
        }

        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        self.llm = ChatGoogleGenerativeAI(**llm_kwargs)

        # Initialize middleware stack with LLM
        self._middleware_stack = BuiltInMiddlewares().get()

        # Build the agent executor
        self.system_prompt = self._build_prompt_template()
        self.executor = self._build_agent()

        logger.info(
            "HeathcliffAgent initialized with %d tools (max_iterations=%d).",
            len(self._original_langchain_tools),
            self.max_iterations,
        )

    def _build_prompt_template(self) -> str:
        """Create system prompt for react agent."""
        master_info = Config.MASTER_INFO
        system_prompt_text = build_system_prompt(master_info)
        return system_prompt_text

    def _build_agent(self):
        """Build LangChain agent using create_agent."""

        agent_graph = create_agent(
            name="Heathcliff",
            model=self.llm,
            tools=self._original_langchain_tools,
            middleware=self._middleware_stack,
            system_prompt=self.system_prompt,
        )

        return agent_graph

    def _format_chat_history(
        self,
        context_messages: List[Dict[str, Any]],
        memories: List[str],
    ) -> List:
        """Format memories and context as chat history for the agent."""
        chat_history = []

        # Add memories as a system message
        if memories:
            memories_str = "\n".join(f"- {m}" for m in memories)
            chat_history.append(
                SystemMessage(content=f"Long-term memories:\n{memories_str}")
            )

        # Add recent conversation context (chronologically ordered)
        if context_messages:
            for msg in context_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                if role == "user":
                    chat_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    chat_history.append(AIMessage(content=content))

        return chat_history

    def ask(
        self,
        query: str,
        session_id: Optional[str] = None,
        additional_callbacks: Optional[List[Any]] = None,
    ) -> str:
        """
        Simple alias for invoke(). Ask the agent a question.

        Args:
            query: User's text query
            session_id: Optional session ID (generates UUID if not provided)
            additional_callbacks: Optional list of callback handlers

        Returns:
            str: Agent's response text
        """
        return self.invoke(
            query, session_id=session_id, additional_callbacks=additional_callbacks
        )

    def invoke(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        additional_callbacks: Optional[List[Any]] = None,
    ) -> str:
        """
        Process user input using LangGraph ReAct agent.

        Args:
            user_input: User's text query
            session_id: Optional session ID (generates UUID if not provided)
            additional_callbacks: Optional list of callback handlers

        Returns:
            str: Agent's response text

        Raises:
            ValueError: If user_input is empty or exceeds 10k chars
        """
        # Validation
        if not user_input or not user_input.strip():
            raise ValueError("User input cannot be empty")

        if len(user_input) > 10000:
            raise ValueError("User input exceeds maximum length of 10000 characters")

        # Generate session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        logger.info(f"Processing input: '{user_input[:50]}...' session: {session_id}")

        # Get context from memory manager
        context_messages = self.memory_manager.get_recent_chats(
            session_id=session_id, n=10
        )
        memories_results = self.memory_manager.recall(query=user_input, n=3)

        # Extract memories
        memories = []
        if memories_results and memories_results.get("documents"):
            docs = (
                memories_results["documents"][0]
                if memories_results["documents"]
                else []
            )
            memories = list(docs)

        # Format chat history as messages (for agent's memory)
        chat_history = self._format_chat_history(context_messages, memories)

        # Build dynamic user context with datetime, memories, and conversation
        timezone = Config.MASTER_INFO.get("timezone", "America/New_York")
        enhanced_input = build_user_context(
            user_input=user_input,
            timezone=timezone,
            memories=memories,
            recent_messages=context_messages,
        )
        messages = chat_history + [HumanMessage(content=enhanced_input)]

        # Merge callbacks
        all_callbacks = list(self.callbacks)
        if additional_callbacks:
            all_callbacks.extend(additional_callbacks)

        # Invoke graph
        try:
            result = self.executor.invoke(
                {"messages": messages},
                config={"callbacks": all_callbacks} if all_callbacks else None,
            )
        except Exception as e:
            # Log the error but re-raise it so the UI can handle it (e.g. for approvals)
            logger.error(f"Agent invocation failed: {e}", exc_info=True)
            raise e

        # Extract response from final message
        final_messages = result.get("messages", [])
        if final_messages:
            last_message = final_messages[-1]
            content = last_message.content

            # Handle Gemini's structured content format
            if isinstance(content, list):
                text_parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                response = "".join(text_parts)
            else:
                response = str(content) if content else ""
        else:
            response = ""

        if not response:
            response = "I encountered an error processing your request."

        # Save to memory
        self.memory_manager.save_chat(user_input, response, session_id)

        logger.info(f"Response: '{response[:50]}...'")
        log_langfuse_interaction(
            session_id=session_id,
            user_input=user_input,
            response=response,
            status="success",
        )

        return response


if __name__ == "__main__":
    # Simple test
    from core.memory_manager import MemoryManager

    memory_mgr = MemoryManager()
    session_id = "test_session"
    agent = HeathcliffAgent(memory_manager=memory_mgr)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"} or not user_input:
            print("Bye!")
            break
        response = agent.invoke(user_input=user_input, session_id=session_id)
        print(f"Heathcliff: {response}")
