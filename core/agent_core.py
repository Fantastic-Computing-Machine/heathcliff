# ABOUTME: Supervisor agent orchestrator using the LangChain subagents pattern
# ABOUTME: Each domain (music, email, calendar, info, contacts, comms) is a
# ABOUTME: sub-agent wrapped as a single @tool; supervisor sees ~8 tools total.
# ABOUTME: HeathcliffAgent is a singleton — call HeathcliffAgent.instance() or
# ABOUTME: construct once; subsequent calls to __init__ return the same object.

import uuid
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from instructions.prompts import build_system_prompt
from logger import logger
from core.memory_manager import AgentMemoryError
from utils.langfuse_client import (
    get_langfuse_callback_handler,
    log_langfuse_interaction,
)
from core.middleware import create_middleware_stack
from config import Config

_instance: Optional["HeathcliffAgent"] = None


class HeathcliffAgent:
    """
    Singleton supervisor agent orchestrator.

    Tools (subagents + skills) are assembled internally — no tool wiring
    needed at the call site.  Pass ``extra_tools`` to extend the default set.

    Usage:
        agent = HeathcliffAgent(memory_manager=mm)
        # or re-use the same instance anywhere:
        agent = HeathcliffAgent.instance()
    """

    _instance: Optional["HeathcliffAgent"] = None

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def instance(cls) -> "HeathcliffAgent":
        """Return the singleton, raising if not yet initialised."""
        if cls._instance is None:
            raise RuntimeError(
                "HeathcliffAgent has not been initialised yet. "
                "Call HeathcliffAgent(memory_manager=...) first."
            )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton (useful for testing)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Internal tool assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_default_tools() -> List[Any]:
        """Load all domain subagents + skill tools."""
        from core.subagents import get_all_subagent_tools
        from skills.skill_tools import get_skill_tools

        tools: List[Any] = []
        try:
            tools.extend(get_all_subagent_tools())
        except Exception as exc:
            logger.warning("Failed to load subagent tools: %s", exc)
        try:
            tools.extend(get_skill_tools())
        except Exception as exc:
            logger.warning("Failed to load skill tools: %s", exc)
        return tools

    # ------------------------------------------------------------------
    # Initialisation (idempotent on singleton re-use)
    # ------------------------------------------------------------------

    def __init__(
        self,
        memory_manager=None,
        extra_tools: Optional[List[Any]] = None,
    ):
        """
        Initialise the Heathcliff supervisor agent.

        Args:
            memory_manager: MemoryManager instance. Required on first call;
                            ignored on subsequent calls (singleton re-use).
            extra_tools: Optional list of additional BaseTool objects to
                         append to the default subagent + skill tools.
        """
        # Guard: already initialised — skip re-init
        if getattr(self, "_initialised", False):
            return
        self._initialised = True

        self.memory_manager = memory_manager

        # Assemble tools: defaults + any caller-supplied extensions
        default_tools = self._assemble_default_tools()
        extension = list(extra_tools) if extra_tools else []
        self._tools: List[Any] = default_tools + extension

        self.max_iterations = Config.MAX_ITERATIONS
        self.callbacks: List[Any] = []

        # Langfuse callback
        langfuse_handler = get_langfuse_callback_handler()
        if langfuse_handler:
            self.callbacks.append(langfuse_handler)
            logger.info("Langfuse callback handler enabled")
        else:
            logger.info(
                "Langfuse callback handler unavailable; falling back to manual trace events only"
            )

        # Middleware
        self.middleware_stack: List[Any] = []
        self.llm = ChatGoogleGenerativeAI(
            model=Config.MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=Config.TEMPERATURE,
            max_output_tokens=Config.MAX_TOKENS,
            top_p=Config.TOP_P,
        )
        self.middleware_stack = create_middleware_stack(llm=self.llm)
        self.callbacks.extend(self.middleware_stack)

        # Build supervisor graph
        self.prompt = self._build_prompt_template()
        self.executor = self._build_agent()

        logger.info(
            "HeathcliffAgent (supervisor) initialised with %d tools "
            "(max_iterations=%d)",
            len(self._tools),
            self.max_iterations,
        )

    def _build_prompt_template(self) -> SystemMessage:
        """Create supervisor system prompt."""
        master_info = Config.MASTER_INFO
        system_prompt_text = build_system_prompt(master_info)
        return SystemMessage(content=system_prompt_text)

    def _build_agent(self):
        """Build LangChain supervisor agent with subagent + skill tools."""
        agent_graph = create_agent(
            model=self.llm,
            tools=self._tools,
            system_prompt=self.prompt,
        )
        logger.info(
            f"Supervisor built with tools: {[getattr(t, 'name', str(t)) for t in self._tools]}"
        )
        return agent_graph

    def _format_chat_history(
        self, context_messages: List[Dict[str, Any]], memories: List[str]
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

        # Add critical instruction to focus on current query
        # This prevents the agent from picking tools based on previous queries
        if chat_history:  # Only add if there's context
            chat_history.append(
                SystemMessage(
                    content="IMPORTANT: Focus ONLY on the user's CURRENT query. "
                    "The above messages are for context only. Do NOT use tools or take actions "
                    "based on previous queries - only respond to what the user just asked."
                )
            )

        return chat_history

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

        try:
            # Get context from memory manager
            # Use chronological retrieval instead of semantic search
            # to prevent picking up old tool calls from similar queries
            # REDUCED from n=5 to n=2 to avoid context confusion where agent
            # picks tools for previous queries instead of current query
            context_messages = self.memory_manager.get_recent_chats(
                session_id=session_id, n=2
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

            # Format chat history as messages
            chat_history = self._format_chat_history(context_messages, memories)

            # Build message list for graph state
            messages = chat_history + [HumanMessage(content=user_input)]

            # Merge callbacks
            all_callbacks = list(self.callbacks)  # Langfuse already included
            if additional_callbacks:
                all_callbacks.extend(additional_callbacks)

            # Invoke graph - handles iterations, tool calls, everything!
            result = self.executor.invoke(
                {"messages": messages},
                config={"callbacks": all_callbacks} if all_callbacks else None,
            )

            # Extract response from final message
            final_messages = result.get("messages", [])
            if final_messages:
                # Last message should be the assistant's response
                last_message = final_messages[-1]
                content = last_message.content

                # Handle Gemini's structured content format
                if isinstance(content, list):
                    # Extract text from [{'type': 'text', 'text': '...'}] format
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

        except Exception as e:
            if isinstance(e, AgentMemoryError):
                error_message = "Memory Not found, Heathcliff shutting down."
                logger.error(error_message)
                return error_message

            logger.error(f"Agent invocation failed: {e}", exc_info=True)
            error_message = (
                "I encountered an error processing your request. Please try again."
            )
            log_langfuse_interaction(
                session_id=session_id,
                user_input=user_input,
                response=error_message,
                status="error",
                extra_metadata={"error": str(e)},
            )
            return error_message

    def stream_invoke(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        additional_callbacks: Optional[List[Any]] = None,
    ):
        """
        Stream agent execution with status updates.

        Args:
            user_input: User's text query
            session_id: Optional session ID (generates UUID if not provided)
            additional_callbacks: Optional list of callbacks to add for this invocation

        Yields:
            dict: Event dictionaries with structure:
                - {"type": "status", "message": str, "data": dict}
                - {"type": "tool", "message": str, "data": dict}
                - {"type": "response", "data": str}
                - {"type": "complete", "message": str, "data": dict}
                - {"type": "error", "message": str, "data": str}
        """
        # Validation
        if not user_input or not user_input.strip():
            yield {
                "type": "error",
                "message": "User input cannot be empty",
                "data": "Please provide a valid input.",
            }
            return

        if len(user_input) > 10000:
            yield {
                "type": "error",
                "message": "User input exceeds maximum length",
                "data": "Input must be less than 10,000 characters.",
            }
            return

        # Generate session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        logger.info(f"Streaming input: '{user_input[:50]}...' session: {session_id}")

        try:
            # Get context from memory manager
            yield {
                "type": "status",
                "message": "Retrieving memories and context...",
                "data": {"phase": "retrieval"},
            }

            # Use chronological retrieval instead of semantic search
            # to prevent picking up old tool calls from similar queries
            # REDUCED from n=5 to n=2 to avoid context confusion where agent
            # picks tools for previous queries instead of current query
            context_messages = self.memory_manager.get_recent_chats(
                session_id=session_id, n=2
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

            memories_count = len(memories)

            yield {
                "type": "status",
                "message": f"Retrieved {memories_count} memories",
                "data": {"phase": "retrieval_complete", "memories": memories_count},
            }

            # Format chat history as messages
            chat_history = self._format_chat_history(context_messages, memories)

            # DEBUG: Log what context is being sent to agent
            logger.info(f"===== CONTEXT DEBUG =====")
            logger.info(f"Current user input: {user_input}")
            logger.info(f"Number of context messages: {len(chat_history)}")
            for i, msg in enumerate(chat_history):
                logger.info(f"Context msg {i}: {msg.type} - {str(msg.content)[:200]}")
            logger.info(f"========================")

            # Build message list for graph state
            messages = chat_history + [HumanMessage(content=user_input)]

            # Merge callbacks
            all_callbacks = list(self.callbacks)  # Langfuse already included
            if additional_callbacks:
                all_callbacks.extend(additional_callbacks)

            # LangGraph graph has built-in streaming!
            final_response = ""
            tools_used = []

            for chunk in self.executor.stream(
                {"messages": messages},
                config={"callbacks": all_callbacks} if all_callbacks else None,
            ):
                # Debug: Log raw chunk structure
                logger.info(
                    f"Received chunk: {type(chunk).__name__}, keys: {chunk.keys() if isinstance(chunk, dict) else 'N/A'}, content: {str(chunk)[:500]}"
                )

                # Graph yields state updates with messages
                # LangGraph wraps messages in chunk['model']['messages']
                chunk_messages = None
                if (
                    "model" in chunk
                    and isinstance(chunk["model"], dict)
                    and "messages" in chunk["model"]
                ):
                    chunk_messages = chunk["model"]["messages"]
                elif "messages" in chunk:
                    chunk_messages = chunk["messages"]

                if chunk_messages:
                    last_msg = chunk_messages[-1]

                    # Debug logging
                    logger.info(
                        f"Stream chunk - msg type: {last_msg.type}, has tool_calls: {hasattr(last_msg, 'tool_calls')}, tool_calls value: {getattr(last_msg, 'tool_calls', None)}"
                    )

                    # Check for tool calls
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for tool_call in last_msg.tool_calls:
                            tool_name = tool_call.get("name", "unknown")
                            tools_used.append(tool_name)
                            yield {
                                "type": "tool",
                                "message": f"Calling tool: {tool_name}",
                                "data": {
                                    "name": tool_name,
                                    "args": tool_call.get("args", {}),
                                },
                            }

                    # Check for tool results
                    if last_msg.type == "tool":
                        yield {
                            "type": "tool",
                            "message": "Tool completed",
                            "data": {"result": str(last_msg.content)[:200]},
                        }

                    # Check for final AI response (AI message without tool calls, or with empty tool_calls)
                    if last_msg.type == "ai" and (
                        not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls
                    ):
                        content = last_msg.content
                        # Handle Gemini's structured content format
                        if isinstance(content, list):
                            text_parts = [
                                part.get("text", "")
                                for part in content
                                if isinstance(part, dict) and part.get("type") == "text"
                            ]
                            final_response = "".join(text_parts)
                        else:
                            final_response = str(content) if content else ""

                        if final_response:
                            yield {"type": "response", "data": final_response}

            if not final_response:
                final_response = "I encountered an error processing your request."
                yield {"type": "response", "data": final_response}

            # Save to memory
            self.memory_manager.save_chat(user_input, final_response, session_id)

            # Yield completion
            yield {
                "type": "complete",
                "message": "Processing complete",
                "data": {
                    "session_id": session_id,
                    "tools_used": tools_used,
                    "tool_count": len(tools_used),
                },
            }

            logger.info(f"Stream complete: '{final_response[:50]}...'")
            log_langfuse_interaction(
                session_id=session_id,
                user_input=user_input,
                response=final_response,
                status="success",
                extra_metadata={"tools_used": tools_used},
            )

        except Exception as e:
            if isinstance(e, AgentMemoryError):
                error_message = "Memory Not found, Heathcliff shutting down."
                yield {"type": "error", "message": error_message, "data": error_message}
                return

            logger.error(f"Agent streaming failed: {e}", exc_info=True)
            error_message = (
                "I encountered an error processing your request. Please try again."
            )

            yield {
                "type": "error",
                "message": f"Error: {str(e)}",
                "data": error_message,
            }

            log_langfuse_interaction(
                session_id=session_id,
                user_input=user_input,
                response=error_message,
                status="error",
                extra_metadata={"error": str(e)},
            )

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"<HeathcliffAgent model={Config.MODEL}>"
