# ABOUTME: Supervisor agent orchestrator using the LangChain subagents pattern
# ABOUTME: Each domain (music, email, calendar, info, contacts, comms) is a
# ABOUTME: sub-agent wrapped as a single @tool; supervisor sees ~8 tools total.
# ABOUTME: HeathcliffAgent is a singleton — call HeathcliffAgent.instance() or
# ABOUTME: construct once; subsequent calls to __init__ return the same object.

import uuid
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import Config
from core.memory_manager import AgentMemoryError
from core.middleware import create_middleware_stack
from instructions.prompts import (
    USER_PROMPT_TEMPLATE,
    build_system_prompt,
    get_current_temporal_context,
)
from logger import logger
from utils.langfuse_client import (
    get_langfuse_callback_handler,
    log_langfuse_interaction,
)

INPUT_MAX_LENGTH = 10000


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
        """Initialise the Heathcliff supervisor agent.

        Args:
            memory_manager: MemoryManager instance. Required on first call;
                            ignored on subsequent calls (singleton re-use).
            extra_tools: Optional list of additional BaseTool objects to
                         append to the default subagent + skill tools.
        """
        if getattr(self, "_initialised", False):
            return
        self._initialised = True

        self.memory_manager = memory_manager

        # Assemble tools: defaults + any caller-supplied extensions
        self._tools: List[Any] = self._assemble_default_tools() + list(
            extra_tools or []
        )

        self.max_iterations = Config.MAX_ITERATIONS
        self.callbacks: List[Any] = []

        langfuse_handler = get_langfuse_callback_handler()
        if langfuse_handler:
            self.callbacks.append(langfuse_handler)
            logger.info("Langfuse callback handler enabled")
        else:
            logger.info(
                "Langfuse callback handler unavailable; "
                "falling back to manual trace events only"
            )

        self.llm = init_chat_model(
            api_key=Config.AI_KEY,
            model=Config.SUPERVISOR_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            top_p=Config.TOP_P,
            max_retries=3,
            timeout=Config.TIMEOUT_SECONDS,
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
        tool_names = [getattr(t, "name", str(t)) for t in self._tools]
        logger.info("Supervisor built with tools: %s", tool_names)
        return agent_graph

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _ROLE_TO_MSG = {
        "user": HumanMessage,
        "assistant": AIMessage,
    }

    @staticmethod
    def _extract_memories(memories_results: Optional[Dict[str, Any]]) -> List[str]:
        """Pull flat list of memory strings from a ChromaDB query result."""
        if not memories_results:
            return []
        docs = memories_results.get("documents") or []
        return list(docs[0]) if docs else []

    @staticmethod
    def _extract_gemini_content(content) -> str:
        """Normalise Gemini's structured response to a plain string."""
        if isinstance(content, list):
            return "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return str(content) if content else ""

    @staticmethod
    def _build_memories_block(memories: List[str]) -> str:
        """Build dynamic long-term memory block for the user prompt."""
        cleaned_memories = [m.strip() for m in memories if isinstance(m, str) and m]
        if not cleaned_memories:
            return "None."
        return "\n".join(f"- {memory}" for memory in cleaned_memories)

    def _format_chat_history(self, message_history: List[Dict[str, Any]]) -> List:
        """Format pair-based history as LangChain message objects."""
        chat_history: list = []

        for msg in message_history:
            role = str(msg.get("role", ""))
            if role in self._ROLE_TO_MSG:
                msg_cls = self._ROLE_TO_MSG[role]
                chat_history.append(msg_cls(content=msg.get("content", "")))

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

        if len(user_input) > INPUT_MAX_LENGTH:
            raise ValueError(
                f"User input exceeds maximum length of {INPUT_MAX_LENGTH} characters"
            )

        # Generate session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        logger.info("Processing input: '%.50s...' session: %s", user_input, session_id)

        try:
            # Build message history: semantic pairs + recent chronological pairs
            message_history = self.memory_manager.build_message_history(
                query=user_input, session_id=session_id
            )
            memories = self._extract_memories(
                self.memory_manager.recall(query=user_input, n=3)
            )

            # Format chat history as messages
            chat_history = self._format_chat_history(message_history)
            memories_block = self._build_memories_block(memories)

            # Build message list for graph state
            temporal_context = get_current_temporal_context()
            formatted_input = USER_PROMPT_TEMPLATE.format(
                memories_block=memories_block,
                user_input=user_input,
                **temporal_context,
            )
            messages = chat_history + [HumanMessage(content=formatted_input)]

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
            response = (
                self._extract_gemini_content(final_messages[-1].content)
                if final_messages
                else ""
            ) or "I encountered an error processing your request."

            # Save to memory
            self.memory_manager.save_chat(user_input, response, session_id)

            logger.info("Response: '%.50s...'", response)
            log_langfuse_interaction(
                session_id=session_id,
                user_input=user_input,
                response=response,
                status="success",
            )

            return response

        except AgentMemoryError:
            error_message = "Memory Not found, Heathcliff shutting down."
            logger.error(error_message)
            return error_message
        except Exception as e:
            logger.error("Agent invocation failed: %s", e, exc_info=True)
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

        if len(user_input) > INPUT_MAX_LENGTH:
            yield {
                "type": "error",
                "message": "User input exceeds maximum length",
                "data": f"Input must be less than {INPUT_MAX_LENGTH:,} characters.",
            }
            return

        # Generate session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        logger.info("Streaming input: '%.50s...' session: %s", user_input, session_id)

        try:
            yield {
                "type": "status",
                "message": "Retrieving memories and context...",
                "data": {"phase": "retrieval"},
            }

            # Build message history: semantic pairs + recent chronological pairs
            message_history = self.memory_manager.build_message_history(
                query=user_input, session_id=session_id
            )
            memories = self._extract_memories(
                self.memory_manager.recall(query=user_input, n=3)
            )

            yield {
                "type": "status",
                "message": f"Retrieved {len(memories)} memories",
                "data": {"phase": "retrieval_complete", "memories": len(memories)},
            }

            chat_history = self._format_chat_history(message_history)
            memories_block = self._build_memories_block(memories)

            logger.debug(
                "CONTEXT DEBUG — input: %s | %d context msgs",
                user_input,
                len(chat_history),
            )
            for i, msg in enumerate(chat_history):
                logger.debug("  ctx[%d] %s: %.200s", i, msg.type, msg.content)

            # Build message list for graph state
            temporal_context = get_current_temporal_context()
            formatted_input = USER_PROMPT_TEMPLATE.format(
                memories_block=memories_block,
                user_input=user_input,
                **temporal_context,
            )
            messages = chat_history + [HumanMessage(content=formatted_input)]

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
                logger.debug(
                    "Received chunk: %s, keys: %s, content: %.500s",
                    type(chunk).__name__,
                    chunk.keys() if isinstance(chunk, dict) else "N/A",
                    chunk,
                )

                # LangGraph wraps messages in chunk['model']['messages']
                model_data = chunk.get("model")
                chunk_messages = (
                    model_data.get("messages")
                    if isinstance(model_data, dict)
                    else chunk.get("messages")
                )

                if chunk_messages:
                    last_msg = chunk_messages[-1]

                    logger.debug(
                        "Stream chunk — type: %s, tool_calls: %s",
                        last_msg.type,
                        getattr(last_msg, "tool_calls", None),
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

                    # Final AI response (no tool calls)
                    if last_msg.type == "ai" and not getattr(
                        last_msg, "tool_calls", None
                    ):
                        final_response = self._extract_gemini_content(last_msg.content)
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

            logger.info("Stream complete: '%.50s...'", final_response)
            log_langfuse_interaction(
                session_id=session_id,
                user_input=user_input,
                response=final_response,
                status="success",
                extra_metadata={"tools_used": tools_used},
            )

        except AgentMemoryError:
            error_message = "Memory Not found, Heathcliff shutting down."
            yield {"type": "error", "message": error_message, "data": error_message}

        except Exception as e:
            logger.error("Agent streaming failed: %s", e, exc_info=True)
            error_message = (
                "I encountered an error processing your request. Please try again."
            )
            yield {
                "type": "error",
                "message": f"Error: {e}",
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
        return f"<HeathcliffAgent model={Config.SUPERVISOR_MODEL}>"
