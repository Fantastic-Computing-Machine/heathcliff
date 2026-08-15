# ABOUTME: Coordinator-based agent orchestrator using LangGraph
# ABOUTME: Replaces the single supervisor ReAct loop with a coordinator graph
# ABOUTME: that plans, dispatches, executes subtasks, aggregates, and responds.
# ABOUTME: HeathcliffAgent is a singleton — call HeathcliffAgent.instance() or
# ABOUTME: construct once; subsequent calls to __init__ return the same object.

import uuid
from typing import Any, Dict, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langfuse import propagate_attributes

from config import Config
from core.coordinator_graph import (
    build_coordinator_graph,
    invoke_coordinator,
    resume_coordinator,
    stream_coordinator,
)
from core.delegation.registry import build_default_registry
from db.memory_manager import MemoryManager
from instructions.prompts import (
    USER_PROMPT_TEMPLATE,
    get_current_temporal_context,
)
from logger import logger
from utils.errors import AgentMemoryError
from utils.langfuse_client import get_langfuse_callback_handler

INPUT_MAX_LENGTH = 10000


class HeathcliffAgent:
    """
    Singleton supervisor agent orchestrator.

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
    # Initialisation (idempotent on singleton re-use)
    # ------------------------------------------------------------------

    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
    ):
        """Initialise the Heathcliff supervisor agent.

        Args:
            memory_manager: MemoryManager instance. Required on first call;
                            ignored on subsequent calls (singleton re-use).
        """
        if getattr(self, "_initialised", False):
            return
        self._initialised = True

        self.memory_manager = memory_manager

        self.callbacks: List[Any] = []

        self.llm = init_chat_model(
            api_key=Config.get_ai_api_key(),
            model=Config.SUPERVISOR_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            top_p=Config.TOP_P,
            max_retries=3,
            timeout=Config.TIMEOUT_SECONDS,
        )
        self.coordinator = self._build_coordinator()
        logger.info("HeathcliffAgent initialised")

    def _build_coordinator(self):
        """Build coordinator graph with capability registry."""
        self._registry = build_default_registry()
        coordinator = build_coordinator_graph(registry=self._registry, llm=self.llm)
        logger.info(
            "Coordinator built with %d registered agents",
            len(self._registry.agent_names()),
        )
        return coordinator

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_memories(memories_results: Optional[Dict[str, Any]]) -> List[str]:
        """Pull flat list of memory strings from a ChromaDB query result."""
        if not memories_results:
            return []
        docs = memories_results.get("documents") or []
        return list(docs[0]) if docs else []

    @staticmethod
    def _build_memories_block(memories: List[str]) -> str:
        """Build dynamic long-term memory block for the user prompt."""
        cleaned_memories = [m.strip() for m in memories if isinstance(m, str) and m]
        if not cleaned_memories:
            return "None."
        return "\n".join(f"- {memory}" for memory in cleaned_memories)

    def _build_callbacks(
        self, additional_callbacks: Optional[List[Any]] = None
    ) -> Optional[List[Any]]:
        """Assemble the callbacks list, injecting the Langfuse handler if available."""
        callbacks = list(self.callbacks)
        langfuse_handler = get_langfuse_callback_handler()
        if langfuse_handler:
            callbacks.append(langfuse_handler)
        if additional_callbacks:
            callbacks.extend(additional_callbacks)
        return callbacks or None

    def invoke(
        self,
        user_input: str,
        conversation_id: Optional[str] = None,
        additional_callbacks: Optional[List[Any]] = None,
    ) -> str:
        """
        Process user input using LangGraph ReAct agent.

        Args:
            user_input: User's text query
            conversation_id: Optional conversation ID (generates UUID if not provided)

        Returns:
            str: Agent's response text

        Raises:
            ValueError: If user_input is empty or exceeds 10k chars
        """
        if not user_input or not user_input.strip():
            raise ValueError("User input cannot be empty")
        if len(user_input) > INPUT_MAX_LENGTH:
            raise ValueError(
                f"User input exceeds maximum length of {INPUT_MAX_LENGTH} characters"
            )

        if self.memory_manager is None:
            raise AgentMemoryError("MemoryManager not initialised")
        conversation_id = conversation_id or str(uuid.uuid4())
        logger.info(
            "Processing input: '%.50s...' conversation: %s", user_input, conversation_id
        )

        try:
            chat_history = self.memory_manager.build_langchain_history(
                query=user_input, conversation_id=conversation_id
            )
            memories = self._extract_memories(
                self.memory_manager.recall(query=user_input, n=3)
            )
            memories_block = self._build_memories_block(memories)

            formatted_input = USER_PROMPT_TEMPLATE.format(
                memories_block=memories_block,
                user_input=user_input,
                **get_current_temporal_context(),
            )
            messages = chat_history + [HumanMessage(content=formatted_input)]

            with propagate_attributes(
                session_id=conversation_id, user_id=Config.LANGFUSE_USER_ID
            ):
                callbacks = self._build_callbacks(additional_callbacks)
                response = invoke_coordinator(
                    compiled_graph=self.coordinator,
                    user_input=user_input,
                    session_id=conversation_id,
                    messages=messages,
                    callbacks=callbacks,
                )

            response = response or "I encountered an error processing your request."

            self.memory_manager.save_turn(user_input, response, conversation_id)
            logger.info("Response: '%.50s...'", response)
            return response

        except AgentMemoryError:
            error_message = "Memory Not found, Heathcliff shutting down."
            logger.error(error_message)
            return error_message
        except Exception as e:
            logger.error("Agent invocation failed: %s", e, exc_info=True)
            return "I encountered an error processing your request. Please try again."

    def stream_invoke(
        self,
        user_input: str,
        conversation_id: Optional[str] = None,
        additional_callbacks: Optional[List[Any]] = None,
    ):
        """
        Stream agent execution with status updates.

        Args:
            user_input: User's text query
            conversation_id: Optional conversation ID (generates UUID if not provided)
            additional_callbacks: Optional list of callbacks to add for this invocation

        Yields:
            dict: Event dictionaries with structure:
                - {"type": "plan", "message": str, "data": dict}
                - {"type": "dispatch", "message": str, "data": dict}
                - {"type": "subtask_complete", "message": str, "data": dict}
                - {"type": "quality_retry", "message": str, "data": dict}
                - {"type": "response", "data": str}
                - {"type": "complete", "message": str, "data": dict}
                - {"type": "error", "message": str, "data": str}
        """
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

        if self.memory_manager is None:
            yield {
                "type": "error",
                "message": "MemoryManager not initialised",
                "data": "",
            }
            return
        conversation_id = conversation_id or str(uuid.uuid4())
        logger.info(
            "Streaming input: '%.50s...' conversation: %s", user_input, conversation_id
        )

        try:
            chat_history = self.memory_manager.build_langchain_history(
                query=user_input, conversation_id=conversation_id
            )
            memories = self._extract_memories(
                self.memory_manager.recall(query=user_input, n=3)
            )
            memories_block = self._build_memories_block(memories)

            logger.debug(
                "CONTEXT DEBUG — input: %s | %d context msgs",
                user_input,
                len(chat_history),
            )
            for i, msg in enumerate(chat_history):
                logger.debug("  ctx[%d] %s: %.200s", i, msg.type, msg.content)

            formatted_input = USER_PROMPT_TEMPLATE.format(
                memories_block=memories_block,
                user_input=user_input,
                **get_current_temporal_context(),
            )
            messages = chat_history + [HumanMessage(content=formatted_input)]

            final_response = ""
            approval_pending = False
            callbacks = self._build_callbacks(additional_callbacks)

            with propagate_attributes(
                session_id=conversation_id, user_id=Config.LANGFUSE_USER_ID
            ):
                for event in stream_coordinator(
                    compiled_graph=self.coordinator,
                    user_input=user_input,
                    session_id=conversation_id,
                    messages=messages,
                    callbacks=callbacks,
                ):
                    yield event

                    event_type = event.get("type", "")
                    if event_type == "response":
                        final_response = event.get("data", "")
                    elif event_type == "approval_required":
                        approval_pending = True
                    elif event_type == "complete":
                        event_response = event.get("data", {}).get("response", "")
                        if event_response:
                            final_response = event_response

            if approval_pending:
                logger.info(
                    "Coordinator paused for approval: conversation %s",
                    conversation_id,
                )
                return

            if not final_response:
                final_response = "I encountered an error processing your request."
                yield {"type": "response", "data": final_response}

            self.memory_manager.save_turn(user_input, final_response, conversation_id)
            logger.info("Stream complete: '%.50s...'", final_response)

        except AgentMemoryError:
            error_message = "Memory Not found, Heathcliff shutting down."
            yield {"type": "error", "message": error_message, "data": error_message}

        except Exception as e:
            logger.error("Agent streaming failed: %s", e, exc_info=True)
            yield {
                "type": "error",
                "message": f"Error: {e}",
                "data": "I encountered an error processing your request. Please try again.",
            }

    def resume_approval(
        self,
        *,
        conversation_id: str,
        user_input: str,
        approved: bool,
        modified_input: Optional[str] = None,
    ) -> str:
        """Resume the coordinator action paused for Streamlit approval."""
        if self.memory_manager is None:
            raise AgentMemoryError("MemoryManager not initialised")

        with propagate_attributes(
            session_id=conversation_id, user_id=Config.LANGFUSE_USER_ID
        ):
            callbacks = self._build_callbacks()
            response = resume_coordinator(
                compiled_graph=self.coordinator,
                session_id=conversation_id,
                approved=approved,
                modified_input=modified_input,
                callbacks=callbacks,
            )

        response = response or "I encountered an error processing your request."
        self.memory_manager.save_turn(user_input, response, conversation_id)
        return response

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"<HeathcliffAgent model={Config.SUPERVISOR_MODEL}>"
