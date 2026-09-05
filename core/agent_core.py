# ABOUTME: Coordinator-based agent orchestrator using LangGraph
# ABOUTME: Replaces the single supervisor ReAct loop with a coordinator graph
# ABOUTME: that plans, dispatches, executes subtasks, aggregates, and responds.
# ABOUTME: HeathcliffAgent is a singleton — call HeathcliffAgent.instance() or
# ABOUTME: construct once; subsequent calls to __init__ return the same object.

import uuid
from contextlib import contextmanager
from itertools import chain
from typing import Any, Dict, Generator, List, Optional

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
from core.runtime_profile import RuntimeProfile
from core.runtime.compat import RuntimeV2CompatibilityAdapter
from db.memory_manager import MemoryManager
from instructions.prompts import (
    USER_PROMPT_TEMPLATE,
    get_current_temporal_context,
)
from logger import logger
from utils.errors import AgentMemoryError
from utils.langfuse_client import (
    flush_langfuse,
    get_langfuse_callback_handler,
    get_langfuse_client,
)
from utils.langfuse_client import (
    trace_tags as resolve_trace_tags,
)

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
        runtime_profile: Optional[RuntimeProfile] = None,
        runtime_profile_revision: int = 0,
        runtime_v2: Optional[Any] = None,
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
        self.runtime_profile = runtime_profile or RuntimeProfile.defaults()
        self.runtime_profile.validate()
        self.runtime_profile_revision = runtime_profile_revision
        self.runtime_v2 = (
            RuntimeV2CompatibilityAdapter(runtime_v2) if runtime_v2 is not None else None
        )

        self.callbacks: List[Any] = []

        self.llm = init_chat_model(
            api_key=Config.get_ai_api_key(),
            model=self.runtime_profile.supervisor_model,
            temperature=self.runtime_profile.temperature,
            max_tokens=self.runtime_profile.max_tokens,
            top_p=Config.TOP_P,
            max_retries=3,
            timeout=Config.TIMEOUT_SECONDS,
        )
        self.coordinator = self._build_coordinator()
        logger.info("HeathcliffAgent initialised")

    def _build_coordinator(self):
        """Build coordinator graph with capability registry."""
        self._registry = build_default_registry(
            self.runtime_profile.enabled_agents,
            tool_model=self.runtime_profile.tool_model,
        )
        coordinator = build_coordinator_graph(registry=self._registry, llm=self.llm)
        logger.info(
            "Coordinator built with %d registered agents",
            len(self._registry.agent_names()),
        )
        return coordinator

    @contextmanager
    def _trace_request(
        self,
        user_input: str,
        run_id: str,
        conversation_id: str,
        requested_tags: Optional[List[str]] = None,
    ) -> Generator[tuple[Any | None, str | None], None, None]:
        """Create a Langfuse root observation when tracing is configured."""
        client = get_langfuse_client()
        if client is None:
            yield None, None
            return

        metadata = {
            str(key): str(value)[:200]
            for key, value in self.runtime_profile.metadata(
                self.runtime_profile_revision
            ).items()
        }
        tags = resolve_trace_tags(requested_tags)
        try:
            with propagate_attributes(
                trace_name=Config.TRACE_NAME,
                session_id=conversation_id,
                user_id=Config.LANGFUSE_USER_ID,
                environment=Config.ENVIRONMENT,
                version=Config.LANGFUSE_VERSION,
                metadata=metadata,
                tags=tags or None,
            ):
                try:
                    observation_context = client.start_as_current_observation(
                        name=Config.TRACE_NAME,
                        as_type="agent",
                        input={"query": user_input, "run_id": run_id},
                        metadata=metadata,
                    )
                except Exception as exc:
                    logger.warning("Unable to start Langfuse trace: %s", exc)
                    yield None, None
                    return
                with observation_context as observation:
                    yield observation, client.get_trace_url()
        finally:
            flush_langfuse(client)

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
        trace_tags: Optional[List[str]] = None,
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

        if self.runtime_v2 is not None:
            return self.runtime_v2.invoke(
                user_input, conversation_id or str(uuid.uuid4())
            )

        if self.memory_manager is None:
            raise AgentMemoryError("MemoryManager not initialised")
        conversation_id = conversation_id or str(uuid.uuid4())
        logger.info(
            "Processing input: '%.50s...' conversation: %s", user_input, conversation_id
        )

        run_id = str(uuid.uuid4())
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

            with self._trace_request(
                user_input, run_id, conversation_id, trace_tags
            ) as (observation, _):
                callbacks = self._build_callbacks(additional_callbacks)
                response = invoke_coordinator(
                    compiled_graph=self.coordinator,
                    user_input=user_input,
                    session_id=conversation_id,
                    messages=messages,
                    callbacks=callbacks,
                )
                response = response or "I encountered an error processing your request."
                if observation is not None:
                    observation.update(output={"response": response})

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
        trace_tags: Optional[List[str]] = None,
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

        if self.runtime_v2 is not None:
            conversation_id = conversation_id or str(uuid.uuid4())
            response = self.runtime_v2.invoke(user_input, conversation_id)
            yield {"type": "response", "data": response}
            yield {"type": "complete", "data": {"response": response}}
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

        run_id = str(uuid.uuid4())
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
            execution_events: List[Dict[str, Any]] = []
            approval_pending = False

            with self._trace_request(
                user_input, run_id, conversation_id, trace_tags
            ) as (observation, trace_url):
                callbacks = self._build_callbacks(additional_callbacks)
                event_stream = iter(
                    stream_coordinator(
                        compiled_graph=self.coordinator,
                        user_input=user_input,
                        session_id=conversation_id,
                        messages=messages,
                        callbacks=callbacks,
                    )
                )
                first_event = next(event_stream, None)
                if (
                    isinstance(first_event, dict)
                    and first_event.get("type") == "approval_required"
                ):
                    # Preserve the established interrupt contract for callers
                    # that only need to render an immediate approval prompt.
                    approval_pending = True
                    execution_events.append(dict(first_event))
                    yield first_event
                else:
                    yield {
                        "type": "run_started",
                        "message": "Run started",
                        "data": {
                            "run_id": run_id,
                            "profile_revision": self.runtime_profile_revision,
                            "trace_url": trace_url,
                        },
                    }
                    execution_events.append(
                        {
                            "type": "run_started",
                            "message": "Run started",
                            "data": {
                                "run_id": run_id,
                                "profile_revision": self.runtime_profile_revision,
                                "trace_url": trace_url,
                            },
                        }
                    )
                for event in (
                    ()
                    if approval_pending
                    else chain(
                        [first_event] if first_event is not None else [],
                        event_stream,
                    )
                ):
                    enriched_event = dict(event)
                    enriched_event["run_id"] = run_id
                    enriched_event["profile_revision"] = self.runtime_profile_revision
                    enriched_event["trace_url"] = trace_url
                    yield enriched_event

                    event_type = event.get("type", "")
                    if event_type != "response":
                        execution_events.append(
                            {
                                "type": event_type,
                                "message": str(event.get("message", "")),
                                "data": event.get("data", {}),
                            }
                        )
                    if event_type == "response":
                        final_response = event.get("data", "")
                    elif event_type == "approval_required":
                        approval_pending = True
                    elif event_type == "complete":
                        event_response = event.get("data", {}).get("response", "")
                        if event_response:
                            final_response = event_response

                if observation is not None and final_response:
                    observation.update(output={"response": final_response})

            if approval_pending:
                logger.info(
                    "Coordinator paused for approval: conversation %s",
                    conversation_id,
                )
                return

            if not final_response:
                final_response = "I encountered an error processing your request."
                yield {
                    "type": "response",
                    "data": final_response,
                    "run_id": run_id,
                    "profile_revision": self.runtime_profile_revision,
                }

            self.memory_manager.save_turn(
                user_input,
                final_response,
                conversation_id,
                execution_events=execution_events,
            )
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
        execution_events: Optional[List[Dict[str, Any]]] = None,
        trace_tags: Optional[List[str]] = None,
    ) -> str:
        """Resume the coordinator action paused for Streamlit approval."""
        if self.runtime_v2 is not None:
            return self.runtime_v2.resume_approval(conversation_id, approved)
        if self.memory_manager is None:
            raise AgentMemoryError("MemoryManager not initialised")

        run_id = str(uuid.uuid4())
        with self._trace_request(
            user_input, run_id, conversation_id, trace_tags
        ) as (observation, _):
            callbacks = self._build_callbacks()
            response = resume_coordinator(
                compiled_graph=self.coordinator,
                session_id=conversation_id,
                approved=approved,
                modified_input=modified_input,
                callbacks=callbacks,
            )
            if observation is not None:
                observation.update(output={"response": response})

        response = response or "I encountered an error processing your request."
        resolution_event = {
            "type": "approval_resolved",
            "message": "Action approved" if approved else "Action rejected",
            "data": {"modified_input": modified_input or ""},
        }
        if execution_events is None:
            self.memory_manager.save_turn(user_input, response, conversation_id)
        else:
            self.memory_manager.save_turn(
                user_input,
                response,
                conversation_id,
                execution_events=[*execution_events, resolution_event],
            )
        return response

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"<HeathcliffAgent model={Config.SUPERVISOR_MODEL}>"
