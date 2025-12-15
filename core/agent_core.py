# ABOUTME: LangGraph-based agent orchestrator for Heathcliff assistant
# ABOUTME: Manages conversation flow through retrieval, reasoning, tool calling, and output nodes

import re
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, TypedDict, Union

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from logger import logger
from instructions.prompts import build_system_prompt
from utils.langfuse_client import (
    get_langfuse_callback_handler,
    log_langfuse_interaction,
    log_langfuse_tool_event,
)


class Message(TypedDict):
    """Represents a single message in the conversation."""

    role: str  # "user", "assistant", or "tool"
    content: str
    timestamp: float
    tool_calls: List[dict]


class AgentState(TypedDict):
    """State object passed through the LangGraph workflow."""

    messages: List[Message]
    user_input: str
    session_id: str
    context: str
    memories: List[str]
    tool_calls: List[dict]
    final_response: str
    iteration_count: int  # Track reasoning iterations to prevent loops


TOOL_NAME_ALIASES: Dict[str, Set[str]] = {
    "play_track": {"spotify", "music_player", "play_music"},
    "pause_playback": {"pause_music", "spotify_pause"},
    "current_track": {"now_playing", "spotify_status"},
    "search_web": {"search", "web", "google_search"},
    "wikipedia_search": {"wikipedia", "wiki"},
    "get_news": {"news", "headlines"},
    "get_weather": {"weather"},
}


class HeathcliffAgent:
    """
    Main agent orchestrator using LangGraph.

    Manages the conversation flow through:
    1. Retrieval - fetch relevant context and memories
    2. Reasoning - process with LLM, determine next action
    3. Tool Calling - execute requested tools
    4. Output - save conversation and return response
    """

    def __init__(
        self,
        config,
        memory_manager,
        tools: Optional[Union[Dict[str, Callable[..., str]], Iterable]] = None,
    ):
        """
        Initialize the Heathcliff agent.

        Args:
            config: Config object with gemini_key and LLM settings
            memory_manager: MemoryManager instance for context retrieval
        """
        self.config = config
        self.memory_manager = memory_manager
        self.prompt = self._build_prompt_template()
        self.tools = self._prepare_tools(tools)

        # Keep original LangChain Tool objects for structured calling
        self._original_langchain_tools = []
        if tools:
            if not isinstance(tools, dict):
                self._original_langchain_tools = [
                    tool for tool in tools if hasattr(tool, "name") and hasattr(tool, "description")
                ]

        self.max_iterations = config.get(
            "llm.max_iterations", 20
        )  # Prevent infinite tool loops
        self.callbacks: List[Any] = []

        langfuse_handler = get_langfuse_callback_handler()
        if langfuse_handler:
            self.callbacks.append(langfuse_handler)
            logger.info("Langfuse callback handler enabled")
        else:
            logger.info(
                "Langfuse callback handler unavailable; falling back to manual trace events only"
            )

        # Initialize Gemini LLM
        llm_kwargs: Dict[str, Any] = {
            "model": config.get("llm.model", "gemini-2.0-flash-exp"),
            "google_api_key": config.gemini_key,
            "temperature": config.get("llm.temperature", 0.7),
            "max_output_tokens": config.get("llm.max_tokens", 1536),
            "top_p": config.get("llm.top_p", 0.9),
        }

        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        self.llm = ChatGoogleGenerativeAI(**llm_kwargs)

        # Bind tools for structured function calling (Gemini native tool use)
        # This enables the LLM to return properly formatted tool calls
        # instead of relying on text parsing
        if self._original_langchain_tools:
            self.llm_with_tools = self.llm.bind_tools(self._original_langchain_tools)
            logger.info(f"✓ Bound {len(self._original_langchain_tools)} tools to LLM for structured function calling")
        else:
            self.llm_with_tools = self.llm
            logger.warning("No LangChain tools found - falling back to text-based tool calling only")

        # Build the LangGraph workflow
        self.graph = self._build_graph()

        # self.graph.get_graph().draw_mermaid_png(output_file_path="i3.png")

        logger.info(
            "HeathcliffAgent initialized with max_iterations=%d", self.max_iterations
        )

    def _prepare_tools(
        self, tools: Optional[Union[Dict[str, Callable[..., str]], Iterable]]
    ) -> Dict[str, Callable[..., str]]:
        """Normalize tool registry to a simple name -> callable mapping."""

        normalized: Dict[str, Callable[..., str]] = {}

        if not tools:
            return normalized

        if isinstance(tools, dict):
            iterator = tools.items()
            for name, func in iterator:
                if callable(func):
                    self._register_tool(normalized, name, func)
            return normalized

        for tool in tools:
            tool_name = getattr(tool, "name", None)
            if tool_name and callable(getattr(tool, "invoke", None)):
                wrapped = self._wrap_langchain_tool(tool)
                self._register_tool(normalized, tool_name, wrapped)
            elif callable(tool):
                self._register_tool(
                    normalized, getattr(tool, "__name__", "tool"), tool
                )

        return normalized

    def _wrap_langchain_tool(self, tool: Any) -> Callable[..., Any]:
        """Wrap LangChain BaseTool instances so they match the callable contract."""

        def _runner(*args, **kwargs):
            tool_input: Any

            if kwargs:
                tool_input = kwargs
            elif len(args) == 1:
                tool_input = args[0]
            elif len(args) > 1:
                tool_input = list(args)
            else:
                tool_input = {}

            try:
                return tool.invoke(tool_input)
            except TypeError:
                # Fallback to the original invocation style if the tool expects kwargs
                if kwargs:
                    return tool.invoke(**kwargs)
                return tool.invoke(*args)

        return _runner

    def _register_tool(
        self, registry: Dict[str, Callable[..., Any]], raw_name: str, func: Callable
    ) -> None:
        """Register a tool plus any aliases for easier LLM routing."""

        if not raw_name:
            raw_name = getattr(func, "__name__", "tool")

        canonical_name = raw_name.lower()
        registry[canonical_name] = func

        alias_target = canonical_name
        for key, aliases in TOOL_NAME_ALIASES.items():
            if canonical_name == key or canonical_name in aliases:
                alias_target = key
                break

        # Ensure canonical alias target points to the same function
        registry[alias_target] = func

        for alias in TOOL_NAME_ALIASES.get(alias_target, set()):
            registry.setdefault(alias, func)

    def _build_prompt_template(self) -> ChatPromptTemplate:
        """Create the shared prompt template for Gemini reasoning calls."""

        # Load master information from config and build dynamic system prompt
        master_info = self.config.get("master", {})
        system_prompt = build_system_prompt(master_info)

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("system", "Memories:\n{memories_block}"),
                ("system", "Recent chat context:\n{context_block}"),
                (
                    "system",
                    "Tool feedback (DO NOT re-call these tools):\n{tool_results_block}",
                ),
                ("system", "Live transcript this session:\n{message_block}"),
                ("human", "{user_input}"),
            ]
        )

    def _build_graph(self) -> CompiledStateGraph:
        """
        Construct the LangGraph workflow with nodes and edges.

        Flow:
        START -> retrieval -> reasoning -> [routing] -> tool_node OR output_node -> END

        Returns:
            Compiled StateGraph ready to invoke
        """
        # Create the state graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("retrieval_node", self._retrieval_node)
        workflow.add_node("reasoning_node", self._reasoning_node)
        workflow.add_node("tool_node", self._tool_calling_node)
        workflow.add_node("output_node", self._output_node)

        # Set entry point
        workflow.set_entry_point("retrieval_node")

        # Add edges
        workflow.add_edge("retrieval_node", "reasoning_node")

        # Conditional edge from reasoning based on tool_calls
        workflow.add_conditional_edges(
            "reasoning_node",
            self._route_after_reasoning,
            {"tool_node": "tool_node", "output_node": "output_node"},
        )

        # Tool node loops back to reasoning
        workflow.add_edge("tool_node", "reasoning_node")

        # Output node goes to END
        workflow.add_edge("output_node", END)

        # Compile and return
        return workflow.compile()

    def _route_after_reasoning(self, state: AgentState) -> str:
        """
        Routing function to decide next node after reasoning.

        Args:
            state: Current agent state

        Returns:
            str: Either "tool_node" or "output_node"
        """
        iteration_count = state.get("iteration_count", 0)

        # Prevent infinite loops - force output after max iterations
        if iteration_count >= self.max_iterations:
            logger.warning(
                f"Max iterations ({self.max_iterations}) reached, forcing output"
            )
            return "output_node"

        if state.get("tool_calls") and len(state["tool_calls"]) > 0:
            return "tool_node"
        return "output_node"

    def _retrieval_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Node 1: Query memory for context and relevant memories.

        Args:
            state: Current execution state

        Returns:
            dict with updated 'context' and 'memories' fields
        """
        user_input = state["user_input"]
        session_id = state["session_id"]

        context_str = ""
        memories_list = []

        try:
            # Get relevant chat history
            chat_results = self.memory_manager.get_chat_context(
                query=user_input, session_id=session_id, n=5
            )

            if chat_results and chat_results.get("documents"):
                docs = chat_results["documents"][0] if chat_results["documents"] else []
                metas = chat_results.get("metadatas", [[]])[0]

                context_parts = []
                for i, doc in enumerate(docs):
                    role = (
                        metas[i].get("role", "unknown") if i < len(metas) else "unknown"
                    )
                    context_parts.append(f"[{role}]: {doc}")

                context_str = "\n".join(context_parts)

        except Exception as e:
            logger.warning(f"Failed to retrieve chat context: {e}")

        try:
            # Get relevant long-term memories
            memory_results = self.memory_manager.recall(query=user_input, n=3)

            if memory_results and memory_results.get("documents"):
                docs = (
                    memory_results["documents"][0]
                    if memory_results["documents"]
                    else []
                )
                memories_list = list(docs)

        except Exception as e:
            logger.warning(f"Failed to retrieve memories: {e}")

        logger.debug(
            f"Retrieved {len(memories_list)} memories, context length: {len(context_str)}"
        )

        return {**state, "context": context_str, "memories": memories_list}

    def _reasoning_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Node 2: Call Gemini with context, determine next action.

        Args:
            state: Current execution state with context/memories injected

        Returns:
            dict with 'final_response' if no tools, or 'tool_calls' if tools needed
        """
        user_input = state["user_input"]
        context = state.get("context", "")
        memories = state.get("memories", [])
        messages = state.get("messages", [])
        iteration_count = state.get("iteration_count", 0)

        # Increment iteration count
        new_iteration_count = iteration_count + 1

        logger.debug(f"Reasoning iteration {new_iteration_count}/{self.max_iterations}")

        memories_str = (
            "\n".join(f"- {m}" for m in memories)
            if memories
            else "No relevant memories."
        )

        # Build tool results block with emphasis on already-called tools
        tool_messages = [msg for msg in messages if msg.get("role") == "tool"]
        if tool_messages:
            tool_results_parts = [
                "IMPORTANT: You have already called these tools. DO NOT call them again:"
            ]
            for msg in tool_messages:
                tool_results_parts.append(f"✓ {msg.get('content', '')}")
            tool_results = "\n".join(tool_results_parts)
        else:
            tool_results = "No tools called yet in this turn."

        conversation_block = "\n".join(
            f"{msg.get('role', 'unknown').title()}: {msg.get('content', '')}"
            for msg in messages
            if msg.get("role") in {"user", "assistant"}
        )

        prompt_messages = self.prompt.format_messages(
            memories_block=memories_str,
            context_block=context or "No prior conversation in this session.",
            tool_results_block=tool_results,
            message_block=conversation_block or "No live transcript yet.",
            user_input=user_input,
        )

        try:
            # Use LLM with bound tools for structured function calling
            response = self.llm_with_tools.invoke(prompt_messages)

            # Extract response text properly from Gemini's structured format
            if hasattr(response, "content"):
                content = response.content

                # Handle structured content blocks (Gemini native function calling format)
                if isinstance(content, list) and len(content) > 0:
                    # Extract text from first text block
                    first_block = content[0]
                    if isinstance(first_block, dict) and first_block.get("type") == "text":
                        response_text = first_block.get("text", "")
                    else:
                        # Fallback: join all text blocks
                        text_parts = [
                            block.get("text", "")
                            for block in content
                            if isinstance(block, dict) and block.get("type") == "text"
                        ]
                        response_text = " ".join(text_parts) if text_parts else str(content)

                # Handle plain string content (legacy format)
                elif isinstance(content, str):
                    response_text = content

                # Fallback for unknown formats
                else:
                    response_text = str(content)
            else:
                response_text = str(response)

            logger.debug(f"LLM response: {response_text[:100]}...")

            # APPROACH 1: Try structured tool calls first (Gemini native)
            tool_calls = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                # Gemini returned structured tool calls - much more reliable!
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "").lower()
                    tool_args = tool_call.get("args", {})
                    tool_calls.append({"name": tool_name, "args": tool_args})

                logger.info(f"Structured tool calls detected: {tool_calls}")
            else:
                # APPROACH 2: Fall back to regex parsing from text (legacy support)
                tool_calls = self._parse_tool_calls(response_text)
                if tool_calls:
                    logger.info(f"Text-based tool calls detected (fallback): {tool_calls}")

            # Filter out duplicate tool calls from previous iterations
            called_tools = {msg.get("tool_name") for msg in tool_messages}
            filtered_tool_calls = [
                tc for tc in tool_calls if tc.get("name") not in called_tools
            ]

            if filtered_tool_calls != tool_calls:
                logger.warning(
                    f"Filtered out {len(tool_calls) - len(filtered_tool_calls)} duplicate tool calls"
                )

            if filtered_tool_calls:
                logger.info(f"Tool calls detected: {filtered_tool_calls}")
                return {
                    **state,
                    "tool_calls": filtered_tool_calls,
                    "final_response": "",
                    "iteration_count": new_iteration_count,
                }
            else:
                # Clean up response (remove any accidental tool markers)
                clean_response = re.sub(r"\[TOOL:.*?\]", "", response_text).strip()
                return {
                    **state,
                    "tool_calls": [],
                    "final_response": clean_response,
                    "iteration_count": new_iteration_count,
                }

        except Exception as e:
            logger.error(f"LLM invocation failed: {e}", exc_info=True)
            return {
                **state,
                "tool_calls": [],
                "final_response": "I encountered an error processing your request. Please try again.",
                "iteration_count": new_iteration_count,
            }

    def _parse_tool_calls(self, response_text: str) -> List[dict]:
        """
        Parse tool call requests from LLM response.

        Args:
            response_text: Raw LLM response text

        Returns:
            List of tool call dictionaries with 'name' and 'args'
        """
        tool_calls = []

        # Pattern: [TOOL: name param=value param2=value2]
        pattern = r"\[TOOL:\s*(\w+)(?:\s+([^\]]+))?\]"
        matches = re.findall(pattern, response_text)

        for match in matches:
            tool_name = match[0].lower()
            args_str = match[1] if len(match) > 1 else ""

            # Parse arguments
            args = {}
            if args_str:
                # Handle key=value pairs with support for:
                # 1. Quoted values: query="taylor swift love story"
                # 2. Unquoted multi-word values: query=taylor swift love story (until next param or end)

                # First try to match quoted values: param="value with spaces"
                quoted_pattern = r'(\w+)="([^"]*)"'
                quoted_matches = re.findall(quoted_pattern, args_str)

                if quoted_matches:
                    # Use quoted values
                    for key, value in quoted_matches:
                        args[key] = value
                else:
                    # No quotes - try to parse intelligently
                    # If there's an = sign, assume everything after it (until next param or ]) is the value
                    if '=' in args_str:
                        # Split on first = to get key and the rest
                        parts = args_str.split('=', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            # Value is everything after = (trimmed)
                            value = parts[1].strip()
                            args[key] = value
                    else:
                        # Positional argument (no = at all)
                        args["value"] = args_str.strip()

            tool_calls.append({"name": tool_name, "args": args})

        return tool_calls

    def _tool_calling_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Node 3: Execute requested tools and return results.

        Args:
            state: Current state with tool_calls list

        Returns:
            dict with results appended to messages, tool_calls cleared
        """
        tool_calls = state.get("tool_calls", [])
        messages = list(state.get("messages", []))
        session_id = state.get("session_id", "unknown")

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "").lower()
            args = tool_call.get("args", {})

            logger.info(f"Executing tool: {tool_name} with args: {args}")

            tool_func = self.tools.get(tool_name)

            if not tool_func:
                result = f"Unknown tool: {tool_name}"
                status = "error"
            else:
                status = "success"
                try:
                    if args:
                        # Support both keyword args and shorthand "value"
                        if set(args.keys()) == {"value"}:
                            result = tool_func(args["value"])
                        else:
                            result = tool_func(**args)
                    else:
                        result = tool_func()
                except TypeError:
                    # Fallback to positional-only call with single value
                    value = args.get("value") or next(iter(args.values()), None)
                    result = tool_func(value) if value is not None else tool_func()
                except Exception as e:  # pragma: no cover - defensive guard
                    logger.error(f"Tool execution failed: {e}", exc_info=True)
                    result = f"Tool {tool_name} failed: {str(e)}"
                    status = "error"

            logger.info(f"Tool result: {result}")
            log_langfuse_tool_event(session_id, tool_name, args, result, status=status)

            messages.append(
                {
                    "role": "tool",
                    "content": result,
                    "timestamp": datetime.now().timestamp(),
                    "tool_calls": [],
                    "tool_name": tool_name,  # Track which tool was called
                }
            )

        return {**state, "messages": messages, "tool_calls": []}

    def _output_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Node 4: Save conversation to memory and prepare final output.

        Args:
            state: Final state with 'final_response' set

        Returns:
            dict with saved data, ready to return to user
        """
        user_input = state["user_input"]
        final_response = state["final_response"]
        session_id = state["session_id"]

        messages = list(state.get("messages", []))

        try:
            # Save conversation to memory
            self.memory_manager.save_chat(user_input, final_response, session_id)
            logger.debug(f"Saved chat to session {session_id}")

            messages.append(
                {
                    "role": "assistant",
                    "content": final_response,
                    "timestamp": datetime.now().timestamp(),
                    "tool_calls": [],
                }
            )

        except Exception as e:
            logger.error(f"Failed to save chat: {e}")

        tool_names = [
            msg.get("tool_name")
            for msg in messages
            if msg.get("role") == "tool" and msg.get("tool_name")
        ]

        metadata = {
            "tool_count": len(tool_names),
            "tool_names": tool_names,
            "iteration_count": state.get("iteration_count", 0),
        }

        log_langfuse_interaction(
            session_id=session_id,
            user_input=user_input,
            response=final_response,
            status="success",
            extra_metadata=metadata,
        )

        return {**state, "messages": messages}

    def stream_invoke(self, user_input: str, session_id: Optional[str] = None):
        """
        Generator method that yields status updates during agent execution.

        This is the streaming version of invoke() that provides real-time visibility
        into agent processing stages for UI integration (e.g., Streamlit status containers).

        Args:
            user_input: User's text query
            session_id: Optional session ID (generates UUID if not provided)

        Yields:
            dict: Event dictionaries with structure:
                - {"type": "status", "message": str, "data": dict} - Status updates
                - {"type": "tool", "message": str, "data": dict} - Tool execution events
                - {"type": "response", "data": str} - Final response text
                - {"type": "complete", "message": str, "data": dict} - Completion with metadata
                - {"type": "error", "message": str, "data": str} - Error events

        Example:
            >>> for event in agent.stream_invoke("What's the weather?"):
            ...     if event["type"] == "status":
            ...         print(f"Status: {event['message']}")
            ...     elif event["type"] == "response":
            ...         print(f"Response: {event['data']}")
        """
        # Validate input
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

        user_message = {
            "role": "user",
            "content": user_input.strip(),
            "timestamp": datetime.now().timestamp(),
            "tool_calls": [],
        }

        # Initialize agent state
        state = {
            "messages": [user_message],
            "user_input": user_input.strip(),
            "session_id": session_id,
            "context": "",
            "memories": [],
            "tool_calls": [],
            "final_response": "",
            "iteration_count": 0,
        }

        try:
            # Step 1: Retrieval node
            yield {
                "type": "status",
                "message": "Retrieving memories and context...",
                "data": {"phase": "retrieval"},
            }

            state = self._retrieval_node(state)

            memories_count = len(state.get("memories", []))
            context_length = len(state.get("context", ""))

            yield {
                "type": "status",
                "message": f"Retrieved {memories_count} memories",
                "data": {
                    "phase": "retrieval_complete",
                    "memories": memories_count,
                    "context_length": context_length,
                },
            }

            # Step 2: Reasoning loop (up to max_iterations)
            while state.get("iteration_count", 0) < self.max_iterations:
                iteration = state.get("iteration_count", 0) + 1

                yield {
                    "type": "status",
                    "message": f"Reasoning iteration {iteration}/{self.max_iterations}...",
                    "data": {
                        "phase": "reasoning",
                        "iteration": iteration,
                        "max_iterations": self.max_iterations,
                    },
                }

                # Call reasoning node
                state = self._reasoning_node(state)

                # Check if we have tool calls
                tool_calls = state.get("tool_calls", [])

                if not tool_calls:
                    # No tools needed, we have final response
                    break

                # Step 3: Tool execution
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "unknown")
                    tool_args = tool_call.get("args", {})

                    yield {
                        "type": "tool",
                        "message": f"Calling tool: {tool_name}",
                        "data": {"name": tool_name, "args": tool_args, "status": "starting"},
                    }

                # Execute tools
                state = self._tool_calling_node(state)

                # Report tool results
                tool_messages = [
                    msg for msg in state.get("messages", []) if msg.get("role") == "tool"
                ]

                if tool_messages:
                    last_tool_msg = tool_messages[-1]
                    tool_result = last_tool_msg.get("content", "")
                    tool_name = last_tool_msg.get("tool_name", "unknown")

                    yield {
                        "type": "tool",
                        "message": f"Tool '{tool_name}' completed",
                        "data": {
                            "name": tool_name,
                            "result": tool_result,
                            "status": "success",
                        },
                    }

            # Check if we hit max iterations
            if state.get("iteration_count", 0) >= self.max_iterations:
                yield {
                    "type": "status",
                    "message": f"Max iterations ({self.max_iterations}) reached",
                    "data": {
                        "phase": "max_iterations",
                        "iteration_count": state.get("iteration_count", 0),
                    },
                }

            # Step 4: Output node
            yield {
                "type": "status",
                "message": "Generating final response...",
                "data": {"phase": "output"},
            }

            state = self._output_node(state)

            final_response = state.get("final_response", "")

            if not final_response:
                final_response = "I encountered an error processing your request."

            # Yield the response
            yield {"type": "response", "data": final_response}

            # Extract tool names used
            tool_names = [
                msg.get("tool_name")
                for msg in state.get("messages", [])
                if msg.get("role") == "tool" and msg.get("tool_name")
            ]

            # Yield completion
            yield {
                "type": "complete",
                "message": "Processing complete",
                "data": {
                    "session_id": session_id,
                    "iteration_count": state.get("iteration_count", 0),
                    "tools_used": tool_names,
                    "tool_count": len(tool_names),
                },
            }

            logger.info(f"Stream complete: '{final_response[:50]}...'")

        except Exception as e:
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

    def invoke(self, user_input: str, session_id: Optional[str] = None) -> str:
        """
        Main entry point: Process user input and return agent response.

        Args:
            user_input: User's text query
            session_id: Optional session ID (generates UUID if not provided)

        Returns:
            str: Agent's response text

        Raises:
            ValueError: If user_input is empty or exceeds 10k chars

        Example:
            >>> response = agent.invoke("What's the weather?")
            >>> response
            "The weather is 72F and sunny."
        """
        # Validate input
        if not user_input or not user_input.strip():
            raise ValueError("User input cannot be empty")

        if len(user_input) > 10000:
            raise ValueError("User input exceeds maximum length of 10000 characters")

        # Generate session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        logger.info(f"Processing input: '{user_input[:50]}...' session: {session_id}")

        user_message = {
            "role": "user",
            "content": user_input.strip(),
            "timestamp": datetime.now().timestamp(),
            "tool_calls": [],
        }

        # Initialize agent state
        initial_state = {
            "messages": [user_message],
            "user_input": user_input.strip(),
            "session_id": session_id,
            "context": "",
            "memories": [],
            "tool_calls": [],
            "final_response": "",
            "iteration_count": 0,
        }

        try:
            # Invoke the graph
            graph_config: Dict[str, Any] = {}
            if self.callbacks:
                graph_config["callbacks"] = self.callbacks

            result = self.graph.invoke(initial_state, config=graph_config or None)
            response = result.get("final_response", "")

            if not response:
                response = "I encountered an error processing your request."

            logger.info(f"Response: '{response[:50]}...'")
            return response

        except Exception as e:
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

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"<HeathcliffAgent model={self.config.get('llm.model', 'gemini-2.0-flash-exp')}>"
