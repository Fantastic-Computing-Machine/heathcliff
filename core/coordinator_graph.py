# ABOUTME: Coordinator graph replacing the single supervisor ReAct loop
# ABOUTME: Nodes: plan -> execute_subtasks -> aggregate

from __future__ import annotations

import json
import re
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextvars import copy_context
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, cast

from langchain_community.callbacks.human import HumanRejectedException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import TypedDict

from core.approval_handler import requires_approval
from core.delegation.contracts import (
    ErrorType,
    TaskResult,
    TaskSpec,
    TaskStatus,
)
from core.delegation.registry import CapabilityRegistry
from core.middleware import DelegationBudget, create_execution_budget
from core.subagents._runner import use_agent_callbacks
from logger import logger
from utils.langfuse_client import is_langfuse_callback_handler, trace_observation


class CoordinatorState(TypedDict, total=False):
    """Typed state flowing through the coordinator graph."""

    messages: List[Any]
    user_input: str
    session_id: str
    task_specs: List[Dict[str, Any]]
    task_results: List[Dict[str, Any]]
    final_response: str
    error: Optional[str]
    coordinator_started_at: float


@dataclass(frozen=True)
class CoordinatorContext:
    """Run-scoped values that must not be persisted in checkpoints."""

    callbacks: Sequence[Any] = ()


class PlannerTaskModel(BaseModel):
    """Strict schema for planner task objects."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(..., min_length=1)
    target_agent: str = Field(..., min_length=1)
    depends_on: List[int] = Field(default_factory=list)
    parallelizable: bool = Field(default=False)


_PLAN_SYSTEM = """\
You are a task planner for Heathcliff, a personal AI assistant.
Given a user request, decompose it into one or more subtasks. Each subtask
targets exactly one of the available agents.

Use the supplied conversation history to resolve short follow-up answers and
continue the user's original task. A task that consumes another task's output
must declare that task in "depends_on". For example, researching a topic and
emailing a named contact without a known address requires research, contact
lookup, then an email task that depends on both results.

Available agents and their capabilities:
{agent_descriptions}

Respond with a JSON array of subtask objects. Each object must have:
- "goal": what the subtask should accomplish (detailed natural-language instruction)
- "target_agent": agent name from the list above
- "depends_on": list of indices (0-based) of subtasks this depends on (empty if none)
- "parallelizable": true if this subtask can run in parallel with others at the same level

If the request is simple and maps to exactly one agent, return a single-element array.
Return ONLY valid JSON, no other text."""

_PLAN_REPAIR_SYSTEM = """\
Your previous planner output failed schema validation.
Return ONLY valid JSON. Do not include markdown fences or commentary.
Output must be a JSON array of objects with exactly these keys:
"goal", "target_agent", "depends_on", "parallelizable"."""

_AGGREGATE_SYSTEM = """\
You are a response aggregator for Heathcliff, a personal AI assistant.
Given a user's original request and the results from one or more subtasks,
synthesise a single coherent response.

Rules:
- Be concise but complete
- If only one subtask produced a result, return it directly (possibly lightly edited)
- If multiple results, merge them into a unified answer
- Maintain Heathcliff's British-accented, polished persona
- Only mention errors if no useful result was obtained

Return ONLY the final response text."""

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B-\x1F\x7F]")
_MAX_DEP_CONTEXT_CHARS = 1600


def _extract_llm_text(content: Any) -> str:
    """Extract string content from LLM response, handling multi-part lists."""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", str(part)) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _extract_json_payload(raw: str) -> str:
    """Strip markdown code fences if present."""
    cleaned = raw.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    lines = [line for line in lines if not line.strip().startswith("```")]
    return "\n".join(lines).strip()


def _parse_planner_tasks(raw: str) -> List[PlannerTaskModel]:
    """Parse and validate planner output against strict schema."""
    payload = json.loads(_extract_json_payload(raw))
    if not isinstance(payload, list):
        payload = [payload]
    return [PlannerTaskModel.model_validate(item) for item in payload]


def _fallback_single_task(
    user_input: str, session_id: str, registry: CapabilityRegistry
) -> List[Dict[str, Any]]:
    """Fallback path when planning fails entirely."""
    agent_names = registry.agent_names()
    target = (
        "info_agent_tool"
        if registry.get("info_agent_tool")
        else agent_names[0]
        if agent_names
        else "info_agent_tool"
    )
    spec = TaskSpec(
        goal=user_input,
        target_agent=target,
        depends_on=[],
        session_id=session_id,
    )
    return [spec.to_dict()]


def _detect_cycle_nodes(dependencies: Dict[int, List[int]]) -> set[int]:
    """Return task indices that are part of a directed cycle."""
    color: Dict[int, int] = {}
    stack: List[int] = []
    cycle_nodes: set[int] = set()

    def dfs(node: int) -> None:
        color[node] = 1
        stack.append(node)
        for dep in dependencies.get(node, []):
            dep_color = color.get(dep, 0)
            if dep_color == 0:
                dfs(dep)
            elif dep_color == 1:
                if dep in stack:
                    start = stack.index(dep)
                    cycle_nodes.update(stack[start:])
        stack.pop()
        color[node] = 2

    for node in dependencies:
        if color.get(node, 0) == 0:
            dfs(node)
    return cycle_nodes


def _validate_dependencies(
    planner_tasks: Sequence[PlannerTaskModel],
) -> tuple[Dict[int, List[int]], Dict[int, List[str]], set[int]]:
    """Validate planner dependency indices and detect cycles."""
    count = len(planner_tasks)
    valid_deps: Dict[int, List[int]] = {idx: [] for idx in range(count)}
    dep_errors: Dict[int, List[str]] = {idx: [] for idx in range(count)}
    graph: Dict[int, List[int]] = {idx: [] for idx in range(count)}

    for idx, task in enumerate(planner_tasks):
        for dep in task.depends_on:
            if dep < 0 or dep >= count:
                dep_errors[idx].append(f"out_of_range:{dep}")
                continue
            if dep == idx:
                dep_errors[idx].append(f"self_ref:{dep}")
                continue
            graph[idx].append(dep)
            if dep > idx:
                dep_errors[idx].append(f"forward_ref:{dep}")
                continue
            valid_deps[idx].append(dep)

    cycle_nodes = _detect_cycle_nodes(graph)
    return valid_deps, dep_errors, cycle_nodes


def _repair_planner_output(
    llm: Any, user_input: str, raw_output: str, error_message: str
) -> List[PlannerTaskModel]:
    """Try one constrained repair pass for invalid planner output."""
    repair_messages = [
        SystemMessage(content=_PLAN_REPAIR_SYSTEM),
        HumanMessage(
            content=(
                f"User request:\n{user_input}\n\n"
                f"Invalid output:\n{raw_output}\n\n"
                f"Validation error:\n{error_message}"
            )
        ),
    ]
    with trace_observation(
        "coordinator.plan_repair",
        as_type="chain",
        input={"user_input": user_input, "invalid_output": raw_output},
    ) as observation:
        repair_result = llm.invoke(repair_messages)
        repair_raw = _extract_llm_text(
            repair_result.content
            if hasattr(repair_result, "content")
            else repair_result
        )
        if observation is not None:
            observation.update(output={"planner_response": repair_raw})
    return _parse_planner_tasks(repair_raw)


def _plan(
    state: CoordinatorState,
    registry: CapabilityRegistry,
    llm: Any,
    budget: DelegationBudget,
) -> Dict[str, Any]:
    """Parse user intent into TaskSpecs via strict LLM planning."""
    user_input = state["user_input"]
    session_id = state.get("session_id", "")

    agent_descriptions = "\n".join(
        f"- {d.name}: {', '.join(d.capabilities)}"
        for d in registry.all_agents()
        if d.name not in ("recent_context", "load_skill", "update_master_info")
    )

    plan_messages = [
        SystemMessage(
            content=_PLAN_SYSTEM.format(agent_descriptions=agent_descriptions)
        )
    ]
    conversation_messages = list(state.get("messages", []))
    if conversation_messages:
        plan_messages.extend(conversation_messages)
    else:
        plan_messages.append(HumanMessage(content=user_input))

    try:
        with trace_observation(
            "coordinator.plan",
            as_type="chain",
            input={
                "user_input": user_input,
                "available_agents": registry.agent_names(),
            },
        ) as observation:
            result = llm.invoke(plan_messages)
            raw = _extract_llm_text(
                result.content if hasattr(result, "content") else result
            )
            if observation is not None:
                observation.update(output={"planner_response": raw})
        planner_tasks = _parse_planner_tasks(raw)
    except (ValidationError, json.JSONDecodeError, TypeError) as exc:
        logger.warning(
            "[coordinator:plan] Parse/validation failed, attempting repair: %s", exc
        )
        try:
            planner_tasks = _repair_planner_output(
                llm=llm,
                user_input=user_input,
                raw_output=raw if "raw" in locals() else "",
                error_message=str(exc),
            )
        except (ValidationError, json.JSONDecodeError, TypeError) as repair_exc:
            logger.warning(
                "[coordinator:plan] Repair failed (%s); falling back to single task",
                repair_exc,
            )
            return {
                "task_specs": _fallback_single_task(user_input, session_id, registry),
            }
    except Exception as exc:
        logger.warning("[coordinator:plan] Planner error (%s); fallback", exc)
        return {
            "task_specs": _fallback_single_task(user_input, session_id, registry),
        }

    try:
        budget.check_task_count(len(planner_tasks))
    except ValueError:
        logger.warning(
            "[coordinator:plan] Task count %d exceeds budget %d; truncating",
            len(planner_tasks),
            budget.max_tasks_per_request,
        )
        planner_tasks = planner_tasks[: budget.max_tasks_per_request]

    valid_deps, dep_errors, cycle_nodes = _validate_dependencies(planner_tasks)
    task_specs: List[Dict[str, Any]] = []

    for idx, item in enumerate(planner_tasks):
        spec = TaskSpec(
            goal=item.goal,
            target_agent=item.target_agent,
            depends_on=[
                task_specs[dep_index]["task_id"]
                for dep_index in valid_deps.get(idx, [])
            ],
            session_id=session_id,
            constraints={
                "planner_index": idx,
                "planner_parallelizable": item.parallelizable,
                "dependency_errors": dep_errors.get(idx, []),
                "dependency_cycle": idx in cycle_nodes,
            },
        )
        task_specs.append(spec.to_dict())

    logger.info(
        "[coordinator:plan] Decomposed into %d subtask(s): %s",
        len(task_specs),
        [s["target_agent"] for s in task_specs],
    )
    return {"task_specs": task_specs}


def _safe_callback_call(callback: Any, method_name: str, **kwargs: Any) -> None:
    """Invoke callback hook defensively without breaking coordinator flow."""
    method = getattr(callback, method_name, None)
    if not callable(method):
        return
    try:
        method(**kwargs)
        return
    except HumanRejectedException:
        raise
    except TypeError:
        pass
    except Exception as exc:
        logger.warning("[coordinator:callbacks] %s failed: %s", method_name, exc)
        return

    try:
        if method_name == "on_tool_start":
            method(kwargs.get("serialized", {}), kwargs.get("input_str", ""))
        elif method_name == "on_tool_end":
            method(kwargs.get("output", ""))
        elif method_name == "on_tool_error":
            method(kwargs.get("error"))
    except HumanRejectedException:
        raise
    except Exception as exc:
        logger.warning(
            "[coordinator:callbacks] %s fallback failed: %s", method_name, exc
        )


def _outer_task_callbacks(callbacks: Sequence[Any]) -> Sequence[Any]:
    """Keep Langfuse's callback inside the specialist agent it is tracing."""
    return tuple(
        callback for callback in callbacks if not is_langfuse_callback_handler(callback)
    )


def _sanitize_dependency_output(text: str, max_chars: int) -> str:
    """Normalize and sanitize dependency output before prompt injection."""
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = _CONTROL_CHAR_RE.sub("", normalized)
    normalized = re.sub(r"\r\n?", "\n", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized).strip()
    if max_chars <= 0:
        return ""
    if len(normalized) > max_chars:
        return normalized[:max_chars].rstrip() + "..."
    return normalized


def _build_dependency_context(dep_results: List[TaskResult]) -> str:
    """Build sanitized literal dependency context bounded to 1600 chars total."""
    if not dep_results:
        return ""

    per_dep_limit = max(1, _MAX_DEP_CONTEXT_CHARS // len(dep_results))
    blocks: List[str] = []
    for dep_res in dep_results:
        snippet = _sanitize_dependency_output(dep_res.output, per_dep_limit)
        if not snippet:
            continue
        blocks.append(
            f"[{dep_res.producer_agent or 'dependency'}]\n```text\n{snippet}\n```"
        )
    if not blocks:
        return ""
    return "Context from prior subtasks (literal):\n" + "\n\n".join(blocks)


def _dependency_precheck(
    spec: TaskSpec, completed_results: Dict[str, TaskResult]
) -> TaskResult | None:
    """Return an immediate dependency failure result when applicable."""
    constraint_errors = list(spec.constraints.get("dependency_errors", []))
    if spec.constraints.get("dependency_cycle"):
        constraint_errors.append("cycle")
    if constraint_errors:
        return TaskResult(
            task_id=spec.task_id,
            status=TaskStatus.DEPENDENCY_FAILED,
            errors=[
                f"Invalid dependency specification: {', '.join(constraint_errors)}"
            ],
            error_type=ErrorType.DEPENDENCY_FAILED,
            producer_agent=spec.target_agent,
        )

    deps_ok = all(
        completed_results.get(dep_id, TaskResult(task_id=dep_id)).is_success
        for dep_id in spec.depends_on
    )
    if deps_ok:
        return None
    return TaskResult(
        task_id=spec.task_id,
        status=TaskStatus.DEPENDENCY_FAILED,
        errors=["Dependency failed"],
        error_type=ErrorType.DEPENDENCY_FAILED,
        producer_agent=spec.target_agent,
    )


def _execute_with_timeout(task_callable: Any, timeout_ms: int) -> TaskResult:
    """Execute callable with timeout; return timeout TaskResult via exception sentinel."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(copy_context().run, task_callable)
    try:
        return cast(TaskResult, future.result(timeout=max(timeout_ms, 1) / 1000.0))
    except TimeoutError:
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _is_agent_failure(output: str) -> bool:
    """Recognise the explicit failure contract used by specialist wrappers."""
    return (
        output.strip()
        .lower()
        .startswith(
            (
                "research failed:",
                "music control failed:",
                "email operation failed:",
                "calendar operation failed:",
                "contacts lookup failed:",
                "communications failed:",
                "research agent is currently unavailable.",
                "music agent is currently unavailable.",
                "email agent is currently unavailable.",
                "calendar agent is currently unavailable.",
                "contacts agent is currently unavailable.",
                "communications agent is currently unavailable.",
            )
        )
    )


def _log_task_result(spec: TaskSpec, result: TaskResult) -> None:
    """Log structured per-task telemetry fields."""
    logger.info(
        "[coordinator:task] task_id=%s agent=%s status=%s error_type=%s latency_ms=%s",
        spec.task_id,
        spec.target_agent,
        result.status.value,
        result.error_type.value if result.error_type else "",
        result.latency_ms,
    )


def _execute_single_task(
    spec: TaskSpec,
    registry: CapabilityRegistry,
    callbacks: Sequence[Any],
    timeout_ms: int,
) -> TaskResult:
    """Run one task with callback bridge, validation, and timeout handling."""
    descriptor = registry.get(spec.target_agent)
    if descriptor is None:
        return TaskResult.failure(
            task_id=spec.task_id,
            error=f"Unknown agent: {spec.target_agent}",
            error_type=ErrorType.VALIDATION_ERROR,
            producer_agent=spec.target_agent,
        )

    if descriptor.locality != "local":
        return TaskResult.failure(
            task_id=spec.task_id,
            error=f"Agent locality '{descriptor.locality}' is not enabled in this phase.",
            error_type=ErrorType.VALIDATION_ERROR,
            producer_agent=descriptor.name,
        )

    run_id = uuid.uuid4()
    serialized = {"name": descriptor.name, "id": ["tool", descriptor.name]}
    outer_callbacks = _outer_task_callbacks(callbacks)
    for callback in outer_callbacks:
        try:
            _safe_callback_call(
                callback,
                "on_tool_start",
                serialized=serialized,
                input_str=spec.goal,
                run_id=run_id,
                parent_run_id=None,
            )
        except HumanRejectedException as exc:
            return TaskResult(
                task_id=spec.task_id,
                status=TaskStatus.APPROVAL_REJECTED,
                errors=[str(exc) or "Tool execution rejected by approval handler."],
                error_type=ErrorType.APPROVAL_REJECTED,
                producer_agent=descriptor.name,
            )

    start = time.monotonic()

    try:

        def invoke() -> TaskResult:
            try:
                with trace_observation(
                    descriptor.name,
                    as_type="agent",
                    input={"task_id": spec.task_id, "request": spec.goal},
                ) as observation:
                    with use_agent_callbacks(callbacks):
                        if isinstance(descriptor.invoke_fn, BaseTool):
                            output = descriptor.invoke_fn.invoke({"request": spec.goal})
                        else:
                            output = descriptor.invoke_fn(request=spec.goal)
                    output_text = str(output) if output else "No response generated."
                    if observation is not None:
                        observation.update(output={"response": output_text})
                    if _is_agent_failure(output_text):
                        return TaskResult.failure(
                            task_id=spec.task_id,
                            error=output_text,
                            error_type=ErrorType.EXECUTION_ERROR,
                            producer_agent=descriptor.name,
                            latency_ms=int((time.monotonic() - start) * 1000),
                        )
                    return TaskResult(
                        task_id=spec.task_id,
                        status=TaskStatus.COMPLETED,
                        output=output_text,
                        producer_agent=descriptor.name,
                        latency_ms=int((time.monotonic() - start) * 1000),
                    )
            except Exception as exc:
                logger.error(
                    "[coordinator:task] %s failed: %s",
                    descriptor.name,
                    exc,
                    exc_info=True,
                )
                return TaskResult.failure(
                    task_id=spec.task_id,
                    error=str(exc),
                    error_type=ErrorType.EXECUTION_ERROR,
                    producer_agent=descriptor.name,
                    latency_ms=int((time.monotonic() - start) * 1000),
                )

        # A Python worker thread cannot be killed. Never allow an approved
        # side-effecting agent to outlive the coordinator after a timeout.
        result = (
            invoke()
            if requires_approval(descriptor.name, spec.goal)
            else _execute_with_timeout(invoke, timeout_ms=timeout_ms)
        )
    except TimeoutError:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        timeout_result = TaskResult(
            task_id=spec.task_id,
            status=TaskStatus.TIMEOUT,
            errors=["Task execution timed out."],
            error_type=ErrorType.TIMEOUT,
            producer_agent=descriptor.name,
            latency_ms=elapsed_ms,
        )
        for callback in outer_callbacks:
            _safe_callback_call(
                callback,
                "on_tool_error",
                error=TimeoutError("Task execution timed out."),
                run_id=run_id,
                parent_run_id=None,
            )
        return timeout_result

    result.output = _extract_llm_text(result.output)
    if not result.producer_agent:
        result.producer_agent = descriptor.name
    for callback in outer_callbacks:
        if result.is_success:
            _safe_callback_call(
                callback,
                "on_tool_end",
                output=result.output,
                run_id=run_id,
                parent_run_id=None,
            )
        else:
            err = result.errors[0] if result.errors else "Task failed."
            _safe_callback_call(
                callback,
                "on_tool_error",
                error=RuntimeError(err),
                run_id=run_id,
                parent_run_id=None,
            )
    return result


def _execute_subtasks(
    state: CoordinatorState,
    registry: CapabilityRegistry,
    budget: DelegationBudget,
    callbacks: Sequence[Any] = (),
) -> Dict[str, Any]:
    """Run task specs with enforced budgets, timeouts, and dependency semantics."""
    task_specs = [TaskSpec.from_dict(d) for d in state.get("task_specs", [])]
    started_at = state.get("coordinator_started_at", time.monotonic())
    completed_results: Dict[str, TaskResult] = {}
    results: List[Dict[str, Any]] = []

    approval_specs = [
        spec for spec in task_specs if requires_approval(spec.target_agent, spec.goal)
    ]
    approval_task_ids = {spec.task_id for spec in approval_specs}
    approval_response: Any = {"approved": True}
    if approval_specs:
        actions = [
            {
                "task_id": spec.task_id,
                "tool_name": spec.target_agent,
                "tool_input": spec.goal,
            }
            for spec in approval_specs
        ]
        approval_response = interrupt(
            {
                "type": "approval_required",
                "session_id": state.get("session_id", ""),
                "tool_name": actions[0]["tool_name"],
                "tool_input": actions[0]["tool_input"],
                "actions": actions,
            }
        )

    approved = (
        bool(approval_response.get("approved"))
        if isinstance(approval_response, dict)
        else bool(approval_response)
    )
    rejected_task_ids = set() if approved else approval_task_ids
    modified_input = (
        approval_response.get("tool_input")
        if approved and len(approval_specs) == 1 and isinstance(approval_response, dict)
        else None
    )

    for spec in task_specs:
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        remaining_ms = budget.max_total_runtime_ms - elapsed_ms
        if remaining_ms <= 0:
            logger.warning(
                "[coordinator:execute] Max runtime reached (%d ms), stopping scheduling.",
                budget.max_total_runtime_ms,
            )
            break

        if spec.task_id in rejected_task_ids:
            rejected = TaskResult(
                task_id=spec.task_id,
                status=TaskStatus.APPROVAL_REJECTED,
                errors=["Approval rejected by user."],
                error_type=ErrorType.APPROVAL_REJECTED,
                producer_agent=spec.target_agent,
            )
            completed_results[spec.task_id] = rejected
            results.append(rejected.to_dict())
            _log_task_result(spec, rejected)
            continue

        precheck = _dependency_precheck(spec, completed_results)
        if precheck is not None:
            completed_results[spec.task_id] = precheck
            results.append(precheck.to_dict())
            _log_task_result(spec, precheck)
            continue

        dep_context_source = [
            completed_results[dep_id]
            for dep_id in spec.depends_on
            if dep_id in completed_results and completed_results[dep_id].is_success
        ]
        enhanced_goal = (
            str(modified_input)
            if spec.task_id in approval_task_ids and modified_input
            else spec.goal
        )
        dep_context = _build_dependency_context(dep_context_source)
        if dep_context:
            enhanced_goal = f"{enhanced_goal}\n\n{dep_context}"

        exec_spec = TaskSpec(
            task_id=spec.task_id,
            goal=enhanced_goal,
            target_agent=spec.target_agent,
            inputs=spec.inputs,
            constraints=spec.constraints,
            depends_on=spec.depends_on,
            session_id=spec.session_id,
            parent_task_id=spec.parent_task_id,
        )

        timeout_ms = min(budget.per_task_timeout_ms, remaining_ms)
        result = _execute_single_task(
            exec_spec, registry, callbacks, timeout_ms=timeout_ms
        )
        completed_results[spec.task_id] = result
        results.append(result.to_dict())
        _log_task_result(spec, result)

    logger.info(
        "[coordinator:execute] Completed %d subtask(s): %s",
        len(results),
        [(r.get("producer_agent"), r.get("status")) for r in results],
    )
    return {"task_results": results}


def _aggregate(state: CoordinatorState, llm: Any) -> Dict[str, Any]:
    """Merge TaskResults into a candidate response."""
    task_results = [TaskResult.from_dict(d) for d in state.get("task_results", [])]
    user_input = state.get("user_input", "")

    successful = [r for r in task_results if r.is_success]
    failed = [r for r in task_results if not r.is_success]

    if not successful:
        error_msgs = "; ".join(
            error for r in failed for error in (r.errors or ["Unknown error"])
        )
        return {
            "final_response": f"I wasn't able to complete that request. {error_msgs}"
        }

    if len(successful) == 1:
        base_response = successful[0].output
    else:
        result_summaries = "\n\n".join(
            f"[{r.producer_agent}]: {r.output}" for r in successful
        )
        agg_messages = [
            SystemMessage(content=_AGGREGATE_SYSTEM),
            HumanMessage(
                content=f"User request: {user_input}\n\nSubtask results:\n{result_summaries}"
            ),
        ]
        try:
            with trace_observation(
                "coordinator.aggregate",
                as_type="chain",
                input={"user_input": user_input, "subtask_results": result_summaries},
            ) as observation:
                agg_result = llm.invoke(agg_messages)
                base_response = _extract_llm_text(
                    agg_result.content if hasattr(agg_result, "content") else agg_result
                ).strip()
                if observation is not None:
                    observation.update(output={"response": base_response})
        except Exception as exc:
            logger.warning("[coordinator:aggregate] LLM merge failed: %s", exc)
            base_response = "\n\n---\n\n".join(r.output for r in successful)

    if failed and len(successful) < len(task_results):
        caveat = (
            f" Note: I couldn't complete {len(failed)} of {len(task_results)} subtasks, "
            "so some details may be missing."
        )
        return {"final_response": (base_response + caveat).strip()}
    return {"final_response": base_response}


def build_coordinator_graph(
    registry: CapabilityRegistry,
    llm: Any,
) -> Any:
    """Build the graph with process-local approval checkpoints.

    InMemorySaver survives Streamlit reruns but not process restarts.
    """
    budget = create_execution_budget()

    def plan_node(state: CoordinatorState) -> Dict[str, Any]:
        return _plan(state, registry, llm, budget)

    def execute_node(
        state: CoordinatorState, runtime: Runtime[CoordinatorContext]
    ) -> Dict[str, Any]:
        context = runtime.context or CoordinatorContext()
        return _execute_subtasks(state, registry, budget, callbacks=context.callbacks)

    def aggregate_node(state: CoordinatorState) -> Dict[str, Any]:
        return _aggregate(state, llm)

    graph = StateGraph(CoordinatorState, context_schema=CoordinatorContext)
    graph.add_node("plan", plan_node)
    graph.add_node("execute_subtasks", execute_node)
    graph.add_node("aggregate", aggregate_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute_subtasks")
    graph.add_edge("execute_subtasks", "aggregate")
    graph.add_edge("aggregate", END)

    # InMemorySaver keeps approvals resumable across Streamlit reruns in this
    # process. It intentionally does not survive a process restart.
    compiled = graph.compile(checkpointer=InMemorySaver()).with_config(
        {"configurable": {"thread_id": f"direct-{uuid.uuid4()}"}}
    )
    logger.info("[coordinator] Graph compiled with %d nodes", 3)
    return compiled


def invoke_coordinator(
    compiled_graph: Any,
    user_input: str,
    session_id: str,
    messages: Optional[List[Any]] = None,
    callbacks: Optional[List[Any]] = None,
) -> str:
    """Invoke the coordinator graph and return the final response string."""
    state: CoordinatorState = {
        "messages": messages or [],
        "user_input": user_input,
        "session_id": session_id,
        "task_specs": [],
        "task_results": [],
        "final_response": "",
        "error": None,
        "coordinator_started_at": time.monotonic(),
    }

    config = {"configurable": {"thread_id": session_id}}
    result = compiled_graph.invoke(
        state,
        config=config,
        context=CoordinatorContext(callbacks=list(callbacks or [])),
    )
    if result.get("__interrupt__"):
        return "Approval is required before I can complete that action."
    return result.get(
        "final_response", "I encountered an error processing your request."
    )


def resume_coordinator(
    compiled_graph: Any,
    session_id: str,
    approved: bool,
    modified_input: Optional[str] = None,
    callbacks: Optional[List[Any]] = None,
) -> str:
    """Resume a paused coordinator run under its original LangGraph thread."""
    resume_value: Dict[str, Any] = {"approved": approved}
    if modified_input is not None:
        resume_value["tool_input"] = modified_input
    config = {"configurable": {"thread_id": session_id}}
    result = compiled_graph.invoke(
        Command(resume=resume_value),
        config=config,
        context=CoordinatorContext(callbacks=list(callbacks or [])),
    )
    return result.get(
        "final_response", "I encountered an error processing your request."
    )


def stream_coordinator(
    compiled_graph: Any,
    user_input: str,
    session_id: str,
    messages: Optional[List[Any]] = None,
    callbacks: Optional[List[Any]] = None,
):
    """Stream coordinator graph execution with coordinator-native events."""
    state: CoordinatorState = {
        "messages": messages or [],
        "user_input": user_input,
        "session_id": session_id,
        "task_specs": [],
        "task_results": [],
        "final_response": "",
        "error": None,
        "coordinator_started_at": time.monotonic(),
    }

    agents_used: List[str] = []
    seen_agents: set[str] = set()
    final_response = ""

    config = {"configurable": {"thread_id": session_id}}
    interrupted = False
    for chunk in compiled_graph.stream(
        state,
        config=config,
        context=CoordinatorContext(callbacks=list(callbacks or [])),
    ):
        if not isinstance(chunk, dict):
            continue
        for node_name, update in chunk.items():
            if update is None:
                continue
            if node_name == "__interrupt__":
                interrupts = list(update) if isinstance(update, (list, tuple)) else []
                payload = interrupts[0].value if interrupts else {}
                interrupted = True
                yield {
                    "type": "approval_required",
                    "message": "Approval required",
                    "data": payload,
                }
            elif node_name == "plan":
                specs = update.get("task_specs", [])
                yield {
                    "type": "plan",
                    "message": f"Planned {len(specs)} subtask(s)",
                    "data": {
                        "task_count": len(specs),
                        "agents": [s.get("target_agent") for s in specs],
                    },
                }
                yield {
                    "type": "dispatch",
                    "message": "Dispatching sequentially",
                    "data": {"strategy": "sequential"},
                }
            elif node_name == "execute_subtasks":
                results = update.get("task_results", [])
                for r in results:
                    agent = r.get("producer_agent") or ""
                    if agent and agent not in seen_agents:
                        seen_agents.add(agent)
                        agents_used.append(agent)
                    yield {
                        "type": "subtask_complete",
                        "message": f"{agent or 'agent'} completed",
                        "data": {
                            "task_id": r.get("task_id"),
                            "agent": agent,
                            "status": r.get("status"),
                            "latency_ms": r.get("latency_ms", 0),
                        },
                    }
            elif node_name == "aggregate":
                response = update.get("final_response", "")
                if response:
                    final_response = response
                    yield {"type": "response", "data": response}

    if interrupted:
        return

    yield {
        "type": "complete",
        "message": "Processing complete",
        "data": {
            "session_id": session_id,
            "agents_used": agents_used,
            "agent_count": len(agents_used),
            "response": final_response,
        },
    }
