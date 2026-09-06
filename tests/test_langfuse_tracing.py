# ABOUTME: Langfuse request-scoping and coordinator-step instrumentation tests.

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch


def test_v4_client_receives_process_environment_and_release():
    from config import Config
    from utils.langfuse_client import _build_client_kwargs

    with (
        patch.object(Config, "ENVIRONMENT", "local-dev"),
        patch.object(Config, "LANGFUSE_RELEASE", "release-1"),
    ):
        kwargs = _build_client_kwargs()

    assert kwargs["environment"] == "local-dev"
    assert kwargs["release"] == "release-1"


def test_trace_tags_mark_pytest_and_merge_without_duplicates():
    from utils.langfuse_client import trace_tags

    with patch.dict("os.environ", {"PYTEST_CURRENT_TEST": "test_x"}, clear=True):
        assert trace_tags(["test", "live-integration"]) == [
            "test",
            "live-integration",
            "pytest",
        ]


def test_normal_trace_tags_do_not_include_test():
    from utils.langfuse_client import trace_tags

    with patch.dict("os.environ", {}, clear=True):
        assert trace_tags(["manual"]) == ["manual"]


def test_flush_failure_is_non_fatal():
    from utils.langfuse_client import flush_langfuse

    client = Mock()
    client.flush.side_effect = RuntimeError("offline")

    flush_langfuse(client)

    client.flush.assert_called_once_with()


def test_callback_handler_is_request_scoped():
    from config import Config
    from utils.langfuse_client import get_langfuse_callback_handler

    first_handler = Mock()
    second_handler = Mock()
    with (
        patch("utils.langfuse_client._is_enabled", return_value=True),
        patch("utils.langfuse_client.get_langfuse_client", return_value=Mock()),
        patch(
            "utils.langfuse_client.CallbackHandler",
            side_effect=[first_handler, second_handler],
        ) as handler_type,
    ):
        first = get_langfuse_callback_handler()
        second = get_langfuse_callback_handler()

    assert first is first_handler
    assert second is second_handler
    assert handler_type.call_count == 2
    handler_type.assert_called_with(public_key=Config.LANGFUSE_PUBLIC_KEY)


def test_trace_observation_records_a_named_nested_step():
    from utils.langfuse_client import trace_observation

    observation = Mock()
    active_span = Mock()
    active_span.get_span_context.return_value.is_valid = True

    @contextmanager
    def observation_context():
        yield observation

    client = Mock()
    client.start_as_current_observation.return_value = observation_context()
    with (
        patch("utils.langfuse_client.trace.get_current_span", return_value=active_span),
        patch("utils.langfuse_client.get_langfuse_client", return_value=client),
    ):
        with trace_observation(
            "coordinator.plan", as_type="chain", input={"user_input": "hello"}
        ) as current:
            assert current is not None
            current.update(output={"planner_response": "[]"})

    client.start_as_current_observation.assert_called_once_with(
        name="coordinator.plan",
        as_type="chain",
        input={"user_input": "[redacted]"},
    )
    observation.update.assert_called_once_with(output={"planner_response": "[redacted]"})


def test_trace_observation_does_not_create_a_root_without_an_agent_request():
    from utils.langfuse_client import trace_observation

    inactive_span = Mock()
    inactive_span.get_span_context.return_value.is_valid = False
    with (
        patch(
            "utils.langfuse_client.trace.get_current_span", return_value=inactive_span
        ),
        patch("utils.langfuse_client.get_langfuse_client") as get_client,
    ):
        with trace_observation("coordinator.plan") as observation:
            assert observation is None

    get_client.assert_not_called()


def test_outer_dispatch_skips_the_langfuse_callback():
    from core.coordinator_graph import _outer_task_callbacks

    langfuse_callback = Mock()
    approval_callback = Mock()

    with patch(
        "core.coordinator_graph.is_langfuse_callback_handler",
        side_effect=lambda callback: callback is langfuse_callback,
    ):
        assert _outer_task_callbacks([langfuse_callback, approval_callback]) == (
            approval_callback,
        )


def test_request_attributes_wrap_root_callbacks_and_flush():
    from config import Config
    from core.agent_core import HeathcliffAgent

    events = []
    observation = Mock()
    client = Mock()
    client.get_trace_url.return_value = "https://trace"

    @contextmanager
    def observation_context():
        events.append("root-enter")
        yield observation
        events.append("root-exit")

    @contextmanager
    def propagation_context(**kwargs):
        events.append(("propagate-enter", kwargs))
        yield
        events.append("propagate-exit")

    def start_observation(**kwargs):
        events.append(("root-create", kwargs))
        return observation_context()

    client.start_as_current_observation.side_effect = start_observation
    agent = object.__new__(HeathcliffAgent)
    agent.runtime_profile = SimpleNamespace(
        metadata=lambda revision: {"profile_revision": revision}
    )
    agent.runtime_profile_revision = 7

    with (
        patch("core.agent_core.get_langfuse_client", return_value=client),
        patch("core.agent_core.propagate_attributes", propagation_context),
        patch("core.agent_core.resolve_trace_tags", return_value=["test"]),
        patch("core.agent_core.flush_langfuse") as flush,
        agent._trace_request("hello", "run-1", "session-1", ["test"]) as current,
    ):
        events.append("request-body")
        assert current == (observation, "https://trace")

    propagation = events[0][1]
    assert propagation == {
        "trace_name": Config.TRACE_NAME,
        "session_id": "session-1",
        "user_id": Config.LANGFUSE_USER_ID,
        "environment": Config.ENVIRONMENT,
        "version": Config.LANGFUSE_VERSION,
        "metadata": {"profile_revision": "7"},
        "tags": ["test"],
    }
    assert events[1][0] == "root-create"
    assert events[2:] == [
        "root-enter",
        "request-body",
        "root-exit",
        "propagate-exit",
    ]
    flush.assert_called_once_with(client)


def test_runtime_v2_trace_uses_its_own_root_and_flushes():
    from config import Config
    from utils.langfuse_client import trace_runtime_turn

    observation = Mock()
    client = Mock()

    @contextmanager
    def propagation_context(**kwargs):
        assert kwargs["trace_name"] == f"{Config.TRACE_NAME}.v2"
        assert kwargs["session_id"] == "thread-1"
        assert kwargs["metadata"]["runtime"] == "v2"
        yield

    @contextmanager
    def observation_context():
        yield observation

    client.start_as_current_observation.return_value = observation_context()
    with (
        patch("utils.langfuse_client.get_langfuse_client", return_value=client),
        patch("utils.langfuse_client.propagate_attributes", propagation_context),
        patch("utils.langfuse_client.trace_tags", return_value=["runtime-v2"]),
        patch("utils.langfuse_client.flush_langfuse") as flush,
    ):
        with trace_runtime_turn(
            thread_id="thread-1", turn_id="turn-1", user_input="hello"
        ) as current:
            assert current is not None

    client.start_as_current_observation.assert_called_once_with(
        name="heathcliff.runtime.v2",
        as_type="agent",
        input={"user_input": "[redacted]", "turn_id": "turn-1"},
        trace_context={"trace_id": __import__("utils.langfuse_client", fromlist=["runtime_trace_id"]).runtime_trace_id("thread-1", "turn-1")},
    )
    flush.assert_called_once_with(client)


def test_generation_metadata_usage_and_private_payloads():
    from utils.langfuse_client import trace_observation

    client, observation, active_span = Mock(), Mock(), Mock()
    active_span.get_span_context.return_value.is_valid = True

    @contextmanager
    def span_context():
        yield observation

    client.start_as_current_observation.return_value = span_context()
    with (
        patch("utils.langfuse_client.get_langfuse_client", return_value=client),
        patch("utils.langfuse_client.trace.get_current_span", return_value=active_span),
    ):
        with trace_observation(
            "runtime.model", as_type="generation", model="gemini-test",
            model_parameters={"temperature": 0.2, "system_instruction": "private"},
            input={"provider_state": {"parts": "opaque"}, "arguments": {"to": "private"}},
        ) as current:
            current.update(
                model="gemini-effective",
                usage_details={"input": 10, "output": 3, "total": 13},
                output={"text": "private", "tool_calls": ["mail_read"],
                        "error": "private", "provider_state": {"parts": "opaque"}},
                metadata={"effective_config": {"temperature": 0.2, "api_key": "private"}},
                status_message="private",
            )

    started = client.start_as_current_observation.call_args.kwargs
    assert started["model"] == "gemini-test"
    assert started["model_parameters"] == {"temperature": 0.2}
    updated = observation.update.call_args.kwargs
    assert updated["model"] == "gemini-effective"
    assert updated["usage_details"] == {"input": 10, "output": 3, "total": 13}
    assert updated["output"]["tool_calls"] == ["mail_read"]
    assert "private" not in repr((started, updated))
    assert "opaque" not in repr((started, updated))


def test_resume_trace_id_is_stable_and_turn_specific():
    from utils.langfuse_client import runtime_trace_id

    trace_id = runtime_trace_id("thread-1", "turn-1")
    assert trace_id == runtime_trace_id("thread-1", "turn-1")
    assert len(trace_id) == 32 and int(trace_id, 16) > 0
    assert trace_id != runtime_trace_id("thread-1", "turn-2")
    assert trace_id != runtime_trace_id("thread-2", "turn-1")
    assert runtime_trace_id("ab", "c") != runtime_trace_id("a", "bc")


def test_usage_normalization_rejects_invalid_counts_and_content():
    from utils.langfuse_client import normalize_usage_details

    assert normalize_usage_details({
        "prompt_token_count": 10, "candidates_token_count": 3,
        "total_token_count": 15, "thoughts_token_count": 2,
        "cached_content_token_count": 4, "secret": "private",
    }) == {"input": 10, "output": 5, "total": 15, "input_cached": 4}
    assert normalize_usage_details({"input": -1, "output": True, "total": "4"}) == {}


def test_sdk_failures_never_replace_application_errors_or_log_payloads():
    import pytest
    from utils.langfuse_client import trace_observation

    for failing_phase in ("start", "enter", "update", "exit"):
        client, observation, active_span = Mock(), Mock(), Mock()
        active_span.get_span_context.return_value.is_valid = True
        observed_errors = []

        @contextmanager
        def span_context():
            if failing_phase == "enter":
                raise RuntimeError("secret-sdk-error")
            try:
                yield observation
            except BaseException as exc:
                observed_errors.append(str(exc))
                raise
            finally:
                if failing_phase == "exit":
                    raise RuntimeError("secret-sdk-error")

        client.start_as_current_observation.return_value = span_context()
        if failing_phase == "start":
            client.start_as_current_observation.side_effect = RuntimeError("secret-sdk-error")
        if failing_phase == "update":
            observation.update.side_effect = RuntimeError("secret-sdk-error")
        with (
            patch("utils.langfuse_client.get_langfuse_client", return_value=client),
            patch("utils.langfuse_client.trace.get_current_span", return_value=active_span),
            patch("utils.langfuse_client.logger") as logger,
        ):
            with pytest.raises(ValueError, match="application-private"):
                with trace_observation("runtime.model", as_type="generation") as current:
                    if current:
                        current.update(output="private")
                    raise ValueError("application-private")
        assert observed_errors == []
        assert "secret-sdk-error" not in repr(logger.mock_calls)
        assert "application-private" not in repr(logger.mock_calls)


def test_flush_from_event_loop_is_nonblocking_and_coalesces():
    import asyncio
    import threading
    from utils.langfuse_client import flush_langfuse

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    loop_thread = threading.get_ident()
    workers = []

    def flush():
        workers.append(threading.get_ident())
        entered.set()
        release.wait(2)
        finished.set()

    async def run():
        client = Mock(flush=flush)
        flush_langfuse(client)
        try:
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            assert not finished.is_set(), "flush blocked the event loop"
            for _ in range(10):
                flush_langfuse(client)
            assert workers == [workers[0]]
            assert workers[0] != loop_thread
        finally:
            release.set()

    asyncio.run(run())
    assert finished.wait(2)


def test_async_flush_waits_off_loop():
    import asyncio
    import threading
    from utils.langfuse_client import async_flush_langfuse

    loop_thread = threading.get_ident()
    workers = []
    asyncio.run(async_flush_langfuse(Mock(flush=lambda: workers.append(threading.get_ident()))))
    assert workers and workers[0] != loop_thread
