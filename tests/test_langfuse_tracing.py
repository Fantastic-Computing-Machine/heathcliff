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
        input={"user_input": "hello"},
    )
    observation.update.assert_called_once_with(output={"planner_response": "[]"})


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
