# ABOUTME: Langfuse request-scoping and coordinator-step instrumentation tests.

from contextlib import contextmanager
from unittest.mock import Mock, patch


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
