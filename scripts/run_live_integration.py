"""Run bounded real-service checks and write a JSONL trace for review."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.subagents._runner import capture_agent_invocations
from main import HeathcliffAssistant

READ_ONLY_QUERIES = [
    "What is the weather in Jersey City right now?",
    "Give me the three latest technology headlines, with their sources.",
    "Research Mount Fuji's September and October hiking conditions and cite sources.",
    "Show the sender and subject of my five most recent unread Gmail messages. Do not modify anything.",
    "What is on my calendar in the next 48 hours? Do not create or modify anything.",
    "Find the contact named Aditya Agarwal and list only the saved email addresses.",
    "What is currently playing on Spotify? Do not change playback.",
    (
        "Research Mount Fuji's September and October hiking conditions, then look "
        "at my calendar next month and recommend three open weekend windows for a trip."
    ),
]

APPROVAL_QUERIES = [
    (
        "Draft an email to test@example.com with subject 'Heathcliff approval test' "
        "and body 'This is a test draft.' Do not send it."
    ),
    "Create a one-hour barber appointment tomorrow at 3 PM.",
]


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _run_query(
    assistant: HeathcliffAssistant, query: str, reject_approval: bool
) -> dict[str, Any]:
    conversation_id = str(uuid.uuid4())
    started = datetime.now(UTC)
    events: list[dict[str, Any]] = []
    response = ""
    approval: dict[str, Any] | None = None

    with capture_agent_invocations() as subagent_calls:
        for event in assistant.agent.stream_invoke(
            query, conversation_id=conversation_id
        ):
            event = _jsonable(event)
            events.append(event)
            if event.get("type") == "response":
                response = str(event.get("data") or response)
            elif event.get("type") == "error":
                response = str(event.get("data") or event.get("message") or response)
            elif event.get("type") == "approval_required":
                approval = dict(event.get("data") or {})

        if approval and reject_approval:
            response = assistant.agent.resume_approval(
                conversation_id=conversation_id,
                user_input=query,
                approved=False,
            )

    completed = datetime.now(UTC)
    return {
        "record_type": "live_integration_query",
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "duration_ms": int((completed - started).total_seconds() * 1000),
        "conversation_id": conversation_id,
        "query": query,
        "response": response,
        "approval": approval,
        "approval_rejected_by_harness": bool(approval and reject_approval),
        "events": events,
        "subagent_calls": _jsonable(subagent_calls),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-approvals", action="store_true")
    args = parser.parse_args()

    output = args.output or Path("artifacts") / (
        "live-integration-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    queries = READ_ONLY_QUERIES + (APPROVAL_QUERIES if args.include_approvals else [])
    assistant = HeathcliffAssistant(enable_audio=False)

    with output.open("w", encoding="utf-8") as handle:
        for number, query in enumerate(queries, start=1):
            print(f"[{number}/{len(queries)}] {query}", flush=True)
            try:
                record = _run_query(assistant, query, reject_approval=True)
            except Exception as exc:
                record = {
                    "record_type": "live_integration_query",
                    "query": query,
                    "response": "",
                    "fatal_error": f"{type(exc).__name__}: {exc}",
                }
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
            print(
                f"  -> {record.get('response', record.get('fatal_error', ''))[:160]}",
                flush=True,
            )

    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
