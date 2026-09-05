"""Synchronous client used by interactive Heathcliff surfaces when Runtime V2 is enabled."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class RuntimeV2HttpClient:
    """Compatibility surface over the daemon's HTTP/SSE API; never falls back locally."""

    def __init__(self, base_url: str, timeout_seconds: int = 45) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._approvals: dict[str, str] = {}

    @staticmethod
    def _thread_id(conversation_id: str | None) -> str:
        if conversation_id:
            try:
                return str(uuid.UUID(conversation_id))
            except ValueError:
                return str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"heathcliff:{conversation_id}")
                )
        return str(uuid.uuid4())

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body else {},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read() or b"{}")
        except URLError as exc:
            raise RuntimeError("Runtime V2 daemon is unavailable") from exc

    def _events(self, thread_id: str, after: int) -> Iterator[dict[str, Any]]:
        query = urlencode({"after": after})
        request = Request(f"{self.base_url}/v2/threads/{thread_id}/events?{query}")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                event_name = "message"
                event_id = after
                data: list[str] = []
                for raw_line in response:
                    line = raw_line.decode().rstrip("\r\n")
                    if not line:
                        if data:
                            yield {
                                "event": event_name,
                                "id": event_id,
                                "data": json.loads("\n".join(data)),
                            }
                        event_name, data = "message", []
                    elif line.startswith("id: "):
                        event_id = int(line[4:])
                    elif line.startswith("event: "):
                        event_name = line[7:]
                    elif line.startswith("data: "):
                        data.append(line[6:])
        except URLError as exc:
            raise RuntimeError("Runtime V2 event stream is unavailable") from exc

    def stream_invoke(
        self, user_input: str, conversation_id: str | None = None, **_: Any
    ) -> Iterator[dict[str, Any]]:
        thread_id = self._thread_id(conversation_id)
        admitted = self._request(
            "POST",
            f"/v2/threads/{thread_id}/turns",
            {"content": user_input, "idempotency_key": str(uuid.uuid4())},
        )
        cursor = int(admitted.get("event_cursor", 0))
        yield {"type": "run_started", "data": {"thread_id": thread_id}}
        for event in self._events(thread_id, cursor):
            payload = event["data"].get("payload", {})
            kind = event["event"]
            if kind == "approval.required":
                approval_id = str(payload["id"])
                self._approvals[thread_id] = approval_id
                yield {
                    "type": "approval_required",
                    "data": {
                        "approval_id": approval_id,
                        "session_id": thread_id,
                        "tool_name": payload["tool_call"]["name"],
                        "tool_input": json.dumps(payload["tool_call"]["arguments"]),
                    },
                }
            elif kind == "tool.proposed":
                yield {"type": "dispatch", "data": payload}
            elif kind == "tool.completed":
                yield {"type": "subtask_complete", "data": payload}
            elif kind == "turn.completed":
                yield {"type": "response", "data": payload.get("response", "")}
            elif kind in {"turn.failed", "turn.cancelled"}:
                yield {"type": "error", "data": payload.get("error", kind)}

    def invoke(self, user_input: str, conversation_id: str | None = None) -> str:
        response = "I encountered an error processing your request."
        for event in self.stream_invoke(user_input, conversation_id):
            if event["type"] in {"response", "error"}:
                response = str(event["data"])
        return response

    def resume_approval(
        self,
        conversation_id: str,
        approved: bool,
        **_: Any,
    ) -> str:
        thread_id = self._thread_id(conversation_id)
        approval_id = self._approvals.get(thread_id)
        if approval_id is None:
            raise ValueError("No Runtime V2 approval is pending for this conversation")
        result = self._request(
            "POST", f"/v2/approvals/{approval_id}/decision", {"approved": approved}
        )
        self._approvals.pop(thread_id, None)
        return str(
            result.get("response")
            or ("Action completed." if approved else "Action rejected.")
        )
