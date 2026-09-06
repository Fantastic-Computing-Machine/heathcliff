# Tools implementation handoff

## Controller requests (early)

- WSL `.venv` contains jsonschema 4.26.0; no MCP SDK distribution is installed. Please declare `jsonschema>=4.26,<5` and `mcp>=1.28,<2` in the manifest/lock (controller-owned). The adapter targets the official SDK's v1 ClientSession API, with optional import until configured.
- Proposed common fields: `verification_policy` (`none` / `required`, default `none`) and `verification_arguments: dict[str, Any]` (default empty) on ToolContract. Existing `verification_tool` requests verification using those explicit arguments; no natural-language extraction. Until common fields land, `register(..., verification_arguments=...)` supports this using the existing contract field.
- Registry keeps `execute`, `execute_ready`, `get`, `requires_approval`, and `declarations()`; adds `catalog(offset=0, limit=50)` and `declarations(names=[...])` (max 50 explicit names), plus `drain()` for lifecycle shutdown. Typed ToolResult outcomes are preserved; ordinary strings and dictionaries are data, not failure heuristics.
- Registry execution will wait for dispatched work on cancellation/timeout before returning/raising, preserving caller-owned durable locks as well as local locks. Controller must renew durable mutation/runtime leases while waiting and drain tools before closing clients. Deadlines cannot kill Python threads.
- MCP manager exposes `connect(config)`, `refresh(server_name)`, `close()`; controller wires configuration and lifecycle. No server is contacted without explicit trust and per-tool local ToolContract policies.
- Registry validates before dispatch; controller should also invoke `validate_call(call)` before saving an approval or claiming a mutation lease. Invalid/unknown calls return NOT_STARTED.

## Verification environment

WSL launcher currently fails with `Wsl/Service/E_UNEXPECTED`; UNC checkout access works. Isolated Windows test tooling will be used if WSL remains unavailable. No live external effects or secrets are required.

Implementation/test results and remaining gaps will be recorded here before handoff. Shared MEMORY/TODO and manifests are left to the controller under the assigned file ownership.
