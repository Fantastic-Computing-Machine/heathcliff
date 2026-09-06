# Runtime reliability implementation

Approved specification: the user's PostgreSQL + Qdrant implementation plan, 2026-09-05.

## Constraints

- PostgreSQL is authoritative; Qdrant is a rebuildable semantic index. No pgvector or mandatory S3.
- Preserve existing staged changes and legacy stores; no live migrations, secret rotation, outbound effects, or client cutover without validation.
- Python/Gemini native, no keyword routing. Streamlit, CLI/voice, and Blob UI use one daemon.
- Read-only root, external credentials, encrypted durable OAuth, bounded durable background work.
- Tests precede behavior changes; distinguish local checks from live service acceptance.

## Tasks and ownership

1. Controller: native context/Gemini protocol, runtime coordination, integration, verification.
2. Storage agent: transactional PostgreSQL primitives, job queue, leases, JSON handling; db/runtime_store.py and migrations 001/002.
3. Memory agent: Qdrant recall and durable semantic pipeline; memory modules, recall repository, migration 003.
4. Tools agent: tool validation, execution, MCP catalog; tools/legacy_tools and MCP module.
5. Follow-up: HTTP/SSE and clients, stateless deployment, credentials, tracing, import tooling.

## Interface preflight

| Tasks | Shared boundary | Decision |
|---|---|---|
| 1 / 2 | Store operations and job claiming | Storage exposes typed existing objects; additional queue records are JSON dictionaries. Controller wires execution. |
| 1 / 3 | Model recall and terminal follow-up | Memory service exposes recall(query), index pending work, source validation; no mutation of engine by agent. |
| 1 / 4 | Tool execution and context | Registry preserves public execute/declarations contracts; controller owns new common contract fields. |
| 2 / 3 | SQL migration ordering | Separate ordered migrations; memory has its own tables and shares the existing pool. |
| All | Existing dirty feature checkout | Work in place as requested, disjoint files; do not stage, commit, or reset user changes. |

## Progress

- Baseline audit: 14 focused tests pass but prompt omission, invalid argument execution, JSON decoding, failure terminal state, and unstable projection IDs reproduced.
- Implementation in progress. No production acceptance claimed.
- Deployment prerequisite: PostgreSQL LAN port inaccessible; Qdrant requires an API key. Do not print or embed secrets.
