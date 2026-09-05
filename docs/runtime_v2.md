# Runtime V2 daemon

Runtime V2 is the durable, Gemini-native execution boundary. It supports a
local single-host SQLite/filesystem mode and a portable PostgreSQL/S3 mode.

For a completely local runtime, set:

    RUNTIME_V2_ENABLED=true
    RUNTIME_STORAGE_BACKEND=sqlite
    RUNTIME_SQLITE_PATH=.data/runtime_v2.sqlite3
    RUNTIME_ARTIFACT_DIRECTORY=.data/runtime_artifacts

No PostgreSQL, S3, MinIO, or Docker service is required. Run it with:

    uv run python -m ui.runtime_server

SQLite is intentionally single-host: run one daemon against a database file.
To move machines, stop the daemon, copy both the SQLite database and artifact
directory, then start the new daemon. For multi-host handoff and concurrent
instances, use `RUNTIME_STORAGE_BACKEND=postgres` with `DATABASE_URL` and the
S3 settings below.

## Langfuse

When `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are configured, every V2
turn emits the root trace `heathcliff.runtime.v2` under trace name
`heathcliff.agent.v2`, tagged `runtime-v2` and associated with its runtime
thread UUID. Each Gemini call is a `runtime.model` generation and each tool is
an observation named `runtime.tool.<tool_name>`. Tool input and output are
redacted when the contract is sensitive. The trace flushes after every terminal
turn, including failures.

The daemon listens on port 8700 and exposes /v2/threads, turn admission,
approval decisions, resumable SSE events, health, and readiness endpoints.
Readiness requires PostgreSQL and object storage; mutations acquire a PostgreSQL
lease, so a replacement container must be healthy before the old one drains.
Run a linux/amd64 and linux/arm64 docker build in CI to publish production images.

Set `RUNTIME_V2_ENABLED=true` and `RUNTIME_V2_URL=http://host:8700` for the
Streamlit control panel, `main.py --text`/voice CLI, and `ui/server.py` floating
blob UI. Those clients use HTTP/SSE only and fail closed if the daemon is down;
they do not create a local legacy agent as a fallback.
