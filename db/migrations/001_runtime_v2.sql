CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS runtime_threads (
    id uuid PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    next_event_seq bigint NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS runtime_inputs (
    id uuid PRIMARY KEY,
    thread_id uuid NOT NULL REFERENCES runtime_threads(id),
    content text NOT NULL,
    idempotency_key text NOT NULL,
    admitted_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (thread_id, idempotency_key)
);
CREATE TABLE IF NOT EXISTS runtime_turns (
    id uuid PRIMARY KEY,
    thread_id uuid NOT NULL REFERENCES runtime_threads(id),
    input_id uuid NOT NULL REFERENCES runtime_inputs(id),
    status text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    UNIQUE (input_id)
);
CREATE TABLE IF NOT EXISTS runtime_events (
    id uuid PRIMARY KEY,
    thread_id uuid NOT NULL REFERENCES runtime_threads(id),
    turn_id uuid REFERENCES runtime_turns(id),
    sequence bigint NOT NULL,
    kind text NOT NULL,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (thread_id, sequence)
);
CREATE INDEX IF NOT EXISTS runtime_events_thread_sequence_idx
    ON runtime_events (thread_id, sequence);
CREATE TABLE IF NOT EXISTS runtime_approvals (
    id uuid PRIMARY KEY,
    thread_id uuid NOT NULL REFERENCES runtime_threads(id),
    turn_id uuid NOT NULL REFERENCES runtime_turns(id),
    tool_call jsonb NOT NULL,
    resource_scope jsonb NOT NULL,
    expires_at timestamptz NOT NULL,
    status text NOT NULL,
    decided_at timestamptz
);
CREATE TABLE IF NOT EXISTS runtime_checkpoints (
    id uuid PRIMARY KEY,
    thread_id uuid NOT NULL REFERENCES runtime_threads(id),
    through_sequence bigint NOT NULL,
    payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (thread_id, through_sequence)
);
CREATE TABLE IF NOT EXISTS runtime_memory_jobs (
    id uuid PRIMARY KEY,
    source_event_id uuid NOT NULL REFERENCES runtime_events(id),
    status text NOT NULL DEFAULT 'pending',
    attempts integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now(),
    claimed_at timestamptz,
    completed_at timestamptz
);
CREATE UNIQUE INDEX IF NOT EXISTS runtime_memory_jobs_source_event_idx
    ON runtime_memory_jobs (source_event_id);
CREATE TABLE IF NOT EXISTS personal_memories (
    id uuid PRIMARY KEY,
    kind text NOT NULL,
    subject text NOT NULL,
    content text NOT NULL,
    confidence double precision NOT NULL,
    source_event_id uuid NOT NULL REFERENCES runtime_events(id),
    source_kind text NOT NULL,
    valid_from timestamptz,
    valid_until timestamptz,
    supersedes_id uuid REFERENCES personal_memories(id),
    embedding vector(3072),
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS personal_memories_subject_idx ON personal_memories(subject);
CREATE TABLE IF NOT EXISTS runtime_leases (
    name text PRIMARY KEY,
    holder text NOT NULL,
    expires_at timestamptz NOT NULL
);
CREATE TABLE IF NOT EXISTS runtime_resource_locks (
    name text PRIMARY KEY,
    holder text NOT NULL,
    expires_at timestamptz NOT NULL
);
CREATE TABLE IF NOT EXISTS runtime_artifacts (
    content_hash text PRIMARY KEY,
    uri text NOT NULL,
    content_type text NOT NULL,
    size_bytes bigint NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS runtime_credentials (
    id uuid PRIMARY KEY,
    account_scope text NOT NULL,
    provider text NOT NULL,
    encrypted_refresh_token bytea NOT NULL,
    key_version text NOT NULL,
    expires_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (account_scope, provider)
);
