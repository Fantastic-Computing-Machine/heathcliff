-- PostgreSQL is authoritative. Qdrant contains only rebuildable vector projections.
CREATE TABLE IF NOT EXISTS runtime_recall_sources (
    id uuid PRIMARY KEY,
    scope text NOT NULL,
    kind text NOT NULL CHECK (kind IN ('history', 'fact')),
    source_event_id uuid NOT NULL REFERENCES runtime_events(id),
    revision integer NOT NULL CHECK (revision > 0),
    payload jsonb NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    valid_until timestamptz,
    deleted_at timestamptz
);
CREATE INDEX IF NOT EXISTS runtime_recall_sources_event_idx
    ON runtime_recall_sources (source_event_id);
CREATE INDEX IF NOT EXISTS runtime_recall_sources_expiry_idx
    ON runtime_recall_sources (valid_until) WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS runtime_recall_revisions (
    source_id uuid NOT NULL REFERENCES runtime_recall_sources(id),
    revision integer NOT NULL CHECK (revision > 0),
    payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source_id, revision)
);

-- Receipts are durable even when every extracted fact has been tombstoned.
CREATE TABLE IF NOT EXISTS runtime_recall_extractions (
    source_event_id uuid PRIMARY KEY REFERENCES runtime_events(id),
    completed_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runtime_recall_jobs (
    id uuid PRIMARY KEY,
    source_id uuid NOT NULL REFERENCES runtime_recall_sources(id),
    revision integer NOT NULL CHECK (revision > 0),
    action text NOT NULL CHECK (action IN ('upsert', 'delete')),
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed')),
    attempts integer NOT NULL DEFAULT 0,
    claim_token uuid,
    available_at timestamptz NOT NULL DEFAULT now(),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source_id, revision, action)
);
CREATE INDEX IF NOT EXISTS runtime_recall_jobs_due_idx
    ON runtime_recall_jobs (available_at, created_at) WHERE status <> 'completed';
