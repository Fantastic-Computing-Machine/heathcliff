-- Additive upgrade: existing vector columns, extensions and embeddings are retained.
ALTER TABLE runtime_turns ADD COLUMN IF NOT EXISTS cancel_requested boolean NOT NULL DEFAULT false;
ALTER TABLE runtime_leases ADD COLUMN IF NOT EXISTS generation bigint NOT NULL DEFAULT 1;
ALTER TABLE runtime_resource_locks ADD COLUMN IF NOT EXISTS generation bigint NOT NULL DEFAULT 1;

CREATE TABLE IF NOT EXISTS runtime_jobs (
    id uuid PRIMARY KEY,
    kind text NOT NULL,
    payload jsonb NOT NULL,
    idempotency_key text NOT NULL,
    status text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','running','completed','failed','cancelled')),
    attempts integer NOT NULL DEFAULT 0,
    max_attempts integer NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
    holder text,
    generation bigint NOT NULL DEFAULT 0,
    expires_at timestamptz,
    cancel_requested boolean NOT NULL DEFAULT false,
    result jsonb,
    error text,
    created_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    UNIQUE (kind, idempotency_key)
);
CREATE INDEX IF NOT EXISTS runtime_jobs_claim_idx ON runtime_jobs(status, created_at);
CREATE TABLE IF NOT EXISTS runtime_resource_quarantine (
    name text PRIMARY KEY,
    holder text NOT NULL,
    generation bigint NOT NULL,
    reason text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
