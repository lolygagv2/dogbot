"""Schema for /data/wimz.db — VERBATIM from .claude/WIMZ_Data_Architecture_Spec.md.

Do not edit the DDL here. The spec is the source of truth: change it there
first (version bump + changelog), then mirror it here. See spec §10.
"""

SCHEMA_VERSION = "0.3"

# Spec §3 — connection pragmas, applied on every open.
PRAGMAS = [
    "PRAGMA journal_mode = WAL;",        # concurrent reads while writing
    "PRAGMA synchronous = NORMAL;",      # good durability/wear balance under WAL
    "PRAGMA busy_timeout = 5000;",       # ms, avoid lock errors under contention
    "PRAGMA foreign_keys = ON;",
]

# Spec §4 — core schema, verbatim.
SPEC_DDL = """
-- A physical robot.
CREATE TABLE device (
  device_id        TEXT PRIMARY KEY,        -- UUIDv7, stable per unit
  hardware_rev     TEXT,                    -- e.g. 'rpi5-hailo8-v3'
  firmware_version TEXT,
  created_at       INTEGER NOT NULL,
  updated_at       INTEGER NOT NULL
);

-- The human owner / household. PII lives here and is treated specially on sync.
CREATE TABLE user (
  user_id          TEXT PRIMARY KEY,        -- UUIDv7
  display_name     TEXT,                    -- PII
  contact          TEXT,                    -- PII (email/phone)
  consent_version  TEXT,                    -- which consent they accepted
  consent_scope    TEXT,                    -- JSON: {behavioral:true, media:false,...}
  consent_at       INTEGER,
  created_at       INTEGER NOT NULL,
  updated_at       INTEGER NOT NULL
);

-- An individual dog. This identity must be STABLE over time. It is the
-- backbone of the longitudinal moat.
CREATE TABLE dog (
  dog_id           TEXT PRIMARY KEY,        -- UUIDv7, never recycled
  user_id          TEXT REFERENCES user(user_id),
  name             TEXT,                    -- PII-adjacent; not used as a key
  qr_code_id       TEXT,                    -- app-generated marker id; the fleet's physical markers are ArUco, called "QR" throughout
  id_method        TEXT,                    -- 'qr' | 'direct_trained' | 'manual'  ('qr' covers ArUco markers)
  breed            TEXT,
  birthdate        INTEGER,                 -- epoch ms, nullable
  weight_g         INTEGER,                 -- nullable
  color            TEXT,                    -- v0.3: app-authoritative, human-entered; robot-consumed
  treats_per_reward INTEGER,                -- v0.3: app-authoritative reward config; robot-consumed (null -> robot defaults to 1)
  signature        TEXT,                    -- optional visual embedding ref/hash
  created_at       INTEGER NOT NULL,
  updated_at       INTEGER NOT NULL
);
CREATE INDEX idx_dog_qr ON dog(qr_code_id);

-- Every model that ever produces a label is registered here, so every
-- machine label is attributable. This is label provenance.
CREATE TABLE model_registry (
  model_id         TEXT PRIMARY KEY,        -- UUIDv7
  name             TEXT NOT NULL,           -- e.g. 'dogdetector_14'
  kind             TEXT NOT NULL,           -- 'detection' | 'pose' | 'audio' | ...
  version          TEXT NOT NULL,           -- e.g. '14' or semver
  artifact_hash    TEXT,                    -- HEF / weights hash
  deployed_at      INTEGER NOT NULL
);

-- A continuous run on the device: a training block, a monitoring window,
-- a piloted session, a daycare shift.
CREATE TABLE session (
  session_id       TEXT PRIMARY KEY,        -- UUIDv7
  device_id        TEXT NOT NULL REFERENCES device(device_id),
  mode             TEXT NOT NULL,           -- 'training' | 'monitor' | 'play' | 'daycare' | 'pilot'
  initiated_by     TEXT NOT NULL,           -- 'autonomous' | 'user_pilot' | 'scheduled'
  app_version      TEXT,
  model_versions   TEXT,                    -- JSON snapshot of active model_ids
  started_at       INTEGER NOT NULL,
  ended_at         INTEGER,
  created_at       INTEGER NOT NULL,
  updated_at       INTEGER NOT NULL
);
CREATE INDEX idx_session_device_time ON session(device_id, started_at);

-- The atomic log row. EVERYTHING observed or done becomes an event,
-- including non-selected poses and low-confidence detections.
CREATE TABLE event (
  event_id         TEXT PRIMARY KEY,        -- UUIDv7
  session_id       TEXT NOT NULL REFERENCES session(session_id),
  device_id        TEXT NOT NULL REFERENCES device(device_id),
  dog_id           TEXT REFERENCES dog(dog_id),  -- NULL if unidentified
  ts               INTEGER NOT NULL,        -- epoch ms
  seq              INTEGER,                 -- monotonic within session
  event_type       TEXT NOT NULL,           -- see Section 5 taxonomy
  payload          TEXT,                    -- JSON, typed per event_type
  confidence       REAL,                    -- 0..1 for machine-produced events
  model_id         TEXT REFERENCES model_registry(model_id),  -- producer, if machine
  label_source     TEXT NOT NULL DEFAULT 'machine', -- 'machine' | 'human' | 'auto_rule'
  media_id         TEXT REFERENCES media_asset(media_id),     -- NULL if none
  synced           INTEGER NOT NULL DEFAULT 0,
  created_at       INTEGER NOT NULL
);
CREATE INDEX idx_event_session ON event(session_id, ts);
CREATE INDEX idx_event_dog_type ON event(dog_id, event_type, ts);
CREATE INDEX idx_event_unsynced ON event(synced) WHERE synced = 0;

-- THE MOAT TABLE. One row per training attempt, linking cue -> response ->
-- reward -> outcome as a single queryable unit. Reference the underlying
-- events by id so the raw signal is preserved, but denormalize the key
-- fields here for fast longitudinal queries.
CREATE TABLE training_attempt (
  attempt_id       TEXT PRIMARY KEY,        -- UUIDv7
  session_id       TEXT NOT NULL REFERENCES session(session_id),
  dog_id           TEXT REFERENCES dog(dog_id),
  trick_label      TEXT NOT NULL,           -- 'sit' | 'down' | 'quiet' | ...
  cue_event_id     TEXT REFERENCES event(event_id),
  cue_type         TEXT,                    -- 'voice' | 'visual' | 'llm_audio'
  cue_ts           INTEGER NOT NULL,
  response_event_id TEXT REFERENCES event(event_id),
  detected_response TEXT,                   -- detected pose/behavior label
  response_ts      INTEGER,
  latency_ms       INTEGER,                 -- response_ts - cue_ts
  success          INTEGER,                 -- 0/1, or graded 0..100
  confidence       REAL,
  reward_dispensed INTEGER NOT NULL DEFAULT 0,
  dispense_id      TEXT REFERENCES dispense_log(dispense_id),
  model_versions   TEXT,                    -- JSON snapshot
  label_source     TEXT NOT NULL DEFAULT 'machine',
  media_id         TEXT REFERENCES media_asset(media_id),
  synced           INTEGER NOT NULL DEFAULT 0,
  created_at       INTEGER NOT NULL,
  updated_at       INTEGER NOT NULL
);
CREATE INDEX idx_attempt_dog_trick_time ON training_attempt(dog_id, trick_label, cue_ts);

-- Physical treat dispenses (44-slot carousel).
CREATE TABLE dispense_log (
  dispense_id      TEXT PRIMARY KEY,        -- UUIDv7
  session_id       TEXT NOT NULL REFERENCES session(session_id),
  dog_id           TEXT REFERENCES dog(dog_id),
  ts               INTEGER NOT NULL,
  slot             INTEGER,                 -- 0..43
  trigger          TEXT,                    -- 'attempt' | 'manual_pilot' | 'schedule'
  attempt_id       TEXT,                    -- soft ref to training_attempt
  dispensed_confirmed INTEGER NOT NULL DEFAULT 0, -- v0.2: IR through-beam broke = treat physically ejected
  confirm_latency_ms  INTEGER,                    -- v0.2: fire -> beam-break in ms; also a jam-trend signal
  synced           INTEGER NOT NULL DEFAULT 0,
  created_at       INTEGER NOT NULL
);

-- Index of media files. The bytes live on disk; this row is the pointer.
CREATE TABLE media_asset (
  media_id         TEXT PRIMARY KEY,        -- UUIDv7
  session_id       TEXT REFERENCES session(session_id),
  dog_id           TEXT REFERENCES dog(dog_id),
  kind             TEXT NOT NULL,           -- 'video' | 'image'
  rel_path         TEXT NOT NULL,           -- relative to /data
  codec            TEXT,                    -- 'h264' | 'h265' | 'jpeg'
  width            INTEGER,
  height           INTEGER,
  duration_ms      INTEGER,
  size_bytes       INTEGER,
  sha256           TEXT,                    -- integrity + dedupe on sync
  start_ts         INTEGER,
  end_ts           INTEGER,
  retention_class  TEXT NOT NULL DEFAULT 'standard', -- 'permanent'|'standard'|'ephemeral'
  synced           INTEGER NOT NULL DEFAULT 0,
  created_at       INTEGER NOT NULL
);
CREATE INDEX idx_media_retention ON media_asset(retention_class, created_at);

-- Periodic per-dog, per-trick rollups: the longitudinal response curve,
-- precomputed so progress queries are cheap and so the curve survives
-- even if old raw rows are pruned on the cloud side.
CREATE TABLE outcome_snapshot (
  snapshot_id      TEXT PRIMARY KEY,        -- UUIDv7
  dog_id           TEXT NOT NULL REFERENCES dog(dog_id),
  trick_label      TEXT NOT NULL,
  window_start     INTEGER NOT NULL,
  window_end       INTEGER NOT NULL,
  attempts         INTEGER NOT NULL,
  successes        INTEGER NOT NULL,
  success_rate     REAL NOT NULL,
  avg_latency_ms   INTEGER,
  mastery_level    REAL,                    -- derived 0..1
  created_at       INTEGER NOT NULL
);
CREATE INDEX idx_snapshot_dog_trick ON outcome_snapshot(dog_id, trick_label, window_start);

-- v0.2: Natural-language summary of one session, generated by the Relay LLM
-- layer from already-structured rows. Relay-generated, app-read. No always-on
-- model; one cheap per-session API call. See the Queryable-Store proposal.
CREATE TABLE session_report (
  report_id        TEXT PRIMARY KEY,        -- UUIDv7, generated on Relay
  session_id       TEXT NOT NULL REFERENCES session(session_id),
  dog_id           TEXT NOT NULL REFERENCES dog(dog_id),  -- denormalized for fast read
  generated_at     INTEGER NOT NULL,        -- epoch ms, UTC
  model_id         TEXT NOT NULL,           -- API model string, e.g. 'claude-haiku-4-5' (literal, NOT a model_registry FK)
  input_hash       TEXT NOT NULL,           -- sha256 of the structured input, for idempotency
  summary_text     TEXT NOT NULL,           -- the report shown in the app
  stats_json       TEXT,                    -- the raw numbers the summary was built from
  created_at       INTEGER NOT NULL,
  updated_at       INTEGER NOT NULL
);
-- Idempotency guard: same session + same input never produces a duplicate report.
CREATE UNIQUE INDEX idx_session_report_idem ON session_report(session_id, input_hash);

-- Tracks what has been pushed to the cloud, per table, for incremental sync.
CREATE TABLE sync_state (
  table_name       TEXT PRIMARY KEY,
  last_synced_at   INTEGER NOT NULL DEFAULT 0,  -- high-water mark (created_at/updated_at)
  last_attempt_at  INTEGER,
  last_error       TEXT
);

-- Schema version for migrations.
CREATE TABLE schema_meta (
  key              TEXT PRIMARY KEY,
  value            TEXT
);
INSERT INTO schema_meta(key, value) VALUES ('schema_version', '0.3');
"""
