# WIM-Z Data Refactor — Design (v1 draft, 2026-09-01)

**Status:** DESIGN ONLY — no code until Morgan + App Claude approve.
**Owner:** Robot side. **Reviewer:** App Claude (against their consumer
requirements, received 2026-09-01). **Parent doc:** `.claude/WIMZ_Data_Architecture_Spec.md`
(the spec stays authoritative; everything here lands as additive spec bumps).

**Standing decisions (not up for debate in review):**
- No backfill of legacy rows in v1 (Morgan). The existing `scripts/backfill_wimz.py`
  ledger stands as-is; no further backfill.
- The live WS/relay event contract the app parses does NOT change. This refactor
  shapes storage and analytics only.

---

## 1. Current state (audited 2026-09-01)

Four SQLite databases coexist, all under the shared data dir:

| DB | Size / hot tables | Status |
|---|---|---|
| `treatbot.db` (`core/store.py`, `core/bark_store.py`, `core/weekly_summary.py`) | 80 MB — telemetry 815k rows, events 52k, rewards 1.2k, sg_interventions 201, coaching_sessions 91, sg_sessions 10 | LIVE, primary legacy store |
| `wimz.db` (`core/data/`, spec store) | 1.8 MB — event 3.7k, session 153, training_attempt 177, dispense_log 131, media_asset 171 | LIVE, dual-write, spec-conformant |
| `dogbot.db` (`core/dog_database.py`) | dogs 2, behavior_events 2.5k | ORPHANED — no live callers since Aug 18 |
| `missions.db` (`missions/__init__.py`) | 4 small tables | LEGACY — live missions write treatbot.db |

Plus JSON state files (robot_state, night_mode, schedules, audio_state) — out of
scope for v1 except where noted.

### Why the refactor is needed (concrete defects found)

1. **Four incompatible timestamp conventions**: unix float (`time.time()`) in most
   treatbot tables; local-naive ISO in `dog_events`; SQLite UTC text in `barks`
   (queried with *local* cutoffs — a silent UTC/local skew bug in
   `bark_store.py:110-117` and weekly_summary); epoch-ms INTEGER in wimz.db.
2. **Five dog-identity formats**: `dog_N` tracker slots, `aruco_<marker>`,
   the app/cloud canonical UUID, a *different* wimz-local UUIDv7, and free-text
   junk. The same dog (Elsa) has two unrelated UUIDs in treatbot.db vs wimz.db;
   nothing reconciles them.
3. **The SG bark-type intelligence is lost on process exit.**
   `session_bark_log` (per-bark ts/label/group/confidence/loudness/dog) is
   in-memory only. The `quiet_periods_json` / `dog_bark_counts_json` columns are
   always written as `'[]'` / `'{}'` (call sites omit the args). Raw `barks`
   rows are pruned at 24 h and carry `emotion`, not the SG reporting group.
   Net: exactly the data the app wants per-bark rows of is the data we discard.
4. **Retention is broken both directions**: `cleanup_old_data()` (events +
   telemetry) has ZERO callers — hence 815k telemetry rows and an 80 MB DB —
   while barks/dog_events are aggressively pruned at 24 h. wimz.db has no sweep
   at all (`retention_class` is written, never enforced).
5. **Version drift**: spec changelog says 0.5, spec §4 DDL still inserts '0.3',
   `core/data/schema.py` says 0.4, live wimz.db is stamped 0.4, wimz_store
   docstring says 0.3. All additive so nothing is broken, but it must be
   squared before new migrations land.
6. treatbot.db never sets `PRAGMA foreign_keys=ON`; `barks.duration_ms` and
   `session_id` are never populated; treatbot `dogs` holds only test fixtures
   (real profiles live in the `settings['dog_profiles']` JSON blob).

---

## 2. Target state (v1)

**One behavioral store: `wimz.db`, per the spec.** The refactor's endgame was
already declared in `core/data/__init__.py`: retire `core/store.py`'s behavioral
tables, `bark_store.py`, `dog_database.py`, and the missions/__init__ DB into
wimz.db. This design fills the gaps between the spec-as-implemented and the
app's five consumer requirements, then sequences the cutover.

### R1 — Per-bark rows

Per-bark data stays in the spec's `event` table (`event_type='bark'`) — no new
table. The bark payload contract is extended (spec §5 bump, additive):

```json
{
  "db": -12.4, "duration_ms": 900, "class": "bark",
  "emotion": "anxious", "gate": "passed",
  "bark_type": "distress",          // SG reporting group (the config mapping)
  "bark_label": "anxious",          // raw classifier label
  "escalation_level": 2,            // SG escalation level AT the bark; null outside SG
  "sg_state": "intervention"        // SG FSM state at the bark; null outside SG
}
```

- `event.confidence`, `event.dog_id`, `event.session_id`, `event.ts` already
  carry the rest of the app's required columns.
- Note the live bus/relay `bark` events already carry `bark_type`/`bark_label`
  as of 2026-09-01 (commit 8068ef3); this makes the *stored* row match.
- **"Did an intervention/treat follow?"** — resolved by query, not stored:
  interventions become `event_type='sg_intervention'` rows (below) in the same
  session timeline, and treats are already `dispense_log` + `treat_dispensed`
  events. A documented export query joins bark → next intervention/dispense
  within the session. Rationale: `event` rows stay immutable (a bark can't know
  its future at write time), and the join is cheap with `idx_event_session`.
  If App Claude needs it precomputed, the export layer (not the row) adds a
  `followed_by` field.

### R2 — Sessions first-class

The spec `session` table already exists and SG/coach write it. Additions:

- `session.mode` vocabulary gains explicit `'sg'` and `'coach'` (today they map
  onto 'monitor'/'training'; the export layer needs the real mode).
- New nullable `session.outcome_json` column: at session end the owning mode
  writes the same numbers `build_summary_payload()` computes today (total barks,
  bark-type counts, interventions, treats, successful quiets, max escalation,
  panic episodes, trend). The sg_summary *event* keeps flowing to the app
  unchanged; the column makes it a queryable row. Weekly Summary then reads
  sessions + events instead of the legacy tables — which is what makes its
  accuracy validatable.
- New `event_type='sg_intervention'` (spec §5): payload `{escalation_level,
  barks_triggering, quiet_achieved, quiet_duration_sec, treat_given,
  dispense_id, music_played}`. Replaces `sg_interventions` writes.
- `silent_guardian_sessions` / `sg_interventions` / `coaching_sessions` /
  treatbot `rewards` become read-only legacy after cutover (kept, not migrated
  — no-backfill rule).

### R3 — IDs and timestamps

- **Storage stays epoch-ms UTC INTEGER** (spec §4 rule, unchanged). The app's
  "ISO8601 UTC everywhere" requirement is satisfied at every boundary: exports,
  REST responses, and events serialize `ts` as ISO8601 UTC (`...Z`), exactly as
  the live event contract already does. Storage format ≠ wire format; this doc
  makes that explicit so review can confirm it's acceptable.
- **Canonical dog identity = the app/cloud UUID** (the one the relay accepts).
  Additive `dog.app_dog_id TEXT` column + unique index in wimz.db; the profile
  sync path (`dog_profile_manager`) populates it. All exports and analytics key
  on `app_dog_id`; the wimz-local `dog_id` PK stays for FK integrity (no PK
  rewrite, no FK churn). Unattributed rows stay `dog_id=NULL` (the existing
  `_TAG_RE` guard already keeps `dog_N` junk out of identity).
- `dog_N` slot ids and `aruco_*` tags never enter analytics as identity — they
  remain attribution *inputs* resolved to the canonical UUID or NULL.

### R4 — Live event contract: untouched

No change to any WS/relay payload shape, event name, or endpoint. (The only
recent wire change — the per-bark `bark_type` stamp — was the app's own ask and
is already shipped.)

### R5 — No backfill

Nothing migrates from treatbot/dogbot/missions rows. Legacy DBs stay on disk
read-only until Morgan approves archival (cleanup protocol: list, approve,
archive — never silent deletion).

---

## 3. Sequencing (each phase independently shippable)

**Phase 1 — hygiene + stop the bleeding (no schema change):**
square the version drift (spec DDL insert → current, schema.py → 0.5, docstrings);
persist the SG per-bark rows by enriching `_log_bark_event`'s wimz payload with
`bark_type`/`bark_label`/`escalation_level`/`sg_state`; fix the
`end_silent_guardian_session` call sites that drop `quiet_periods`/`dog_bark_counts`;
wire telemetry pruning (call `cleanup_old_data()` from the same maintenance loop
that calls `cleanup_old_events()`, or cap telemetry by row count).

**Phase 2 — spec bump v0.6 (additive):** `session.outcome_json`, `dog.app_dog_id`
(+ index), `mode` vocabulary, `sg_intervention` event type, bark payload contract
(§2/R1). Mirror into `core/data/schema.py` with a `"0.5"` migration entry.

**Phase 3 — cutover:** SG, coach, and Weekly Summary write/read wimz.db only;
legacy behavioral writers (`bark_store.log_bark`, `log_dog_event`,
`silent_guardian_sessions`/`sg_interventions`/`coaching_sessions` inserts) are
retired. Telemetry stays in treatbot.db (it's operational, not behavioral).

**Phase 4 — decommission (needs explicit approval):** archive `dogbot.db` +
`core/dog_database.py` and the `missions/__init__.py` DB layer per cleanup
protocol.

Validation gate per phase: run SG + coach live, confirm sg_summary /
weekly report numbers match a hand query of wimz.db before retiring the
legacy source it replaced.

---

## 4. Open questions for review

1. **App Claude:** is "followed_by intervention/treat" as a *documented export
   query / export-layer field* (not a stored column) acceptable? (§2/R1)
2. **App Claude:** confirm ISO8601-at-the-boundary over epoch-ms storage
   satisfies requirement 3. (§2/R3)
3. **Morgan:** telemetry retention — prune to 30 days (~70 MB reclaimed), or
   keep longer for the freeze RCA? (Power-watch CSV is separate and unaffected.)
4. **Morgan:** Phase 4 archival approval can wait; flagging now so it's not a
   surprise.
5. **App Claude:** does the app ever need `bark_timeline` buckets server-side
   from *storage* (i.e., a query), or only via the live `sg_summary` payload?
   (Determines whether the export layer ships a canned timeline query in v1.)

---

*Drafted by robot-side Claude, 2026-09-01, from a full storage audit
(4 DBs, all writers/readers, retention paths). Review → approve → Phase 1.*
