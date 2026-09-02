# WIM-Z Data Refactor — Design (v1 draft, 2026-09-01)

**Status:** APPROVED by App Claude 2026-09-01 ("approved with notes", all
five requirements met; see Review Outcome at bottom). Phase 1 implemented
2026-09-01. **Phase 2 implemented 2026-09-01** (spec v0.6, live-verified on
tb5: DB migrated, `outcome_json` + `app_dog_id` writes confirmed, Elsa's
dual-UUID reconciled via profile sync). **Phase 3 implemented 2026-09-01**
(see Phase 3 notes below). **Phase 4 implemented 2026-09-01 (Morgan's
go: "finish phase 4 and call all open work closed") — THE REFACTOR IS
COMPLETE.** Phase 4 notes:
- Endpoint migration: /events/dog, /events/dog/summary, /events/dog/latest,
  and /sg/sessions/recent now read wimz.db (core/data/legacy_views.py),
  wire shapes preserved except: dog_events timestamps are ISO8601 Z (were
  local-naive), dog ids are canonical app UUIDs, and the SG catch-up 'id'
  is the session UUID string (was a legacy int) with 'interventions_count'
  carrying the count. Live-verified.
- Writers retired: coaching_sessions, dog_events, sg_interventions. The
  spec store rows are the single record for each.
- RETAINED by decision (operational, not analytical): rewards + missions
  tables and their writers (7 live reward-path call sites; wimz carries
  the same data natively + backfilled), telemetry (30d-pruned), and the
  silent_guardian_sessions writer — a wire-compat shim carrying the int
  session_id in the sg_summary payload and the L4 summary_sent guard;
  retire it when the app drops int session ids.
- Archived: data/dogbot.db + data/missions.db -> data/archive/ (sources
  also in the fleet tarball); backfill script guards against the moved
  files. core/dog_database.py and missions/ stay in-tree, dormant, no
  importers.
Phase 3 notes — scope adjusted from the original sketch after mapping REST
readers: Weekly Summary (all report/trends/progress/compare paths) now reads
wimz.db for barks, SG sessions (outcome_json), interventions, coaching, and
dog identity — live-verified via GET /reports/weekly and /reports/trends;
adds by_bark_type; fixes the UTC/local skew. The bark pipeline is
single-writer: bark_detector now writes wimz bark rows in every non-SG mode
(ambient session, gate:'detector'), SG keeps writing its FSM-context rows,
and `bark_store.log_bark` is retired (the legacy `barks` table has zero
writers and zero readers). SG's outcome payload gained
successful_quiets/max_escalation_level (additive). KEPT for Phase 4:
`rewards` + `missions` reads/writes (still consistent, no skew), and the
`coaching_sessions` / `dog_events` / `silent_guardian_sessions` writers —
live REST endpoints (/dog_events*, dog stats) and the L4 summary_sent guard
still read them, so they retire in Phase 4 together with those endpoint
migrations.
Phase 2 notes: coach sessions write no `outcome_json` — their
`training_attempt` rows are already first-class, so a coach summary is a
query, not a stored blob. SG's wimz session now also rolls over at the 8-hour
reset (previously it silently spanned resets), each closing with its
`sg_summary` outcome.
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

---

## Review outcome (App Claude, 2026-09-01) — and dispositions

- **Q1 ACCEPTED** with two conditions, both met: the `followed_by` join window
  is now DEFINED IN SPEC §5 (same session_id, 120 s, else null), and the export
  layer will emit `followed_by` as a field on exported bark rows — no consumer
  re-implements the join.
- **Q2 ACCEPTED** with one hard condition, recorded in spec changelog 0.5.1:
  every serialized timestamp carries `Z`/explicit offset, never naive-local.
  The serializer unit test ships with the export layer (Phase 2/3).
- **Q5 DEFERRED** per review: no canned bark_timeline storage query in v1;
  ship it with the historical-charts app slice, bucket size as a query
  parameter (not sg_summary's hardcoded offset_min).
- **Keep `emotion` alongside `bark_label`** during transition (app history
  renderer keys on it) — done; noted in spec §5.
- Q3 input on record: telemetry ≥ freeze-RCA window on tb2 until RCA closes,
  30 days elsewhere. Phase 1 shipped a 30-day prune fleet-wide (power_watch.csv
  untouched; 30 d comfortably covers post-freeze look-back). Morgan can widen
  it if the RCA wants deeper history.
- App-side follow-up (theirs): label live/history bark rows with `bark_type` —
  queued app slice, no robot action.

**Phase 1 shipped (2026-09-01):** per-bark wimz rows now carry
bark_type/bark_label/escalation_level/sg_state; quiet_periods_json /
dog_bark_counts_json call-site bug fixed (SG now tracks and passes both);
version drift squared (spec 0.5.1 = schema.py = DB stamp via stepwise
migrations); telemetry/events 30-day prune wired into the hourly
maintenance loop.

---

## Amendment A — Backfill (R5 REVERSED by Morgan, 2026-09-01 evening)

**Status: DESIGN FOR REVIEW — no code until App Claude + Morgan approve
this amendment.** Morgan overturned the no-backfill rule: legacy rows ARE
retrofitted into wimz.db so per-dog stats and weekly summaries cover real
history.

**What is actually recoverable (counted 2026-09-01):**

| Source | Rows | Lands as |
|---|---|---|
| treatbot `rewards` | 1,176 (5 are bark_reward-contaminated) | `event` treat rows + `dispense_log` |
| treatbot `coaching_sessions` | 91 | `training_attempt` (+ synthetic per-day coach `session` rows) |
| treatbot `silent_guardian_sessions` | 11 | `session` (mode='sg') with `outcome_json` built from the row |
| treatbot `sg_interventions` | 203 | `sg_intervention` events (phase='outcome') |
| dogbot `behavior_events` | 2,540 | `pose` events (partially done by the old STORE-A backfill — ledger says 380; resume from ledger) |
| missions.db `detections` | 341 | `pose` events (STORE-A ledger says done — verify, skip) |
| treatbot `barks` | **0 — nothing to recover** | — |

**Hard truth to set expectations:** per-bark history is GONE — the 24h
prune already deleted every legacy bark row. Bark history exists only as
session-level totals inside `silent_guardian_sessions` (11 sessions), which
backfill preserves via `outcome_json`. Per-bark rows exist from 2026-09-01
(Phase 1) forward only. The app's per-bark charts will honestly start there.

**Design answers to the five app requirements:**

1. **Provenance** — spec v0.7 (additive): new nullable `origin TEXT` column
   on `event`, `training_attempt`, `dispense_log`, `session`. NULL = native;
   backfilled rows carry `'backfill:<source_table>'`
   (e.g. `backfill:rewards`). Queryable, indexable, survives export.
2. **Timestamp normalization** — per-source conversion at migration time,
   never copied raw: unix-float REAL → epoch-ms; SQLite `CURRENT_TIMESTAMP`
   UTC text → epoch-ms; local-naive ISO (`dog_events`-style) → interpreted
   in the robot's local tz → epoch-ms UTC. Unit test per source format.
3. **Identity mapping** — resolution ladder per row: canonical app UUID
   (matches `dog.app_dog_id`, incl. the get_or_create_dog resolution shipped
   2026-09-01) → `aruco_*` tag via `qr_code_id` → case-insensitive
   name-match against profiles → else `dog_id = NULL`, never a guess.
   `dog_N` slot ids always resolve to NULL.
4. **bark_reward contamination — fully identifiable, better than feared:**
   the removed lottery logged `rewards.behavior = 'bark_<emotion>'`.
   Exactly 5 rows / 5 treats. They migrate with
   `origin='backfill:rewards:bark_reward'` and are EXCLUDED from treat
   totals in owner-facing summaries. No "counts are inflated" caveat needed.
5. **Coverage honesty** — weekly/dog summaries gain a `coverage` field:
   `{"treats_since": <iso>, "coaching_since": <iso>, "per_bark_since":
   "2026-09-01T00:00:00Z"}` so the app can caption how far back each
   number really goes.

**Mechanics:** extend `scripts/backfill_wimz.py` (STORE-A) — same
idempotency ledger in `schema_meta` (`backfill:<table>` keys), same
run-once semantics, resumable. Runs offline/idle only. Spec bump v0.7
(origin column + coverage field definition) lands with the amendment
approval, before the script.

**APPROVED by App Claude 2026-09-01 evening, with four binding flags
(all implemented):**

1. **Coverage gates comparisons AND phrasing** — `change_percent`/`trend`
   go null and the headline drops its comparison clause whenever
   `previous_week` predates that metric's coverage (the app renders "—"
   for null change_percent).
2. **No invented per-dog bark splits** — legacy SG bark totals are
   household-level (`dog_bark_counts_json` was always `'{}'`); backfill
   never fabricates per-dog bark rows. Per-dog bark coverage is computed
   from the data actually present. (Correction found while implementing:
   the August STORE-A backfill had already migrated 1,434 per-bark rows
   before the prune destroyed the source, most with single-dog
   attribution — so some honest per-bark history predates 2026-09-01;
   bark-TYPE labels still start 2026-09-01.)
3. **Storage-only** — the backfill writes SQL rows only. It never
   publishes bus events, never touches the relay, never triggers pushes.
   `scripts/backfill_wimz.py` imports no bus/relay modules by design.
4. **`coverage` ships in the dog_weekly_summary payload** (not just
   exports): `{per_bark_since, bark_type_since, treats_since,
   coaching_since}` — absent/null fields mean no caption, per the app's
   lenient rendering.

*Implemented 2026-09-01 evening: spec v0.7 (origin column), backfill
sources rewards/coaching_sessions/sg_sessions/sg_interventions + STORE-A
retro-tagging, coverage + gating in the per-dog summary.*

---

*Drafted by robot-side Claude, 2026-09-01, from a full storage audit
(4 DBs, all writers/readers, retention paths).*
