#!/usr/bin/env python3
"""Evidence pack generator: canned analysis exports from wimz_fleet.db.

Writes CSVs + the exact SQL used (queries.sql) + README with the standard
filters, into fleet_archive/evidence_pack/. Re-run any time the archive is
rebuilt — outputs are fully derived.

Standard filters (documented in every output):
- origin NOT LIKE '%:log'      log-mined rows overlap DB rows for the same
                               period; excluded from counts by default
- origin NOT LIKE '%bark_reward%'  treats from the removed bark-lottery are
                               excluded from owner/evidence treat totals
"""
import csv
import json
import sqlite3
import sys
from pathlib import Path

ARCHIVE = Path(sys.argv[1] if len(sys.argv) > 1
               else '/home/morgan/wimz/fleet_archive/wimz_fleet.db')
OUT = ARCHIVE.parent / 'evidence_pack'
OUT.mkdir(exist_ok=True)

NO_LOG = "e.origin NOT LIKE '%:log'"
NO_LOTTERY = "e.origin NOT LIKE '%bark_reward%'"

QUERIES = {
 'weekly_barks_per_dog': f'''
    SELECT strftime('%Y-%W', e.ts/1000, 'unixepoch') AS week,
           COALESCE(dg.name, 'unattributed') AS dog,
           d.hardware_rev AS robot, COUNT(*) AS barks,
           ROUND(AVG(json_extract(e.payload,'$.db')), 1) AS avg_loudness_db
    FROM event e
    JOIN device d ON e.device_id = d.device_id
    LEFT JOIN dog dg ON e.dog_id = dg.dog_id
    WHERE e.event_type='bark'
      AND COALESCE(json_extract(e.payload,'$.class'),'bark') != 'notbark'
      AND {NO_LOG}
    GROUP BY week, dog, robot ORDER BY week, dog''',

 'weekly_bark_types': f'''
    SELECT strftime('%Y-%W', e.ts/1000, 'unixepoch') AS week,
           COALESCE(json_extract(e.payload,'$.bark_type'),
                    'untyped_pre_2026-09') AS bark_type,
           COUNT(*) AS barks
    FROM event e
    WHERE e.event_type='bark'
      AND COALESCE(json_extract(e.payload,'$.class'),'bark') != 'notbark'
      AND {NO_LOG}
    GROUP BY week, bark_type ORDER BY week''',

 'weekly_treats_per_robot': f'''
    SELECT strftime('%Y-%W', e.ts/1000, 'unixepoch') AS week,
           d.hardware_rev AS robot,
           COUNT(*) AS dispenses,
           SUM(COALESCE(json_extract(e.payload,'$.treats_dispensed'), 1)) AS treats
    FROM event e JOIN device d ON e.device_id = d.device_id
    WHERE e.event_type='treat_dispensed' AND {NO_LOG} AND {NO_LOTTERY}
    GROUP BY week, robot ORDER BY week, robot''',

 'sg_sessions_over_time': '''
    SELECT date(s.started_at/1000, 'unixepoch') AS date,
           d.hardware_rev AS robot,
           ROUND((COALESCE(s.ended_at, s.started_at) - s.started_at)
                 / 3600000.0, 2) AS hours,
           json_extract(s.outcome_json,'$.total_barks') AS barks,
           json_extract(s.outcome_json,'$.interventions_triggered') AS interventions,
           json_extract(s.outcome_json,'$.successful_quiets') AS quiets,
           json_extract(s.outcome_json,'$.treats_dispensed') AS treats,
           json_extract(s.outcome_json,'$.max_escalation_level') AS max_escalation,
           json_extract(s.outcome_json,'$.panic_episodes') AS panic_episodes,
           s.origin
    FROM session s JOIN device d ON s.device_id = d.device_id
    WHERE s.mode='sg' AND s.outcome_json IS NOT NULL
      -- the same real session can exist twice (legacy-backfill row + native
      -- v0.6 row): keep one per (robot, start-second), preferring native
      AND NOT EXISTS (
        SELECT 1 FROM session s2
        WHERE s2.mode='sg' AND s2.outcome_json IS NOT NULL
          AND s2.device_id = s.device_id
          AND s2.started_at/1000 = s.started_at/1000
          AND s2.session_id != s.session_id
          AND s.origin LIKE '%silent_guardian_sessions'
          AND s2.origin NOT LIKE '%silent_guardian_sessions')
    ORDER BY s.started_at''',

 'sg_intervention_outcomes_weekly': f'''
    SELECT strftime('%Y-%W', e.ts/1000, 'unixepoch') AS week,
           d.hardware_rev AS robot,
           COUNT(*) AS outcomes,
           SUM(CASE WHEN json_extract(e.payload,'$.quiet_achieved')
               THEN 1 ELSE 0 END) AS quiet_achieved,
           ROUND(100.0 * SUM(CASE WHEN json_extract(e.payload,'$.quiet_achieved')
               THEN 1 ELSE 0 END) / COUNT(*), 1) AS success_pct
    FROM event e JOIN device d ON e.device_id = d.device_id
    WHERE e.event_type='sg_intervention'
      AND json_extract(e.payload,'$.phase') = 'outcome' AND {NO_LOG}
    GROUP BY week, robot ORDER BY week''',

 'coaching_progress': '''
    SELECT strftime('%Y-%W', ta.cue_ts/1000, 'unixepoch') AS week,
           COALESCE(dg.name, 'unattributed') AS dog,
           ta.trick_label AS trick,
           COUNT(*) AS attempts,
           SUM(CASE WHEN ta.success >= 1 THEN 1 ELSE 0 END) AS completed,
           ROUND(100.0 * SUM(CASE WHEN ta.success >= 1 THEN 1 ELSE 0 END)
                 / COUNT(*), 1) AS success_pct,
           ROUND(AVG(ta.latency_ms) / 1000.0, 1) AS avg_response_sec
    FROM training_attempt ta
    JOIN session s ON ta.session_id = s.session_id
    LEFT JOIN dog dg ON ta.dog_id = dg.dog_id
    -- coach sessions only: SG 'quiet' attempts live in sg/monitor sessions
    -- and are covered by sg_intervention_outcomes_weekly
    WHERE s.mode IN ('coach','training') AND ta.trick_label != 'quiet'
    GROUP BY week, dog, trick ORDER BY week, dog, trick''',

 'mode_usage': f'''
    SELECT date(e.ts/1000, 'unixepoch') AS date,
           d.hardware_rev AS robot,
           json_extract(e.payload,'$.to') AS entered_mode,
           COUNT(*) AS times
    FROM event e JOIN device d ON e.device_id = d.device_id
    WHERE e.event_type='mode_change'
    GROUP BY date, robot, entered_mode ORDER BY date''',

 'monthly_rollup': f'''
    SELECT strftime('%Y-%m', e.ts/1000, 'unixepoch') AS month,
           d.hardware_rev AS robot,
           SUM(e.event_type='bark') AS barks,
           SUM(CASE WHEN e.event_type='treat_dispensed' AND {NO_LOTTERY}
               THEN 1 ELSE 0 END) AS treats,
           SUM(e.event_type='sg_intervention') AS sg_interventions,
           SUM(e.event_type IN ('pose','detection')) AS vision_events,
           COUNT(*) AS total_events
    FROM event e JOIN device d ON e.device_id = d.device_id
    WHERE {NO_LOG}
    GROUP BY month, robot ORDER BY month, robot''',

 'data_provenance': '''
    SELECT COALESCE(e.origin, 'NATIVE') AS origin, e.event_type,
           COUNT(*) AS rows,
           date(MIN(e.ts)/1000, 'unixepoch') AS from_date,
           date(MAX(e.ts)/1000, 'unixepoch') AS to_date
    FROM event e GROUP BY origin, event_type ORDER BY rows DESC''',
}

def main():
    db = sqlite3.connect(f'file:{ARCHIVE}?mode=ro', uri=True)
    manifest = {}
    sql_lines = ['-- Evidence pack queries (source: wimz_fleet.db)',
                 '-- Standard filters: see README.md', '']
    for name, q in QUERIES.items():
        cur = db.execute(q)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
        with open(OUT / f'{name}.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        manifest[name] = len(rows)
        sql_lines += [f'-- {name}.csv', q.strip() + ';', '']
        print(f'  {name}.csv: {len(rows)} rows')
    (OUT / 'queries.sql').write_text('\n'.join(sql_lines))
    (OUT / 'README.md').write_text(f'''# WIM-Z Fleet Evidence Pack

Generated from wimz_fleet.db ({ARCHIVE}). Regenerate any time with:
`env_new/bin/python scripts/fleet_evidence_pack.py [path/to/wimz_fleet.db]`

## Standard filters (already applied where noted)
- **Log-mined rows excluded from counts** (`origin NOT LIKE '%:log'`):
  log lines overlap database rows for the same period; including both
  double-counts. Log rows remain in the archive for gap analysis.
- **Bark-lottery treats excluded** (`origin NOT LIKE '%bark_reward%'` and
  payload behavior `bark_%`): treats dispensed by the removed ambient
  bark-reward path are not honest training rewards.
- **Clone dedupe is upstream**: the ingest already removed the shared
  cloned-SD-image prefix; each robot's rows are genuinely its own
  (canonical lineage = tb5).

## Reading the files
{json.dumps(manifest, indent=2)}

## Honesty notes
- Per-dog bark attribution starts 2026-09-01 (older bark rows are
  household-level, dog = "unattributed").
- Bark TYPE labels start 2026-09-01 (older rows = "untyped_pre_2026-09").
- The 73MB source attributed to tb3 lacks an internal device marker;
  its rows are tagged backfill:tb3:* and re-attributable in one UPDATE.
- data_provenance.csv is the full audit trail: every row family, its
  origin tag, row count, and date span.
''')
    print(f'\\nPack written to {OUT}')

if __name__ == '__main__':
    main()
