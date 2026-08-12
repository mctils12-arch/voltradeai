/**
 * feedDeadAir.ts — THROUGHPUT liveness for the continuously-ingested
 * position feeds (aircraft, vessels, trains).
 *
 * WHY THIS EXISTS (2026-08-12, from the human-supplied AIS dead-air
 * runbook). The AIS vessel feed went silent at 2026-08-05 ~13:31 UTC and
 * nobody noticed for SEVEN DAYS. Not because monitoring was missing —
 * because monitoring watched the wrong variable. The reconnect loop
 * SUCCEEDED: every 60s the dial worked, the socket opened, the
 * subscription was accepted, and zero frames arrived. Connection state
 * read "healthy" the whole time. ~6.6 days of global ship positions were
 * lost to a green light, and an archive gap never refills (CLAUDE.md
 * Priority 1: "a dead system learns nothing, and an archive gap never
 * refills").
 *
 * The runbook's conclusion, quoted: "Uptime is the wrong signal here;
 * throughput is the right one... This applies to every feed we ingest,
 * not just AIS — the aircraft feed has the same blind spot today."
 *
 * ROOT-CAUSE-AGNOSTIC BY CONSTRUCTION. This module makes NO claim about
 * why a feed went quiet (provider outage, credential, subscription
 * payload, our own bug). It only refuses to call silence "healthy". That
 * is deliberate: CLAUDE.md's RECURRENCE ESCALATES rule forbids a third
 * speculative patch to a twice-"fixed" subsystem, and a detector that
 * asserts nothing about causation is not such a patch.
 *
 * WHY IT READS THE ARCHIVE AND NOT AN IN-PROCESS COUNTER. Every previous
 * liveness attempt on this feed was defeated by a clock reset:
 *   - fix #1 (2026-08-06) tracked the last frame — but a socket that
 *     never received a FIRST frame has no timestamp, so it never aged.
 *   - fix #2 (2026-08-11) added the connect time as the fallback clock —
 *     but the watchdog redials every ~3 min and each redial RESET that
 *     clock, so a feed that never delivered a single frame still read
 *     "live" for 179 of every 180 seconds.
 * Both reset on deploy as well. The archive on disk is the one clock that
 * cannot be reset by a redial, a restart, or a deploy: it is the record of
 * data that actually landed. That is also the exact thing whose loss is
 * permanent, so it is the honest thing to alarm on.
 *
 * PURE CORE (node:test safe): the verdict functions below take observations
 * and a clock. The single fs-touching helper is isolated at the bottom.
 */

import fs from "fs";
import path from "path";
import { archiveBaseDir } from "./datacoreArchive";

/** The continuously-ingested position feeds — the ones for which silence
 *  is unambiguously a fault. aircraft: globalScopes (120s) + trackedPlanes
 *  (30s) + the 10-minute archiveTick. trains: the same 10-minute
 *  archiveTick. vessels: the aisstream websocket firehose. All three write
 *  every hour they are alive, with no dependence on visitor traffic.
 *
 *  DELIBERATELY NOT COVERED: the episodic archive kinds (EDGAR filings,
 *  FIRMS fires, 8-K earnings, macro releases...). Those have legitimately
 *  bursty natural cadences — no filings on a weekend is not a fault — and
 *  alarming on them without per-source cadence models would manufacture
 *  false alarms, which is how an alarm becomes ignorable. Extending this
 *  to episodic sources needs a per-source expected-cadence table; filed
 *  rather than guessed. */
export type FeedKind = "aircraft" | "vessels" | "trains";
export const CONTINUOUS_FEEDS: FeedKind[] = ["aircraft", "vessels", "trains"];

/** Hours of silence before a continuous feed is declared dead.
 *
 *  Chosen honestly against the measurement's own granularity: the archive
 *  buckets rows into hour files, so a perfectly healthy feed reads between
 *  0 and ~1h of "silence" depending on where in the hour it is sampled. 3h
 *  means at least two whole hour-buckets went unwritten — unambiguous, and
 *  survives deploy gaps and brief upstream hiccups without crying wolf.
 *  Against the Aug-5 outage this fires the same morning instead of on day
 *  seven. */
export const FEED_DEAD_AIR_HOURS = 3;

export interface FeedObservation {
  kind: FeedKind;
  /** Newest raw hour file in this kind's archive dir, or null if the dir is
   *  empty/absent. Filenames are `YYYY-MM-DD-HH.jsonl[.gz]`. */
  newestHourFile: string | null;
}

export interface FeedVerdict {
  kind: FeedKind;
  /** Hours since the START of the newest archived hour bucket, or null when
   *  nothing was ever archived. Deliberately named for what it measures: it
   *  over-states by up to 1h for a live feed (hour-bucket granularity),
   *  which is why the threshold is 3h and not 1h. */
  silentHours: number | null;
  dead: boolean;
  detail: string;
}

const HOUR_FILE_RE = /(\d{4})-(\d{2})-(\d{2})-(\d{2})\.jsonl(\.gz)?$/;

/** `2026-08-05-13.jsonl.gz` -> epoch ms of 2026-08-05T13:00:00Z. Null for
 *  any name that is not an hour file — the kind directories also acquire
 *  unrelated files over time, and a name we cannot parse must never be
 *  mistaken for fresh data. UTC end to end, no local-time parsing. */
export function parseArchiveHourMs(fileName: string | null | undefined): number | null {
  if (!fileName) return null;
  const m = HOUR_FILE_RE.exec(fileName);
  if (!m) return null;
  const ms = Date.parse(`${m[1]}-${m[2]}-${m[3]}T${m[4]}:00:00Z`);
  return Number.isFinite(ms) ? ms : null;
}

/** The verdict, pure. */
export function feedDeadAir(
  observations: FeedObservation[],
  nowMs: number,
  thresholdHours: number = FEED_DEAD_AIR_HOURS,
): FeedVerdict[] {
  return observations.map((o) => {
    const hourMs = parseArchiveHourMs(o.newestHourFile);
    if (hourMs === null) {
      // No archived hour at all. The archive's own stance (archiveStats:
      // "a missing position archive must be loud, not absent") applies —
      // a continuous feed that has never written is a fault, not a blank.
      // This is also the state the vessels dir REACHES on its own: raw
      // hours roll up and get deleted after ~7 days, so the frozen
      // 2026-08-05-13 file disappears and staleness would otherwise become
      // invisible exactly when the outage got worse.
      return {
        kind: o.kind,
        silentHours: null,
        dead: true,
        detail: `${o.kind}: no archived hour files at all — a continuously-ingested feed has written nothing`,
      };
    }
    const silentHours = (nowMs - hourMs) / 3_600_000;
    const dead = silentHours > thresholdHours;
    return {
      kind: o.kind,
      silentHours: +silentHours.toFixed(2),
      dead,
      detail: dead
        ? `${o.kind}: DEAD AIR — newest archived hour is ${o.newestHourFile} ` +
          `(${silentHours.toFixed(1)}h ago, threshold ${thresholdHours}h). The connection may look ` +
          `healthy; no data has landed. Cause unknown from this signal alone.`
        : "",
    };
  });
}

export interface FeedDeadAirCheck {
  status: "ok" | "degraded";
  /** kinds in dead air, for the at-a-glance line in a health payload */
  dead: FeedKind[];
  feeds: Record<string, { silent_hours: number | null; dead: boolean }>;
  detail: string;
}

/** Shapes the verdicts into an /api/health check block. Mirrors the
 *  LIVENESS ALARM convention (server/liveness.ts): one `status`, a compact
 *  machine-readable body, and a single human-readable `detail` that a DAILY
 *  session can put at the top of a report without further formatting. */
export function feedDeadAirCheck(verdicts: FeedVerdict[]): FeedDeadAirCheck {
  const dead = verdicts.filter((v) => v.dead);
  const feeds: Record<string, { silent_hours: number | null; dead: boolean }> = {};
  for (const v of verdicts) feeds[v.kind] = { silent_hours: v.silentHours, dead: v.dead };
  return {
    status: dead.length ? "degraded" : "ok",
    dead: dead.map((v) => v.kind),
    feeds,
    detail: dead.length
      ? `FEED DEAD-AIR ALARM: ${dead.map((v) => v.kind).join(", ")} ingesting nothing. ` +
        dead.map((v) => v.detail).join(" ")
      : "",
  };
}

// ── the one impure helper ────────────────────────────────────────────────────

/** Newest raw hour file for a kind, by UTC hour (not by mtime, and not by
 *  raw lexicographic order — both can be wrong: a compressed `.jsonl.gz`
 *  is rewritten long after its hour, and non-hour files sort unpredictably).
 *  readdir only, no statSync per file: /api/health is unauthenticated and
 *  frequently polled, so this stays three cheap directory reads. */
export function newestArchiveHourFile(kind: FeedKind, baseDir?: string): string | null {
  const dir = path.join(baseDir || archiveBaseDir(), kind);
  let best: string | null = null;
  let bestMs = -Infinity;
  try {
    for (const f of fs.readdirSync(dir)) {
      const ms = parseArchiveHourMs(f);
      if (ms !== null && ms > bestMs) { bestMs = ms; best = f; }
    }
  } catch {
    return null; // dir absent === nothing archived, handled as dead upstream
  }
  return best;
}

/** Collect + judge in one call, for the health handler. */
export function observeFeedDeadAir(
  nowMs: number, baseDir?: string, thresholdHours: number = FEED_DEAD_AIR_HOURS,
): FeedDeadAirCheck {
  const obs = CONTINUOUS_FEEDS.map((kind) => ({
    kind, newestHourFile: newestArchiveHourFile(kind, baseDir),
  }));
  return feedDeadAirCheck(feedDeadAir(obs, nowMs, thresholdHours));
}
