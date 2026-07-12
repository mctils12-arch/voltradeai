// EVENT-LOOP LAG MONITOR (REPAIR 2026-07-12, KNOWN BROKEN #18 follow-up)
//
// research/open_questions.md item #18: TIER2-ERROR "daemon run_full_scan
// failed: Daemon timeout" has recurred on a roughly-30-minute cadence
// across two separate days (2026-07-11: 14:48/15:18/15:48Z; 2026-07-12:
// 14:43:42/15:13:44Z). The v1.0.277 envelope-unwrap fix finally captured a
// real active_dispatches reading at the moment of both 2026-07-12
// occurrences — 1 both times, well under the MAX_INFLIGHT_REQUESTS=8 cap —
// which REFUTES the zombie-thread-pileup theory item #18 had been tracking.
//
// A NEW correlation surfaced instead: both 2026-07-12 TIER2-ERROR entries
// landed within 17-35ms of a STREAM-DISCONNECT audit entry
// (14:43:42.554/14:43:42.537 and 15:13:44.239/15:13:44.274). The daemon
// timeout is a Node-side setTimeout(300000) armed when the RPC call
// starts; the WS disconnect fires when Node processes a 'close' event on
// the Alpaca stream socket. These are two independently-scheduled events
// (a ~300s RPC timeout with a scan-dependent start time vs. a ~600s WS
// reconnect cycle observed to be regular to within a few seconds) — for
// both to land within tens of milliseconds of each other, on both
// occurrences observed, is closer than coincidence plausibly explains.
//
// One shared cause fits both symptoms: if Node's event loop stalls
// (synchronous work — GC pause, a large JSON.parse/stringify, etc. —
// blocking the loop for a meaningful stretch), an already-elapsed
// setTimeout and an already-arrived-but-unprocessed socket 'close' event
// both stay queued until the loop resumes, then fire back-to-back. This
// module makes that hypothesis checkable on the next occurrence instead
// of guessed at: it measures how late a scheduled tick actually fires
// (the standard event-loop-lag technique) and audits any lag that would
// be large enough to plausibly explain the observed near-simultaneity.
// Pure measurement — no trading, sizing, or scheduling behavior changes.

export const EVENTLOOP_LAG_CHECK_MS = 2000;
export const EVENTLOOP_LAG_ALERT_THRESHOLD_MS = 500;

// A setInterval(fn, intervalMs) callback that fires after
// (intervalMs + lagMs) wall-clock ms has drifted from schedule by lagMs —
// near-zero under normal load, large when something blocked the loop.
// Clamped at 0: a tick firing early (should not happen, but clock drift or
// test doubles could produce a negative delta) is not "negative lag".
export function computeLagMs(intervalMs: number, actualElapsedMs: number): number {
  return Math.max(0, actualElapsedMs - intervalMs);
}

export function lagExceedsThreshold(
  lagMs: number,
  thresholdMs: number = EVENTLOOP_LAG_ALERT_THRESHOLD_MS,
): boolean {
  return lagMs >= thresholdMs;
}
