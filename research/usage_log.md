# Usage Calibration Log — plan quota vs. routine schedule

Purpose: close the loop between Claude plan usage and the routine
schedule. The human pastes their Plan usage panel screenshot into any
session on this repo; the session reads the 5-hour and weekly
percentages from the image, appends a row here with context, and applies
the calibration rule below (standing behavior note in CLAUDE.md KNOWN
STATE, human-approved 2026-07-03).

## Calibration rule
- 2+ consecutive weekly readings under 50% AND queue depth nonzero →
  recommend specific routine slots to ADD from the expansion menu below.
- Readings approaching 90%+ → recommend slots to DROP per the drop order.

## Schedule reference (A5 design, delivered 2026-07-03)
Full 8-run/day menu (all ET): daily-am 7:00 · product-am 9:00 ·
daily-midday 12:30 (hold merges to close) · product-pm 14:00 (hold
merges) · daily-close 16:15 · edge 17:30 · product-eve 20:00 ·
edge-late 22:30. Plus voltrade-weekly-review Sun 10:00 (C1 briefing +
Gmail draft; not a build slot).

- DROP ORDER under quota pressure: product-pm → edge-late → product-eve.
  Irreducible core (4/day): daily-am, product-am, daily-close, edge.
- ADD ORDER with headroom: reverse of the drop order — fill from the
  currently active subset toward the full 8-run table.
- STARVED valve (armed since #98): 10+ consecutive STARVED sessions
  auto-flags Agent-SDK continuous operation in wishlist.md.

## voltrade-weekly-review routine (Sun 10:00 ET) — canonical prompt text

> [WEEKLY-REVIEW] Read CLAUDE.md fully, then research/experiments.md,
> research/open_questions.md, research/wishlist.md,
> research/usage_log.md. Report only — no code changes, no PRs.
>
> 1. Produce the weekly briefing: (1) session tags of the last 10
> experiments.md entries and the health verdict per HEALTH OF THE LOOP;
> (2) everything in wishlist.md awaiting the human's decision, each with
> a one-line cost/benefit; (3) paper performance vs SPY since last
> review; (4) live-vs-backtest divergence if measurable; (5) the single
> most important thing the human should know this week; (6) usage
> calibration — latest usage_log.md readings + current queue depth; if
> the calibration rule in usage_log.md fires, name the exact routine
> slots to add or drop per its add/drop order.
>
> 2. Email it: with the Gmail connector, create a DRAFT to
> mctils12@gmail.com, subject "VolTrade Weekly — reply with usage
> screenshot". Body = the briefing from step 1, then this reminder:
> "REMINDER: paste your current Plan usage panel screenshot into any
> Claude Code session on voltradeai — it will be appended to
> research/usage_log.md and the schedule recalibrated per the standing
> rule." Note: the connector is draft-only (verified 2026-07-03) — the
> email lands in the Drafts folder, not the Inbox. If Gmail tools are
> unavailable in this routine context, state that prominently at the
> top of the session output and continue — the briefing itself is the
> deliverable.

## Log (append-only; newest row last)

| date | 5-hour peak % | weekly % | sessions that week | STARVED count | schedule changes made |
|---|---|---|---|---|---|
| 2026-07-03 | 40% | 15% (resets Jul 5) | bootstrap day: 1 interactive + routines (2× DAILY, 1× EDGE active) | 0 | weekly-review routine being added (Sun 10:00 ET) |

- 2026-07-03 context: first reading, taken during the bootstrap day
  (~21 PRs from the interactive session — not a representative week; the
  weekly window is also partial, resetting Jul 5). Queue depth is
  nonzero: MAP V2 roadmap R2 (transit counters), R3 (environmental), R4
  (globe), KNOWN BROKEN #3/#4/#5/#6/#9, options-backtest wishlist item.
  Per the 2-consecutive-readings rule, a formal slot-add recommendation
  waits for the next Sunday reading — but at 15% weekly with a deep
  queue, the expected recommendation is to fill toward the 8-run table
  (first adds: product-eve 20:00, edge-late 22:30, daily-midday 12:30).
