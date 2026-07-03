# Usage Calibration Log — plan quota vs. routine schedule

Purpose: close the loop between Claude plan usage and the routine
schedule. The human pastes their Plan usage panel screenshot into any
session on this repo; the session reads the 5-hour and weekly
percentages from the image, appends a row here with context, and applies
the calibration rule below (standing behavior note in CLAUDE.md KNOWN
STATE, human-approved 2026-07-03).

## Calibration rule (DAILY AGGRESSIVE MODE, human-approved 2026-07-03)
- Screenshot pasted → log the row and respond SAME DAY: if trajectory
  shows clear headroom against the queue, name exact routine slots to
  add NOW (up to the platform's daily cap and 1-hour spacing); if
  approaching limits, name what to throttle — fall-through depth first
  (CLAUDE.md SESSION BUDGET ladder), then slots per the drop order.
- Bias toward AGGRESSIVE slot addition while weekly readings are <50%.
- Revisit cadence ~2026-07-24 (2–3 weeks in): once readings flatten,
  drop the daily check back to weekly; the weekly-mode rule then applies
  (2+ consecutive weekly readings <50% → add; ~90%+ → drop).

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

## voltrade-usage-check routine (DAILY 21:30 ET) — canonical prompt text

Routine description field: "Daily usage nudge — two-line day summary +
screenshot reminder draft. REVISIT CADENCE ~2026-07-24: once usage
readings flatten, delete this daily routine and let
voltrade-weekly-review carry the loop."

> [USAGE-CHECK] Read research/usage_log.md and today's entries in
> research/experiments.md. Report only — no code changes, no PRs.
>
> 1. Two-line summary: (line 1) sessions run today and PRs merged today
> with their tags ([REPAIR]/[PIPELINE]/[PRODUCT]/...); (line 2) the
> current top queued item and whether any session today logged STARVED.
>
> 2. With the Gmail connector, create a DRAFT to mctils12@gmail.com,
> subject "VolTrade Daily — usage screenshot". Body = the two-line
> summary, then: "REMINDER: paste your current Plan usage panel
> screenshot into any Claude Code session on voltradeai — same-day
> schedule recalibration per the USAGE-CALIBRATION LOOP." The connector
> is draft-only (verified 2026-07-03): the draft lands in the Drafts
> folder, not the Inbox — the Claude Code Notifications tab is the
> reliable place to see this routine completed. If Gmail tools are
> unavailable in this routine context, state that prominently at the top
> of the session output and continue — the summary itself is the
> deliverable.

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
  (globe), KNOWN BROKEN #3/#4/#5/#6, provider-redundancy research,
  options fill realism, strategy tournament, dual-momentum OOS.
- 2026-07-03 SAME-DAY RECOMMENDATION (daily aggressive mode, human
  directive "scale now"): headroom is decisive — weekly 15% two days
  before reset with only 3 routines active, on the heaviest interactive
  day the repo has had. CREATE TODAY (completes the A5 8-run table +
  the daily nudge; all ≥1h apart, within the per-account daily cap):
  1. voltrade-product-am — 9:00 ET — PRODUCT prompt (B4)
  2. voltrade-daily-midday — 12:30 ET — DAILY prompt (B1; holds merges
     until close per the A5 mid-market rule)
  3. voltrade-product-pm — 14:00 ET — PRODUCT prompt (B4; holds merges)
  4. voltrade-product-eve — 20:00 ET — PRODUCT prompt (B4)
  5. voltrade-usage-check — 21:30 ET — prompt above (this file)
  6. voltrade-edge-late — 22:30 ET — EDGE prompt (B3)
  Rationale: routines alone are a small share of the 15%; ~2.7× routine
  load stays far under the 50% aggression threshold even with
  fall-through making each run heavier. Watch the next 2-3 daily
  readings; if 5-hour peaks spike past ~80%, throttle per drop order
  (product-pm → edge-late → product-eve) before touching the core.
