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
edge-late 22:30. Plus voltrade-weekly-review Sun 10:00 (C1 briefing;
not a build slot).

DELIVERY CHANNEL (fixed 2026-07-03, human directive): routine
deliverables go in the FINAL SESSION OUTPUT, which lands in the Claude
Code Notifications tab — the one channel verified to reach the human.
The Gmail connector is draft-only (no send tool); drafts sat unread in
the Drafts folder, so the Gmail step was dropped from both routine
prompts as a dead letterbox.

NORTH-STAR LINE (human-approved 2026-07-04; UPDATED same day — GIP.md
joined the rule): every PRODUCT (B4) and EDGE (B3) routine prompt must
include, immediately after its CLAUDE.md read instruction: "Then read
VISION.md and GIP.md (repo root) — the product north star for WHAT to
build; CLAUDE.md still governs HOW." The B1/B3/B4 canonical texts live
only in the routine platform (delivered 2026-07-03 as session output,
never committed) — HUMAN ACTION NEEDED: append the quoted line to the
voltrade-product-am/pm/eve and voltrade-edge/edge-late routine
prompts (or update it if the earlier VISION.md-only line was already
added). Sessions cannot edit the routine platform.

- DROP ORDER under quota pressure: product-pm → edge-late → product-eve.
  Irreducible core (4/day): daily-am, product-am, daily-close, edge.
- ADD ORDER with headroom: reverse of the drop order — fill from the
  currently active subset toward the full 8-run table.
- STARVED valve (armed since #98): 10+ consecutive STARVED sessions
  auto-flags Agent-SDK continuous operation in wishlist.md.

## voltrade-usage-check routine (DAILY 21:30 ET) — canonical prompt text

Routine description field: "Daily usage nudge — two-line day summary +
screenshot reminder in session output (Notifications tab). REVISIT
CADENCE ~2026-07-24: once usage readings flatten, delete this daily
routine and let voltrade-weekly-review carry the loop."

> [USAGE-CHECK] Read research/usage_log.md and today's entries in
> research/experiments.md. Report only — no code changes, no PRs.
>
> 1. Two-line summary: (line 1) sessions run today and PRs merged today
> with their tags ([REPAIR]/[PIPELINE]/[PRODUCT]/...); (line 2) the
> current top queued item and whether any session today logged STARVED.
>
> 2. END the session with exactly this block as your final output — it
> is the deliverable, read from the Claude Code Notifications tab (do
> NOT use the Gmail connector; it cannot send, only draft):
>
> "USAGE CHECK <date> — <the two-line summary>.
> ACTION: paste your current Plan usage panel screenshot into any
> session on voltradeai — same-day schedule recalibration per the
> USAGE-CALIBRATION LOOP."

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
> 2. END the session with the briefing as your final output, opening
> with the line "VOLTRADE WEEKLY <date> — reply with usage screenshot"
> and closing with: "ACTION: paste your current Plan usage panel
> screenshot into any session on voltradeai — it will be appended to
> research/usage_log.md and the schedule recalibrated per the standing
> rule." The final output IS the deliverable — it is read from the
> Claude Code Notifications tab. Do NOT use the Gmail connector; it
> cannot send, only draft, and drafts go unread.

## VELOCITY metric (throughput directive 2026-07-04; sessions append a
row when they close; the weekly review reads the trend)

| date | PRs merged (main commits that day) | queue depth (actionable open_questions/roadmap items) | note |
|---|---|---|---|
| 2026-07-03 | 32 | ~14 | bootstrap day |
| 2026-07-04 | 40 (through the GIP directive session) | ~18 (GIP expanded the queue faster than builds drain it — intended) | 4 concurrency collisions, all recovered; partition amendment proposed |

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

## 2026-07-18 (incidental, session screenshot ~20:35Z)
Banner in the human's screenshot: "You've used 87% of your Fable 5
limit — resets Sun, Jul 19, 2:00 AM". Session-tier limit, not the
weekly plan reading. Response (same-day per the loop): throttled
fall-through for the rest of this session — finish queued work only
(catalog-mirror auth fix + B3 integration already in flight), no new
speculative slices until after the reset.
