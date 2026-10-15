# MASTER PROMPT — VOLTRADEAI BOT COCKPIT

> Provenance: human directive received 2026-10-15, installed verbatim. The
> Stage 1 deliverable it orders is `research/bot_cockpit_audit.md`.

## 0. The ambition bar (read first, non-negotiable)

This is not to be built as a modest or understated product. Do not play it safe, do not simplify it into "just another bot dashboard," do not strip out the hard parts to ship faster. Every other bot UI shows a column of numbers you scroll through and learn nothing from — that failure is exactly what this replaces. If a requirement below is hard to build (live self-arranging layout, full-bleed zoning, real-time rendering under load, leak attribution), that difficulty is the point, not a reason to downscope. Build the serious, ambitious instrument described here. Where you must make a tradeoff, surface it and ask — never silently tone it down.

## 1. The process (how this work travels — do not skip or collapse stages)

This moves through three seats in order:

1. **Claude Code — AUDIT.** Read the entire repo and produce the ground-truth map (Stage 1 below). Nothing gets designed until this map exists. Design built on mocked data is decoration; this audit is what makes the design honest.
2. **Claude Design — DISCOVERY + DESIGN.** Fed the audit *and* the repo. First it researches the best way to build an instrument like this and returns a short recommendation with tradeoffs (Stage 2 discovery). Then it designs the cockpit against the *real fields from the audit only* (Stage 2 spine + UI). Mike reviews and tweaks visuals here.
3. **Claude Code — BUILD.** Implements the approved design against real data, with the Stage 3 principles marked non-negotiable so they can't be quietly simplified. Output is one thing: **the bot's page is replaced.**

Scope: this is **only the bot side of VolTradeAI.** The map / data-intelligence platform is untouched. One page, swapped for the real cockpit.

## 2. Stage 1 — SYSTEM AUDIT (run in Claude Code, against the repo)

Read the entire codebase. Miss nothing. You cannot build a better version of a system you do not understand, so understand it completely first. Produce a single structured document that inventories the bot **as it actually exists today** — real file, real function, real data each piece emits. Report what it *does*, not what it should do.

For every stage, report: what it does, the real fields it emits, its healthy-state values, and how you'd know it broke. Cover:

- **Every data feed / input:** source, update frequency, where it lands, and its behavior when stale or failed.
- **Feature pipeline:** each feature, its inputs, its live range vs. its training range.
- **ML stage:** model type, what it predicts, the exact triple-barrier labeling, the purged walk-forward CV setup, and **the precise condition that triggers retraining and where that's logged.**
- **Every internal rule:** entries, vetoes/filters, position sizing, stop-loss logic — with the actual thresholds.
- **Execution path:** signal → order → broker, and what is recorded at each step.
- **State:** where live positions, equity, and trade history live, and in what shape. Identify whether it writes to a database and which one.
- **Current dashboard:** what it shows and where it reads from.

Deliverable: a per-stage map — stage name, real fields emitted, healthy values, failure signature. This vocabulary (stage names, field names) becomes the shared language Design and Build must reuse exactly.

## 3. Stage 2 — DISCOVERY, then the SPINE, then the detailed UI

### 3a. Discovery first (before designing anything)

Research and propose the best way to build an instrument like this, and return a short recommendation with tradeoffs for Mike to pick from before committing:
- Rendering approach for many live-updating panels on one screen without choking (canvas / WebGL vs. DOM).
- Layout system for dense, zoned, full-bleed dashboards.
- How real NOC / observability / trading terminals solve density-without-scrolling.
- Whether anything off-the-shelf fits, or it must be custom.
- Whether a system like this already exists to learn from.

### 3b. The spine — the questions the UI must answer at a glance

The tool is not a status display. Its job every day is to answer one question at the top: **is the edge real right now — or is it gone, and where?** It won't beat the market on day one; early on its real job is a live profit-leak autopsy.

- **Edge:** Is expectancy per trade positive after costs? What win rate do we need to break even given our payoff ratio, and are we above it? How do we compare to just holding the market over the same window? (The card-counter's readout: is there an edge, and where does it live.)
- **Where the edge leaks:** attribute the gap between the *ideal* trade (what the signal said to do) and the *real* result to a named cause — entry slippage, exit slippage, **each scale-out leg separately** (e.g. shedding in thirds), fees/spread, timing/regime. Every dollar of difference between theoretical and real must be attributed to a bucket. Show it as a waterfall/bars so the biggest leak is obvious.
- **Why each trade (the five-layer decision chain):** the trigger (what data woke it, threshold crossed) → the features (what the model saw, each value against its normal band) → the reasoning (which rules fired and which **vetoed**, in order — a skipped trade must be as visible as an executed one) → the conviction (model score, probability, the triple-barrier label it's betting on) → the action (size, entry, exits set, what actually happened). Reasoning must not be repetitive narration — it's evidence, reachable by drilling into a losing cluster.
- **Verification:** when it says "I bought because of X," let the raw numbers behind X be inspectable, so it can't lie to itself.
- **Health:** each stage shows whether it's producing valid, fresh output vs. a **frozen known-good baseline**, so silent failure (ML stopped predicting, retraining stopped firing, feed went stale) surfaces on its own. (Define where "known-good" comes from — training run or set by hand.)
- **History / replay:** scrub back across yesterday / last week / last month and replay a day — often where the leak actually shows itself.

### 3c. The detailed UI (the instrument itself)

One screen, no page-switching, control-room zones:
- **Top — edge scorecard:** expectancy after costs, win rate vs. breakeven win rate, return vs. holding the market. One glance = green or red.
- **Leak-attribution zone:** the ideal-vs-real buckets above, biggest leak obvious.
- **Live reasoning zone:** the five-layer chain for the current / latest decision.
- **Living terminal underneath:** chart populating live with bands and stops drawn on it, open positions, equity, what it's doing in the background right now.
- **Health strip:** each stage green / stale / broken vs. baseline.
- **Drill-down everywhere:** click a leak or a losing cluster → the actual trades and their full reasoning. Detail on demand, never a wall of numbers.

## 4. Stage 3 — DESIGN PRINCIPLES (non-negotiable, do not tone down)

- **Reasoning and edge first.** P&L is shown as a *consequence*, not the headline. The bot's thinking and its edge are the centerpiece.
- **The page is alive and self-arranging.** One page, never switched by hand, always in motion. Calm resting layout on a healthy day. When something happens — new alert, a leak opening, a stage going stale, a high-conviction trade — the page **reconfigures to bring that forward** (the zone grows / surfaces / spotlights), then recedes when resolved. New information can summon new panels that leave when there's nothing to say. You never navigate; the page decides what to put in front of you. Like a control-room wall rearranging around an incident.
- **But you can stop it and inspect.** Click to freeze the motion, pull a panel aside, push something away, pin what you care about, dig into any element — all without the live system failing underneath. Freezing the *view* never freezes the *bot*; it keeps running and recording while you inspect. Primary input is mouse.
- **Full-bleed, no scrolling.** Fill the whole screen edge to edge — no centered column, no wasted margins, no band down the middle. The current UI is a narrow strip you scroll endlessly; that is banned. No vertical scrolling on desktop — depth comes from drill-down and the self-arranging layout, not a feed.
- **Detect the screen and adapt live.** Sense the actual viewport — size, dimensions, orientation, pixel density — and reflow to fill it. Plug a laptop into a big monitor and it expands into the new space. Tiers, not one squeezed design: big external monitor = full multi-zone cockpit; laptop = denser arrangement of the same zones; phone = a *rethought, prioritized* layout (the few things that matter), never the desktop crammed down. Handle different phones, resolutions, aspect ratios, portrait/landscape. Re-adapt on rotate / resize / monitor-switch. **Built for the big screen first.**
- **End-of-day logic.** Account for the trading day as a whole — market hours, the full session, context like a nearby holiday / shortened session affecting expected behavior — and let Mike review the day at its close.
- **Conversational / voice layer (secondary — NOT the main feature).** A layer on top, not the centerpiece. You can talk to the cockpit and it acts: navigate ("scroll back to Tuesday," "jump to the biggest losing day"), ask about its own state ("why did it sell at 10:40," "where are we leaking most this week," "is the edge positive today") and it answers from real data and surfaces the panel that proves it, and rearrange by voice ("pull up the reasoning," "freeze this"). It does the scrolling-back and digging for you. Build the instrument to full strength first; voice must not come at the expense of the core cockpit.

## 5. Hunt every failure variable (non-negotiable)

Find every variable under the sun where this could break and give each a **defined, visible behavior** — never a blank or frozen panel pretending everything's fine. The origin of this whole project: "it broke and I didn't find out for three weeks." Cover at least: a feed goes stale or dies; the ML stops predicting; retraining silently stops firing; the websocket drops; a value leaves its known band; the broker connection fails; latency spikes; a panel gets no data; the market is closed; conflicting or missing state. Each surfaces itself.

## 6. Open decisions to surface (raise these, don't guess)

- **Read-only, or can it also act** (pause the bot, flatten a position, kill a strategy — by mouse or voice)? Note the safety implications.
- **Off-screen alerts** — push to phone / text / email when a feed dies or the edge goes negative? (Directly addresses the three-weeks problem.)
- **Real money vs. paper vs. backtest** — make it unmistakable on-screen which one is being watched.
- **Data freshness** — real-time tick-by-tick vs. every few seconds (drives the rendering choice).
- **Baseline source** for drift detection — training run or hand-set.
- **Persistence** — do pinned / rearranged views survive to next session or reset.
