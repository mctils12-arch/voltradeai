# Bot page — Stage 2 design brief v2 (discovery recommendation + the spine the mockup implements)

> Provenance: written 2026-10-16 after the owner rejected mockup v1 ("not a bot
> cockpit, unnecessary wording… ever changing… click and stop one portion…
> windows on windows that snap back… a visual way of showing what the ML is
> doing is the key"). Inputs: `research/bot_cockpit_audit.md` (Stage 1), the
> directive `research/directives/bot_cockpit_prompt_2026-10-15.md`, nine
> research reports (trading terminals, NOC walls, ML-monitoring UIs, windowing
> libraries, attention-driven cockpits, bot dashboards + explainability,
> refined dense visual language, a compliance check of v1, the ML-zone spec),
> and the live readings of 2026-10-15. The mockup v2 is the Design canvas
> https://claude.ai/artifact/CmisUR2DzG3kxVgnJSMDcM (private to the owner).
> Every number on the canvas is a production reading; nothing is invented.

## 1. What the page is for

One question at the top every day: is the edge real, and if not, where is it
leaking. Today the honest answer is: expectancy −1.20 %/trade on 36 trades,
win rate 27.8 % against a 36.2 % breakeven, and the model's score has no
discriminating power — the top score quintile wins 42.5 % at +5 d, the bottom
quintile 56.2 % (shadow log, 20 000 candidates). The page's job is to make
that visible without a sentence, keep it visible while the bot works, and
turn every problem into condition → consequence → what clears it → action.

## 2. What changed from v1, point by point (owner feedback → v2)

| owner | v2 |
|---|---|
| no "bot cockpit", no unnecessary wording | No wordmark or page name; the site nav already names the site (tab "Bot"). Zero sentences inside panels; labels are nouns; explanations live on hover. Stage codes, rule ids, audit references and glyph alphabets are gone. |
| look at other systems, make it the best | §6 lists what was borrowed and from where. |
| match the site, more refined, not gimmicky | DESIGN.md tokens only; grey is the default state colour; red/green only on numbers and dots, never on containers; no status chips (dot + word instead); one hero number; panels are tint only, 6 px radius; no glows, pulses or gradients. |
| pertinent information present, not explained | Dense by default: stages with their last ten expected ticks, the full book, orders, the 34-feature view, score-pipe, retrain ledger, timeline. Detail is one hover or one window away. |
| the UI is ever changing — searching / found / idle | A page state machine driven by real audit rows (§3). The centre element exists only while the bot is doing something; idle collapses it. The canvas shows the same page in five states using the real 15 Oct sequence. |
| click and stop one portion while the rest runs | Per-panel hold (view only; the bot and buffers continue; badge shows "held hh:mm:ss · live continues · +N ticks"), per-panel pin, and a page-level hold of the self-arranging layout. Acting controls (resume / stay halted) are separate and confirm with their consequence. |
| windows on windows, like Windows 11, snap back | Any panel → window (drag, resize, zoom) over the page; a ghost slot stays in the grid; Dock / Esc / drop-on-ghost springs it home. §5. |
| issue → problem → how to solve | The Issues panel and the incident element are condition · consequence · clears-when · action, each with the live number. |
| a visual way of showing what the ML is doing | The Model zone is the centre of the page (§4): what it is, what it saw, how the number was made, what the score actually predicts, whether it is still learning. |

## 3. The page state machine (real triggers only)

| state | entered by | centre element | leaves when |
|---|---|---|---|
| closed | `/api/bot/market-status.isOpen=false` | day review | isOpen |
| idle | `TIER2 "Scanned N stocks, K trade candidates"`, timer to next scan | none (model zone full height) | `TIER2 "Starting scan"` |
| scanning | `TIER2 "Starting scan (interval…)"` | phase strip from `/api/diag/timings` (universe 0.9 s → snapshot 3.8 → deep score 41.4 → …) with elapsed / budget and candidates so far | `Scanned …` / `TIER2-ERROR` / `TIER2-FAIL` |
| candidate | `Scanned … K>0`, `QUEUE`, `/api/bot/signals` | the decision chain: trigger → features → rules fired / vetoed → conviction → action | `TRADE`/`MORNING-TRADE`/veto row / next scan |
| placed | `TRADE`, `ETF-TRADE`, `OPTIONS-TRADE`, `MORNING-TRADE` | order card with lifecycle words (working · may still fill · filled · cancelled) | fill / `SWEEP` / `SHUTDOWN` |
| exit | `WS-EXIT`, `POS-KILL*`, `EXIT-FILL`, options exit reasons | the leg being shed, realised vs intended | feedback record written |
| incident | `DRAWDOWN-KILL`, `DD-HALT`, `TIER2-FAIL`, `STREAM-DISCONNECT` > 60 s, `TIER3-ML-ERROR`, model age > 24 h, liveness dark > 2 market h, health degraded | condition · consequence · clears when · action; takes the centre; everything else shrinks | the clearing condition, then recedes to a one-line status ("normal · checked hh:mm:ss") |

Rules from the attention / NOC research: only the top alert level reconfigures
the layout; a candidate is a surfacing, not an alarm; the stage rail shows
presence of expected output per cadence (a missing box, never a calm zero);
re-layouts pass through hysteresis (min dwell, recede grace, never while the
pointer is down); halted / closed / deploy periods are shaded behind every
time axis; time-to-breach (equity vs kill line, model age vs 24 h) replaces
binary flags.

## 4. The Model zone (the centre)

Reads today, with zero new code: `/api/ml/status`, `/api/diag/ml`,
`/api/diag/shadow` + `shadow_bands`, the `TIER3` audit rows, the blend
tracker, the shadow candidate records (the only persisted serve-time
feature vectors). Four columns, left to right:

1. **What it is** — model type and regime route (bull unreachable while
   `vxx_ratio` is pinned at 1.0), age vs the 24 h fallback line with the 20 h
   warning, blend weights and what set them (recent win rate), label
   definitions, CV geometry, the retrain ledger as a status history of hourly
   checks, and the label bet vs the live exit rules.
2. **How the number was made** (candidate state) — the score pipe: rules →
   ML p → score → gate → blend → final vs the two thresholds (Node 75, Python
   63). ML cells read "not recorded" until the shadow record carries `ml.*`.
3. **What it saw** — the 34 features against the training band (scaler
   mean ± 1σ, the only persisted training statistic), each marked tracks /
   differs / constant / derived / dead. Today: 7 constant, 4 dead, 8 differ.
4. **What the score predicts** — score quintile vs forward win rate at +5 d
   and +20 d with the labeler's breakeven (66.7 %) drawn; taken vs rejected
   buckets; taken × day-move band (down > 10 % that day wins 79.7 %, up
   10–20 % wins 38.9 %: the bot buys spikes that mean-revert). This is the
   problem statement, on screen, from real data.

Purple is reserved for values the model produced (one categorical colour per
page). Drift is variance collapse, not mean shift: the per-feature view
shows variance ratio and %-constant once the baseline sidecar exists.

## 5. Handling: hold, pin, window, snap back

Panel states: docked · floating (ghost slot kept) · zoomed · parked · hidden.
Flags: held (display paused at T; bot and buffers continue; implicitly
pinned) · pinned (the director may not move, resize or evict it). Page-level
hold pauses the director; everything stays live; a white outline marks
human-induced state (dark cockpit rule: lit means act, white means you did it).

Pull out: drag the header 6 px → the panel lifts onto the float layer and a
ghost outline stays in its slot; drop anywhere, clamped to the page, edges
snap at 12 px. Enlarge: step (+/−, borrows from the lowest-priority
neighbour, undo chain) or zoom (fills the page, others keep subscriptions but
render nothing; the KPI strip and the liveness state stay visible in every
state). Snap back: drop on the ghost, Dock, Esc, or the director's recede;
one FLIP spring (≈450 ms, critically damped) per panel, transform-only during
motion, one real relayout on settle. Tiers: ≥1920 full · ≥1280 dense ·
≥768 two-column · <768 a prioritised list, never the desktop squeezed;
tier changes snap without animation. Persistence: per-tier, user pins /
holds / float rects only; director state never persisted; "Reset layout".

## 6. Discovery — recommendation with tradeoffs (directive §3a)

- **Rendering.** DOM panels under the one `frameCore` rAF loop (Law I),
  springs for layout, `lightweight-charts` (canvas) for the price chart and
  equity curve, SVG for sparklines and bars. Canvas/WebGL for everything would
  cost accessibility and the editor-grade text density for no measured gain
  at ≤ 30 panels; the per-frame cost is the chart, not the DOM.
- **Layout.** A small custom zone engine (CSS `grid-template-areas` per tier +
  a director with hysteresis + FLIP springs + a float layer; ≈600–900 lines
  TypeScript, no new dependency). Fallback: `dockview-react` (MIT) driven by
  the same director — it has floating groups, maximise and lock, but no
  animated re-layout and VS-Code chrome that fights DESIGN.md. Rejected:
  react-grid-layout (scrolling row packing), Golden Layout (stalled),
  FlexLayout (same gaps), Swapy (GPL), Muuri (dead).
- **Density without scrolling.** From the terminals and NOC research: one
  focus context (ticker / trade / time) that every panel follows; state by
  presence (a running scan is a row with elapsed / budget that disappears);
  tables as monitors (≤14 columns, tabular figures, signed colour on the
  cell); sparklines with a grey normal band as the drift and health visual;
  every number carries formula, source and age on hover.
- **Off-the-shelf.** Nothing fits whole. Borrowed forms: Grafana state
  timeline / status history (serving mode, expected ticks); Honeycomb
  compressed trace strip (the gate sequence, vetoes red, missing span = alarm);
  QuantConnect's scored Insight object joined to the order (what the bot must
  persist); Few's bullet graph (feature vs band, win rate vs breakeven);
  Perold's implementation-shortfall decomposition (the leak waterfall, residual
  printed); Windows 11 Snap Layouts / zellij swap layouts / tmux zoom (the
  windowing model); Airbus dark-cockpit and ISA-18.2 (alarm levels,
  consequence + action, shelving with expiry); IBKR / Sierra order-state
  words; Tufte sparklines; Vercel/Linear/Carbon colour and chip restraint.
- **Does a system like this exist?** No live bot UI surveyed shows skipped
  trades, attribution or model outputs; the closest pieces are Freqtrade's
  offline backtesting-analysis table and QuantConnect's Insight/Orders tabs.

## 7. Failure variables → visible behaviour (directive §5)

Feed stale/dead → stage box turns late/missing with age; neutral defaults
render as absent, never as data. ML stops predicting → serving-mode strip
shows fallback as a different model version; age line crosses 24 h. Retrain
silently stops → missing boxes in the hourly ledger. WebSocket drops →
feeds stage dead + bars/min gauge. Value leaves its band → feature marker
outside the band, count in the header. Broker fails → feeds stage dead,
fabricated-equity sizing flagged. Latency spike → scan strip over budget,
event-loop stalls on the timeline. Panel gets no data → "not recorded"
outlined dot + last-known value with age, never $0.00. Market closed →
session clock + day review; idle-closed and idle-broken are distinguished by
the expected-tick boxes. Conflicting state → the three peaks side by side
with their owner. Loop alive but not trading → "stock entries 0 · 4th day"
in the day review and an Issue.

## 8. Open decisions (directive §6) — raised, not guessed

1. Read-only or act: the mockup shows act controls (resume / re-baseline /
   stay halted) that confirm with their consequence; every act endpoint is
   set-not-toggle and persisted so a deploy cannot undo it. Decide.
2. Off-screen alerts: push/text on L3 (latch, loop dark, edge negative) in
   addition to the existing email. Decide the channel.
3. Paper / live / backtest: "Paper · Alpaca" plain text in the nav rail on
   every tier; a replay is labelled "replay" on the timeline.
4. Freshness: every panel prints its own age in the same top-right slot;
   SSE for audit rows, 5–15 s polling elsewhere; tick-by-tick only for the
   chart. Fine for DOM + springs.
5. Drift baseline: the training run writes `voltrade_ml_feature_baseline.json`
   (per-feature n, mean, std, p1/p5/p50/p95/p99, %-constant, per regime);
   hand-set only for hard zeros (fallback count, calibrator newer than model).
6. Persistence: per-tier user layout only, director state never; "Reset".
7. Which threshold is primary in the score pipe: Node 75 (execution truth)
   proposed; Python 63 drawn second.

## 9. What the bot must persist before the page can be honest (small T-BOT PRs)

`ml.*` on the shadow record (model_type, regime_used, win_probability,
ml_score, gated, weights, error); `scan_id` + `decision_trace[] {rule_id,
passed, threshold, value}` for every candidate including vetoed ones;
`position_id` carried as Alpaca `client_order_id` on every order and leg;
the feature baseline sidecar; an append-only retrain ledger with
`insufficient_data` reclassified as an error; SPY daily closes + sleeve tag
for the vs-market cell; per-leg exit fills with decision price and timestamp.
Each is its own PR with a test; the page renders "not recorded" until each
lands, never a stand-in number.

## 10. Build notes (Stage 3 hand-off)

Replace the page at `/app#/bot` under the site's fixed 56 px nav (phone: 48 px
top bar, 64 px bottom bar). `client/src/cockpit/`: `zoneModel.ts`,
`director.ts`, `flip.ts`, `floatLayer.tsx`, panels. Reuse `render/frameCore.ts`,
`render/freshness.ts`, `perfHud` (`?perf=1`), `lightweight-charts`. Gates:
visual harness at 390/768/1440, counters in `ci/counter_baseline.txt`, Law IV
teardown test, p95 frame time under 16.7 ms during a scripted summon/recede.
Everything above is the owner's to tweak on the canvas before the build.
