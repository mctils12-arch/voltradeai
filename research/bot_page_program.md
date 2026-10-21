# BOT PAGE PROGRAM — the human's 2026-10-21 directive, mapped honestly to what the bot has, and the slice order

> Provenance: installed 2026-10-21 from the human's in-session directive
> (spoken-style, paraphrased below with the phrases kept where they carry
> the intent). This file sequences the work; CLAUDE.md governs how every
> slice ships (one logical change per PR, visual harness at 390/768/1440,
> tests for every behaviour change, DESIGN.md tokens, Rendering & Motion
> Law for anything animated). research/bot_cockpit_design_brief_v2.md
> remains the page's design spine — this program extends it, it does not
> replace it.

## 1. What the human asked (and the one-line reading of each)

1. "Charts never load … I'd rather have a larger chart … I don't want
   the ones from that other company … build our own charting software"
   → the page's chart is ours: a canvas engine in the repo, no
   third-party chart library on the Bot page, the chart gets real room.
2. "see the entry, the stop loss, the targets, the areas where you would
   phase in and phase out" → every level the engine actually uses is
   drawn on the chart, from the engine's own formulas (§2 says which
   exist and which do not).
3. "charts for the things you're possibly looking at buying … what
   you're looking for out there and what you'd buy after the review" →
   a persistent Watch surface: the last scan's candidates with score,
   action and reasons, the refusals and why, the morning queue; any of
   them chartable.
4. "look at the page … see if there's anywhere we could optimize" →
   §3 is the audit of the page as it stands on 2026-10-21.
5. "use it as a tool for my personal investing … if something shows up
   as good I research it on my own and invest in my brokerage" → the
   Watch surface and the chart carry enough context (score vs threshold,
   reasons, the bot's own live expectancy) to start that research; the
   page never calls a candidate advice.
6. "beat the market as a whole, as if you just bought the S&P" → the
   existing vs-SPY cell stays the benchmark; the tax slice adds the
   after-tax version of the same comparison.
7. "accounting logic … long-term vs short-term capital gains … some
   maneuvers might not make mathematical sense … year-end offsets …
   rebuying" → a tax-aware ledger: realized gains by holding-period
   term, wash-sale detection, estimated tax drag at the human's rates,
   after-tax account vs after-tax buy-and-hold. The paper account pays
   no tax; the ledger models what the same trades would cost in a
   taxable account, labelled as an estimate.
8. "every aspect of the system has to be built into that UI to a degree"
   → §4 inventories what the page shows today and what it does not, and
   which slice closes each gap (or files it).
9. "a small question mark by terminology, the same size as the font,
   hover over it, a window pops up and explains … and it should function
   even if it's detached" → a help glossary component on every term, in
   docked panels, in-page windows and pop-out windows alike.
10. "I have multiple monitors … detach one of these blocks, like the
    model block or the chart block, and move it to my other monitor" →
    a real browser-window pop-out per panel (the in-page float stays for
    the single-monitor case).
11. "see this change over time … if something changes it pops up and
    changes how the view looks, not static" → the director already
    re-arranges the page on real state changes; the live-change slice
    adds visible change on every number and a change strip for events.

## 2. Honest mapping — what the engine actually has behind each chart level

The chart may draw ONLY what the engine computes. Read from
`bot_engine.py manage_positions` (2026-10-21) and `server/bot.ts
/api/bot/positions`:

| ask | exists? | source of truth |
|---|---|---|
| entry | yes | Alpaca `avg_entry_price` (the position's average fill) |
| stop | yes | phase 1: entry × (1 − ATR%·2 … capped 1.5–10 %); phases 2–4: trailing from the high-water mark (1.5×, 1.0×, trail×ATR, regime-scaled). bot.ts recomputes the price from `voltrade_stop_state.json` (`current_stop_pct`, `phase`) |
| target (take-profit) | yes, phases 1–2 only | bot.ts: entry × (1 + ATR%·1.5); the engine's own TP check (`tp_pct = max(4, min(3×ATR%, 15))`) fires only after both scale-outs — the chart labels the line "target (after ⅔ out)" |
| "phase out" areas | yes — scale-OUT in thirds | ⅓ at +1R (R = `initial_risk_pct` = 2×ATR% at entry), ⅓ at +2R·regime (3R bull/neutral-bull, 1.5R bear/panic/caution, 2R otherwise), remainder on the trailing stop; breakeven stop after the first scale-out; hard floor −20 % in phase 1; time stop at 7 days flat (`abs(pnl) < 2 %`) |
| "phase in" areas | NO | the engine enters a position in one piece (scan → order, or the morning queue). `upgrade_candidates` is produced by manage_positions but nothing consumes it. The chart draws no phase-in level; the help text says so |
| options legs | partly | an OCC leg has no bars; the chart shows the UNDERLYING with the strike and expiry drawn; the leg's own P&L stays in the Book |
| candidates (Watch) | price + score only | a stop/target is computed at ENTRY from that day's ATR; drawing one for a candidate would be invention — the chart shows price, the score vs the admission threshold and the reasons |

Finding filed while mapping (open_questions.md, 2026-10-21): the time
stop is hard-coded `days_held >= 7` in the engine while
`system_config.TIME_STOP_DAYS` says 10 (regime-adaptive 3/5/10) —
the config value is never read on that path. Not changed here (a rule
change needs its evidence); logged for RULE-REVIEW.

## 3. Audit of the page on 2026-10-21 (what "charts never load" actually was)

Read from `client/src/bot/*` and the harness screenshots at 1440:

1. **Layout starvation.** `ChartBody` stacked one 320 px third-party
   chart per position (8 positions → 2,560 px) inside a panel that gets
   ~205 px of the right column; the panel body hides overflow, so the
   visible strip was the first chart's header. The candle canvas was
   below the fold of a container nobody could scroll. "Never loads" was
   mostly "never visible".
2. **Option legs chart nothing.** `/api/bot/bars/:ticker` returns
   `bars: []` for OCC symbols; six of the eight positions are option
   legs → six blank canvases.
3. **Upstream errors were silent.** The bars route mapped any Alpaca
   error body to `{bars: []}` (no `r.ok` check) — a rejected feed or a
   rate-limit rendered as an empty chart, never as an error.
4. **No fetch timeout.** The chart's `apiRequest` had none; a hung bars
   call left "Loading chart…" on screen indefinitely (every other Bot
   page read carries a 20 s cap).
5. The chart library (lightweight-charts, TradingView's) brought its own
   colours, font and chrome into a DESIGN.md page.

Other UI findings worth a slice later (not in this program's first
wave): the Model zone's feature list truncates names at 112 px
(`cross_s…`) with the full name only on hover; the Stages rail's
value column truncates on the L tier; the Book shows no per-position
stop/target (the chart now carries them); the timeline's marks have no
legend.

## 4. Inventory — every part of the system vs the page

| subsystem | on the page today | gap → slice |
|---|---|---|
| scan loop, cadence, phases | Stages rail, scan strip | — |
| candidates + decision chain | centre element only while "candidate" | persistent Watch panel → slice B |
| ML model (age, blend, features, bands) | Model zone | help text on each term → slice C |
| positions (qty, price, P&L) | Book | levels + chart → slice A |
| stop state (phase, R, scale-outs, breakeven) | not shown | chart levels + legend → slice A |
| options manager state (strategy, setup, DTE) | partly (Book label) | underlying chart + strike/expiry → slice A |
| orders working / cancelled by deploy | Orders | — |
| kill switch, halts, auto-resume | incident element + Issues | — |
| audit log | timeline + replay | change strip → slice F |
| P&L by sleeve, by underlying, by hold time, reconciliation | Book summary | tax terms → slice E |
| vs SPY benchmark | KPI | after-tax comparison → slice E |
| counterfactual rule P&L (RULE REVIEW logger) | not built | out of scope here; stays in open_questions.md |
| floor exposure schedule / regime | not shown | file (needs the ablation first, task #34) |
| research artifacts (experiments, tests) | not shown | out of scope (research/ is the record) |

## 5. Slice order (each its own PR, T-CLIENT territory; the two small
server edits in slice A — stop-state fields on `/api/bot/positions`,
`r.ok` on `/api/bot/bars` — belong to this program per the
cross-territory rule, wholly in its PR)

- **A. Chart engine** — `client/src/bot/chart/`: `scales.ts` (pure:
  bar↔x, price↔y, nice ticks, viewport), `levels.ts` (pure: the §2
  table from a position), `draw.ts` (canvas renderer), `BotChart.tsx`
  (one frameCore loop, springs for pan/zoom targets — Law I; Pointer
  Events; designed loading / error / empty states; a 30 s watchdog
  note). Symbol strip = open positions (+ Watch candidates once slice B
  lands); Book row click focuses the chart. The chart takes the right
  column's largest share and zooms to the page. `TradeChart.tsx` and
  the `lightweight-charts` dependency are removed. Harness: a bars
  fixture; the page battery asserts a drawn canvas.
- **B. Watch panel** — last scan top-10 (score vs threshold, action,
  reasons, price), sizing blocks, QUEUE rows, leveraged refusals; a row
  click charts the ticker. Zone templates gain the panel.
- **C. Help glossary** — `help.tsx` (`?` glyph at 1 em; hover / focus /
  tap; clamped popover; Esc) + `glossary.ts` (term → one plain line;
  a test asserts every key used in the panels exists). Applied across
  KPI, panels, chart legend, Book, Model.
- **D. Pop-out windows** — `window.open('/app#/bot/pane/<id>')`; the
  pane route renders one panel with its own data hooks; the main page
  keeps a ghost slot and restores on close; BroadcastChannel carries
  dock/close; protocol in a pure tested module.
- **E. Tax-aware ledger** [RULE-REVIEW] — `server/taxLedger.ts` from
  the same `realizedEvents` walk: term by holding period, wash-sale
  detection (loss + re-open of the same symbol within ±30 days),
  estimated drag at the human's rates (the taxes page's defaults),
  after-tax account vs after-tax SPY buy-and-hold (tax deferred until
  sale). Shown in the Book with help text; own PR, before/after and
  bias direction stated.
- **F. Live change** — `useFlash` (value change → brief highlight,
  reduced-motion aware) on KPI values and Book rows; a change strip for
  new TRADE / EXIT / QUEUE / SIZE-BLOCK / incident rows that never
  covers a control.

Definition of done for every slice: tests for the pure modules; the
visual harness at 390/768/1440 with screenshots reviewed against
DESIGN.md; counters and tsc ratchets green; an experiments.md entry
tagged [PRODUCT] (or [RULE-REVIEW] for E).

## 6. RESUME STATE

- 2026-10-21: program installed; slice A in build.
- 2026-10-21 (scheduled routine): slice A SHIPPED (#1416, v1.0.1135). Slice B SHIPPED
  (v1.0.1136): `watch.ts` (pure) + a Watch panel (left column, between Stages and
  Issues; phone/tablet stack after Issues) — the last scan's top 10 plus any trade or
  sizing refusal outside it, each marked taken / blocked / not taken; a row click
  charts the ticker as a price-only chip tagged "watch" (no stop/target — the engine
  sets those at entry). NOT in slice B, deliberately: the morning-queue rows (audit
  `QUEUE` events, not in the scan result — a follow-up reading the audit) and an
  admission-score line (MIN_SCORE is regime-adaptive and the scan result does not
  carry it; adding `min_score` to scan_market's return is a small T-BOT change, filed
  in open_questions.md). NEXT: slice C (help glossary).
