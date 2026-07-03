# Data / Access Wishlist — human reviews weekly

- **Historical options prices** (EOD chains + marks, ~2016→present) to backtest
  the options leg honestly. Without it, only the equity/ETF logic can be
  validated — and the options leg is the suspected main performance drag.
  Candidates: ORATS, CBOE DataShop, historicaloptiondata.com.

- **[APPROVED BY HUMAN 2026-07-03 — queued as next [REPAIR], see open_questions #7]**
  **Persist the max-drawdown high-water mark** (`state.equityPeak`,
  server/bot.ts:359/862/2482): in-memory only today, so every
  deploy/restart re-bases the drawdown kill switch from current equity —
  frequent autonomous deploys silently defang it. Proposal: save/restore
  equityPeak via the existing /data/voltrade state files. Touches frozen
  kill-switch machinery -> needs explicit human approval (this entry).
  Evidence: /api/health shows equityPeak 0 after today's deploys.
- **Read-only diagnostics access for autonomous sessions**: all diagnostic
  routes are owner-cookie gated (auth.ts — frozen). Options for human:
  (a) paste /api/bot/audit + /api/bot/ml-status JSON into sessions when
  diagnosis is needed, (b) approve a scoped read-only token path in
  auth.ts, or (c) a nightly job that snapshots key state JSON into the
  repo. Until then KNOWN BROKEN #3/#4 can only be verified by the human.

- **[APPROVED BY HUMAN 2026-07-03 — applied same message]** Constitutional
  amendments A1-A3 + STARVED metric (PROMPTS.md Section A): SPINOUT-READY
  DATA LAYER, RAW-DATA vs SIGNALS surface rules, [PRODUCT] session tag,
  starvation signal in HEALTH OF THE LOOP. Proposal and approval recorded
  here per the amendment rule; applied in the same PR.
- **aisstream.io API key** (for the /data map's live vessel overlay — A4
  build): free signup at aisstream.io -> set the key in Railway as
  AISSTREAM_KEY. The vessels layer ships scaffolded and activates when the
  key exists. HUMAN ACTION NEEDED.

- **[APPROVED BY HUMAN 2026-07-03 — applied same message]** Constitutional
  amendment: PROMOTION RULES gain rule 6 (visual verification) — client/
  PRs must run the DESIGN.md harness at 390/768/1440 and self-review
  screenshots before opening. DESIGN.md + scripts/visual_check.mjs are the
  standard and its enforcement. Proposal+approval recorded here per the
  amendment rule.

- **[APPROVED BY HUMAN 2026-07-03 — applied same message]** Constitutional
  amendment: EDGE DOCTRINE gains the BUILD-FIRST RULE (paid is the last
  resort; 4-step free-alternative assessment; honesty clause; every spend
  proposal must attach the analysis). Also DESIGN.md gains the PERFORMANCE
  BUDGET + FEATURE COMPLETENESS CHECKLIST sections. Bookkept per the
  amendment rule.
- **OpenSky free account** (HUMAN ACTION, $0): anonymous OpenSky is
  rate-limited AND currently rejects Railway egress entirely (we run on
  the adsb.lol fallback). A free OpenSky account (OAuth2 client
  credentials -> OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET in Railway)
  raises limits ~4x and may restore the primary feed. BUILD-FIRST
  analysis: raw material already free via adsb.lol; this is a $0 signup
  that adds redundancy, not spend.
- **FlightAware AeroAPI / FAA SWIM (filed flight plans + routes) — PRICED,
  deferred.** BUILD-FIRST analysis attached per the new rule: (1) raw
  material (filed plans) is NOT freely receivable; (2) accumulation
  substitute BUILT: our own position archive gives track history free;
  (3) inference substitute BUILT: destination PREDICTION from trajectory +
  per-tail route history, labeled predicted, self-scored against observed
  landings; (4) what paid adds over our free version: filed (not
  predicted) routes, ETAs, schedules, pre-departure intent. Price: AeroAPI
  personal tier ~$100/mo class. Recommendation: defer until the predicted
  version's measured accuracy (archive self-scoring) proves insufficient
  for a gated signal.
- **Position-archive volume watch** (standing): archive grows on the
  Railway volume; adaptive thinning + rollups built in R1. FLAG HERE if
  growth trends toward plan limits (est. <100MB/mo at current thinning —
  monitor via /api/data/archive/stats once live).
