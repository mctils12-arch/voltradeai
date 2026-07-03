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
- **⚠ FLAGGED CONSTRAINT — aircraft-feed licensing (HUMAN DECISION NEEDED,
  filed 2026-07-03). Analysis only per your instruction; NO provider or
  code change made.** While verifying the new OpenSky credentials we read
  all three providers' actual terms:
  - **OpenSky Network** (current primary): the license grants use "solely
    for the purpose of non-profit research and non-profit education," and
    two independent tripwires both fire for us: (1) "Any use by a
    for-profit or commercial entity requires written permission and a
    license granted by the OpenSky Network"; (2) "Use of the REST API in
    any operational capacity — including integration into a live product,
    service, or automated system (even if only internal) — requires a
    previous written agreement, even for non-profit or governmental
    entities." VolTradeAI has paid features (billing) and integrates the
    feed into a live product plus an automated archive — commercial AND
    operational. The new free account raises rate limits but does not
    change the license; continued use as primary is a terms violation
    unless written permission is obtained (contact@opensky-network.org).
  - **adsb.lol** (fallback 1): API and data licensed **ODbL 1.0**,
    "available to everyone" — commercial use permitted, with attribution
    (the map already shows source attribution) and share-alike on
    derivative *databases*. The only provider terms-compatible with
    commercial display today. Spinout note: our position archive is a
    derivative database — any future redistribution/sale of
    archive-derived products built on adsb.lol data must carry ODbL
    attribution + share-alike (fine for display/signals we keep internal).
  - **airplanes.live** (fallback 2): the free REST API is explicitly
    "Non-Commercial Use" (educational purposes, 1 req/s, no SLA).
    Commercial access exists via direct arrangement
    (airplanes.live/commercial-use/, RapidAPI "coming soon") — same
    incompatibility as OpenSky until arranged.
  - **Recommendation (pending your approval, ~15-min change once
    approved):** make adsb.lol PRIMARY; remove OpenSky from the chain
    unless/until you obtain written permission (it is also still failing
    from Railway egress even with credentials — see the entry above);
    keep airplanes.live as emergency-only fallback while you email their
    commercial contact, or drop it too for strict compliance. If you want
    OpenSky's global-bbox capability legitimately, their non-commercial
    research license does not cover us — the honest paths are written
    permission or a commercial ADS-B aggregator (would join FlightAware
    entry below as a priced item).
  - Sources: opensky-network.org/about/terms-of-use (§1 LICENSE, §3(vi));
    adsb.lol/docs/open-data/api (ODbL 1.0) + adsb.lol privacy-license;
    airplanes.live/api-guide + airplanes.live/commercial-use.

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
- **Position-archive volume watch** (standing, LIVE 2026-07-03 — see
  experiments.md): 30-min sample interval per kind, compact positional
  (not object) JSONL records, 90-day raw retention with a permanent
  rollup surviving pruning. Computed estimate at these parameters:
  aircraft ~40MB/mo + vessels ~65MB/mo ≈ **105MB/mo combined** (math in
  `server/dataArchive.ts` header comment) — this is the actual design
  figure, not a guess; revise the interval if the real
  `/api/data/archive/stats` numbers, once the deploy has run for a few
  days, come in materially higher (e.g. from aircraft/vessel counts near
  the 800/1500 per-request caps more often than assumed). FLAG HERE if
  growth trends toward Railway volume plan limits.
