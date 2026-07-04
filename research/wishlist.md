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
- **[DONE BY HUMAN 2026-07-03 — verification NEGATIVE, see below]**
  OpenSky free account ($0): credentials set in Railway as
  OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET. Same-day verification:
  production STILL serves from community fallbacks — 6+ fresh-bbox
  probes spanning ~30 minutes (longer than the 15-min max backoff, so at
  least one live OpenSky attempt was guaranteed inside the window) all
  returned adsb.lol or airplanes.live; OpenSky never served a request.
  The API itself is up (HTTP 200 anonymously from a non-Railway
  network), so the pre-credentials Railway egress rejection appears to
  persist with OAuth. Not distinguishable from outside: (a) IP-level
  block also covering the auth endpoint, (b) states/all rejecting
  Railway even authenticated, or (c) service never restarted after the
  env vars were set. Railway deploy logs disambiguate — look for
  "[datacore] opensky auth:" lines (token fetch failing) around aircraft
  requests; if no restart happened since setting the vars, redeploy once
  and re-check. MOOT UNTIL THE LICENSING DECISION BELOW: the terms
  analysis means OpenSky should not be our primary even if it worked.
- **[APPROVED BY HUMAN 2026-07-03 — applied same message]** Constitutional
  amendment: CLAUDE.md KNOWN STATE gains the USAGE-CALIBRATION LOOP note
  (usage-screenshot readings → research/usage_log.md; 2+ consecutive
  weekly readings <50% with nonzero queue → recommend slot adds; ≥90% →
  recommend drops per the established drop order). usage_log.md carries
  the A5 schedule reference (8-run menu, drop/add order, STARVED valve)
  and the canonical voltrade-weekly-review routine prompt. Gmail
  connector verified draft-only (no send) — weekly email lands in
  Drafts; routine-context availability unverifiable until the first
  Sunday run. Bookkept per the amendment rule.

- **⚠ FLAGGED CONSTRAINT — aircraft-feed licensing (MONETIZATION
  TRIPWIRE, filed 2026-07-03; corrected same day per human: the site is a
  proof of concept with NO paid product today — billing code exists but
  nothing is charged). Analysis only; NO provider or code change made.**
  Provider terms, assessed against the corrected commercial status:
  - **OpenSky Network** (current primary): the license grants use "solely
    for the purpose of non-profit research and non-profit education."
    As a no-revenue POC we are plausibly inside "non-profit research"
    on the commercial clause — but a second, independent clause still
    fires TODAY regardless of revenue: "Use of the REST API in any
    operational capacity — including integration into a live product,
    service, or automated system (even if only internal) — requires a
    previous written agreement, even for non-profit or governmental
    entities." Our bot + site + automated archive are exactly that. So
    OpenSky technically requires a written agreement even for the POC
    (contact@opensky-network.org — plausibly granted free for research).
    The new free account raises rate limits but does not change this.
  - **adsb.lol** (fallback 1): **ODbL 1.0**, "available to everyone" —
    compatible today AND after monetization, with attribution (already
    shown on the map) and share-alike on derivative *databases*. The
    only provider that survives monetization unchanged. Spinout note:
    the position archive is a derivative database — redistribution/sale
    of archive-derived products built on adsb.lol data must carry ODbL
    attribution + share-alike (internal display/signals are fine).
  - **airplanes.live** (fallback 2): free REST API is "Non-Commercial
    Use" (educational, 1 req/s, no SLA) — **compatible with today's
    no-revenue POC**, incompatible the day the site charges anyone
    (commercial access exists via direct arrangement:
    airplanes.live/commercial-use/, RapidAPI "coming soon").
  - **DECIDED BY HUMAN 2026-07-03 (executed same day, v1.0.45):**
    OpenSky dropped from the runtime chain — adsb.lol primary,
    airplanes.live fallback; removes the ~12s dead OpenSky attempt on
    every fresh viewport. The human has emailed
    contact@opensky-network.org requesting a research agreement.
    IF/WHEN GRANTED: reinstate OpenSky in the chain (git history of
    v1.0.43 has the OAuth + states/all implementation to restore) AND
    re-verify Railway connectivity at that time — the egress block is
    independent of the license and may still bite. THE TRIPWIRE stands:
    before enabling billing, ads, or any paid feature, re-run this
    compliance check — at that moment airplanes.live must be dropped or
    upgraded to a commercial arrangement, and adsb.lol becomes the only
    lawful free provider.
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

- **[APPROVED BY HUMAN 2026-07-03 — applied same message]** Constitutional
  amendments batch: (1) SESSION BUDGET replaced by the PRODUCTIVE
  FALL-THROUGH ladder (queued item -> filed-artifact research -> decision
  request never idles a session; hard limits preserved: own PR/log per
  action, read-before-write, anti-churn, [NO-ACTION] only on empty
  queue); (2) DEAD CODE POLICY (stale code is debt; same-PR removal;
  likely-returner adapters only with zero runtime cost + review-by date +
  open_questions log; 30-day staleness audit as fall-through action);
  (3) CONSTITUTIONAL HYGIENE (monthly rule audit files consolidation
  proposals here, never self-applies; live conflicts resolved by GOAL
  priority order and filed). Bookkept per the amendment rule.

- **⚖ FIRST CONSTITUTIONAL AUDIT (2026-07-03) — [APPROVED BY HUMAN
  2026-07-04: Findings 1 AND 2, shipped as one docs PR same day. F1 =
  STANDING BEHAVIORS section added, rule paragraphs moved verbatim out
  of KNOWN STATE. F2 = delivered via the AUDIT CYCLE register in
  experiments.md (the AUDIT CYCLE proposal's concrete superseding form —
  one register, not two).]**
  - **Finding 1 — rules living in KNOWN STATE.** KNOWN STATE now hosts
    four standing behavior RULES (SPINOUT-READY DATA LAYER, RAW-vs-SIGNAL
    surface rules, USAGE-CALIBRATION LOOP, MONETIZATION TRIPWIRE, plus
    the product-routine mandate). The self-edit rule permits sessions to
    append "factual updates" to KNOWN STATE — rules living there blur
    the facts-vs-rules boundary the amendment lockdown depends on.
    PROPOSAL: add a "STANDING BEHAVIORS (each human-approved, dated)"
    section; MOVE those rule paragraphs there verbatim (zero wording
    change); KNOWN STATE returns to pure facts. Preserves: all rule
    text. Drops: nothing. Resolves: self-edit ambiguity.
  - **Finding 2 — two identical-cadence periodic audits.** DEAD CODE
    POLICY's staleness audit and CONSTITUTIONAL HYGIENE's rule audit
    share trigger (fall-through research tier, 30+ days) but live in
    separate sections; the December market_calendar year-add is a third
    scattered periodic duty. PROPOSAL: one "PERIODIC AUDITS" register
    (subsection of SESSION BUDGET) listing all recurring obligations +
    cadences + last-run dates, each pointing at its governing section.
    Preserves: every audit's content/cadence. Drops: nothing. Resolves:
    scatter — future sessions check one place.
  - Factual drift found and corrected directly in this PR (allowed as
    factual update, not part of the proposal): KNOWN STATE + CODEBASE
    MAP still called backtest.py a STUB hours after the engine was
    rebuilt; both now state the rebuilt reality.
  - Interactions checked, no action needed: STARVED's definition
    survives fall-through unchanged (capacity exhausted with queue
    nonzero); BUILD-FIRST sits correctly as an EDGE DOCTRINE subsection;
    the tripwire rule vs. the FLAGGED CONSTRAINT entry are rule vs.
    decision-record, cross-referenced, not redundant.

- **[APPROVED BY HUMAN 2026-07-03 — applied same message]** USAGE-
  CALIBRATION LOOP switched to DAILY AGGRESSIVE MODE: usage-screenshot
  readings get a SAME-DAY recommendation (headroom → name exact slots to
  add NOW up to the platform cap; near limits → throttle fall-through
  first, then drop order); aggressive-add bias while weekly <50%. New
  voltrade-usage-check routine (DAILY 21:30 ET) — canonical prompt in
  usage_log.md; description carries the ~2026-07-24 revisit note (drop
  back to weekly once readings flatten). Gmail re-verified this session:
  connector remains DRAFT-ONLY (no send tool exists) — daily nudge lands
  in Drafts; the Claude Code Notifications tab is the recommended
  completion signal instead. Bookkept per the amendment rule.

- **⚖ CONSOLIDATION PROPOSAL — AUDIT CYCLE (filed 2026-07-03; [APPROVED
  BY HUMAN 2026-07-04 — applied same day, one docs PR with audit
  Findings 1+2: clause in SESSION BUDGET, register at top of
  experiments.md, both trigger sentences trimmed to pointers]).** Three periodic hygiene duties live in three places:
  (1) DEAD CODE POLICY's staleness sweep ("fall-through reaches the
  research tier and the codebase hasn't had a staleness audit in 30+
  days"); (2) CONSTITUTIONAL HYGIENE's rule audit ("monthly, or as a
  fall-through action when 30+ days since last review"); (3) the
  December market_calendar year-add (FROZEN PATHS exception + KNOWN
  STATE note). PROPOSED AFTER-TEXT — one clause appended to SESSION
  BUDGET, replacing neither policy body (only the scattered TRIGGERS):
  "AUDIT CYCLE: when a session's fall-through reaches the research
  tier, check the audit register at the top of research/experiments.md
  {audit · cadence · last run}: staleness audit (code/deps/config/
  expired adapters — 30d; DEAD CODE POLICY governs), constitutional
  audit (rules — 30d; CONSTITUTIONAL HYGIENE governs), market_calendar
  year-add (December; FROZEN PATHS exception governs). Run the most
  overdue one and update the register." Preserves: every cadence and
  both policy bodies verbatim. Drops: nothing. Resolves: three triggers
  nobody checks in one place; also supersedes the first audit's
  Finding-2 sketch (PERIODIC AUDITS register) with a concrete location.
  If approved: one docs PR adds the clause + the register, and trims
  the two in-place trigger sentences to point at it.

- **Satellite AIS (mid-ocean vessel coverage) — PRICED, deferred; filed
  per the ships directive 2026-07-04.** Verified: our aisstream.io
  subscription is already configured GLOBAL (BoundingBoxes ±90/±180),
  so the coverage gap is physical, not configuration — aisstream
  aggregates TERRESTRIAL receivers, which see ~40-60nm offshore; ships
  mid-ocean go dark between coasts. BUILD-FIRST analysis: (1) the raw
  material (satellite AIS downlink) is inaccessible free — genuinely
  paid class per the EDGE DOCTRINE; (2) accumulation helps at the
  EDGES: our archive records port arrivals/departures + coastal
  transits, which is where R2 transit-analytics value concentrates;
  (3) inference substitute EXISTS for specific questions: a ship that
  left port A heading for port B (destination field) can be
  dead-reckoned mid-ocean and confirmed on coastal reacquisition —
  label as predicted track, never ground truth. (4) Paid adds: true
  mid-ocean positions. Vendors: Spire Maritime, Kpler/exactEarth,
  ORBCOMM — pricing is quote-only, entry commonly $500+/mo class.
  RECOMMENDATION: do not buy unless a specific gated signal needs
  mid-ocean truth (none does today; port-transit signals don't).

- **[APPROVED BY HUMAN 2026-07-04 — applied same message]** DESIGN.md
  amendment: SELF-SEE RULE — "UI changes must verify their own
  rendering: after any change to a panel or overlay, the harness
  screenshots must show ALL registered content reachable (visible or
  behind an on-screen expand control) at all three widths. A component
  that exists in code but can't be reached on screen is a failed
  build." Enforcement shipped in visual_check.mjs (SELF-SEE block) and
  proven against the actual defect by A/B (old CSS -> harness FAILS
  with "panel bottom past viewport"). Bookkept per the amendment rule.
