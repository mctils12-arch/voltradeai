# datacore/ — the spinout-ready data layer

Constitution: CLAUDE.md → KNOWN STATE → SPINOUT-READY DATA LAYER
(human-approved 2026-07-03).

## Boundary rules (non-negotiable)

1. **No trading knowledge.** Nothing in `datacore/` imports from or knows
   about trading logic (`bot_engine.py`, `system_config.py`, `strategies/`,
   `server/bot.ts`). Pipelines here produce *data and signals*; consumers
   decide what to do with them.
2. **API boundary only.** Signals and overlay data are exposed exclusively
   through the `/api/data/*` routes (`server/routes.ts` reads from
   `datacore/` outputs). The bot consumes signals the same way an external
   API customer would. The site frontend never calls external data sources
   directly — everything flows through the boundary.
   (One scoped exception: the map's base imagery *tiles* load from the tile
   CDN directly in the browser — raster tiles are static imagery with
   attribution, not overlay data, and proxying them would only add cost
   and latency.)
3. **RAW vs SIGNAL labeling.** Every product surface layer is labeled:
   - RAW-DATA OVERLAY — displayed as-is with source attribution, no
     predictive claim, ships without ladder gating.
   - SIGNAL — an interpreted reading (tank-fill %, yard change, flow
     anomaly). May not appear on the surface before passing ROOT
     VALIDATION LADDER gate 2 (statistical predictive power).
4. **Ladder discipline.** Every pipeline records its ladder gate status
   here and its experiments in `research/experiments.md`. Failed roots log
   their layer of death. No gate-skipping.

## Layout

- `sites/` — static reference data (strategic sites: tank farms, mills,
  ports) with coordinates, metadata, and sources. RAW.
- In practice every pipeline shipped so far (aircraft, vessels, position
  archive, `edgarForm4`) lives directly in `server/*.ts` as a pure,
  independently-testable module exposing fetch + cache + gate status —
  the `pipelines/` Python package layout below was the original plan but
  was never followed; documented here for history, not as the live shape.
  Original plan: `pipelines/` — one module per data root (future:
  sentinel2_tanks, adsb_flows, ais_vessels, trends_demand), each exposing
  `fetch()` (raw), `latest()` (cached most-recent), and gate status.
- Node-side serving lives in `server/routes.ts` under `/api/data/*` — thin
  proxies/caches over this package's outputs. Keep them dumb.
- `server/edgarForm4.ts` — SEC EDGAR Form 4 (insider transactions) pipeline.
  No API key required. Ladder gate 1 (DATA) PASSED: the dependency-free XML
  parser was verified field-by-field against two real, live-fetched Form 4
  filings (see `server/edgarForm4.test.ts`). Polls the public "getcurrent"
  feed on a 15-min background timer started at boot, dedupes the feed's
  filer/issuer entry pairs by accession number, and serves the latest N
  parsed filings at `/api/data/insider` as a RAW-DATA overlay (as-filed
  display, no predictive claim). Gate 2 (does insider-buy clustering predict
  forward returns better than base rate) is open — see
  research/open_questions.md.
- `server/datacoreArchive.ts` — the position archive (MAP V2 ROADMAP R1;
  ARCHIVE EVERYTHING amendment): adaptive thinning (full resolution near
  strategic sites and low-altitude flight / near-port vessels; sparser
  oceanic cruise; per-entity cadence clocks), hourly JSONL under
  `${DATA_DIR}/datacore_archive/` (kinds: aircraft, vessels, trains), gzip after 2h, 7-day raw retention with
  rollup into per-entity daily track summaries (bbox + coarse polyline).
  Growth estimate at current thinning: ~100MB/mo combined (volume watch in
  wishlist; observe via `/api/data/archive/stats`). Recent trails served
  at `/api/data/track/:kind/:id` (powers the map's click-through history).
  Raw material for the archive-enabled signal hypotheses in
  open_questions.md and R2 maritime transit analytics.
  (Supersedes the parallel-built `dataArchive.ts` from PR #107 — uniform
  30-min sampling didn't meet the amendment's adaptive-thinning and
  compression requirements; its growth-estimate discipline was adopted.)

## Spinout trigger (decided by the human)

A root passes ladder gate 2 AND (external demand exists OR processing
needs dedicated infrastructure). Until then: one loop, one repo.
