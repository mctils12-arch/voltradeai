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
- `pipelines/` — one module per data root (future: sentinel2_tanks,
  opensky_flows, ais_vessels, edgar_form4, trends_demand). Each exposes
  `fetch()` (raw), `latest()` (cached most-recent), and gate status.
- Node-side serving lives in `server/routes.ts` under `/api/data/*` — thin
  proxies/caches over this package's outputs. Keep them dumb.
- `server/datacoreArchive.ts` — the position archive (MAP V2 ROADMAP R1;
  ARCHIVE EVERYTHING amendment): adaptive thinning (full resolution near
  strategic sites and low-altitude flight / near-port vessels; sparser
  oceanic cruise; per-entity cadence clocks), hourly JSONL under
  `${DATA_DIR}/datacore_archive/`, gzip after 2h, 7-day raw retention with
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
