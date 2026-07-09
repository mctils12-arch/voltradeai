# ANALYST CONSOLE program ("WorldView, but with our goal") — charter

INSTALLED 2026-07-07 by human directive: "so the system we have i want
stuff like this https://youtu.be/rXvU7bPJ8n4 but with our goal overall"
— the video is Bilawal Sidhu's WorldView demo: a browser-based
Palantir-Gotham-style geospatial command center (3D globe, fused live
OSINT feeds — air traffic, satellite orbits, camera feeds — with an
AI-analyst experience on top). This charter maps that experience onto
VolTradeAI's mission. CLAUDE.md governs HOW everything here ships;
this file names WHAT the console builds toward. Multi-session program
like GRID VISION; RESUME STATE at the bottom is authoritative.

## What "with our goal" means (the non-negotiable tie to the mission)

WorldView is a demo of LOOKING. Our console is an instrument for
KNOWING: every pane is backed by our own archives (history nobody can
buy back), every number carries freshness/provenance/confidence chips
(the honesty machinery IS the brand), and every entity ties toward
tradable/sellable context (site → company → ticker → events). The
console is the front door of the data platform (Amendment 5:
experience is the door); the bot remains customer zero; nothing
predictive surfaces without its ladder gate. Focus stays on
INFRASTRUCTURE AND MARKETS — facilities, fleets, flows, grids — never
person-level surveillance.

## What we already have (audit, 2026-07-07)

/data map with 20+ live/atlas layers (aircraft, vessels, trains,
fires, weather, alerts, power plants + grid, strategic sites, port
dwell, shadow fleet, atlas set), JSONL archives behind every stream,
entity spine (aircraft→FAA owners→operators), site event timelines
(Everything Graph R1), /api/v1 product surface with key scaffolding,
landing-page 3D hero globe, DESIGN.md + visual harness (390/768/1440),
perf gates. This is most of WorldView's substance — what's missing is
the EXPERIENCE layer: 3D, time, cross-layer query, and the analyst.

## Build order (each W-item = own PR(s), serial merges, own log entry)

- W4 (FIRST — foundation, server-only, no keys): CROSS-LAYER QUERY
  ENGINE. One internal API: {bbox|point+radius, time window, layer
  set} → entities/events/counts from the archives + live caches, with
  per-layer provenance+freshness in the envelope. This is the muscle
  every later feature (dossiers, chat, alerts) calls. Deterministic,
  fully testable, ships tonight-grade.
- W2: SATELLITE ORBITS layer. CelesTrak GP/TLE (free, public),
  satellite.js propagation client-side; TLE snapshots ARCHIVED daily
  (orbit history = another accumulating asset). Filters: starlink /
  imaging / GPS / stations. Licensing check first (CelesTrak terms are
  permissive; verify + attribute).
- W1: 3D GLOBE MODE for /data. MapLibre globe projection (free, no
  key, current stack) as the default cinematic mode; graceful 2D
  fallback; mobile-flawless at 390px per the harness. OPTIONAL
  upgrade behind a key: Google Photorealistic 3D Tiles (has a free
  tier — wishlist entry for Mike; NOT required for the program).
- W3: TIME SCRUBBER ("4D"). Playback over the archives we already
  record (aircraft trails, vessel snapshots, fires, alerts, port
  events): pick a window, scrub, watch the world move. Pure archive
  readout — zero new data cost, impossible to fake, uniquely OURS
  (nobody else has our recorded history).
- W5: ENTITY DOSSIER v2. Click anything → one panel: identity (spine),
  cross-layer history (W4), related filings/contracts (EDGAR/
  USAspending streams), nearest strategic sites, ticker linkage where
  it exists. The Everything Graph, surfaced.
- W6 (THE CENTERPIECE — key-gated): THE ANALYST. Chat pane where an
  LLM with tool access to W4/W5 answers questions ("what changed at
  Cushing this week?", "which carriers' fleets flew less than usual
  before earnings?"), and drives the map (fly-to, toggle layers, draw
  results) via a typed command protocol. Server-side tool loop;
  ANTHROPIC_API_KEY lives in Railway only; ACTIVATES ON KEY DETECT
  like every other keyed stream. Every answer renders with source
  chips (layer + freshness + confidence); the model is CONSTRAINED to
  our data tools — no invented facts; "I don't have that" is a valid
  answer and renders honestly. Predictive-sounding output is blocked
  server-side unless the underlying signal is ladder-gate-2 validated.
- W7 (research-first): PUBLIC CAMERA/WEBCAM layer. Licensing decides:
  state DOT traffic cams (public), Windy webcams API (free tier,
  attribution). Build-first analysis files before any code.

## Paid/key boundary (BLOCKED-FOR-MIKE items, routed around meanwhile)

1. ANTHROPIC_API_KEY (Railway) — unlocks W6 the moment it lands.
   Estimated cost at hobby usage with a small model for the tool loop
   (Haiku-class) and short answers: single-digit $/month; heavier use
   or a bigger model: tens of $/month. Everything else in W6 (tools,
   protocol, UI, tests) builds WITHOUT the key.
2. OPTIONAL Google Maps Platform key for Photorealistic 3D Tiles
   (free monthly tier exists; card required) — cinematic city-level
   3D. MapLibre globe ships regardless; this is polish, not blocker.
3. OPTIONAL Windy webcams key (free tier) — W7, pending licensing
   research.

## Honesty rules specific to the console

- A layer that is stale renders as stale (freshness chip), never
  hidden. The analyst cites what it queried, with timestamps.
- RAW overlays vs SIGNALS labeling carries into every console pane
  and every analyst answer (existing standing behavior, restated).
- The analyst never fabricates an entity, number, or date; tool
  results are the only ground truth it may assert; refusal-to-guess
  renders as a first-class UI state.
- Person-level tracking is out of scope, permanently: entities are
  aircraft, vessels, facilities, companies, markets.

## RESUME STATE (update every session that touches this program)

- 2026-07-07: charter installed. Wishlist entries for the three keys
  filed same day.
- 2026-07-07 (same session): W4 SHIPPED (v1.0.195) —
  server/queryEngine.ts + /api/data/query. Six layers (aircraft,
  vessels, trains, fires, alerts, gauges — exactly the archives that
  exist), point+radius+window fold with per-layer provenance +
  freshness, caps stated, rejected_layers surfaced, LRU cache (5 min,
  50 entries, 0.05-deg rounding). Built by subagent, session-reviewed
  (read-before-write) before integration. FOLLOW-UP noted: add a
  concurrent-scan gate if /api/data/query sees real traffic (public
  endpoint, uncached scans are disk-heavy; mitigations today: LRU +
  coord rounding + Cache-Control 300s). NEXT: W2 satellite orbits
  (CelesTrak licensing check first), then W1 globe mode.
- 2026-07-07 (same session): W2 SERVER HALF SHIPPED (v1.0.196) —
  server/satellites.ts + /api/data/satellites + manifest. LICENSING
  VERIFIED (quotes in wishlist.md): CelesTrak data freely available,
  no redistribution restriction, courtesy limits binding (GP updates
  2h; our cadence 6h/group, non-200 never retried until next sweep,
  M2M rule implemented). Groups: stations, starlink, gps-ops, geo
  (charter's "active-geosynchronous" does not exist — geo is the real
  group). OMM JSON format (NOT TLE — CelesTrak's 5-digit catalog
  numbers exhaust ~2026-07-12; TLE would have broken within days).
  Archive = orbit HISTORY: dedup NORAD_CAT_ID|EPOCH, every epoch
  advance accumulates. Built by subagent, session-reviewed. NEXT: W1
  globe mode (T-CLIENT, visual harness at 390/768/1440) + the client
  satellite layer (satellite.js SGP4 propagation) — consider a
  field-projection param on the route first (starlink full payload is
  a few MB); then W3 time scrubber.
- 2026-07-07 (same session): W6 CLIENT SHIPPED (v1.0.201) — Analyst
  chat pane on /data (client/src/components/AnalystPane.tsx, lazy
  chunk = zero analyst code until first open, never polls). Third
  top-left control ([data-vt-analyst]); 390px bottom sheet / desktop
  side panel (clears controls + open layers panel; harness occlusion
  + self-see pass). All 8 server states rendered distinctly
  (awaiting_key / budget_exhausted / 429 / 401 sign-in / 400 / 502 /
  network / success). Success = answer + source chips (tool +
  freshness) + collapsible "how I got this" trace + tokens/budget/
  model footline. map_commands EXECUTE live: fly_to via map ref,
  toggle_layer via the SAME enabled-state the layer panel uses (no
  parallel state; honest "can't toggle" note for unwired/keyless
  layers). Harness ANALYST battery added (open/close/layout, asserts
  NO POST fires); [data-vt-analyst] in both occlusion lists; no
  assertion weakened. Built by subagent, session re-ran harness +
  reviewed screenshots. THE ANALYST CONSOLE FRONT-END IS COMPLETE:
  activates end-to-end the moment ANTHROPIC_API_KEY lands in Railway.
  NEXT: W2 client satellite layer, W3 time scrubber, W5 dossier v2.
- 2026-07-07 (same session): W1 SHIPPED (v1.0.197) — 3D globe mode is
  the /data map DEFAULT (MapLibre v5 native projection, zero new
  deps). Toggle stacked under fullscreen ([data-vt-globe],
  localStorage vt-map-globe), projection baked into the bootstrap
  style (first paint correct), zero-cost-when-off (flat pref = no
  projection API calls), degradation = mercator + disabled-with-reason
  toggle. Mobile default GLOBE kept on A/B evidence (390px globe
  median 33ms vs flat 17-33ms, overlapping p95, all inside gates). 19
  layers verified toggling clean in globe mode incl. rasters +
  hillshade (synthetic-DEM pixel test). Harness gains a GLOBE MODE
  battery (default assertion, round-trip, persistence, aria-pressed)
  + the toggle added to both occlusion hit-tests; no assertion
  weakened. Built by subagent; session re-ran the harness and
  reviewed screenshots (curved vs flat field A/B at 390 confirmed).
  NEXT: W2 client satellite layer (satellite.js SGP4 on the globe;
  field-projection param on the route first), then W3 time scrubber.
- 2026-07-07 (same session): W6 SERVER HALF SHIPPED (v1.0.199) —
  server/analyst.ts + POST /api/analyst (session-gated). Anthropic
  tool-use loop, ANALYST_MODEL default claude-haiku-4-5 (cheapest
  tier), 7 tools (query_window, satellites, nws_alerts, grid_stress,
  eu_load, site_timeline, map_command) — all cache/archive reads,
  port-dwell/shadow-fleet deliberately EXCLUDED (their exports
  trigger per-call archive scans, the R4/R5 hazard). Budgets enforced
  in code: 8 tool calls + 4 round-trips per question (force-close
  stated), ANALYST_DAILY_TOKENS/day persisted across deploys,
  2-concurrent cap. Key-gated awaiting_key honesty — ACTIVATES ON
  ANTHROPIC_API_KEY DETECT. Envelope: answer + tool_trace +
  map_commands + source chips + usage + budget. grid_stress carries
  predictive:false verbatim; system prompt forbids memory facts,
  requires citations, allows refusal. Every envelope key-scrubbed.
  Built by subagent, session-reviewed line-by-line. NEXT: W6 client
  chat pane (T-CLIENT, renders the envelope: answer + chips + map
  command execution; harness at 3 widths), then W2 client satellite
  layer, then W3 time scrubber.
- 2026-07-09: CROSS-PROGRAM NOTE — W2 (client satellite layer) is DONE,
  shipped under the separate ORBITAL program (research/orbital_program.md
  O1-O3, #359/#361/#363/#396) rather than logged here; that program's own
  RESUME STATE has the detail. Recorded here only because this file's own
  "NEXT" line never got updated — a MEMORY PROTOCOL miss worth naming so a
  future session doesn't re-plan already-shipped work.
- 2026-07-09 — [PRODUCT] W3 SHIPPED (v1.0.250): TIME SCRUBBER —
  server/queryEngine.ts's querySnapshot() (GET /api/data/snapshot) reads
  ONE archived hour (aircraft/vessels/trains) or day (fires/alerts/gauges)
  bucket exactly as recorded — reuses W4's LAYER_SOURCES and the SCALE S1
  viewport helper (bbox optional), zero new data cost, bounded to the same
  RAW_RETENTION_DAYS window, point-capped and stated never silent.
  client/src/components/TimeScrubber.tsx is the fourth top-left control
  (lazy chunk, zero-cost-when-off like the analyst pane): pick a layer,
  drag a slider across the past week (hour resolution), Play steps
  forward toward now, points paint as a distinct amber circle layer on
  the map with a "historical replay — not live" note, honesty status line
  (count/capped/off-screen-dropped/provenance). Visual harness battery
  added (open/geometry/SELF-SEE/occlusion/close, asserts >=1 GET fires —
  the deliberate mirror of the analyst battery's zero-POST assertion,
  since this panel SHOULD fetch on open); occlusion selector lists
  updated in all 4 places. Gates: tsc 64 baseline unchanged, test:node
  542/542 (18 new querySnapshot tests), pytest 583/1skip (no Python
  touched), build OK (TimeScrubber chunk 4.06kB gzip 1.87kB), visual
  --page data 3 widths. NEXT: W5 entity dossier v2 is the one remaining
  charter item.
- 2026-07-09 — [PRODUCT] W5 SHIPPED (v1.0.258): ENTITY DOSSIER v2 — click
  any entity, get identity + cross-layer Everything Graph neighborhood +
  ticker-matched USAspending contracts + nearest strategic sites, all in
  one card. Extends the EXISTING click-to-detail `Detail`/`vt-site-card`
  pattern (no new panel/button) rather than inventing new UX. Server:
  server/dossier.ts (new, pure function, every IO source injected) composes
  entityGraph.ts's already-built identity/neighborhood/filings join (no new
  join logic needed there — insider_of edges already ARE "related
  filings") with ONE new join (USAspending contracts by resolved ticker)
  and haversineKm-based nearest-sites (reused from firesFacilities.ts, not
  a 4th local copy). entityGraph.ts's graph cache lifted from a private
  routes.ts closure to shared `cachedGraphSync()`/`bootGraphPoll()` exports
  so the new /api/data/dossier route reads the SAME 15-min cache as
  /api/data/graph instead of triggering a second independent 168h-AIS-fold
  rebuild (avoids the R4/R5 hazard). Client: 7 of 9 entity kinds wired
  (site/powerplant/vessel get real graph entity ids; aircraft/train/fire/
  gauge/alert get lat/lon-only nearest-sites, honestly — they aren't graph
  nodes yet; satellite deliberately skipped, ground-proximity is
  meaningless at orbital altitude). Live-verified against real
  datacore/*.json (not just fixtures) post-build. Gates: tsc 66 baseline
  confirmed identical via git-stash A/B diff, test:node 533/533 (9 new),
  client libs 88/88, build clean, visual --page data 3 widths 0 hard
  failures. THE ANALYST CONSOLE CHARTER'S W1-W6 BUILD ORDER IS NOW FULLY
  SHIPPED. Two small honest gaps logged, not blocking: aircraft aren't
  graph nodes yet (no ticker linkage even for a publicly-traded FAA
  registrant); queryEngine.ts/siteTimeline.ts still carry their own local
  kmBetween duplicates of haversineKm (small future cleanup).
