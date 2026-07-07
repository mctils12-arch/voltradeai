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
