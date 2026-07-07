# ORBITAL PROGRAM — full satellite population: visualization + data layer + cross-system ties

INSTALLED 2026-07-07 by human directive (verbatim intent preserved).
Multi-session program like GRID VISION / ANALYST CONSOLE / SCALE;
RESUME STATE at the bottom is authoritative. CLAUDE.md governs HOW
everything ships; this names WHAT the orbital program builds toward.

## THE DATA-PATH REALITY (lead with this — it splits the whole program)

CelesTrak is FIREWALLED from Railway (R17: connect timeouts from our
datacenter IP range; verified, filed). So the program splits cleanly by
where the TLEs are fetched:

- CLIENT-FETCH PATH — UNBLOCKED, ships now. The visitor's BROWSER can
  reach celestrak.org (browser IPs, not Railway's). The hero-globe
  visual and the /data satellite layer fetch TLEs + SATCAT client-side
  and propagate with satellite.js (SGP4) in the browser. No server
  dependency, no firewall problem. This is the bulk of the "whole sky
  alive" experience and it is not blocked by anything.
- SERVER-ARCHIVE / CROSS-TIE PATH — BLOCKED on the relay decision. The
  orbit-HISTORY archive and any cross-system tie that needs TLEs
  server-side (entity-graph joins computed server-side, pass-over
  prediction as a served signal, launch-event detection) need TLEs on
  the server, which the firewall blocks. This waits on the
  session-relay-ingest decision already sitting in wishlist.md (or the
  ties compute client-side where feasible).

Honest consequence: the beautiful visualization is unblocked; the
archive + server-computed signals are relay-gated. Do not claim the
archive is accumulating until the relay lands.

## DATA SOURCES + LICENSING

- CelesTrak GP (orbits) — verified clean (R17 licensing record): freely
  available, courtesy limits (2h update cycle). Full active catalog
  GROUP=active (~10k+ objects). OMM JSON format (NOT TLE — 5-digit
  catalog IDs exhausted ~2026-07-12; JSON/OMM is the mandated format).
- CelesTrak SATCAT (metadata) — satcat.csv: name, NORAD ID, int'l
  designator, operator/owner, country, launch date, object type
  (PAYLOAD/ROCKET BODY/DEBRIS), operational status, orbit class, RCS
  size. Free, same source. Join to GP on NORAD ID. ~few MB.
- Space-Track (full ~25k+ catalog incl. all debris/analyst objects) —
  authoritative but requires a FREE ACCOUNT + a USER AGREEMENT that
  restricts redistribution. NOT used in v1. Filed as a gated extension:
  its own licensing review before any use; CelesTrak's ~10k active set
  is the clean no-account source for the first build. Debris-completeness
  beyond CelesTrak's sets is the only thing Space-Track adds.

## HONESTY / PROVENANCE (standing rules, applied)

- REAL POSITIONS ONLY — SGP4 propagation of real TLEs. Never a faked or
  interpolated-beyond-validity orbit.
- FRESHNESS chip per object: TLE epoch age. Orbit uncertainty grows with
  epoch age — flag TLEs older than a stated threshold as "stale orbit".
- DECAYING / UNCERTAIN orbits FLAGGED not hidden: low perigee / high drag
  (decay risk) and stale-epoch objects carry a visible caveat.
- NO SILENT DROPS: if the render decimates, the count shown vs total is
  stated (the platform's no-silent-caps rule).

## PERFORMANCE (the hard constraint — 10k+ moving objects)

- MANDATORY GPU/WebGL. NOT DOM markers, NOT 10k MapLibre symbol features
  with per-frame position rewrites. The population renders as GPU points
  (instanced), positions updated from a Web Worker running SGP4 off the
  main thread (propagate at ~1-2 Hz, interpolate between; propagate the
  full set for the globe, the visible set at high zoom).
- KEY TECHNICAL DECISION (O1 spike answers it): a custom MapLibre
  CustomLayer with raw WebGL/regl (no heavy new dep) vs deck.gl
  ScatterplotLayer (built for millions of GPU points, but a real
  dependency). Pick on evidence from the spike.
- DECIMATION: globe/low-zoom shows the full population as points (cheap
  on GPU); high-zoom resolves individual clickable satellites + labels.
  If 10k can't stay smooth even as points, render full-set-low-detail
  and progressively resolve on zoom — NEVER drop objects silently.
- PERF HARNESS GATES IT at 390/768/1440 (mobile-flawless rule). The
  shipping bar: smooth frame budget with the FULL population on screen.

## BUILD ORDER (each slice its own PR; O1 gates the rest)

- O1 — FEASIBILITY SPIKE (T-CLIENT, gating). Prove 10k+ SGP4-propagated
  points render smoothly on the globe: choose the render path (custom
  WebGL layer vs deck.gl), Web-Worker propagation, perf-harness-gated at
  3 widths. Nothing visual ships until this passes with the full
  population. Report the frame numbers.
- O2 — HERO-GLOBE VISUAL (T-CLIENT). The full population orbiting in real
  time on the landing globe, matching the existing globe-symbol
  aesthetic — the "command-center whole sky alive" look. Client-fetch
  TLEs, worker SGP4, the O1 render path.
- O3 — /data SATELLITES LAYER (T-CLIENT + tiny server). Toggleable,
  GPU-rendered; click a satellite → panel: name, NORAD ID, type,
  operator/country, launch date, altitude, orbit class (LEO/MEO/GEO),
  status (GP + SATCAT joined on NORAD ID, client-side). Orbit-class +
  type filters; freshness chip; decay/stale honesty flags. Registry
  layer like the others; perf-gated.
- O4 — ORBIT-HISTORY ARCHIVE (SERVER, RELAY-GATED). The server stream
  (built, W2) + relay ingest → daily orbit-history accumulation; unblocks
  the server-computed ties. Waits on the relay decision.

## CROSS-SYSTEM TIES — each filed as its own hypothesis (open_questions.md)

HONEST FRAMING (per the directive): "visualization + brand value is a
legitimate justification on its own." Below, each tie is labeled REAL
SIGNAL / OPERATIONAL VALUE / SHOWCASE. Most orbital ties are showcase or
operational, not trading alpha — stated plainly, no manufactured signal.

(a) SENTINEL-2 / EO PASS-OVER PREDICTION over our monitored sites —
    OPERATIONAL VALUE (strongest real tie). From EO-satellite TLEs
    (Sentinel-2A/2B, Landsat 8/9), compute the next overpass time over
    each site (ports, grid, strategic sites) → "next fresh-imagery
    opportunity here". Directly feeds the tank-fill / imagery-acquisition
    workflow (know when to expect a new usable scene). NOT a trading
    signal — an operational utility that makes the imagery pipeline
    smarter. Real. Ladder: n/a (operational), but validate overpass
    predictions against actual scene timestamps (gate-1-style accuracy
    check).
(b) ENTITY GRAPH via operator→company→ticker — REAL STRUCTURAL / SHOWCASE
    value. SATCAT operator/owner → company → public ticker where it
    exists (launch: RKLB, ASTS; comms: VSAT, IRDM, GSAT; defense/gov:
    LMT/NOC/BA + non-public). A company lookup shows its orbital
    footprint beside its jets/vessels/plants — the Everything Graph
    connective tissue. Structural value is real. TRADING signal is
    SPECULATIVE and slow (constellation size changes over months) —
    gate it separately; do not claim alpha from a slow-moving count.
(c) LAUNCH ACTIVITY as an event overlay — SHOWCASE + WEAK SIGNAL. New-
    object appearances (fresh launches) tied to launch-provider/operator
    companies, cross-referenced with the entity graph + news. Honest:
    launches are largely pre-scheduled/announced (efficient, low
    surprise) so cadence is weak alpha; launch FAILURES / cadence
    anomalies are the only plausibly-tradeable edge and are rare. Value
    is mostly a real event overlay on the graph, not a signal. Gate
    honestly.
(d) GEO COMMS as an infrastructure layer — SHOWCASE. GEO comms sats as
    the space tier of the global-infrastructure map alongside grid +
    vessels + plants. Pure visualization/completeness value; no trading
    claim. Legitimate on brand grounds per the directive.
(e) OTHER TIES FOUND (honest):
    - Reentry/decay events overlay — minor real event (a decaying asset),
      showcase; no market relevance.
    - Conjunction/collision-risk near active assets — a real space-domain
      signal but hard and not market-relevant. Note, don't build.
    - Comms coverage over monitored sites (which sat covers a port) —
      tenuous; parked.
    VERDICT stated plainly: (a) is the one genuinely useful operational
    tie; (b) is real structure + showcase; (c)/(d)/(e) are showcase or
    weak. NONE is strong trading alpha — and that is acceptable here.

## GENERAL PRINCIPLE — CROSS-SYSTEM INTEGRATION (human-directed 2026-07-07)

Every system ties into the others where it adds real value; the ENTITY
GRAPH is the connective tissue. No isolated silos. When adding or
proposing ANY stream/feature, assess and wire its cross-system links and
file them. NEVER fabricate a tie that isn't real — state honestly which
links are real signal vs operational vs pure showcase. (Recorded in
KNOWN STATE; woven into every future build's assessment.)

## RESUME STATE (update every session that touches this program)
- 2026-07-07: charter installed. NOTHING BUILT YET (the server satellite
  stream W2 exists but is firewall-blocked on prod). NEXT: O1 feasibility
  spike (10k+ GPU points smooth on the globe) — the gating unknown.
  Cross-tie hypotheses filed in open_questions.md. Server-tie slices
  (O4 + server-computed ties) blocked on the CelesTrak relay decision
  (wishlist).

## MASTER-BUILD EXTENSION (human directive 2026-07-07 — fidelity tiers + coverage geometry)

Extends the base program above. Two additions: honest per-design-class
FIDELITY, and real COVERAGE/FOOTPRINT geometry.

### FIDELITY BY DESIGN CLASS (honest tiers — same discipline as the aircraft silhouette)

- CONSTELLATIONS (Starlink, GPS, Iridium, OneWeb, Planet…): thousands
  of near-identical units → model each DESIGN/GENERATION ONCE from real
  published GROUND imagery/specs (photos taken ON EARTH before launch —
  display units, press kits, spec sheets), then GPU-INSTANCE that one
  real model across every member at its true SGP4 position. Honest (they
  genuinely ARE that design) and efficient (one model, thousands of
  placements). NEVER photograph satellites in orbit; NEVER invent a
  model to fill a gap.
- NAMED UNIQUE (ISS, Hubble…): their own real 3D model from published
  imagery.
- UNPHOTOGRAPHED (classified payloads, random debris): honest SYMBOLIC
  markers typed by class (payload / rocket-body / debris), sized/styled
  by type. No fabricated model.
- The layer info STATES the fidelity/coverage boundary honestly (how many
  designs have real models vs symbolic-only).

### THE LOD REALITY (honest engineering truth — affirms the "progressive resolve" instruction)

You cannot render 10k detailed models (or splats) at once — that melts
any GPU, phone first. So detail is a FOCUS asset, not a whole-sky asset:
- GLOBE / FAR ZOOM: the full population renders as GPU points/billboards
  (the O1 render path). This is the "whole sky alive" look. Every object
  present, none dropped.
- FOCUS / ZOOM-IN / CLICK: the focused (and nearby) satellites resolve to
  their real design-class 3D model, live-rotating on true orbital data,
  with the full detail panel. This is exactly the directive's "render all
  at low detail, progressively resolve on zoom, never drop silently."

### GAUSSIAN SPLAT — HONEST SCOPE (the one place to reality-check the ambition)

4D Gaussian splats are real but the heaviest, most speculative slice:
they need (1) multi-angle GROUND imagery of a given design (exists for
marquee units like Starlink — display models, published photos), (2) an
OFFLINE photogrammetry/splat-training step producing a .splat/.ply asset,
(3) an in-browser splat renderer (a real dependency, used ONLY in the
focus/detail view — never for the 10k population). HONEST RECOMMENDATION:
standard glTF 3D models get ~90% of the "premium per-design detail" value
at a fraction of the cost/risk and ship first; a splat is a MARQUEE
UPGRADE for one or two designs (Starlink) IF the ground imagery + pipeline
pan out. Do not promise splats across the board — gate them behind a
dedicated pipeline spike after the base fidelity ships.

### COVERAGE / FOOTPRINT ANALYSIS (real spherical-cap geometry — genuine value)

From each satellite's true position + altitude + published beam/elevation
mask, compute the ground-coverage cone (a spherical cap on the earth).
Real math, client-computed from SGP4. Tools:
- STARLINK COVERAGE + BLACKOUT: union of all Starlink footprints (~25°
  min-elevation mask, public) → where covered / where the gaps are, live.
- GPS GEOMETRY over a point: which GPS sats are above the horizon + their
  geometry → DOP (dilution of precision) affecting positioning accuracy.
- EO VISIBILITY / NEXT PASS: which imaging sats can see a monitored site
  and when the next pass is (this IS cross-tie (a) — the imagery workflow
  feed).
- "WHAT'S OVERHEAD / WHAT COVERS THIS SPOT NOW": a general map query.
HONESTY: beam/elevation parameters are published where known (Starlink
25°, GPS horizon) and labeled ESTIMATED where they are inferred — never a
guessed cone presented as exact.

### EXTENDED BUILD ORDER (folds into O1–O4)
- O1 (running) base 10k GPU-points perf spike — still gates everything.
- O2 hero-globe points population · O3 /data clickable layer + metadata
  (as above; detail view resolves the design-class model on focus).
- O5 (NEW) DESIGN-CLASS MODEL LOD: per-design glTF 3D models, GPU-
  instanced, LOD (points far → model near). Standard 3D first.
- O6 (NEW, gated) SPLAT PIPELINE: marquee-design 4D splat from ground
  imagery + in-browser splat renderer for the detail view only. Own
  spike (imagery sourcing + pipeline + renderer dep) before any build.
- O7 (NEW) COVERAGE/FOOTPRINT TOOLS: spherical-cap footprint geometry →
  Starlink coverage/blackout, GPS DOP, EO visibility/next-pass (ties to
  cross-tie (a)), "what's overhead here now" query. Client-computed.
- O4 orbit-history archive + server-computed ties — RELAY-GATED.
COVERAGE-TOOL VALUE (honest labels): EO next-pass = OPERATIONAL (imagery
feed); GPS-DOP + Starlink-coverage + what's-overhead = SHOWCASE + real
utility, no trading alpha claimed.

## O1 SPIKE RESULT (2026-07-07) — FEASIBLE with wide margin; the render approach is LOCKED

Durable record of the O1 feasibility spike (workstream b). Reference
implementation lives in the session scratchpad (orbital-spike/) — the
load-bearing files to PORT into client/src for O2/O3 are
`render_harness.html` (the globe-aware custom WebGL point layer) and
`sat_worker.js` (the Web-Worker SGP4 loop). If a future session lacks
the scratchpad, re-derive from this record.

VERDICT: 10k+ live satellites on the /data MapLibre globe passes every
perf gate at 390/768/1440 with large headroom; the point-field stays
smooth to ~100k objects (no decimation of the dots needed).
Progressive-resolve-on-zoom is only for per-satellite DETAIL (labels,
orbit lines, click targets), never for keeping the field smooth —
honors never-drop-silently. Mobile 390px is the EASIEST width.

LOCKED APPROACH (O2/O3 build to this):
- Render: MapLibre v5 CustomLayer (renderingMode "2d"), instanced
  gl.POINTS, projected via the v5 shaderData vertexShaderPrelude
  projectTile / projectTileFor3D (globe + mercator + ALTITUDE — bind
  the 5 defaultProjectionData uniforms). Altitude via
  projectTileFor3D so LEO/MEO/GEO shells are visually distinct (GEO at
  ~36,000km = large globe radius). NO deck.gl (rejected — 200KB+ for a
  problem the custom layer clears with margin).
- Propagation: satellite.js (NEW DEP, ~11KB gz, WORKER-ONLY so it
  never enters the main bundle) in a dedicated Web Worker; propagate
  all ~10-16k objects to now at ~1 Hz; post a transferable
  Float32Array(N*3) of lon/lat/alt; the vertex shader interpolates
  along track velocity between ticks for smooth 60fps. 10k propagates
  in ~29ms (V8); worker keeps the main thread free (proven).
- Picking: custom layers have no queryRenderedFeatures → CPU
  nearest-point lookup against the last propagated Float32Array
  (trivial at 10k).
- TLE plumbing: CLIENT-SIDE fetch (CelesTrak firewalls Railway, R17);
  real GROUP=active pull = 15,932 objects.
- Landing hero globe is a SEPARATE canvas (D3/topojson, not MapLibre)
  — satellites there is its own decision, not a drop-in of the /data
  layer.
DEP DECISION FOR THE PARENT AT INTEGRATION: `npm i satellite.js`
(worker-only import). No other new deps.

RESUME-STATE UPDATE: O1 DONE (feasible, approach locked). NEXT after
the parallel workstreams (a data pipeline, d geometry, e-entity, c
model research) land + serial-integrate: O2/O3 build ports the spike
reference code into a real MapLibre custom layer + worker + the /data
satellites layer + click detail UI. O7 coverage tools consume the
geometry engine. O4 + server ties remain relay-gated.
