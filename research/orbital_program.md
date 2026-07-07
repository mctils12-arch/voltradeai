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
