# DESIGN.md — binding design standard for all user-facing work

This is not advisory. Every PR that touches `client/` is bound by this
document and by CLAUDE.md promotion rule 6 (the visual-harness rule).

## Core principle

EVERY page must look and function properly at BOTH phone and desktop
sizes. Neither is the afterthought. The primary human reviews production
on a Galaxy S24 — if it fails on the phone, it failed.

## Canonical test widths

| Width | Device class | Input model | Notes |
|-------|--------------|-------------|-------|
| 390px | Phone (Galaxy S24 class) | Touch | Collapsed controls by default |
| 768px | Tablet / split-screen | Mixed | Spot-check: nothing broken between the extremes |
| 1440px | Desktop | Mouse + keyboard | Full-featured: hover, tooltips, keyboard |

The visual harness (`npm run visual`) renders key pages headless at all
three widths, saves screenshots to `.visual/`, and runs mechanical checks.
Review your own screenshots against this document BEFORE opening the PR.

## Requirements at every width

1. **Full-viewport immersive pages** (the /data map and anything like it):
   100% width, full remaining height below the nav — no letterboxing, no
   fixed-size containers, no page scroll caused by the page itself.
2. **No permanent overlays covering content.** Controls collapse. Phone:
   collapsed by default. Desktop: may default open where space allows.
3. **Alive on first load.** Pages show live content with zero clicks.
   Default-on layers/data, never an empty canvas behind a toggle.
4. **Touch targets ≥ 44×44px on phone.** Desktop gets hover states,
   tooltips, and keyboard support (at minimum: zoom/pan keys on the map,
   Escape closes panels/popovers).
5. **Theme tokens only.** All UI uses the site's existing dark theme and
   typography (canonical values below). No hardcoded off-palette colors,
   no default-library styling visible (a page must look like
   voltradeai.com, never like a library demo).
6. **Every interactive element** has hover (desktop), active, and loading
   states.
7. **Empty / error / awaiting-key states are designed, not accidental.**
   If a feed is down or a key is missing, the user sees an intentional,
   styled state that says what's happening and what (if anything) to do.
8. **Attribution and legal lines** are present but unobtrusive (small,
   tertiary color, never overlapping controls).

## Canonical theme tokens (extracted from client/src/index.css — the
single source of truth; if index.css changes, update this table)

| Token | Value | Use |
|-------|-------|-----|
| `--bg-primary` | `#050a13` | page background |
| `--bg-secondary` | `#0a1628` | raised surfaces |
| `--bg-card` | `rgba(15, 29, 51, 0.6)` | cards/panels |
| `--bg-card-hover` | `rgba(20, 37, 67, 0.75)` | card hover |
| `--text-primary` | `#eef3fb` | headings, primary text |
| `--text-secondary` | `#b3c2d8` | body text |
| `--text-tertiary` | `#6680a0` | captions, attribution |
| `--accent` | `#4d9fff` | primary accent, active toggles |
| `--accent-bright` | `#7cc4ff` | accent hover |
| `--accent-green` | `#4ade80` | positive / live status |
| `--accent-red` | `#ff5a6e` | negative / error status |
| `--accent-orange` | `#fbb24c` | warning / awaiting status |
| `--accent-purple` | `#c084fc` | special categories |
| `--font-body` | `'Geist', -apple-system, system-ui, sans-serif` | all body text |
| `--font-mono` | `'Geist Mono', ui-monospace, 'JetBrains Mono', 'SF Mono', monospace` | numbers, data, code |

Layout constants: desktop top nav is fixed, ~56px tall. Mobile
(≤639px, the site breakpoint): top bar + a fixed 64px bottom tab bar —
full-viewport pages must end above it. Prefer `100dvh` over `100vh` on
mobile so browser chrome doesn't cause overflow.

## Map-specific rules (the /data product surface)

- Layer controls live top-right (standard map UX), as a compact button
  that expands; collapsed by default on phone.
- Every layer row: a toggle switch, live status (active / awaiting key /
  loading / error), and a count badge when active ("1,204 aircraft").
- The RAW DATA / SIGNAL distinction is a small info-icon tooltip, not a
  permanent paragraph.
- A legend explains marker colors/types.
- Loading skeletons over blank canvases while tiles/data fetch.
- Detail interactions: site markers open a clean detail card (name, type,
  what it's watching and why); aircraft/vessels open compact popovers.

## Self-see rule (human-approved 2026-07-04)

UI changes must verify their own rendering: after any change to a panel
or overlay, the harness screenshots must show ALL registered content
reachable (visible or behind an on-screen expand control) at all three
widths. A component that exists in code but can't be reached on screen
is a failed build.

Enforcement (visual_check.mjs SELF-SEE block): opens the panel via its
own control, expands every collapsed group, then asserts — panel bottom
inside the viewport; internal scrolling engaged when content overflows;
every layer in the registry has a reachable row; every toggle scrolls
into view and is hit-testable (nothing rendered on top of it). Precedent:
the 2026-07-04 defect — the panel's height constraint resolved against
an auto-height wrapper, rows below the fold were unreachable, and the
harness passed because it never asserted reachability.

v2.4 additions (human-approved 2026-07-04): with the panel OPEN, no map
control (zoom in/out, fullscreen, FAB) may be occluded at any test
width — same elementFromPoint hit-test as the toggle check (precedent:
the open panel covered the zoom buttons in production). Plus the armed
eternal-spinner check: any layer row in "loading" longer than 30s must
carry a designed note.

## Loading-state rule (human-approved 2026-07-04)

No loading state lives longer than 30s without resolving to a designed
status: retrying / source unavailable / awaiting key. Eternal spinners
are a failed build.

Implementation: every layer-status change is timestamped; a client
watchdog upgrades any bare "loading" older than 30s to an explicit
retrying note, and status notes RENDER on loading rows (precedent: the
2026-07-04 defect — the OWM "key activating, auto-retrying" note
existed in state but the panel dropped notes on loading rows, so
production showed a bare eternal spinner for the whole activation
window).

## Zero-cost-when-off (human-approved 2026-07-04)

A toggled-off layer must do no work — no tile prefetch, no websocket,
no polling, no render pass; initialization is lazy on first enable.
Enforcement: the harness loads the map with ALL layers off and asserts
zero layer-data API calls plus an interactive-time budget (the
regression guard for "the site got slower"). Heavy default-on layers
mount deferred, after the map's first idle.

## Imagery metadata honesty (human-approved 2026-07-04)

Where capture dates are available (Sentinel-2 scenes when that pipeline
lands; any tile source exposing dates), display "imagery as of [date]"
on the map. Where unavailable (Esri base tiles), say "date unavailable"
rather than implying freshness. Standing display rule: no imagery
surface may imply currency it cannot prove.

## Reference data accuracy (human-approved 2026-07-03)

REFERENCE DATA ACCURACY: raw reference data (site coordinates, facility
metadata) must be verified against imagery or an authoritative source
before shipping; RAW layers skip the signal ladder, not fact-checking.
Anything feeding future geofences gets coordinate verification
mandatory.

Tooling: `scripts/site_verify.py` (all sites from a JSON) and
`scripts/site_candidate_verify.py` (ad-hoc id/lat/lon/zoom quads) render
each coordinate on Esri World Imagery with a crosshair — the facility
must be visibly present (a port shows docks/cranes, a tank farm shows
tanks, a mill shows the plant). Requires `pillow` (session-local
install; not a runtime dependency). Precedent: the 2026-07-03 audit
found 11 of 16 strategic sites mispositioned (worst ~18km) — town
centroids and water instead of facilities.

## Performance budget (human-approved 2026-07-03)

- Map interactions stay smooth on mid-range phone hardware at 10k+
  features — rendering is canvas/WebGL layers (MapLibre native or
  deck.gl), never per-marker DOM; off-screen features are culled.
- Initial page interactive under 3 seconds.
- Live layers degrade gracefully: stale data with a visible timestamp
  beats a spinner; a failed feed shows its last-known state and when it
  was fetched, never an empty layer with no explanation.
- Heavy geo processing is CLIENT-side. The server (Railway Hobby plan)
  only proxies and caches feeds — one upstream request shared across all
  visitors, never per-visitor fan-out and never server-side geometry.

## Feature completeness checklist (human-approved 2026-07-03)

Every user-facing feature answers these BEFORE its PR opens — the human
should never discover scale, coverage, or failure-mode gaps in production:

1. Does it work at global scale, not just the demo region?
2. What happens when the feed fails, rate-limits, or returns partial
   data?
3. What does the user see on first load, on error, on empty?
4. Is heavy work client-side?
5. What are the data source's hard limits, and does the UI state them
   honestly (coverage gaps, update cadence, prediction vs. ground truth)?

## The harness (enforcement)

`npm run visual` → `scripts/visual_check.mjs`:
- Builds nothing itself — run `npm run build` first (it serves
  `dist/public` with mocked `/api/*` fixtures for determinism).
- Renders each key page at 390/768/1440, saves `.visual/<page>-<w>.png`.
- Mechanical checks per width: no horizontal overflow; map container
  fills the viewport region (marker: `[data-vt-map]`); no fixed/absolute
  element permanently covering >40% of the map; controls visible and not
  clipped; touch targets ≥44px at 390px.
- Exit code is nonzero on hard failures (use `--soft` for baseline runs).

Per CLAUDE.md promotion rule 6: PRs touching `client/` must include this
run at all three widths; the session reviews its own screenshots against
this document before opening the PR, and attaches or describes them in
the PR description.
