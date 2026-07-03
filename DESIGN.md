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
