# UI Integration — what changed, what broke, what to watch

## Summary

The standalone marketing landing (`landing/`) has been removed. The new design
is now a React route at `/` inside the existing app. The dashboard (Analyze,
Scanner, News, Watchlist, AI Engine bot) moved from `/` to `/app`, and login
moved to `/login`. The Tailwind theme has been re-skinned from the old
"tactical military" palette (cyan #00e5ff on near-black) to the new
"VolTradeAI" palette (electric blue #4d9fff on deep navy #050a13) with
semantic finance accent colors (green/red/purple/amber).

## Routes

| URL              | What                                                    |
|------------------|---------------------------------------------------------|
| `/`              | New marketing landing page (LandingPage component)      |
| `/login`         | Login page (standalone)                                 |
| `/app`           | Dashboard shell — defaults to Analyze tab               |
| `/app/...`       | Dashboard with sub-routes / hash tabs                   |
| Anything else    | Falls through to landing                                |

## Files changed

### Deleted
- `landing/` (entire folder — old standalone HTML/CSS/JS landing)

### Replaced
- `server/static.ts` — removed `/bot` landing serving; React app now owns all
  routes and the SPA fallback returns `index.html`.
- `client/src/App.tsx` — wired `wouter` routing for `/`, `/login`, `/app`.
- `client/src/index.css` — first ~7000 chars (theme block) rewritten with the
  new navy + cyan + semantic palette. `@layer base` and utility classes
  preserved as-is.
- `client/src/pages/home.tsx` — Logo component changed from SVG to text-based
  `vol·trade·/AI` mark.
- `client/index.html` — Inter+JetBrains font links → Geist+Geist Mono.
- `tailwind.config.ts` — `font-sans` resolves to Geist first.

### Added
- `client/src/pages/landing.tsx` — React component that mounts the new
  marketing landing.
- `client/src/pages/_landing_styles.css.txt` — landing CSS (scoped to
  `.vt-landing-root` so it can't leak into the dashboard).
- `client/src/pages/_landing_body.html.txt` — landing body markup, with nav
  Sign-up/Log-in links rewritten to `/login`.
- `client/src/pages/_landing_script.js.txt` — landing JS (world map +
  particles + ticker), runs once after D3 + topojson load from CDN.

### Color migration (333 replacements across 11 files)
Hardcoded tactical colors replaced everywhere in `client/src/`:

| Old (tactical)  | New (VolTradeAI)  | Meaning            |
|-----------------|-------------------|--------------------|
| `#00e5ff` cyan  | `#4d9fff` blue    | Primary accent     |
| `#33ecff`       | `#7cc4ff`         | Brighter accent    |
| `#00ff41` neon  | `#4ade80` emerald | Gain / positive    |
| `#ff3333` red   | `#ff5a6e` coral   | Loss / negative    |
| `#d4a017`       | `#fbb24c` amber   | Warn / highlight   |
| `#a855f7`       | `#c084fc` lavender| AI / Pro tier      |
| `#050a12` bg    | `#050a13` navy    | Background         |
| `#0a0e17`       | `#0a1628`         | Secondary bg       |

## What this does NOT do

- **Page layouts inside the dashboard are unchanged.** Analyze, Scanner, News,
  Watchlist, and Bot pages still have their old layouts. Only the colors,
  fonts, and chrome around them have been updated. Restyling each page to
  feel cohesive with the marketing landing is a separate pass.
- **No light theme.** `:root` and `.light`/`.dark` all currently render the
  same dark navy palette. Adding a real light theme later requires writing a
  new set of HSL tokens in the `.light` block.
- **Shadcn components inherit the new theme automatically** because they read
  HSL values from `--primary`, `--accent`, `--card`, etc. — which are all
  pointing at the new palette now. But complex components (charts, data
  tables) may still have hardcoded tints in their own component files; those
  will need spot fixes.

## How to deploy

1. `npm install` (no new dependencies — `wouter` was already in
   `package.json`).
2. `npm run build` — produces `dist/`.
3. `npm run start` — serves the bundled app via Express on the configured port.
4. Push to GitHub. Railway picks it up via `railway.toml` / `Dockerfile`.

## Smoke tests after deploy

- [ ] `/` renders the new landing (world map, particles, NVDA mockup).
- [ ] `/` does NOT carry navy `vt-landing-root` styles into the dashboard.
- [ ] Click "Sign up free" → routes to `/login`.
- [ ] Log in → lands on `/app` showing the Analyze tab.
- [ ] Tabs (Scanner, News, Watchlist, Bot) load the same way they did before.
- [ ] Auth state persists across reloads.
- [ ] Mobile: landing is responsive (hero collapses to single column under 980px).

## Rollback

If something breaks in production, the cleanest rollback is `git revert` of
this commit — the old landing folder, App.tsx routing, and theme were all
self-contained and removing them all together returns the app to its prior
state.
