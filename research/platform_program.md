# PLATFORM INTEGRATION PROGRAM — one coherent site: trading front-end + data/API product

INSTALLED 2026-07-11 by human directive. Multi-session program like
ORBITAL / GRID VISION / ANALYST CONSOLE. RESUME STATE at the bottom is
authoritative. CLAUDE.md governs HOW everything ships; this names WHAT
the program builds toward. Reads with VISION.md + GIP.md (the platform is
sold via API + subscriptions; the trading bot is customer zero).

## THE DIRECTIVE (human, 2026-07-11, paraphrased — verbatim intent preserved)

Integrate the site so the STOCK side and the DATA-WORLD side each live
under their own drop-down menus; streamline the whole site. Build out the
data side and the API we offer — "like Massive.com but in our own way":
real API, documentation, infrastructure, user accounts with billing
history, and customer features. There must be FREE products and PAID
versions, expanding over time as we keep building ideas.

## GROUND TRUTH AT INSTALL (what already exists — do NOT rebuild)

A surprising amount is already built in PRE-REVENUE form:
- `/api/v1/*` — versioned public data API (`server/apiProduct.ts`, NOT
  frozen): `x-api-key` auth, three tiers (dev/pro/enterprise), sliding-
  window rate limiting, per-request usage metering (JSONL), per-response
  LICENSE_MARKS. Live endpoints: `/api/v1/meta`, `/tracks/:kind/:id`,
  `/stats/portdwell|shadow|archive`. Gated SIGNAL endpoints listed as
  `coming_gated` until their ladder gates pass.
- `/developers` page (`client/src/pages/developers.tsx`) — honest docs:
  endpoint reference, curl example, license table, pricing PREVIEW
  (Developer free / Pro TBA / Enterprise), email waitlist. Not sale-ready
  by design.
- Drafted legal: `datacore/API_TERMS_DRAFT.md`, `datacore/LICENSING_AUDIT.md`.
- Consumer billing FULLY wired (Stripe) for the trading side: Free vs Pro,
  checkout/portal/webhooks, subscription state on `users` row. `/pricing`.
  FILES `server/billing.ts` + `server/auth.ts` are FROZEN.
- Nav: one working dropdown pattern (Analyze → 4 sections) in
  `client/src/pages/home.tsx`; `.analyze-dropdown-*` CSS.

THE REAL GAPS: (1) nav integration — 9 flat tabs, data-world not grouped,
`/developers` unreachable from the app; (2) self-serve API accounts —
keys are env-provisioned, disconnected from login; (3) turning billing ON
for the API — FROZEN + gated by the MONETIZATION READINESS CHECKLIST
(wishlist.md) + the provider-compliance tripwire.

## HUMAN DECISIONS (2026-07-11, via AskUserQuestion)

1. NAV = two-world dropdowns: a "Trade" menu (Analyze sections, Scanner,
   Watchlist, Research, News, Planner, Taxes) and a "Data / Intelligence"
   menu (Live Map, Streams, Everything Graph, Developers/API, Signals-soon).
2. BILLING POSTURE = BUILD NOW, FLIP LATER. Build the full experience
   (accounts, self-serve keys, usage dashboards, pricing pages) but keep
   it PRE-REVENUE — no Stripe flip, no real charges — until the human
   approves the MONETIZATION READINESS CHECKLIST item-by-item and the
   aircraft-provider compliance switch is resolved. Nothing here may
   activate real billing; that stays human-gated per CLAUDE.md.
3. FIRST SLICE = nav restructure + bring the API page into the app.

## PHASES (each = its own PR + tagged log entry; sequence, don't bundle)

- P1 NAV INTEGRATION (this slice, T-CLIENT) — two-world dropdowns in
  home.tsx; Developers/API reachable from the Data menu; visual harness
  green at 390/768/1440. NO billing, NO frozen paths.
- P2 DEVELOPERS/API EXPERIENCE (T-CLIENT + T-DATACORE) — elevate
  `/developers` to the PREMIUM EXPERIENCE STANDARD: live endpoint explorer,
  copyable examples per endpoint, honest freshness/coverage/confidence on
  every sample, clear free-vs-paid tier table (still "preview / join
  waitlist", no checkout).
- P3 SELF-SERVE API ACCOUNTS (pre-revenue) — issue API keys bound to a
  logged-in account (new `api_keys` table, NOT env), an in-app "your keys +
  usage today" panel reading the existing metering. Key ISSUANCE for a
  paying customer stays gated (checklist item 4); preview keys are free and
  clearly labeled preview.
- P4 USAGE & BILLING-HISTORY UI (pre-revenue) — surface metered usage and
  (when billing flips) invoices/history. Reads existing meter; no Stripe
  changes until approval.
- P5 BILLING ACTIVATION (HUMAN-GATED, FROZEN) — only after the human
  approves the MONETIZATION READINESS CHECKLIST item-by-item AND the
  aircraft chain is compliance-clean (drop/upgrade airplanes.live + adsb.fi
  or license commercially — else `/api/health` degrades). Touches FROZEN
  billing.ts/auth.ts → needs the human's hands; propose exact diffs in
  wishlist.md, never self-apply.

## GUARDRAILS (from CLAUDE.md — restated so no session forgets)

- FROZEN: server/auth.ts, server/billing.ts, server/providerCompliance.ts
  MECHANISMS. Never edit; propose via wishlist.md.
- MONETIZATION TRIPWIRE: any session touching billing/pricing/subscriptions
  re-runs the aircraft-provider compliance check in wishlist.md first.
- SPINOUT-READY: datacore/ signals flow only through the internal API
  boundary; RAW vs SIGNAL labels; SIGNALs gated at ladder gate 2 before
  surfacing or selling.
- PREMIUM EXPERIENCE STANDARD governs every user-facing surface; premium
  presentation of wrong numbers is fraud — correctness wins over polish.
- One logical change per PR; visual harness for any client/ touch.

## RESUME STATE (authoritative; update at each session end)

- 2026-07-11: program installed. P1 (nav two-world dropdowns + Developers
  in-app) SHIPPED + merged (v1.0.273, commit a6fc63a).
- 2026-07-11: P2 (developers/API premium pass) SHIPPED (v1.0.276) — live
  endpoint explorer (per-endpoint copyable curl, "Run live example" against
  each stats endpoint's public preview mirror, relative freshness from a
  new `generated_at` stamp on every v1 response + its preview route),
  data-driven tier table (reads `meta.limits`, no more hand-typed numbers
  that could drift from the real rate limiter). Full trace in
  experiments.md.
- 2026-07-11: P3 (self-serve API key accounts, pre-revenue) SHIPPED
  (v1.0.278) — `server/apiKeyAccounts.ts` (`api_keys` table, DB injected
  via a factory so tests never import auth.ts's live singleton), 3 new
  `/api/account/api-keys` routes (list/create/revoke, fixed "dev" preview
  tier), `/apikeys` self-serve panel (generate + one-time reveal + usage-
  today + revoke), nav leaf + /developers cross-link. Found and worked
  around (not fixed) a live vite `base:"./"` bug where multi-segment
  routes 404 their own JS bundle — filed in open_questions.md OPS
  GOTCHAS, this feature's route kept single-segment to sidestep it. Full
  trace in experiments.md. NEXT: P4 (usage & billing-history UI, still
  pre-revenue) or the vite-base-path bug fix (own PR). P5 is HUMAN-GATED.
- 2026-07-11: vite base-path bug FIXED (v1.0.280, own PR, [REPAIR]) —
  `vite.config.ts`'s `base` flipped `"./"` -> `"/"`; A/B-verified against
  the real `/newsletter/:slug` route (3/3 hard failures pre-fix, 0
  post-fix); visual harness gained a generalized "app actually mounted"
  check (`#root` child count) on every page plus a permanent 2+-segment
  PAGES entry, so this bug class can't silently regress on any future
  route. Full trace in experiments.md. NEXT: P4 (usage & billing-history
  UI, still pre-revenue) — the queue is otherwise clear. P5 is
  HUMAN-GATED.
- TERRITORY: P1/P2/P3 are T-CLIENT-primary (P3's server/apiKeyAccounts.ts
  is a small T-DATACORE-adjacent addition, not itself a datacore/ module).
  P4 adds T-DATACORE proper + SHARED (routes.ts, package.json)
  last-and-minimal.
- 2026-07-12: P4 (usage & billing-history UI, pre-revenue) SHIPPED
  (v1.0.289) — `apiKeyAccounts.ts` gained `keyUsageHistory(userId, id,
  days)` (ownership-checked like `revokeApiKey`, days clamped to
  `MAX_USAGE_HISTORY_DAYS`=30, reads the same per-day `apiusage/*.jsonl`
  archive `usageToday` already reads via a shared `countUsageForDay`
  helper — no new metering path); `GET /api/account/api-keys/:id/usage`
  route; `/apikeys` gained a per-key expandable "Usage" panel (14-day
  CSS bar chart, total requests) and a static Billing section stating
  the honest pre-revenue fact ($0 charged, dev tier only) rather than
  fabricating invoice rows — no Stripe/billing code touched. Full trace
  in experiments.md. Program queue is now clear except P5 (HUMAN-GATED).
  NEXT (whoever picks the program back up): either wait for a human P5
  go-ahead, or treat "clear queue" as done and look elsewhere per
  SESSION BUDGET fall-through.
