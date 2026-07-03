# HANDOFF.md — VolTradeAI Autonomous System Setup

Complete checklist to go from current repo → fully autonomous loop.
Work through it top to bottom, once. Total time: under an hour.

## 1. Files into the repo (do this in your existing Claude Code session)

- `CLAUDE.md`                       → repo root
- `ci.yml`                          → `.github/workflows/ci.yml`
- `automerge.yml`                   → `.github/workflows/automerge.yml`
- `HANDOFF.md` (this file)          → repo root
- Create `research/` directory with three files:
  - `research/experiments.md`   (header line only: "# Experiment Log")
  - `research/open_questions.md` (header: "# Open Questions")
  - `research/wishlist.md`       (header: "# Data / Access Wishlist — human reviews weekly")

Commit and push. Confirm the CI workflow runs on GitHub and goes green.
(Expect the 2 known test failures referencing backtest_v2.py — either
skip them now with @pytest.mark.skip(reason="backtest_v2 not ported")
or let the agent fix them in its first session.)

## 2. One-time GitHub settings (repo → Settings)

- General → Pull Requests → enable "Allow auto-merge"
- Branches → add protection rule for `main`:
  - Require status checks to pass: python-tests, node-build, docker-build
- Install the Claude GitHub App on this repository. NOTE: /web-setup in
  Claude Code only grants clone access — it does NOT install the App.
  Webhooks and PR creation need the App installed explicitly.

## 3. One-time Railway setting

- Service → Settings → GitHub → enable "Wait for CI" so Railway deploys
  main only after checks pass.

## 4. First supervised session (run once, watch it)

In Claude Code, give this kickoff:

  "Read CLAUDE.md fully. Your standing top-priority task is rebuilding
  the backtest engine. The output schema to reproduce is in
  backtest_10yr_results.json. bot.ts invokes it as
  `python3 backtest.py <ticker> <strategy> <years>` and JSON-parses
  stdout (current backtest.py is a stub — read its docstring). Use the
  same Alpaca SIP data helpers the live system uses
  (_fetch_alpaca_bars in analyze.py) so backtest and live see identical
  data. When done: fix or skip the two failing tests in
  test_audit_critical.py, add tests for the new engine, log the work in
  research/experiments.md, and open a PR from a claude/ branch."

Watch the PR merge itself once CI is green. That proves the whole loop.

## 5. Create the routine (the continuous loop)

In Claude Code, run `/schedule` (or create a Routine from the web UI):

- Repository: this repo
- Trigger: scheduled — 2 to 4 runs per day (e.g. 7:00, 12:30, 17:00 ET;
  more frequent than hourly is rejected, and runs share your usage
  quota + daily routine caps, so 2-4/day is the sweet spot)
- Keep the default branch safety: push only to claude/* branches.
  Auto-merge (step 2) provides full autonomy anyway.
- Prompt:

  "Read CLAUDE.md completely, then research/experiments.md,
  research/open_questions.md, and research/wishlist.md. Check system
  health and recent performance: fetch the site's /api/health endpoint
  and review the audit log and equity curve via the bot API. Then
  execute the SINGLE highest-value action per the SESSION BUDGET rules
  in CLAUDE.md. Follow the PROMOTION RULES for any change. Open one PR
  from a claude/ branch. Before ending, append your session log to
  research/experiments.md. If nothing needs doing, append a one-line
  'no action' entry and stop."

## 6. Connectors — what you need (short answer: almost none)

NEEDED:
- GitHub (the Claude GitHub App from step 2) — repo access + PRs.
  That's it for launch.

NOT NEEDED:
- Railway connector: none exists / none required. The agent monitors
  deploys through the site itself (/api/health, audit-log endpoints)
  which is more meaningful than deploy status anyway.
- Database connectors: state is JSON files on the Railway volume +
  SQLite; the agent reads them via the site's API routes.
- Web research: Claude Code sessions can search/fetch the web natively
  for the research step. No connector.

OPTIONAL LATER (add only when wanted):
- A notification connector (e.g. Slack) if you want a daily one-line
  summary pushed to you instead of checking the site.
- Alpaca has no MCP connector requirement — the bot already talks to
  Alpaca directly with your keys; the agent never needs broker access
  itself, it only edits code.

## 7. Your recurring role (the only human tasks)

- Weekly: read research/wishlist.md, decide what data to pay for.
- Occasionally: glance at equity curve vs SPY on the site.
- December 2026: confirm the agent added 2027 market holidays
  (it's instructed to, but calendar correctness is worth one glance).
