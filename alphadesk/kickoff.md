# kickoff.md — paste this into Claude Code to start

Copy everything in the block below as your first message to Claude Code (run
`claude` in this folder first). It points the agent at the project docs and
gives it a concrete, verifiable first task with the guardrails it must keep.

---

You are picking up an existing project, AlphaDesk — an explainable equity (and
soon options) research engine. Do this in order:

1. Read `CLAUDE.md` and `TASKS.md` in full, then `README.md`. Summarize back to
   me in 5 bullets: what the engine does, the architecture, what's done vs
   stubbed, the guardrails, and what task #1 is. Wait for my "go" before coding.

2. Confirm the environment. Run `python -m alphadesk keys` and tell me which of
   my vendor keys (Alpaca / Polygon / Finnhub) are detected. If any I expect are
   missing, tell me the exact env-var name to set (see `.env.example`). My keys
   are already created — I just need to export them or put them in `.env`.

3. Task #1 — verify the live adapter (do NOT skip to other tasks):
   - Run `python -m alphadesk AAPL --json` (live) and
     `python -m alphadesk AAPL --sample --json`, then diff them.
   - For every field that is still sample-derived under live (price, history,
     P/E, beta, margins, shares outstanding), trace why: is the vendor call
     failing, or is a response field landing under a different name than the
     mapping in `market.py` / `LiveProvider.fundamentals`?
   - Fix the field mappings so live data actually populates. Finnhub metric
     names are the most likely culprit — check them against a real response.
   - Show me a before/after of one ticker proving live numbers now fill in.

Guardrails you must keep at all times:
- This is a research/education tool. NOT investment advice. Keep the disclaimer
  in all output.
- This repo does not place orders. Do not add any execution/trading code. If we
  ever discuss it, it must be paper/sandbox-first behind an off-by-default flag.
- Keep verdicts explainable — every score must trace to named factors.
- Keep `python -m alphadesk selftest` green; add a check for anything you change.
- Clean-room: do not import or copy from any other project. Only my own vendor
  keys are reused.

Work one task at a time. After task #1 passes, stop and ask me before starting
task #2 (SEC EDGAR filings).

---

Tip: if Claude Code asks permission for each command and it slows you down, you
can let it run autonomously — but only in a folder under git so you can revert.
Run `git init && git add -A && git commit -m "alphadesk handoff"` first.
