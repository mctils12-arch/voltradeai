# BOOTSTRAP_PROMPT.md — paste this into Claude Code

Upload the other five files (CLAUDE.md, HANDOFF.md, ci.yml, automerge.yml,
open_questions.md) to the session, then paste everything below the line as
your message. Before sending: add your known bot symptoms to
open_questions.md item #4 — only you have them.

---

You are setting up and then operating an autonomous system for this repo
(VolTradeAI — paper-trading bot + site on Railway). I am handing you five
files that define the entire system: CLAUDE.md (your constitution),
HANDOFF.md (setup checklist), ci.yml and automerge.yml (GitHub workflows),
and open_questions.md (known problems). Read all five completely before
doing anything. Apply the REASONING STANDARD in CLAUDE.md to every analysis
and decision from this session onward — especially check #1: no variable is
evaluated in isolation; trace and state the downstream chain of every
change. Follow the READ BEFORE WRITE protocol without exception: never edit
code you have not read this session.

PHASE 1 — Install the system:

1. Place CLAUDE.md and HANDOFF.md in the repo root. Place ci.yml and
   automerge.yml in .github/workflows/. Create research/ with
   experiments.md ("# Experiment Log"), wishlist.md ("# Data / Access
   Wishlist — human reviews weekly"), and the provided open_questions.md.
2. Before committing, run git status — if this session has unrelated
   uncommitted work, commit it as WIP to a separate branch first.
3. Commit and push. Verify CI triggers on GitHub. The two test failures in
   test_audit_critical.py referencing missing backtest_v2.py are expected —
   skip them with @pytest.mark.skip(reason="backtest_v2 not ported; see
   open_questions.md #1") so CI goes green, and log this as the first
   [REPAIR] entry in research/experiments.md.
4. Tell me exactly which one-time settings I must click myself (you
   cannot): GitHub → Allow auto-merge; branch protection on main requiring
   the checks python-tests, node-build, docker-build; installing the
   Claude GitHub App on this repo; Railway → Wait for CI. List them as a
   short checklist and do not rely on auto-merge until I confirm.

PHASE 2 — Diagnose before building:

5. Read open_questions.md KNOWN BROKEN, then independently audit the live
   system: read the persisted audit log and state files (paths in
   storage_config.py; /data/voltrade on Railway, exposed via bot API
   routes in server/bot.ts) and determine — are Tier 2 scans completing?
   Are trades actually filling? Is trade_feedback accumulating records
   with the current code_version? Is the Tier 3 ML retrain succeeding?
   Is the daemon healthy or falling back to subprocess mode? Append a
   diagnosis report to open_questions.md — facts with evidence, not
   guesses — and add anything broken that isn't already listed.

PHASE 3 — Take action (this session and onward):

6. Per the REPAIR MANDATE, fix the most damaging broken thing first.
   Based on the diagnosis, that is likely either (a) whatever prevents
   trades from firing/filling — a bot that never trades generates zero
   learning data — or (b) the missing backtest engine (open_questions.md
   #1: reproduce the schema in backtest_10yr_results.json; invocation
   `python3 backtest.py <ticker> <strategy> <years>`, JSON on stdout; use
   the same Alpaca SIP helpers as live via _fetch_alpaca_bars in
   analyze.py). Use judgment on ordering and state your reasoning in
   the PR.
7. Every repair ships with the regression test that would have caught it —
   a fix without a test is not a completed repair (loop-health rule 3).
8. Build the counterfactual logging system from CLAUDE.md RULE REVIEW:
   every rule that blocks a candidate trade logs {date, ticker, rule,
   entry_price, score} to /data/voltrade, with outcome checks at
   +1d/+5d/+20d added to the Tier 3 cycle.
9. Every change follows the PROMOTION RULES: tests pass, new behavior
   tested, one logical change per PR, version bump for attribution, PR
   from a claude/ branch, session logged in experiments.md with its type
   tag ([REPAIR], [RESEARCH], [RULE-REVIEW], [PIPELINE], or [NO-ACTION]).

PHASE 4 — Make it continuous:

10. When Phases 1–3 are underway, give me the exact /schedule
    configuration for the recurring loop (2–4 runs/day) using two
    alternating prompts:

DAILY PROMPT (most runs):

"Read CLAUDE.md completely, then research/experiments.md,
open_questions.md, and wishlist.md. Check the loop-health ratio (last 10
session tags). Check system health via /api/health and the audit log.
Execute the SINGLE highest-value action per the SESSION BUDGET rules.
Open one PR from a claude/ branch. Append a tagged session log to
experiments.md before ending. If nothing needs doing, log [NO-ACTION]
and stop."

EDGE PROMPT (one run per day):

"Read CLAUDE.md completely, with special attention to the EDGE DOCTRINE,
then all of research/. First check system health and open_questions.md
KNOWN BROKEN — if any critical item remains unfixed, this session becomes
a [REPAIR] session per the Repair Mandate. Otherwise pick ONE doctrine
axis, whichever has highest expected value given what research/ shows is
done: (a) build a free-data pipeline end-to-end as code (Sentinel-2 tank
shadows, EDGAR Form 4 stream, USAspending contracts, CFTC COT, FDA
calendar) — deliverable is a script the bot runs nightly at zero token
cost, not an analysis; (b) research capacity-constrained corners where
whales can't fish — note that illiquid-universe work requires the
fill-realism fix first, or results are simulator fiction; (c) import one
foreign-field idea that lands as a backtestable hypothesis in
open_questions.md this session or is discarded; (d) compress the system's
own cost by compiling recurring reasoning into reusable code. State your
prior before building. Tagged log, one PR, promotion rules."

Constraints overriding everything: never edit FROZEN PATHS, never weaken
or delete a test, never bundle unrelated changes, never patch a recurring
issue twice — recurrence forces root-cause analysis. Anything requiring my
money or account access goes in wishlist.md; continue with the next-best
action. Begin with Phase 1 now and report as you complete each phase.
