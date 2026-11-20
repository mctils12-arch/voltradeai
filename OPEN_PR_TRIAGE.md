# Open PR triage — one-time cleanup, 2026-11-19

Owner directive (execution-layer Addendum 1 §A10): list every open PR with
what it does, how far along it is, and whether it still applies; finish the
useful ones; close only what is fully superseded, with a reason; throw
nothing useful away. The standing rule is in `CLAUDE.md` (STANDING
BEHAVIORS, FINISH WHAT WAS STARTED).

Read at 2026-11-19 21:30 ET against `main` at v1.0.1486.

## Open PRs: 1

| PR | Opened | What it does | How far along | Still applies? | Outcome |
|---|---|---|---|---|---|
| #1706 | 2026-11-11 | The daemon's health probe reported its all-time memory peak as its current memory, so the health check raised "deep-score guard latched off" alarms that were not real. The PR made the probe report the current figure and the peak separately. | Complete: code, 10 tests, log entry. Never merged; opened during market hours by a routine and not picked up after the close. | No. The same fix landed five days later as #1770 (v1.0.1413, `666765a4`), with different key names. Later work went further (#1833: the check reads what the guard did; memory by kind). | **Closed 2026-11-19**, with a note on the PR saying where the work went. |

What #1706 had that `main` does not, and why it is not carried over:
- A label for a container where `/proc` is unreadable. Production has `/proc`; `main` falls back to the old figure there.
- Ten tests pinned to #1706's own key names. `main` tests the keys it ships.
- Its open question (the watchdog ends the daemon on a one-time peak past 1 GB) is already filed in `research/open_questions.md`.

## Branches pushed without a PR: 6

The same class of forgotten work, so they were read too. None holds work
that `main` lacks.

| Branch | Last commit | What it holds | On `main` as |
|---|---|---|---|
| `claude/eloquent-dijkstra-3s57ok` | 2026-11-16 | Form 4 / 13F / 8-K archive writers stop re-appending rows after a restart | `8adad785` (2026-11-17) |
| `claude/quirky-hopper-r16sce` | 2026-11-17 | The same change, a later copy | `8adad785` |
| `claude/eloquent-dijkstra-f3m5hx` | 2026-11-17 | Form 4 poll archives only forms of type 4 or 4/A | `5e4ea3f1` (2026-11-17, a re-land) |
| `claude/lucid-keller-0o4jw8` | 2026-11-12 | EDGAR archive pages say which days they show; "Load older days" | #1762 (Form 4, 13F) and #1767 (earnings language), rebuilt as `client/src/lib/filingCoverage.ts` |
| `claude/lucid-keller-u27h63` | 2026-11-15 | Nothing that is not on `main` | merged |
| `claude/gridvision-train-container` | 2026-07-08 | Tower-detector training container | #370 |

These six branches (and `claude/funny-fermat-97o39y`, #1706's) can be
deleted. A session does not delete branches it did not create; that is
the owner's to do, or to delegate.

## Why #1706 was forgotten

It was opened at 15:27 ET by a scheduled run, which may not merge during
market hours. Nothing made the next run look for it. Two later sessions
then fixed the same bug from scratch. The standing rule closes that gap:
open PRs are read before new work starts.

## State after the cleanup

Open PRs: 0 (before tonight's ship queue opens its own, one at a time).
