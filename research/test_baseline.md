# test_baseline.md — what the test suites actually do

**MASTER PROGRAM T1.1 / Q10 deliverable.** Measured 2026-08-14 at
`package.json` 1.0.706, after `npm ci`, `pip install -r requirements.txt` and
`pip install -r requirements-dev.txt`.

This is the baseline the rest of Track 1 needs: T1.2 promotes the green set
into `ci/required.txt` and quarantines the rest with a reason and a date.
Reproduce with the three commands in the table.

## The gap this closes

CI invoked **four** test files out of ~364. `python-tests` names four by hand;
`node-build` runs `npm ci`, the typecheck ratchet, and `npm run build`, and
invokes **no test suite at all**.

So **1000 passing client tests and 1247 passing server tests had never once run
in CI** — and neither had the four ratchet tests added in #823/#824/#825/#826,
which were written specifically to stop regressions and were, until this job,
guarding nothing.

## Measured baseline

| suite | files | tests | pass | fail | wall time | command |
|---|---|---|---|---|---|---|
| server (node) | 147 | 1248 | 1247 | **1** | ~96s | `npx tsx --test server/*.test.ts` |
| client (node) | 96 | 1000 | **1000** | **0** | ~21s | `npx tsx --test $(git ls-files 'client/**/*.test.ts')` |
| python | 121 | 1338 + 54 subtests | **1337** (+1 skip) | **0** | ~64s | `VOLTRADE_CI=1 python -m pytest -q` |
| **total** | **364** | **~3,586** | **3,584** | **1** | **~3m 01s** | |

**One known failure out of ~3,586**, and it is a test asserting the presence of
files that were never committed. That ratio is the argument for Track 1: the
suites are in far better shape than "360 ungated tests" suggests, and the reason
they were not gating was never that they were failing.

## What the baseline run found

### 1. The python suite is green — once CI installs the deps it already declares

The first whole-suite run reported `1 skipped, 1 error` in **4 seconds**:
`test_grid_county_ba.py` loads `scripts/grid_county_ba.py`, which imports
`openpyxl`, and **a collection error aborts the entire pytest run**.

The obvious reading — "openpyxl is undeclared" — is wrong. It *is* declared, in
`requirements-dev.txt`, whose header describes it as the repo's "session-run /
test-only Python deps". The real gap was that **CI installs `requirements.txt`
and `pytest`, and never that file.**

Fixed in this PR by installing it in the test job. Result: **1337 passed,
1 skipped, 0 failed, no collection errors, 64s** — and no
`--continue-on-collection-errors` flag needed, which matters, because as a
permanent setting that flag would absorb the next collection break silently:
the same class of mistake as the `|| true` this program just removed from the
typecheck.

Deliberately *not* fixed by moving `openpyxl` into `requirements.txt`: that file
is what the frozen Dockerfile installs into the Railway image, and
`requirements-dev.txt`'s own header records that nothing in it belongs on a
runtime path.

### 2. The repo's own canary was never armed

`test_collection_health.py::test_full_repo_pytest_collection_succeeds` exists
for exactly the condition above — a test whose entire job is asserting that
full-repo pytest collection succeeds. **It was failing, for precisely the reason
it exists to catch, and it had never run in CI to sound the alarm**, because CI
names four files and this is not one of them.

This is the clearest single argument in the repo for Track 1: the guard was
written, the guard was correct, the guard was never armed. It now passes.

### 3. `server/gridTiles.test.ts` — asserts ≥50 pmtiles, 3 exist *(Q12)*

`assert.ok(files.length >= 50)` over `client/public/tiles/*.pmtiles`. Three
exist, and `git log --all` finds no `power_*.pmtiles` **ever committed** — so
this assertion has failed since the day it was written. It is the confirmation
of `wishlist.md` A4 PHASE 2 item 2 (*"US-full power grid,
boot-fetch-from-Release — filed above, NOT built"*): the test asserts the built
state, and the build never happened.

**The one entry T1.2 must quarantine or resolve.**

### 4. `server/datacoreArchive.test.ts` — a UTC-midnight flake, now confirmed *(Q15)*

Two rollup tests fail by arithmetic, not chance. They compute
`oldMs = now - (RAW_RETENTION_DAYS + 2) * 86400_000` and write cadence-spaced
samples, so within roughly an hour of UTC midnight those samples straddle two
UTC days and the expected 1 rolled day becomes 2 (`expected: 1, actual: 2`; the
sibling asserts 0 and gets 1).

**Confirmed experimentally rather than argued:** both failed at 23:55Z and both
pass at 01:00Z on the same tree, same commit. They are *not* in the failure
count above because that run was taken outside the window — which is exactly
what makes this dangerous. **Must be fixed before T1.2 makes the suite
blocking**, or CI goes red for an hour every night, which is how a new gate
loses its credibility in its first week.

## Cost

**~3m 01s of test execution**, plus `npm ci` (~15s cached),
`pip install` (~40s cached). Roughly **4–5 minutes per run** on top of the
existing jobs, path-filtered so docs-only and research-only PRs skip it
entirely.

Against the 2,000-minute monthly cap documented in `ci.yml`'s header, and with
`concurrency: cancel-in-progress` already reclaiming superseded PR runs, this is
affordable. T1.4 revisits with real numbers once several runs accumulate.

## Risks to settle before T1.2 makes this blocking

1. **Q15 must be fixed first**, or the gate reds nightly for an hour.
2. **Q12 must be quarantined or resolved** — it is the one standing failure.
3. **The python suite reaches the network.** `VOLTRADE_CI=1` is set, but the
   baseline run still produced live yfinance traffic
   (`$DX-Y.NYB: possibly delisted`, `curl: (35) Recv failure`). `ci.yml`'s
   header claims "network-dependent tests are excluded in CI" — that claim is
   **not fully true** for a whole-suite run, and network flakiness is exactly
   the wrong thing to put behind a merge gate. Identify and mark those tests
   before promoting the python suite to required.

## Correction to an earlier note

`PROGRAM_STATE.md` previously said Q12 and Q15 "must be handled in the same PR
[as Q10], or the suite reds on day one." That conflated T1.1 with T1.2. This
job is `continue-on-error` at both job and step level, so **nothing it finds
can red the build**. The quarantine work is required before **T1.2** flips the
suite to blocking — which is where it now sits.
