# T0.0 — the `tsc --noEmit` baseline, triaged

**MASTER PROGRAM Day One item 0.** "Run `npx tsc --noEmit`. Record the count.
Triage every error: bug / noise / dead code. `altScale` came from this list.
This is the highest-yield single action in the program and it uses a tool
already installed."

Run 2026-08-13, at `package.json` 1.0.701, commit `0d3d7d8`, after `npm ci`.
Reproduce with `npx tsc --noEmit`, or `scripts/program_status.sh` for the count.

**83 errors.** Every one is triaged below. None had ever been read: `ci.yml`'s
`node-build` job runs `npx tsc --noEmit || true`, with the trailing comment
*"tighten to hard-fail once existing TS errors are cleared"* — so this list has
printed on every CI run for months into a log nobody opens.

## Headline

| class | count | verdict |
|---|---|---|
| **TS2304** — `Cannot find name` | **5** | **REAL BUGS.** Identifiers that do not exist at runtime. §1 below |
| TS2339 — `'trim' does not exist on type 'Buffer'` | 42 | ONE root cause, one-line fix, runtime-correct today. §2 |
| TS2802 — iteration needs `--target es2015+` | 23 | NOISE — manufactured by a missing `target` in `tsconfig.json`. §3 |
| TS7006 — implicit `any` parameter | 6 | Same root cause as TS2802. §3 |
| TS2353 / TS2345 / TS2349 / TS7016 | 7 | Genuine type issues, individually judged. §4 |

**Two config-level facts dominate this list.** 29 of the 83 errors (§3) are an
artifact of a `tsconfig.json` that omits `target`, and 42 more (§2) trace to a
single untyped function. Together that is **86% of the baseline from two
causes** — which is exactly why "83 errors" was never actionable enough for
anyone to start, and why the count alone was the wrong number to track.

## §1 — TS2304, the real bugs (5)

An identifier that does not resolve. At runtime this is a `ReferenceError`,
not a slow path or a wrong value. All five are in `datamap.tsx`, and all five
are inside `try {} catch {}` blocks that swallow the throw — which is why they
have never been reported as crashes. They are silent dead code paths.

### (a) `altScale` ×3 — F-A, confirmed exactly as the audit predicted

| site | current line | what dies |
|---|---|---|
| `datamap.tsx:4384` (×2 refs) | `alt - (altScale > 0 ? gZ / altScale : 0)` | the flight card's **ALT AGL** readout |
| `datamap.tsx:4421` | `Math.max(0, alt) * altScale` | the follow-camera's **elevation datum** |

`altScale` is declared once, at `datamap.tsx:4089`, inside the flight-track
paint closure that ends at 4234. Both uses are in a *different* function — the
readout tick — which redeclares `terrainOn` (4347) and `gZ` (4348) for itself
but never `altScale`.

*(The audit cited lines 4026/4321/4358; the file has moved since. The finding is
unchanged — verified this session against the current tree.)*

Consequence, in order: the tick throws at 4384 → the `catch { /* readouts must
never break the tick */ }` at 4430 swallows it → **every statement after 4384
is skipped for that tick**, which includes GND SPD, VERT SPD, and the entire
follow-aircraft recenter block at 4412. The comment promising readouts never
break the tick is what guarantees the breakage stays invisible.

**Fix:** recompute at each use site, from the same expression as 4089 —
`const altScale = terrainOn ? terrainExagRef.current : 1;` — using the
`terrainOn` already in scope at 4347.

### (b) `e` ×2 — `datamap.tsx:6889` — NOT IN THE AUDIT, found by this sweep

```
6889:  try { if (e?.originalEvent) e.originalEvent.__vtFeatClaim = true; } catch {}
```

inside `const focusSat = (index: number) => {` (declared 6807), whose only
parameter is `index`. There is no `e`.

This is an **extraction bug**. The identical line lives at 6778 inside
`onClick(e)`, where `e` is real. The comment at 6803 records the extraction:
*"O6-3: one focus path for BOTH entrances — a map click (above) and a search
hit (SatFinder → focusSatByIndexRef)."* When the satellite-focus body moved out
of the click handler into a shared function, this line came with it and lost
its binding.

Consequence: clicking a satellite never stamps `__vtFeatClaim`, so the
deferred click-off handler at 3882 — `if (oe?.__vtFeatClaim) { clearTrail();
return; } // curtain goes; the new card stays` — never takes that branch. The
flight curtain and the satellite card mis-interact on exactly the click the
line exists to handle. Same signature as F-A: real, user-visible, silent.

**Fix:** thread the event through as optional —
`const focusSat = (index: number, e?: { originalEvent?: ... }) => {` — and pass
it at the click call site (6800). The SatFinder entrances (2605, 2634, 6964)
pass nothing and correctly make no claim: a search hit is not a map click and
has no `originalEvent` to stamp.

### Why this class is worth a permanent ratchet

TS2304 is the one error code that is **never** config noise and never a false
positive — the name is either in scope or it is not. Both defects here were
introduced by ordinary refactors (a closure split, a function extraction), both
were masked by an empty `catch`, and one of them survived a full static audit.
`tsc_2304` is therefore broken out as its own counter in
`scripts/program_status.sh` with a target of 0, separate from the general
`tsc_errors` count.

## §2 — TS2339 `Property 'trim' does not exist on type 'Buffer'` (42)

All 42 are in `server/bot.ts`, all on `stdout.trim()`, and all trace to **one
function**:

```ts
server/bot.ts:191   async function execPythonSerialized(cmd: string, opts?: any) {
```

No declared return type. It returns `execAsync(cmd, {...opts})` where
`execAsync = promisify(exec)`, and because `opts` is `any`, TypeScript cannot
tell which `exec` overload applies and falls back to the `Buffer` variant.

**Runtime-correct today.** Node's `child_process.exec` defaults to
`encoding: 'utf8'` and genuinely returns a string; every one of these 42
`.trim()` calls works. This is a typing gap, not a defect — but it is 51% of
the baseline, so it hides everything else.

**Fix (one line, own PR):** annotate the return type —
`: Promise<{ stdout: string; stderr: string }>`. That is a truthful narrowing
of what the function already returns, not a cast that hides anything. Expected:
42 errors and the related TS2345 at 4764 all clear together.

## §3 — TS2802 (23) + TS7006 (6) — a missing `target`, verified

`tsconfig.json` sets `"lib": ["esnext", "dom", "dom.iterable"]` but **never
sets `target`**, so `tsc` falls back to its ES5 default and rejects
`for…of` over a `Map`, `Set`, or `NodeList` — iteration that every runtime and
every build path in this repo supports.

**Verified, not assumed.** Adding `"target": "ES2022"` and clearing the
incremental cache:

```
83 errors  →  54 errors
all 23 TS2802 gone · all 6 TS7006 gone · nothing new appears
```

*(Method note for whoever repeats this: `tsconfig.json` sets
`"incremental": true` with a `tsBuildInfoFile` under `node_modules/typescript/`.
My first attempt at this experiment returned an unchanged 83 because the run
was served from that cache. Delete `node_modules/typescript/tsbuildinfo`
between runs or the result is meaningless.)*

No emit is involved — `noEmit: true`, and the shipped bundles are built by vite
(client) and esbuild (server), neither of which reads this target. So the ES5
default describes no artifact this repo produces; it is checking against a
runtime that does not exist.

**Not fixed in this PR.** It is a config change with its own blast radius
(a real target may surface *new* errors elsewhere in strict mode), so it gets
its own PR under D9, sequenced before T1.6's hard-fail ratchet — the ratchet
should pin an honest number, not one inflated by 29 phantoms.

## §4 — the remaining 7, individually judged

| site | error | verdict |
|---|---|---|
| `server/bot.ts:1272` | `lastEquity` not on the bot-state type | **Real.** Reads a field the declared state shape does not have → `undefined` at runtime. Worth tracing: it is in kill-switch-adjacent state |
| `server/bot.ts:3190` | `volume` not on `QueuedTrade` | **Real**, same class |
| `server/bot.ts:3206` | `instrument` not on `QueuedTrade` | **Real**, same class |
| `server/bot.ts:3519` | `rank` not in an audit-record literal | Real but benign — excess property on a log record |
| `server/billing.ts:117` (×2) | `current_period_end` not on `Subscription` | Stripe SDK type drift. **`billing.ts` is a FROZEN PATH** — do not touch; file it |
| `server/billing.ts:322` | `toLowerCase` on `string \| string[]` | **Real crash risk** on a repeated header. FROZEN PATH — file it |
| `client/.../TradeChart.tsx:178,199` | `textColor` / `scaleMargins` unknown | lightweight-charts API drift; options silently ignored |
| `client/.../datamap.tsx:3337` | `Type 'never' has no call signatures` | Needs a read |
| `client/.../datamap.tsx:13577` | `string` not assignable to the layer-group union | A group id computed as a bare `string` |
| `server/owmTiles.ts:79` | no types for `pngjs` | Noise. `@types/pngjs` or a one-line `declare module` |

The three `server/bot.ts` state-shape errors (1272, 3190, 3206) are the most
interesting residue: each is a property read that the type system says will be
`undefined`. None is on the order-submission path, so none is a stop condition
— but they belong in Track 5, and `bot.ts:1272` touches equity tracking, which
is measurement code (CLAUDE.md MEASUREMENT INTEGRITY) and therefore gets its
own PR and its own scrutiny.

## What this changes about the plan

1. **`tsc_errors` alone is the wrong ratchet metric.** 86% of it is two
   mechanical causes. `tsc_2304` — always-real, currently 5, target 0 — is the
   counter that would have caught both bugs found today, and it is now tracked
   separately.
2. **T1.6 should be sequenced after the `target` fix and the
   `execPythonSerialized` annotation.** Pinning 83 pins 71 phantoms and invites
   a future session to "fix" them by suppression, which §12 names as a failure
   mode. Pin ~12.
3. **The audit's §5.1 admission was right and understated.** "`tsc --noEmit` was
   never run. The error count is unknown and `altScale` came out of that pile."
   One sweep of that pile produced a second live bug of the same class
   (`focusSat`) that a full static audit had missed.
