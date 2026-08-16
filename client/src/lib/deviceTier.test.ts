import { test } from "node:test";
import assert from "node:assert/strict";
import {
  surfacePixelRatio,
  classifyDevice, govInit, govStep, median,
  setOverloaded, isOverloaded, overloadFromState,
  GOV_OVERLOAD_MS, GOV_OVERLOAD_HOLD_MS, GOV_CALM_MS, GOV_CALM_HOLD_MS, GOV_COOLDOWN_MS, GOV_STRETCH_GRACE_MS,
  isFragileGpu,
} from "./deviceTier";

const RES_HOLD = GOV_OVERLOAD_HOLD_MS + GOV_STRETCH_GRACE_MS; // resolution waits out the tick-stretch grace

// ── classifyDevice ──────────────────────────────────────────────────────────

test("software renderers are minimal tier, ratio capped at 1", () => {
  for (const r of [
    "Google SwiftShader",
    "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)...)",
    "llvmpipe (LLVM 15.0.7, 256 bits)",
    "Microsoft Basic Render Driver",
  ]) {
    const t = classifyDevice({ renderer: r, devicePixelRatio: 2 });
    assert.equal(t.tier, "minimal", r);
    assert.equal(t.pixelRatioCap, 1, r);
    assert.ok(t.reasons.length >= 1);
  }
});

test("old integrated GPUs are reduced tier, capped at 1.5", () => {
  const t = classifyDevice({ renderer: "ANGLE (Intel, Intel(R) HD Graphics 520 Direct3D11)", devicePixelRatio: 2 });
  assert.equal(t.tier, "reduced");
  assert.equal(t.pixelRatioCap, 1.5);
});

test("low memory or very few cores reduce the tier", () => {
  const lowMem = classifyDevice({ renderer: "NVIDIA GeForce RTX 3060", deviceMemoryGB: 4, devicePixelRatio: 2 });
  assert.equal(lowMem.tier, "reduced");
  const lowCores = classifyDevice({ renderer: "NVIDIA GeForce RTX 3060", cores: 2, devicePixelRatio: 2 });
  assert.equal(lowCores.tier, "reduced");
});

test("capable hardware is full tier, capped at 2 even on 3x displays", () => {
  const t = classifyDevice({ renderer: "Apple M2", deviceMemoryGB: 16, cores: 8, devicePixelRatio: 3 });
  assert.equal(t.tier, "full");
  assert.equal(t.pixelRatioCap, 2);
  assert.equal(t.reasons.length, 0);
});

test("reduced cap never exceeds the actual device ratio", () => {
  const t = classifyDevice({ renderer: "Intel(R) UHD Graphics 400", devicePixelRatio: 1 });
  assert.equal(t.pixelRatioCap, 1);
});

test("modern UHD 620+ integrated graphics is NOT flagged weak", () => {
  const t = classifyDevice({ renderer: "ANGLE (Intel, Intel(R) UHD Graphics 620 Direct3D11)", devicePixelRatio: 2 });
  assert.equal(t.tier, "full");
});

// ── governor ────────────────────────────────────────────────────────────────

test("sustained overload steps the ratio down one ladder notch, with a note", () => {
  let st = govInit(2, 0);
  // overload begins at t=20s (past cooldown), must hold for HOLD_MS
  let d = govStep(st, 200, 20_000);
  assert.equal(d.apply, undefined); // just started
  d = govStep(d.state, 200, 20_000 + RES_HOLD);
  assert.equal(d.apply, 1.75);
  assert.match(d.note ?? "", /Render resolution lowered/);
  assert.match(d.note ?? "", /all layers and data stay on/);
});

test("a momentary spike does NOT step down (hold requirement)", () => {
  let st = govInit(2, 0);
  let d = govStep(st, 200, 20_000);
  d = govStep(d.state, 20, 21_000); // recovered before the hold elapsed
  d = govStep(d.state, 200, 22_000); // spike again — overload clock restarts
  d = govStep(d.state, 200, 22_000 + RES_HOLD - 1_000);
  assert.equal(d.apply, undefined);
});

test("steps are rate-limited by the cooldown", () => {
  let st = govInit(2, 0);
  let d = govStep(st, 200, 20_000);
  d = govStep(d.state, 200, 20_000 + RES_HOLD); // -> 1.75
  assert.equal(d.apply, 1.75);
  const stepAt = 20_000 + RES_HOLD;
  // still overloaded immediately after: no second step inside the cooldown
  d = govStep(d.state, 200, stepAt + 1_000);
  d = govStep(d.state, 200, stepAt + RES_HOLD);
  assert.equal(d.apply, undefined);
  // overload persisted through the cooldown: next notch lands once the
  // full resolution hold has renewed past the cooldown
  d = govStep(d.state, 200, stepAt + 1_000 + RES_HOLD);
  assert.equal(d.apply, 1.5);
});

test("ratio floors at 1 — no step below the ladder", () => {
  let st = govInit(1, 0);
  let d = govStep(st, 500, 20_000);
  d = govStep(d.state, 500, 20_000 + RES_HOLD);
  assert.equal(d.apply, undefined);
  assert.equal(d.state.ratio, 1);
});

test("sustained calm steps back up but never above the device ceiling", () => {
  // classified ceiling 1.5 (reduced tier); driven down to 1, then calm
  let st = govInit(1.5, 0);
  let d = govStep(st, 200, 20_000);
  d = govStep(d.state, 200, 20_000 + RES_HOLD); // -> 1.25
  assert.equal(d.apply, 1.25);
  const t0 = 20_000 + RES_HOLD + GOV_COOLDOWN_MS + 1;
  d = govStep(d.state, 10, t0);
  d = govStep(d.state, 10, t0 + GOV_CALM_HOLD_MS); // -> back up one notch
  assert.equal(d.apply, 1.5);
  assert.match(d.note ?? "", /restored/);
  // more calm: already at ceiling, no further step
  const t1 = t0 + GOV_CALM_HOLD_MS + GOV_COOLDOWN_MS + 1;
  d = govStep(d.state, 10, t1);
  d = govStep(d.state, 10, t1 + GOV_CALM_HOLD_MS);
  assert.equal(d.apply, undefined);
  assert.equal(d.state.ratio, 1.5);
});

test("between-band readings reset both trend clocks", () => {
  let st = govInit(2, 0);
  let d = govStep(st, 200, 20_000);
  d = govStep(d.state, 60, 21_000); // between calm and overload
  assert.equal(d.state.overloadSince, null);
  assert.equal(d.state.calmSince, null);
});

test("median helper", () => {
  assert.equal(median([]), 0);
  assert.equal(median([3]), 3);
  assert.equal(median([1, 9, 3]), 3);
});

// ── overload flag ───────────────────────────────────────────────────────────

test("overloadFromState: on after overload hold, off after 10s calm, sticky between", () => {
  let st = govInit(1, 0); // at the pixel floor — the flag is the only lever left
  let d = govStep(st, 200, 20_000);
  assert.equal(overloadFromState(d.state, 20_000, false), false); // just began
  d = govStep(d.state, 200, 20_000 + GOV_OVERLOAD_HOLD_MS);
  assert.equal(overloadFromState(d.state, 20_000 + GOV_OVERLOAD_HOLD_MS, false), true);
  // between bands: sticky (keeps previous)
  d = govStep(d.state, 60, 40_000);
  assert.equal(overloadFromState(d.state, 40_000, true), true);
  // calm begins: still sticky until 10s of calm
  d = govStep(d.state, 10, 50_000);
  assert.equal(overloadFromState(d.state, 55_000, true), true);
  d = govStep(d.state, 10, 60_001);
  assert.equal(overloadFromState(d.state, 60_001, true), false);
});

test("setOverloaded/isOverloaded module store round-trips", () => {
  assert.equal(isOverloaded(), false);
  setOverloaded(true);
  assert.equal(isOverloaded(), true);
  setOverloaded(false);
  assert.equal(isOverloaded(), false);
});

test("lever order: tick-stretch flag engages before any resolution step (round-16 quality contract)", () => {
  let st = govInit(2, 0);
  let d = govStep(st, 200, 20_000);
  const atFlag = 20_000 + GOV_OVERLOAD_HOLD_MS;
  d = govStep(d.state, 200, atFlag);
  assert.equal(overloadFromState(d.state, atFlag, false), true, "smoothness lever on at the hold");
  assert.equal(d.apply, undefined, "resolution untouched while the stretch grace runs");
  d = govStep(d.state, 200, atFlag + GOV_STRETCH_GRACE_MS - 1_000);
  assert.equal(d.apply, undefined, "still untouched just inside the grace");
  d = govStep(d.state, 200, atFlag + GOV_STRETCH_GRACE_MS);
  assert.equal(d.apply, 1.75, "pixels sacrificed only after the grace expires");
});

// ── fragile-GPU classification (repair 2026-08-05) ──────────────────────────
// Field defect: a "no-webgl" boot probe (Chrome GPU process down — the exact
// fragile state the mitigations exist for) left INTEGRATED_GPU false and
// silently DISABLED the tile-cache mitigation on the crashing Iris Xe
// machine. The affirmative dead-GPU read must classify as fragile; unknown
// must not.
test('isFragileGpu: Intel integrated yes, Arc no, dead-GPU probe yes (fail-safe), unknown no', () => {
  assert.equal(isFragileGpu('Google Inc. (Intel) | ANGLE (Intel, Intel(R) Iris(R) Xe Graphics (0x00009A49) Direct3D11 vs_5_0 ps_5_0, D3D11)'), true, 'the field machine itself');
  assert.equal(isFragileGpu('Intel | Intel(R) UHD Graphics 630'), true);
  assert.equal(isFragileGpu('Intel | Intel(R) Arc(TM) A770 Graphics'), false, 'Arc is discrete');
  assert.equal(isFragileGpu('no-webgl'), true, 'dead GPU process at boot -> mitigations ON, never silently off');
  assert.equal(isFragileGpu('unavailable'), false, 'unknown stays non-fragile');
  assert.equal(isFragileGpu('NVIDIA | NVIDIA GeForce RTX 4070'), false);
  assert.equal(isFragileGpu('Apple | Apple M2'), false, 'Apple integrated is not the fragile class');
});


// ── F-C: the celestial surfaces were never capped (2026-08-14) ─────────────
// datamap.tsx has always clamped its map canvas to tier.pixelRatioCap. The two
// celestial surfaces — celestialSky.ts (a SECOND WebGL2 context) and
// spaceFrame.ts (2D canvases) — sized their backing stores from raw
// devicePixelRatio and did not import this module at all. On a 3x phone that is
// 9x the backing-store pixels of a 1x surface, for every raster op they do.
//
// SCOPE, stated so nobody re-derives an overclaim: this bounds MEMORY and FILL
// RATE. It does NOT speed up the Moon's CPU raycast — patchBufDims() sizes that
// buffer in CSS px against MOON_PATCH_*_LONG_PX, so it never depended on DPR.

function withGlobals<T>(dpr: number, tier: unknown, fn: () => T): T {
  const g = globalThis as Record<string, unknown>;
  const prevDpr = g.devicePixelRatio;
  const prevTier = g.__vtDeviceTier;
  g.devicePixelRatio = dpr;
  if (tier === undefined) delete g.__vtDeviceTier; else g.__vtDeviceTier = tier;
  try { return fn(); } finally {
    g.devicePixelRatio = prevDpr;
    if (prevTier === undefined) delete g.__vtDeviceTier; else g.__vtDeviceTier = prevTier;
  }
}

test("surfacePixelRatio: honours the shared __vtDeviceTier cap when datamap has published one", () => {
  // One machine, one classification: all three surfaces must agree rather than
  // each computing its own reading from different inputs.
  assert.equal(withGlobals(3, { tier: "minimal", pixelRatioCap: 1 }, () => surfacePixelRatio()), 1);
  assert.equal(withGlobals(3, { tier: "reduced", pixelRatioCap: 1.5 }, () => surfacePixelRatio()), 1.5);
  assert.equal(withGlobals(3, { tier: "full", pixelRatioCap: 2 }, () => surfacePixelRatio()), 2);
});

test("surfacePixelRatio: falls back to classifying the caller's own renderer", () => {
  // celestialSky passes its own GL renderer string, so a software rasterizer is
  // caught even if datamap.tsx never ran (celestial-only routes, direct entry).
  assert.equal(
    withGlobals(3, undefined, () => surfacePixelRatio("Google SwiftShader")), 1,
    "software renderer -> minimal tier, cap 1",
  );
  assert.equal(
    withGlobals(3, undefined, () => surfacePixelRatio("NVIDIA GeForce RTX 4070")), 2,
    "full tier still caps at 2 (the 3rd x is invisible and quadratic)",
  );
});

test("surfacePixelRatio: never exceeds the device, never drops below 1", () => {
  // A 1x device must not be UPscaled by a cap of 2 — the cap is a ceiling only.
  assert.equal(withGlobals(1, { tier: "full", pixelRatioCap: 2 }, () => surfacePixelRatio()), 1);
  // Absent/garbage DPR must not produce 0 (a 0-px backing store is a blank canvas).
  assert.equal(withGlobals(0, { tier: "full", pixelRatioCap: 2 }, () => surfacePixelRatio()), 1);
  assert.equal(withGlobals(undefined as unknown as number, undefined, () => surfacePixelRatio()), 1);
});


test("both celestial surfaces size their backing store through the cap, not raw DPR", async () => {
  // The unit tests above prove the helper is correct; this proves it is WIRED.
  // Without it the helper could be perfect and both files still uncapped.
  const fs = await import("node:fs");
  const path = await import("node:path");
  const url = await import("node:url");
  const here = path.dirname(url.fileURLToPath(import.meta.url));

  // Q19 (2026-08-14): extended past the two celestial files to EVERY client
  // module that sizes a canvas. D8 found three more the moment it was written —
  // DataWorldMap (3 sites), bot.tsx, login.tsx — and login's was the largest of
  // all: a full-viewport animated canvas with its own rAF loop, not the small
  // widget its filename suggests. Naming files individually would have missed
  // the next one, so the companion test below closes the whole class.
  for (const rel of ["celestial/celestialSky.ts", "celestial/spaceFrame.ts"]) {
    const raw = fs.readFileSync(path.join(here, rel), "utf8");
    // STRIP COMMENTS FIRST (PROGRAM_STATE.md L15): the comments explaining this
    // very fix quote `devicePixelRatio`, and a source scan cannot tell code
    // from prose about code. Four checks were broken this way before the rule.
    const code = raw.split("\n").filter((l) => !/^\s*(\/\/|\*|\/\*)/.test(l)).join("\n");

    assert.ok(
      /surfacePixelRatio\s*\(/.test(code),
      `${rel} must size its backing store via surfacePixelRatio()`,
    );
    const resize = code.slice(code.indexOf("function resizeBacking"));
    assert.ok(
      !/devicePixelRatio/.test(resize.slice(0, 600)),
      `${rel}'s resizeBacking() must not read devicePixelRatio directly — ` +
      `that is the uncapped path this fix removed`,
    );
  }
});


test("no client module sizes a canvas from raw devicePixelRatio", async () => {
  // D8 as an assertion rather than a counter. `uncapped_surface` reached 0 in
  // Q19; this is what keeps it there — a counter tells you afterwards, a test
  // stops the merge.
  //
  // deviceTier.ts itself is the ONE legitimate reader: it is where the clamp
  // lives. Everything else goes through surfacePixelRatio().
  const fs = await import("node:fs");
  const path = await import("node:path");
  const cp = await import("node:child_process");
  const url = await import("node:url");
  const here = path.dirname(url.fileURLToPath(import.meta.url));
  const root = path.resolve(here, "..", "..", "..");

  const files = cp.execFileSync("git", ["ls-files", "client/src/**/*.ts", "client/src/**/*.tsx"],
    { cwd: root, encoding: "utf8" })
    .split("\n")
    .filter((f) => f && !f.includes(".test.")
      // Two legitimate readers, and only two:
      //  - lib/deviceTier.ts  — where the clamp itself lives.
      //  - pages/datamap.tsx  — where the tier reading is PRODUCED. It needs the
      //    raw ratio as INPUT to classifyDevice (alongside the GL renderer
      //    string) and publishes the result on __vtDeviceTier for every other
      //    surface to consume. It cannot call surfacePixelRatio() without
      //    circularity. The test below proves it still clamps, so this
      //    exemption cannot quietly become a loophole.
      && !f.endsWith("lib/deviceTier.ts") && !f.endsWith("pages/datamap.tsx"));

  const offenders: string[] = [];
  for (const f of files) {
    const raw = fs.readFileSync(path.join(root, f), "utf8");
    // Strip comments FIRST (PROGRAM_STATE.md L15) — the comments explaining
    // this very fix quote `devicePixelRatio`, and a source scan cannot tell
    // code from prose about code.
    const code = raw.split("\n").filter((l) => !/^\s*(\/\/|\*|\/\*)/.test(l)).join("\n");
    if (/\bwindow\.devicePixelRatio|\bglobalThis\.devicePixelRatio/.test(code)) offenders.push(f);
  }

  assert.deepEqual(
    offenders, [],
    `these modules read devicePixelRatio directly instead of surfacePixelRatio():\n` +
    offenders.map((f) => `  ${f}`).join("\n") +
    `\n\nBacking-store cost is QUADRATIC in the ratio — an uncapped 3x phone ` +
    `allocates 9x the pixels of a 1x surface, for every raster op the module does.`,
  );
});


test("datamap.tsx — the one exempt reader — still clamps to the tier cap", async () => {
  // The exemption above is only safe while this stays true. An allowlist entry
  // that stops honouring the rule is worse than no rule at all: it looks
  // covered while it is not.
  const fs = await import("node:fs");
  const path = await import("node:path");
  const url = await import("node:url");
  const here = path.dirname(url.fileURLToPath(import.meta.url));
  const raw = fs.readFileSync(path.resolve(here, "..", "pages", "datamap.tsx"), "utf8");
  const code = raw.split("\n").filter((l) => !/^\s*(\/\/|\*|\/\*)/.test(l)).join("\n");

  assert.match(code, /classifyDevice\s*\(/,
    "datamap.tsx is exempt because it PRODUCES the tier reading — if it no " +
    "longer calls classifyDevice, it is just another uncapped reader");
  assert.match(code, /Math\.min\(\s*dpr\s*,\s*tier\.pixelRatioCap\s*\)/,
    "datamap.tsx must still clamp the raw ratio to tier.pixelRatioCap before " +
    "sizing anything");
  assert.match(code, /__vtDeviceTier/,
    "datamap.tsx must keep publishing __vtDeviceTier — surfacePixelRatio() " +
    "reads it so all surfaces share one classification");
});
