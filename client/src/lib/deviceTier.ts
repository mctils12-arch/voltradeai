// Device capability tier + adaptive render-fidelity governor.
//
// WHY (live incident 2026-07-21, round 15): a user's machine lost its WebGL
// context outright ("3D rendering lost" banner, blank map) with all layers +
// terrain on — the map must MEASURE what the machine can do and adapt,
// instead of assuming every device can push a 2×-DPR canvas with terrain
// RTT, 10k satellites, and instanced aircraft at 60fps.
//
// HONESTY CONTRACT: adaptation NEVER drops data — every layer stays on and
// every number stays real. The only lever is canvas pixel density
// (map.setPixelRatio): smooth-but-softer beats sharp-but-dead. Every step
// is surfaced to the user (the notice chip), never silent.
//
// Two halves, both pure and unit-tested:
//   classifyDevice  — static reading of the machine at startup (GPU class
//                     from the renderer string, RAM, cores) → starting cap.
//   frame governor  — runtime feedback loop on real rAF frame times with
//                     hysteresis; steps the ratio down under sustained
//                     overload, back up after sustained calm.

export type DeviceTier = "full" | "reduced" | "minimal";

export interface TierReading {
  tier: DeviceTier;
  /** starting cap for map pixelRatio (device may be higher; we never render above this) */
  pixelRatioCap: number;
  /** honest, user-showable reasons for the classification */
  reasons: string[];
}

// Software rasterizers: no GPU at all (VMs, remote desktop, blocklisted
// drivers, corporate lockdowns). Same detection the sky/atmosphere tier
// uses (globeAtmosphere.skyForRenderer).
const SOFTWARE_GL = /swiftshader|llvmpipe|softpipe|software rasterizer|microsoft basic render/i;
// Old low-end integrated GPUs that predate the UHD 620 class — real GPUs,
// but terrain RTT at 2× DPR is beyond them. Deliberately narrow: a wrong
// "full" is recoverable (the governor steps down); a wrong "reduced" would
// permanently soften a capable machine.
const WEAK_IGPU = /intel(\(r\))?\s*(hd|uhd)\s*graphics\s*(4|5)\d\d\b/i;

export function classifyDevice(input: {
  renderer: string;
  deviceMemoryGB?: number;
  cores?: number;
  devicePixelRatio: number;
}): TierReading {
  const reasons: string[] = [];
  const dpr = Math.max(1, input.devicePixelRatio || 1);
  if (SOFTWARE_GL.test(input.renderer)) {
    reasons.push("software 3D renderer (no GPU acceleration available)");
    return { tier: "minimal", pixelRatioCap: 1, reasons };
  }
  if (WEAK_IGPU.test(input.renderer)) reasons.push("older integrated graphics");
  if (input.deviceMemoryGB != null && input.deviceMemoryGB <= 4) reasons.push("limited system memory");
  if (input.cores != null && input.cores <= 2) reasons.push("few CPU cores");
  if (reasons.length) return { tier: "reduced", pixelRatioCap: Math.min(dpr, 1.5), reasons };
  // Full tier still caps at 2×: above that, extra canvas pixels are
  // invisible on a map but quadratic in GPU cost (3× phone DPRs).
  return { tier: "full", pixelRatioCap: Math.min(dpr, 2), reasons };
}

// ── frame governor ─────────────────────────────────────────────────────────
// Feedback loop with hysteresis. Inputs are rolling MEDIAN frame times from
// real requestAnimationFrame deltas (what jank feels like), not GPU timers.
// Steps down fast (a machine at 10fps is drowning), steps up slowly (only
// after sustained calm), and never flaps: one step per cooldown window.

/** ratio ladder — steps the governor walks; floor 1 keeps text readable */
export const GOV_LADDER = [2, 1.75, 1.5, 1.25, 1] as const;
/** median frame time that counts as overload (~11 fps) */
export const GOV_OVERLOAD_MS = 90;
/** overload must persist this long before a step down */
export const GOV_OVERLOAD_HOLD_MS = 8_000;
/** median frame time that counts as calm (~30+ fps) */
export const GOV_CALM_MS = 30;
/** calm must persist this long before a step back up */
export const GOV_CALM_HOLD_MS = 60_000;
/** minimum spacing between any two steps */
export const GOV_COOLDOWN_MS = 10_000;
/** LEVER ORDER (human 2026-07-21 round 16: "not reduce quality … of the
 *  terrain"): the overload FLAG (animation tick-stretch — smoothness
 *  lever, zero quality cost) engages at GOV_OVERLOAD_HOLD_MS; RESOLUTION
 *  steps wait this much longer, so pixels are only sacrificed when idle
 *  gaps alone couldn't save the machine. */
export const GOV_STRETCH_GRACE_MS = 22_000;

export interface GovState {
  /** current applied ratio */
  ratio: number;
  /** ceiling: the device's classified cap — never exceeded on step-up */
  ceil: number;
  overloadSince: number | null;
  calmSince: number | null;
  lastStepAt: number;
}

export function govInit(startRatio: number, now = 0): GovState {
  return { ratio: startRatio, ceil: startRatio, overloadSince: null, calmSince: null, lastStepAt: now };
}

export interface GovDecision {
  state: GovState;
  /** when set, apply this ratio to the map and surface `note` */
  apply?: number;
  note?: string;
}

const nextDown = (r: number): number | null => {
  for (const step of GOV_LADDER) if (step < r - 1e-6) return step;
  return null;
};
const nextUp = (r: number, ceil: number): number | null => {
  for (let i = GOV_LADDER.length - 1; i >= 0; i--) {
    const step = GOV_LADDER[i];
    if (step > r + 1e-6) return Math.min(step, ceil);
  }
  return null;
};

export function govStep(st: GovState, medianFrameMs: number, now: number): GovDecision {
  const s: GovState = { ...st };
  // classify the instant
  if (medianFrameMs > GOV_OVERLOAD_MS) {
    s.overloadSince = s.overloadSince ?? now;
    s.calmSince = null;
  } else if (medianFrameMs < GOV_CALM_MS) {
    s.calmSince = s.calmSince ?? now;
    s.overloadSince = null;
  } else {
    // between bands: neither trend accumulates
    s.overloadSince = null;
    s.calmSince = null;
  }
  const cooled = now - s.lastStepAt >= GOV_COOLDOWN_MS;
  if (s.overloadSince != null && now - s.overloadSince >= GOV_OVERLOAD_HOLD_MS + GOV_STRETCH_GRACE_MS && cooled) {
    const down = nextDown(s.ratio);
    if (down != null) {
      const from = s.ratio;
      s.ratio = down;
      s.lastStepAt = now;
      s.overloadSince = null;
      return {
        state: s, apply: down,
        note: `Render resolution lowered (${from.toFixed(2).replace(/\.?0+$/, "")}× → ${down.toFixed(2).replace(/\.?0+$/, "")}×) to keep the map smooth on this device — all layers and data stay on`,
      };
    }
  }
  if (s.calmSince != null && now - s.calmSince >= GOV_CALM_HOLD_MS && cooled) {
    const up = nextUp(s.ratio, s.ceil);
    if (up != null && up > s.ratio + 1e-6) {
      const from = s.ratio;
      s.ratio = up;
      s.lastStepAt = now;
      s.calmSince = null;
      return {
        state: s, apply: up,
        note: `Render resolution restored (${from.toFixed(2).replace(/\.?0+$/, "")}× → ${up.toFixed(2).replace(/\.?0+$/, "")}×) — this device has headroom again`,
      };
    }
  }
  return { state: s };
}

/** median of an array (probe/governor shared helper) */
export function median(xs: readonly number[]): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

// ── overload flag ───────────────────────────────────────────────────────────
// Second governor lever, for machines already at the 1× pixel floor (probe
// finding 2026-07-21: terrain render cost ∝ loaded tiles is paid on EVERY
// frame, and animation ticks — aircraft glide etc. — keep the renderer
// awake; a drowning machine needs idle gaps, not more frames). While
// overloaded, animation drivers skip alternate ticks: motion updates at
// half cadence, positions stay exact per poll — a fidelity trade, no data
// loss. Module-level store so render-loop code reads it without React.
let overloadedFlag = false;
export const setOverloaded = (v: boolean): void => { overloadedFlag = v; };
export const isOverloaded = (): boolean => overloadedFlag;

/** derive the overload flag from a governor state + current reading:
 *  ON after overload has held its window; OFF once calm has held 10s */
export function overloadFromState(st: GovState, now: number, prev: boolean): boolean {
  if (st.overloadSince != null && now - st.overloadSince >= GOV_OVERLOAD_HOLD_MS) return true;
  if (st.calmSince != null && now - st.calmSince >= 10_000) return false;
  return prev;
}

/** True when the GPU identity string marks a machine that needs the
 *  conservative shared-memory mitigations (smaller tile cache, etc.).
 *  Intel HD/UHD/Iris are shared-memory integrated parts (Arc is discrete;
 *  Apple/ARM integrated GPUs are not fragile the same way — scoped to
 *  Intel). FAIL-SAFE (repair 2026-08-05, GL-loss case file): an
 *  affirmative "no-webgl" probe means Chrome's GPU process was down at
 *  boot — direct evidence of the fragile-driver state the mitigations
 *  exist for; those machines must get them, not lose them. An unknown
 *  probe ("unavailable", empty) stays non-fragile — fail-safe applies
 *  only to the affirmative dead-GPU read. */
export function isFragileGpu(gpuInfo: string): boolean {
  if (gpuInfo === "no-webgl") return true;
  return /intel/i.test(gpuInfo) && !/\barc\b/i.test(gpuInfo);
}
