// layerContract — Law IV, made declarable and checkable.
//
// Law IV says every layer "declares a hard VRAM/heap budget and a max
// feature count, downsamples above it rather than degrading, and
// implements explicit teardown". Auditing this repo against that found a
// split result worth stating precisely, because it changes what this
// module has to do:
//
//   TEARDOWN MOSTLY EXISTS. Every WebGL layer already implements
//   MapLibre's `onRemove` and already deletes its programs and buffers.
//   What is missing is a UNIFORM NAME the harness can assert on, and a
//   path to teardown that does not require the caller to hold a `gl`.
//
//   BUDGETS DO NOT EXIST AT ALL. Zero layers declare a feature cap. A
//   grep for MAX_INSTANCES / MAX_FEATURES / *_CAP across satLayer,
//   airLayer, flightTrackLayer, arcLayer and modelLayer returns nothing.
//   Every one of them renders whatever it is handed.
//
// So this module is mostly about the second half: declaring caps, and
// giving layers ONE downsample path that is honest about what it dropped.
//
// THE NO-SILENT-CAPS RULE (CLAUDE.md) IS THE POINT. A cap that quietly
// truncates is worse than no cap: the map looks complete and is not.
// `downsampleByImportance` therefore always returns what it dropped, and
// publishes it to the perf HUD, so "3,500 of 12,000 aircraft" is visible
// rather than inferred.

import { METRIC, addGauge, setGauge } from "./perfMetrics.ts";

/** Bytes in a megabyte, for budget arithmetic. */
const MB = 1024 * 1024;

/**
 * What every layer module must export. PR10's static assertion checks for
 * exactly these three names.
 */
export interface LayerContract {
  /** Hard cap on rendered features. Above it, downsample — never drop
   *  silently, never degrade the whole layer. */
  readonly maxFeatures: number;
  /** Declared VRAM ceiling in MEGABYTES (Law IV states budgets in MB). */
  readonly vramBudget: number;
  /** Explicit teardown. Must be idempotent and safe on a lost context. */
  dispose(): void;
}

/** The shape of a module under audit — everything optional so a missing
 *  export is a finding rather than a type error. */
export interface LayerModuleShape {
  maxFeatures?: unknown;
  vramBudget?: unknown;
  dispose?: unknown;
}

/**
 * Verify a layer module against the contract. Returns human-readable
 * violations, empty when compliant. Shared by the unit test and by PR10's
 * harness assertion so both check the same predicate.
 */
export function verifyLayerContract(name: string, mod: LayerModuleShape): string[] {
  const out: string[] = [];
  if (typeof mod.maxFeatures !== "number" || !(mod.maxFeatures > 0)) {
    out.push(`${name}: must export a positive numeric \`maxFeatures\` (Law IV)`);
  }
  if (typeof mod.vramBudget !== "number" || !(mod.vramBudget > 0)) {
    out.push(`${name}: must export a positive numeric \`vramBudget\` in MB (Law IV)`);
  }
  if (typeof mod.dispose !== "function") {
    out.push(`${name}: must export a \`dispose()\` function (Law IV teardown)`);
  }
  return out;
}

// ── downsampling ────────────────────────────────────────────────────────────

export interface DownsampleResult<T> {
  /** The features to render, at most `maxFeatures` of them. */
  kept: T[];
  /** How many were dropped. Zero when under cap. */
  dropped: number;
  /** Input size, so a caller can render "N of M". */
  total: number;
  /** True when the cap actually bit. */
  capped: boolean;
}

/**
 * Keep the `maxFeatures` most important items, and SAY how many went.
 *
 * `importanceOf` returns higher-is-more-important. Ties keep input order,
 * so the result is deterministic frame to frame — an unstable sort here
 * would make capped features flicker in and out between frames, which is
 * a Law I violation produced by a Law IV mechanism.
 *
 * Deliberately NOT a random sample and NOT a head-truncate: dropping the
 * first N in array order silently deletes whatever the upstream feed
 * happened to list last, which is how a map ends up quietly missing a
 * whole region.
 */
export function downsampleByImportance<T>(
  items: readonly T[],
  maxFeatures: number,
  importanceOf: (item: T) => number,
): DownsampleResult<T> {
  const total = items.length;
  if (!(maxFeatures > 0)) return { kept: [], dropped: total, total, capped: total > 0 };
  if (total <= maxFeatures) return { kept: items.slice(), dropped: 0, total, capped: false };

  // Decorate-sort-undecorate with the index as the tiebreak, so equal
  // importances resolve identically every frame.
  const decorated = items.map((item, i) => ({ item, i, w: importanceOf(item) }));
  decorated.sort((a, b) => b.w - a.w || a.i - b.i);
  const kept = decorated.slice(0, maxFeatures).map((d) => d.item);
  return { kept, dropped: total - maxFeatures, total, capped: true };
}

/**
 * Apply a cap and REPORT it. The reporting is the contract, not a nicety:
 * CLAUDE.md's no-silent-caps rule exists because a truncated layer looks
 * identical to a complete one.
 */
export function applyFeatureCap<T>(
  layerId: string,
  items: readonly T[],
  contract: Pick<LayerContract, "maxFeatures">,
  importanceOf: (item: T) => number,
): DownsampleResult<T> {
  const res = downsampleByImportance(items, contract.maxFeatures, importanceOf);
  addGauge(METRIC.FEATURES, res.kept.length);
  if (res.capped) {
    setGauge(`${layerId}.dropped`, res.dropped);
    // eslint-disable-next-line no-console
    console.warn(
      `[layer:${layerId}] feature cap ${contract.maxFeatures} bit — rendering ${res.kept.length} of ${res.total}, ${res.dropped} downsampled out by importance`,
    );
  } else {
    setGauge(`${layerId}.dropped`, 0);
  }
  return res;
}

// ── budget arithmetic ───────────────────────────────────────────────────────

/**
 * VRAM a packed vertex/instance buffer occupies.
 * `stride` is FLOATS per feature (not bytes) — every layer here packs
 * Float32Array, and quoting the stride in floats is how the layers
 * already describe themselves (e.g. SAT_STRIDE = 7).
 */
export function packedBufferBytes(featureCount: number, strideFloats: number): number {
  if (!(featureCount > 0) || !(strideFloats > 0)) return 0;
  return featureCount * strideFloats * 4;
}

/**
 * Does a layer's worst case fit its declared budget? Used by the tests to
 * prove each declared `vramBudget` is actually derived from the layer's
 * own arithmetic rather than picked to look reasonable.
 */
export function fitsBudget(worstCaseBytes: number, vramBudgetMB: number): boolean {
  return worstCaseBytes <= vramBudgetMB * MB;
}

export function budgetBytes(vramBudgetMB: number): number {
  return vramBudgetMB * MB;
}

// ── the live registry ───────────────────────────────────────────────────────

export interface RegisteredLayer {
  id: string;
  maxFeatures: number;
  vramBudget: number;
  dispose: () => void;
}

const live = new Map<string, RegisteredLayer>();

/**
 * Register a mounted layer so (a) the harness can toggle and audit every
 * layer generically and (b) a view teardown can dispose everything without
 * knowing what is mounted. Returns an unregister function.
 */
export function registerLayer(layer: RegisteredLayer): () => void {
  live.set(layer.id, layer);
  return () => {
    live.delete(layer.id);
  };
}

export function liveLayers(): RegisteredLayer[] {
  return Array.from(live.values());
}

export function totalDeclaredVramMB(): number {
  let n = 0;
  live.forEach((l) => {
    n += l.vramBudget;
  });
  return n;
}

/**
 * Dispose every registered layer. Each dispose is isolated: one layer
 * throwing during teardown must not strand the rest, or a single bad
 * layer turns a clean unmount into a guaranteed leak.
 */
export function disposeAllLayers(): { disposed: number; errors: unknown[] } {
  const errors: unknown[] = [];
  let disposed = 0;
  Array.from(live.values()).forEach((l) => {
    try {
      l.dispose();
      disposed++;
    } catch (err) {
      errors.push(err);
      // eslint-disable-next-line no-console
      console.error(`[layer:${l.id}] dispose threw`, err);
    }
  });
  live.clear();
  return { disposed, errors };
}

/** Registry reset for tests and the harness's between-case baseline. */
export function resetLayerRegistry(): void {
  live.clear();
}
