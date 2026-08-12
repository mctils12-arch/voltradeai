// freshness — Law V, and the root cause of the OpenWeatherMap non-render.
//
// ── THE ROOT CAUSE (traced 2026-08-12, not a retry) ─────────────────────────
//
// Law V says any bug that has appeared twice gets a root-cause fix, and
// names the OpenWeatherMap non-render specifically. Here is the actual
// failure, traced through datamap.tsx rather than guessed:
//
//  1. The weather layers are added IMPERATIVELY (`map.addSource` /
//     `map.addLayer`) inside a `probe()` that runs on mount and then on a
//     `setInterval` every 10 minutes.
//  2. The map calls `setProjection()` at runtime — the globe/flat toggle.
//     A projection change rebuilds the style, and a style rebuild WIPES
//     every source and layer that was added imperatively.
//  3. The effect that owns the weather layers depends on
//     `[enabled.weather_temp, enabled.weather_wind, mapReady, setStatus]`.
//     A style rebuild changes none of those, so the effect does not re-run.
//  4. Nothing listens for `styledata` to re-add them.
//
// Net effect: toggle the globe with weather on and the layer silently
// disappears for UP TO TEN MINUTES, while the UI still reports "active".
// That is the non-render. It is not an upstream failure, not a key
// problem, and — this is the important part — NOT FIXABLE BY A RETRY,
// because the retry is what is already there and it is on a 10-minute
// timer. The fix is to make layer presence RECONCILED against map state
// rather than re-established on a poll.
//
// Reconciling on `styledata` does not violate Law I. Law I forbids
// recomputing visual state on CAMERA events (zoomend/moveend/move/idle/
// rotate/pitch/dragend) and data ticks. `styledata` is a lifecycle event —
// the map telling us it threw our layers away — and re-adding them is
// restoration, not animation. The existing drape-order guard already
// reconciles on the same event for the same reason.

/** A layer is stale once its data is older than this, by default. */
export const DEFAULT_STALE_AFTER_MS = 15 * 60_000;

/** Past this, the layer is not "stale", it is not live at all. */
export const DEFAULT_DEAD_AFTER_MS = 60 * 60_000;

export type FreshnessLevel = "fresh" | "stale" | "dead" | "unknown";

/**
 * Classify a data age. `unknown` is deliberately distinct from `dead`: a
 * layer that has never reported an age has not failed, it has failed to
 * INSTRUMENT itself — and Law V says a layer that cannot say how old it is
 * may not claim to be live.
 */
export function freshnessLevel(
  ageMs: number | null | undefined,
  staleAfterMs = DEFAULT_STALE_AFTER_MS,
  deadAfterMs = DEFAULT_DEAD_AFTER_MS,
): FreshnessLevel {
  if (ageMs == null || !Number.isFinite(ageMs) || ageMs < 0) return "unknown";
  if (ageMs >= deadAfterMs) return "dead";
  if (ageMs >= staleAfterMs) return "stale";
  return "fresh";
}

/**
 * Compact human age for a status line: "just now", "4m ago", "2h ago".
 * Deliberately coarse — a data age rendered to the second implies a
 * precision the polling interval does not have.
 */
export function formatDataAge(ageMs: number | null | undefined): string {
  if (ageMs == null || !Number.isFinite(ageMs) || ageMs < 0) return "age unknown";
  const s = Math.floor(ageMs / 1000);
  if (s < 45) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

/**
 * Does a layer that SHOULD be on need re-establishing?
 *
 * The predicate behind the fix. Pure so the reconcile rule is unit-tested
 * rather than inferred from a React effect.
 */
export function needsReconcile(input: {
  /** The user has this layer toggled on. */
  wanted: boolean;
  /** The upstream provider is currently serving. */
  providerOk: boolean;
  /** The source is present in the map right now. */
  sourcePresent: boolean;
  /** The layer is present in the map right now. */
  layerPresent: boolean;
}): boolean {
  if (!input.wanted || !input.providerOk) return false;
  // Either half missing is a broken layer: a source with no layer draws
  // nothing, and a layer with no source is an error on the next render.
  return !input.sourcePresent || !input.layerPresent;
}

/**
 * Tracks when a layer last received real data, so it can report its own
 * age (Law V). Separate from the fetch code on purpose — a layer must be
 * able to answer "how old am I" without knowing how it is fed.
 */
export class FreshnessTracker {
  private lastUpdateMs: number | null = null;
  private lastFailureMs: number | null = null;
  private consecutiveFailures = 0;

  constructor(
    readonly id: string,
    private readonly now: () => number = () => Date.now(),
  ) {}

  /** Real data landed. */
  markFresh(atMs?: number): void {
    this.lastUpdateMs = atMs ?? this.now();
    this.consecutiveFailures = 0;
  }

  /** A fetch or provider failed. Does NOT clear the last good timestamp —
   *  Law V wants last-known state rendered, not blanked. */
  markFailure(atMs?: number): void {
    this.lastFailureMs = atMs ?? this.now();
    this.consecutiveFailures++;
  }

  get ageMs(): number | null {
    if (this.lastUpdateMs == null) return null;
    return Math.max(0, this.now() - this.lastUpdateMs);
  }

  get level(): FreshnessLevel {
    return freshnessLevel(this.ageMs);
  }

  get failures(): number {
    return this.consecutiveFailures;
  }

  get lastFailureAtMs(): number | null {
    return this.lastFailureMs;
  }

  /** The status-line string. Always says something — a layer that cannot
   *  report its age says so rather than implying freshness by silence. */
  describe(): string {
    const age = formatDataAge(this.ageMs);
    if (this.level === "unknown") return "age unknown";
    if (this.consecutiveFailures > 0) return `${age} · retrying (${this.consecutiveFailures})`;
    return age;
  }

  reset(): void {
    this.lastUpdateMs = null;
    this.lastFailureMs = null;
    this.consecutiveFailures = 0;
  }
}

/**
 * Log a provider chain that has fallen through to its LAST option.
 *
 * Law V: "A provider chain that falls through to its last option logs the
 * failure loudly. Silent degradation is the reason staleness recurs." The
 * point is that running on the last provider is a DEGRADED state that
 * looks identical to a healthy one from the outside — the chain is still
 * returning data, so nothing else notices.
 */
export function reportChainFallthrough(chainId: string, providerIndex: number, providers: readonly string[]): boolean {
  const isLast = providers.length > 0 && providerIndex === providers.length - 1;
  if (!isLast) return false;
  // eslint-disable-next-line no-console
  console.error(
    `[freshness:${chainId}] provider chain fell through to its LAST option ` +
      `('${providers[providerIndex]}', ${providerIndex + 1}/${providers.length}) — ` +
      `no fallback remains. Chain: ${providers.join(" -> ")}`,
  );
  return true;
}
