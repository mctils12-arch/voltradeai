/**
 * cacheBackfill.ts — shared decision logic for the "cold cache, no on-disk
 * backfill" bug fixed independently, by hand, in the identical shape across
 * 7+ datacore modules since 2026-09-10 (githubOrgActivity, wikiAttention,
 * satellites, euLoad, edgarForm4, nasaFirms, usgsQuakes, ndbcBuoys — see
 * research/experiments.md / research/PROGRAM_STATE.md for the full trail).
 * Every occurrence was the same shape: a module-level in-memory cache fed
 * only by a live poll, staying null/empty across a cold boot or a live
 * outage even when a real on-disk archive already had data to serve —
 * routes.ts then reports `warming_up`/`count:0` while the archive sits
 * unread.
 *
 * RENDERING & MOTION LAW, Law V (Freshness, CLAUDE.md): provider failover
 * must never block first paint — render last-known cached state
 * immediately, swap when live data lands. This is that rule's server-side,
 * pre-render form.
 *
 * REPAIR MANDATE (CLAUDE.md HEALTH OF THE LOOP ITSELF, RECURRENCE
 * ESCALATES): the same bug shape fixed by hand 7+ times in the same
 * subsystem is architecture smell, not 7 coincidences. This compiles the
 * repeated decision into one tested function (EDGE DOCTRINE #3) instead of
 * a pattern each new module's author has to remember and re-derive by hand.
 *
 * SCOPE: only the 4 modules whose refresh function is byte-identical in
 * this decision shape — a flat `{at, items[]}` cache, one live fetch call,
 * backfill attempted on both an empty-but-successful live result and a
 * thrown fetch error — are retrofitted onto this helper: nasaFirms.ts,
 * ndbcBuoys.ts, usgsQuakes.ts, edgarForm4.ts. wikiAttention.ts,
 * satellites.ts and euLoad.ts were read and deliberately NOT retrofitted:
 * they cache a DERIVED aggregate (a picked "latest complete day", a
 * per-group map, computed zone stats) rather than a flat item list, so
 * forcing them onto a `T[]`-shaped helper would either lose that
 * structural difference or require guessing at a wrapping shape nobody
 * asked for. Filed as an open question, not fixed here (scope discipline —
 * one logical change): wikiAttention.ts and euLoad.ts also never attempt a
 * backfill on the thrown-fetch-error path at all (only on an empty-but-
 * non-throwing live result), which may be the same bug's third occurrence
 * in a different shape — see research/open_questions.md.
 */

/**
 * Decides which item list a module's cache should hold after one poll
 * attempt, given the live fetch's result (or `[]` standing in for "the
 * fetch threw, so there is no live result to consider") and how to
 * reconstruct a cache-shaped list from the on-disk archive.
 *
 * - A non-empty live result always wins, regardless of whether a cache
 *   already existed (mirrors every retrofitted module's own first branch:
 *   a real live poll always supersedes whatever was cached).
 * - An empty/failed live result backfills from disk ONLY when there is no
 *   existing cache to fall back on already — a transient empty poll must
 *   never overwrite a good cache with a stale archive read.
 * - Returns `null` to mean "leave the existing cache untouched", so every
 *   call site's own `if (next) cache = {...}` reads as "only touch the
 *   cache when there's something better to put in it."
 *
 * `backfill` is only invoked when actually needed (never speculatively),
 * so a module whose archive read is nontrivial (gunzip, multi-day scan)
 * pays that cost only on the cold/failure path this function exists for.
 */
export function resolveCacheItems<T>(
  hasPrevCache: boolean,
  liveItems: T[],
  backfill: () => T[],
): T[] | null {
  if (liveItems.length > 0) return liveItems;
  if (!hasPrevCache) {
    const archived = backfill();
    if (archived.length > 0) return archived;
  }
  return null;
}
