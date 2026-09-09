/**
 * bisectionFlags.ts — Tier 2 / Tier 3 kill-switches for the KNOWN BROKEN #41
 * (2026-09-08) production crash-loop bisection.
 *
 * WHY THIS EXISTS: production has OOM-crash-looped every ~90-130s during
 * market hours since 2026-09-08. Two independent, confirmed-live fixes
 * (server/portDwellCapture.ts's cooldown guard, server/crashSafeRefresh.ts's
 * shadowstats/portdwell guard) did NOT resolve it — five more full
 * boot/crash cycles were observed after both were live (see
 * research/open_questions.md KNOWN BROKEN #41 and research/wishlist.md's
 * 2026-09-08 incident entry for the full account). Per CLAUDE.md's
 * RECURRENCE ESCALATES rule, a third blind cooldown-style patch is
 * forbidden — the incident needs either Railway's raw stderr/crash logs
 * (not available to this sandbox) or the bisection the wishlist entry
 * names as the free next step: temporarily disable Tier 2 and Tier 3
 * entirely and watch whether the loop stops, which would localize the leak
 * to the scan/strategic-scan path versus something that runs
 * unconditionally (the WebSocket stream, the position monitor, Express
 * itself, or a module-load-time leak).
 *
 * BOTH FLAGS DEFAULT OFF — zero behavior change unless explicitly set.
 * This sandbox has no Railway API/CLI access, so it cannot flip these in
 * production itself; a human (or a future session with that access) sets
 * VOLTRADE_DISABLE_TIER2=1 / VOLTRADE_DISABLE_TIER3=1 in the Railway
 * environment during market hours to run the actual experiment, then
 * unsets them once the incident is closed either way.
 */

export function tier2Disabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return env.VOLTRADE_DISABLE_TIER2 === "1";
}

export function tier3Disabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return env.VOLTRADE_DISABLE_TIER3 === "1";
}
