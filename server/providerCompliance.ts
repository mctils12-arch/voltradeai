// Monetization tripwire — runtime half (CLAUDE.md KNOWN STATE,
// human-approved 2026-07-03). airplanes.live's free API is licensed
// NON-COMMERCIAL: lawful for the current no-revenue proof of concept,
// unlawful the day billing goes live. If billing activates while a
// non-commercial provider is still in the aircraft chain, this module
// makes the violation loud: a COMPLIANCE-WARNING row in the persistent
// audit log (throttled) plus a failing check on /api/health — so even a
// dashboard-only monetization flip (no code change, no session) surfaces
// to the next DAILY routine's health check within hours. The session-side
// rule lives in CLAUDE.md KNOWN STATE; the decision record and provider
// terms analysis in research/wishlist.md (MONETIZATION TRIPWIRE).
//
// Pure module by design (same pattern as vesselStream.ts): no auth/db
// import — auth.ts's top-level sqlite open must not load into tests. The
// audit-log writer is injected by routes.ts at registration time.

// Providers currently in the aircraft chain (server/routes.ts) whose terms
// do NOT permit commercial use. Kept in sync with the chain by
// server/providerCompliance.test.ts. Empty this list only by dropping the
// provider from the chain or upgrading to a commercial arrangement.
export const NON_COMMERCIAL_AIRCRAFT_PROVIDERS = ["airplaneslive", "adsbfi"];

// CREDENTIAL-GATED non-commercial providers (FLIGHT PROGRAM B1,
// 2026-09-28): in the aircraft data path ONLY while their credentials are
// configured, so they count toward the tripwire exactly then. OpenSky
// (server/openskyGlobal.ts — global states/all snapshot feeding
// /api/data/aircraft/global) is licensed for non-profit research/education
// only, and operational REST use needs a written agreement. The env names
// are duplicated here (not imported) to keep this module dependency-free;
// providerCompliance.test.ts pins them to openskyGlobal.ts.
export const CREDENTIAL_GATED_NON_COMMERCIAL_PROVIDERS: { key: string; envAll: string[] }[] = [
  { key: "opensky", envAll: ["OPENSKY_CLIENT_ID", "OPENSKY_CLIENT_SECRET"] },
];

/** Every non-commercial provider actually in the data path under `env`. */
export function activeNonCommercialAircraftProviders(env: NodeJS.ProcessEnv = process.env): string[] {
  const gated = CREDENTIAL_GATED_NON_COMMERCIAL_PROVIDERS
    .filter((p) => p.envAll.every((k) => !!(env[k] && String(env[k]).trim())))
    .map((p) => p.key);
  return [...NON_COMMERCIAL_AIRCRAFT_PROVIDERS, ...gated];
}

// Billing counts as active when the operator flips BILLING_ENABLED
// explicitly, or when Stripe is configured at all — server/billing.ts
// (frozen) activates /api/billing/* on STRIPE_SECRET_KEY presence, so key
// presence IS the earliest observable monetization signal.
export function billingActive(env: NodeJS.ProcessEnv = process.env): boolean {
  return env.BILLING_ENABLED === "true" || !!env.STRIPE_SECRET_KEY;
}

export function aircraftProviderCompliance(env: NodeJS.ProcessEnv = process.env): {
  status: "ok" | "violation";
  detail?: string;
} {
  const active = activeNonCommercialAircraftProviders(env);
  if (billingActive(env) && active.length > 0) {
    return {
      status: "violation",
      detail:
        `billing is active but non-commercial-licensed aircraft provider(s) remain in the chain: ` +
        `${active.join(", ")} — drop or upgrade them before charging anyone ` +
        `(see research/wishlist.md MONETIZATION TRIPWIRE)`,
    };
  }
  return { status: "ok" };
}

// Loud surfacing: console + one audit-log row per throttle window.
// Called at route registration (boot) and on aircraft fetches. The writer
// is injected (routes.ts wires it to the persistent audit_log table).
type AuditWriter = (type: string, message: string) => void;
let auditWriter: AuditWriter | null = null;
export function setComplianceAuditWriter(w: AuditWriter): void {
  auditWriter = w;
}

let lastComplianceWarn = 0;
export function complianceAuditTick(
  env: NodeJS.ProcessEnv = process.env,
  intervalMs = 6 * 3600_000,
  now = Date.now(),
): boolean {
  const c = aircraftProviderCompliance(env);
  if (c.status === "ok") return false;
  if (now - lastComplianceWarn < intervalMs) return false;
  lastComplianceWarn = now;
  console.error("[COMPLIANCE]", c.detail);
  try { auditWriter?.("COMPLIANCE-WARNING", (c.detail || "").slice(0, 500)); } catch {}
  return true;
}
