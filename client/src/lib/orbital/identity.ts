// Satellite identity — EARTH TWIN E4-1 (research/earth_twin_program.md V1
// "IDENTITY BEFORE MODELS"): the click card says "this is a small payload /
// a rocket body / debris, owned by X, launched Y" from SATCAT metadata that
// already ships free with the catalog — no 3D models required to answer
// "what am I looking at?".
//
// Pure formatting only (testable without a DOM): the caller supplies the
// SATCAT record (or null while the catalog is still downloading) and an
// optional resolved operator from ./entityJoin. Every derived claim is
// LABELED derived; absent data reads as an honest gap, never a guess.

import type { SatcatRecord } from './tle.js';
import type { ResolvedOperator } from './entityJoin.js';

/**
 * SATCAT operational-status codes, per CelesTrak's documented legend
 * (celestrak.org/satcat/status.php). Raw code kept alongside the decode —
 * decode tables are documentation, not inference.
 */
export const OPS_STATUS_DECODE: Record<string, string> = {
  '+': 'operational',
  '-': 'nonoperational',
  P: 'partially operational',
  B: 'backup/standby',
  S: 'spare',
  X: 'extended mission',
  D: 'decayed',
  '?': 'unknown',
};

/**
 * Derive a constellation/operator lookup stem from a satellite name:
 * "STARLINK-3021" → "STARLINK", "IRIDIUM 167" → "IRIDIUM", "ISS (ZARYA)" →
 * "ISS". The stem is only a LOOKUP KEY for ./entityJoin's conservative
 * exact-variant resolver — an unmapped stem resolves to null (honest gap),
 * so this derivation can never manufacture an operator.
 */
export function nameStemForOperator(name: string | null | undefined): string | null {
  if (!name) return null;
  const stem = name.replace(/[-\s]+[\d(].*$/, '').trim();
  return stem.length >= 2 ? stem : null;
}

/**
 * Identity lines for the satellite click card. `sc === null` means the
 * SATCAT catalog is not available; `satcatState` says why, honestly.
 */
export function satelliteIdentityLines(
  sc: SatcatRecord | null,
  op: ResolvedOperator | null,
  satcatState: 'loading' | 'ready' | 'error' = 'ready',
): string[] {
  if (!sc) {
    return [
      satcatState === 'loading'
        ? 'Identity (SATCAT) still downloading — reopen this card in a moment for type/owner/launch.'
        : satcatState === 'ready'
          ? 'Not in the SATCAT catalog (fresh launch or analyst object) — a per-object gap, orbital data above is unaffected.'
          : 'Identity (SATCAT) unavailable right now — orbital data above is unaffected.',
    ];
  }
  const lines: (string | null)[] = [
    sc.objectType
      ? `Object type: ${sc.objectType}` +
        (sc.objectType === 'PAYLOAD' && sc.rcsSize === 'SMALL'
          ? ' — smallsat/CubeSat-class size (derived from radar cross-section, not a catalog field)'
          : '')
      : null,
    sc.rcsMeters2 != null
      ? `Radar cross-section: ${sc.rcsMeters2} m²${sc.rcsSize ? ` (${sc.rcsSize} — CelesTrak size buckets)` : ''}`
      : sc.rcsSize
        ? `Radar cross-section class: ${sc.rcsSize} (CelesTrak size buckets)`
        : null,
    sc.owner ? `Owner: ${sc.owner} (SATCAT country/agency code)` : null,
    op
      ? `Operator: ${op.company}` +
        (op.ticker
          ? ` — ${op.ticker}${op.exchange ? ` (${op.exchange})` : ''}`
          : op.is_public
            ? ''
            : ' — no public equity') +
        ` · matched on "${op.matched_on}" (curated map, as of ${op.asof})`
      : null,
    sc.launchDate ? `Launched: ${sc.launchDate}` : null,
    sc.opStatus
      ? `Status: ${OPS_STATUS_DECODE[sc.opStatus] ?? 'code not in decode table'} (SATCAT code ${sc.opStatus})`
      : null,
    sc.intlDes ? `International designator: ${sc.intlDes}` : null,
  ];
  return lines.filter((l): l is string => l != null);
}
