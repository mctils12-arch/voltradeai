/**
 * spaceweatherNormalizer.ts — the ONE home for NOAA SWPC rtsw field names
 * (json/rtsw/rtsw_wind_1m.json + rtsw_mag_1m.json). Space-weather hardening
 * Phase 0.3: every upstream rtsw field name (time_tag, source, active,
 * overall_quality, proton_speed, proton_density, proton_temperature, bt,
 * bz_gsm, phi_gsm, theta_gsm) lives HERE and nowhere else — downstream code
 * (the future 3D solar-wind view, spaceWeather.ts freshness) speaks only the
 * normalized names.
 *
 * Live-probed 2026-07-31: the rtsw files interleave THREE-plus spacecraft at
 * overlapping time_tags (seen that day: ACE, IMAP, SOLAR1 — with active:true
 * on SOLAR1 and DSCOVR absent entirely). `source` is therefore an OPEN string
 * enum — never a hardcoded spacecraft list. Rows with overall_quality 2 can
 * carry null proton_* values; nulls stay null, NaN never escapes this module.
 * File ordering is newest-first upstream but is NEVER trusted — consumers
 * sort by time_tag themselves (latestActive does).
 */

export interface NormRow {
  /** upstream time_tag, kept verbatim (ISO, UTC — often without a Z) */
  time_tag: string;
  /** spacecraft — OPEN string enum (ACE/IMAP/SOLAR1/DSCOVR/…, never a fixed list) */
  source: string;
  /** NOAA's designated-source flag; only rows with active===true are the feed of record */
  active: boolean;
  /** overall_quality as published (2 = degraded, can carry null proton_*) */
  quality: number | null;
  speedKms: number | null; // proton_speed
  densityPcc: number | null; // proton_density (protons/cm^3)
  tempK: number | null; // proton_temperature
  btNt: number | null; // bt
  bzNt: number | null; // bz_gsm
  phiGsm: number | null; // phi_gsm (degrees)
  thetaGsm: number | null; // theta_gsm (degrees)
}

/** null/""/unparseable → null; a finite number stays a number. NaN never leaks. */
const num = (v: unknown): number | null => {
  if (v == null || v === "") return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
};

/**
 * Parse a NOAA time tag as UTC. SWPC feeds are all UTC but routinely omit the
 * timezone marker ("2026-07-31T12:00:00", "2026-07-27 14:53:19.230") — a bare
 * Date.parse would read those as HOST-LOCAL time. Returns epoch ms, or null
 * for missing/unparseable input.
 */
export function noaaUtcMs(s: string | null | undefined): number | null {
  if (typeof s !== "string" || !s.trim()) return null;
  let iso = s.trim().replace(" ", "T");
  if (!/([zZ]|[+-]\d{2}:?\d{2})$/.test(iso)) iso += "Z";
  const ms = Date.parse(iso);
  return Number.isFinite(ms) ? ms : null;
}

/**
 * Validate + normalize a raw rtsw payload. Rows missing a string time_tag or
 * a string source are dropped (shape gate); everything else maps NOAA names →
 * ours with nulls preserved. Non-array input → [].
 */
export function normalizeRtswRows(raw: unknown): NormRow[] {
  if (!Array.isArray(raw)) return [];
  const out: NormRow[] = [];
  for (const r of raw as any[]) {
    const t = r?.time_tag;
    const src = r?.source;
    if (typeof t !== "string" || typeof src !== "string") continue;
    out.push({
      time_tag: t,
      source: src,
      active: r?.active === true,
      quality: num(r?.overall_quality),
      speedKms: num(r?.proton_speed),
      densityPcc: num(r?.proton_density),
      tempK: num(r?.proton_temperature),
      btNt: num(r?.bt),
      bzNt: num(r?.bz_gsm),
      phiGsm: num(r?.phi_gsm),
      thetaGsm: num(r?.theta_gsm),
    });
  }
  return out;
}

/**
 * Newest row with active===true, sorted by time_tag DESC internally — input
 * ordering is never trusted (the live file is newest-first today; that is an
 * observation, not a contract). With requireQuality (default true), rows
 * whose quality===2 or whose speedKms is null are skipped, so the active
 * source's newest QUALITY row wins over a newer degraded row.
 */
export function latestActive(rows: NormRow[], opts?: { requireQuality?: boolean }): NormRow | null {
  const requireQuality = opts?.requireQuality ?? true;
  const sorted = [...rows].sort(
    (a, b) => (noaaUtcMs(b.time_tag) ?? -Infinity) - (noaaUtcMs(a.time_tag) ?? -Infinity),
  );
  for (const r of sorted) {
    if (r.active !== true) continue;
    if (requireQuality && (r.quality === 2 || r.speedKms === null)) continue;
    return r;
  }
  return null;
}

/**
 * Pure staleness check: age of the newest record vs a per-feed threshold.
 * Missing/unparseable timestamp is honestly stale with a null age — a feed
 * that produced nothing has no fresh data.
 */
export function assessStaleness(
  newestTimeTagIso: string | null,
  maxAgeMs: number,
  nowMs: number,
): { stale: boolean; ageMs: number | null } {
  const t = noaaUtcMs(newestTimeTagIso);
  if (t === null) return { stale: true, ageMs: null };
  const ageMs = nowMs - t;
  return { stale: ageMs > maxAgeMs, ageMs };
}
