// dataPanelsView — the sentences and figures of two /data panels added in
// UI coverage batch C (2026-11-20): plants under weather alerts and rail
// carloads. Pure functions with tests; the pages only lay them out.

/** Megawatts are a fixed domain convention in both unit systems (CLAUDE.md
 *  UNITS PREFERENCE). The server writes 0 when the plant database states no
 *  capacity, so 0 reads "not stated", never "0 MW". */
export function mwLabel(mw: number | null | undefined): string {
  if (typeof mw !== "number" || !Number.isFinite(mw) || mw <= 0) return "not stated";
  return `${mw >= 100 ? Math.round(mw).toLocaleString("en-US") : mw.toFixed(1)} MW`;
}

/** "gas 3,241 MW · solar 85 MW", largest first; fuels with no stated capacity dropped. */
export function fuelLine(byFuel: Record<string, number> | null | undefined): string {
  const rows = Object.entries(byFuel ?? {}).filter(([, v]) => typeof v === "number" && v > 0).sort((a, b) => b[1] - a[1]);
  return rows.length ? rows.map(([k, v]) => `${k} ${mwLabel(v)}`).join(" · ") : "no capacity stated";
}

const span = (ms: number): string => {
  const min = Math.round(ms / 60000);
  if (min < 60) return `${min} min`;
  const h = Math.floor(min / 60);
  if (h < 48) return `${h} h ${min % 60} min`;
  return `${Math.floor(h / 24)} d`;
};

/** When the warning ends, against the reader's clock. NWS leaves the end
 *  empty on some warnings (river floods "until further notice"). */
export function endsLabel(ends: string | null | undefined, nowMs: number): string {
  if (!ends) return "no end time stated";
  const t = Date.parse(ends);
  if (!Number.isFinite(t)) return "no end time stated";
  return t <= nowMs ? `end time passed ${span(nowMs - t)} ago` : `ends in ${span(t - nowMs)}`;
}

/** Age of the join: the NWS poll it was computed from. */
export function ageLabel(asOfMs: number | null | undefined, nowMs: number): string {
  if (typeof asOfMs !== "number" || !Number.isFinite(asOfMs)) return "age unknown";
  const d = nowMs - asOfMs;
  return d < 60000 ? "under a minute ago" : `${span(d)} ago`;
}

/** Signed, one decimal; a missing prior week is a dash, not 0%. */
export function signedPct(p: number | null | undefined): string {
  if (typeof p !== "number" || !Number.isFinite(p)) return "—";
  return `${p > 0 ? "+" : ""}${p.toFixed(1)}%`;
}

/** Days between the newest archived week and now: the rail archive is
 *  rebuilt by hand from STB's weekly file, so the page states its lag. */
export function weeksBehind(latestWeek: string | null | undefined, nowMs: number): number | null {
  if (!latestWeek) return null;
  const t = Date.parse(`${latestWeek}T00:00:00Z`);
  if (!Number.isFinite(t)) return null;
  return Math.max(0, Math.floor((nowMs - t) / (7 * 86400000)));
}
