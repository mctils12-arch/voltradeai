// Bot page v2 — number and time formatting (pure). Signed numbers use the
// true minus (U+2212) so colour is never the only channel (DESIGN.md,
// research/bot_cockpit_design_brief_v2.md §6 visual rules).
export const MINUS = "−";

export function fmtSigned(v: number | null | undefined, digits = 2, suffix = ""): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const s = Math.abs(v).toFixed(digits);
  return (v < 0 ? MINUS : v > 0 ? "+" : "") + s + suffix;
}

export function fmtPct(v: number | null | undefined, digits = 2): string {
  return fmtSigned(v, digits, "%");
}

export function fmtMoney(v: number | null | undefined, cents = false): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const abs = Math.abs(v);
  const body = cents
    ? abs.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    : Math.round(abs).toLocaleString("en-US");
  return (v < 0 ? MINUS : "") + "$" + body;
}

export function fmtCompact(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const abs = Math.abs(v);
  const sign = v < 0 ? MINUS : "";
  if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(1)}M`;
  if (abs >= 10_000) return `${sign}$${(abs / 1000).toFixed(1)}k`;
  return `${sign}$${Math.round(abs).toLocaleString("en-US")}`;
}

/** Data age for the uniform top-right slot: "3s", "4m", "2h", "3d". */
export function fmtAge(ageMs: number | null | undefined): string {
  if (ageMs === null || ageMs === undefined || !Number.isFinite(ageMs) || ageMs < 0) return "—";
  const s = Math.round(ageMs / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.round(m / 60);
  if (h < 48) return `${h}h`;
  return `${Math.round(h / 24)}d`;
}

/** "2h02m" style countdown for the session clock. */
export function fmtCountdown(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return "—";
  const total = Math.max(0, Math.round(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h${String(m).padStart(2, "0")}m`;
  if (m > 0) return `${m}m${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

export function fmtClockET(iso: string | number | Date | null | undefined): string {
  if (iso === null || iso === undefined) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false, timeZone: "America/New_York" });
}

/** "Oct 14" — the ET calendar day of a timestamp, for a since / last-close caption. */
export function fmtMonDay(iso: string | number | Date | null | undefined): string {
  if (iso === null || iso === undefined || iso === "") return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "America/New_York" });
}

/** "1,902" — a count with thousands separators; "?" when unknown. */
export function fmtCount(n: number | null | undefined): string {
  return n === null || n === undefined || !Number.isFinite(n) ? "?" : Math.round(n).toLocaleString("en-US");
}

export function fmtClockUTC(iso: string | number | Date | null | undefined): string {
  if (iso === null || iso === undefined) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toISOString().slice(11, 19) + "Z";
}

/** The edge scorecard arithmetic (audit §5): payoff ratio, breakeven win
 *  rate and expectancy per trade from a win rate (0–100 or 0–1), an average
 *  win and an average loss (both as positive magnitudes, in %). */
export function edgeStats(winRate: number | null | undefined, avgWin: number | null | undefined, avgLoss: number | null | undefined) {
  if (winRate === null || winRate === undefined || avgWin === null || avgWin === undefined || avgLoss === null || avgLoss === undefined) return null;
  const wr = winRate > 1 ? winRate / 100 : winRate;
  const w = Math.abs(avgWin);
  const l = Math.abs(avgLoss);
  if (!Number.isFinite(wr) || !Number.isFinite(w) || !Number.isFinite(l) || w + l === 0) return null;
  const payoff = l > 0 ? w / l : null;
  const breakevenWr = l / (w + l);
  const expectancy = wr * w - (1 - wr) * l;
  return { winRate: wr, payoff, breakevenWr, expectancy };
}

export function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/** OCC option symbol → "CRWV 55P 20-Nov"; anything else is returned as is. */
export function fmtOcc(sym: string | null | undefined): string {
  if (!sym) return "—";
  const m = /^([A-Z]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$/.exec(sym.trim().toUpperCase());
  if (!m) return sym;
  const [, root, yy, mm, dd, cp, strike] = m;
  const k = Number(strike) / 1000;
  const mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][Number(mm) - 1] ?? mm;
  return `${root} ${Number.isInteger(k) ? k : k.toFixed(1)}${cp} ${dd}-${mon}${yy !== String(new Date().getUTCFullYear()).slice(2) ? `-${yy}` : ""}`;
}

/** SPY's change on the ET session day of `nowMs`, from the benchmark's last
 *  bar — null when that bar is not today's (weekend, holiday, not fetched),
 *  so an old day's move is never shown as today's. */
export function spyTodayPct(spyLast: { date: string; pct: number } | null | undefined, nowMs: number): number | null {
  if (!spyLast || !Number.isFinite(spyLast.pct)) return null;
  const today = new Date(nowMs).toLocaleDateString("en-CA", { timeZone: "America/New_York" });
  return spyLast.date === today ? spyLast.pct : null;
}
