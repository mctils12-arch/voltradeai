// Bot page — the look-up card's pure half (2026-10-22, human-asked: "look up a
// stock / option chain and see what the bot thinks of it"). The server's
// /api/bot/lookup/:symbol (lookup_ticker.py) walks the scan's own path for
// one symbol; this module turns that answer into the rows the card draws.
// No DOM, no fetch; pinned by lookup.test.ts.

export type GatePass = boolean | null;
export interface LookupGate { gate: string; pass: GatePass; why: string }
export interface LookupDecision { gates: LookupGate[]; stopped_at: string | null; verdict: string }
export interface LookupPick { occ: string | null; strike: number | null; exp_date: string | null; days_out: number | null; bid: number | null; ask: number | null; delta: number | null; delta_real?: boolean; iv: number | null; oi: number | null; premium_pct: number | null }
export interface LookupChain { contracts: number; atm_iv: number | null; iv_rank?: number | null; window_days?: string; csp: LookupPick | null; cc: LookupPick | null; expiries: string[]; scanner_setup?: Record<string, unknown> | null }
export interface LookupResult {
  symbol: string; at: string; path?: string;
  errors: Record<string, string>;
  timings?: Record<string, number>;
  quick?: { price: number; change_pct: number; volume: number; dollar_volume: number; quick_score: number; above_open: boolean };
  floors?: Record<string, unknown>;
  deep?: Record<string, unknown> & { score?: number | null; reasons?: string[]; factors?: Record<string, number> | null; side?: string; action_label?: string; sector?: string | null };
  sources?: Record<string, string>;
  regime?: { label: string; vxx_ratio: number; spy_vs_ma50: number; spy_below_200_days: number; min_score: number };
  held?: boolean;
  sizing?: Record<string, unknown> | null;
  decision?: LookupDecision;
  chain?: LookupChain;
}

/** The symbol box accepts a stock symbol or an OCC option code; an option is looked up by its underlying. */
export function parseLookupInput(raw: string): { symbol: string; underlying: string; option: boolean } | null {
  const s = raw.trim().toUpperCase().replace(/\s+/g, "");
  if (!s) return null;
  const occ = /^([A-Z]{1,6})(\d{6})([CP])(\d{8})$/.exec(s);
  if (occ) return { symbol: s, underlying: occ[1], option: true };
  if (/^[A-Z]{1,6}$/.test(s)) return { symbol: s, underlying: s, option: false };
  return null;
}

export const GATE_WORDS: Record<string, string> = { universe: "in the universe", regime: "regime", cooldown: "stop cooldown", admission: "admission", sector: "sector room", sizing: "sizing" };

/** The gates as the card lists them: the scan's order, each with a status word. */
export function gateRows(d: LookupDecision | undefined): Array<{ gate: string; label: string; status: "pass" | "fail" | "unknown" | "skipped"; why: string }> {
  if (!d) return [];
  let stopped = false;
  return d.gates.map((g) => {
    const status: "pass" | "fail" | "unknown" | "skipped" = stopped ? "skipped" : g.pass === true ? "pass" : g.pass === false ? "fail" : "unknown";
    if (g.pass === false) stopped = true;
    return { gate: g.gate, label: GATE_WORDS[g.gate] ?? g.gate, status, why: g.why };
  });
}

/** Signed factor contributions, largest first — why the score is what it is. */
export function factorBars(factors: Record<string, number> | null | undefined, cap = 6): Array<{ key: string; value: number }> {
  if (!factors) return [];
  return Object.entries(factors)
    .filter(([, v]) => typeof v === "number" && Number.isFinite(v))
    .map(([key, value]) => ({ key, value }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, cap);
}

/** One word the card leads with. */
export function verdictWord(r: LookupResult | undefined): { word: string; tone: "g" | "o" | "r" | "t" } {
  if (!r) return { word: "—", tone: "t" };
  if (r.errors && (r.errors.snapshot || r.errors.symbol)) return { word: "not found", tone: "t" };
  const d = r.decision;
  if (!d) return { word: "unscored", tone: "t" };
  if (d.stopped_at === null && d.verdict === "would place it") return { word: "would place it", tone: "g" };
  if (d.stopped_at) return { word: `stopped at ${GATE_WORDS[d.stopped_at] ?? d.stopped_at}`, tone: d.stopped_at === "admission" ? "o" : "r" };
  return { word: "undecided", tone: "t" };
}

const f1 = (v: number | null | undefined, d = 1) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "—");

/** The chain in one line: ATM IV, IV rank, the put and the call the scanner's delta finder picks. */
export function chainLine(c: LookupChain | undefined): string {
  if (!c) return "chain not fetched";
  if (!c.contracts) return "no contracts in the scanner's window";
  const parts: string[] = [`${c.contracts} contracts`];
  if (c.atm_iv !== null && c.atm_iv !== undefined) parts.push(`ATM IV ${f1(c.atm_iv, 0)}%`);
  if (typeof c.iv_rank === "number") parts.push(`IV rank ${f1(c.iv_rank, 0)}`);
  const pick = (p: LookupPick | null, word: string) => (p && p.strike !== null ? `${word} ${p.strike}${p.exp_date ? ` ${p.exp_date.slice(5)}` : ""}${p.delta !== null ? ` δ${f1(Math.abs(p.delta), 2)}` : ""}${p.bid !== null ? ` $${f1(p.bid, 2)}` : ""}${p.premium_pct !== null ? ` (${f1(p.premium_pct, 1)}%)` : ""}` : null);
  const csp = pick(c.csp, "CSP"), cc = pick(c.cc, "CC");
  if (csp) parts.push(csp);
  if (cc) parts.push(cc);
  return parts.join(" · ");
}

/** The first thing that failed, for the card's honesty line. */
export function lookupErrors(r: LookupResult | undefined): string[] {
  if (!r?.errors) return [];
  return Object.entries(r.errors).map(([k, v]) => `${k}: ${v}`);
}

/** Lookups made on this page, newest first, capped. */
export const LOOKUP_HISTORY_MAX = 8;
export function pushHistory(list: LookupResult[], r: LookupResult): LookupResult[] {
  return [r, ...list.filter((x) => x.symbol !== r.symbol)].slice(0, LOOKUP_HISTORY_MAX);
}
