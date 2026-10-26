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

// ── the look-up ledger line (2026-10-24) ────────────────────────────────────
// /api/bot/lookup-ledger groups every past look-up by the verdict it got and
// settles +1d/+5d/+20d vs SPY. This is the card's one honest line from it: how
// look-ups that got THIS verdict have gone. Thin buckets (n < min_n) are said
// to be thin, never dressed as a result; a bucket with no settled outcome says
// it is waiting.
export interface LedgerCell { n: number; thin: boolean; mean_ret_pct: number | null; win_rate: number | null; n_excess: number; mean_excess_pct: number | null }
export interface LedgerGroup extends Partial<Record<"+1d" | "+5d" | "+20d", LedgerCell>> { count: number }
export interface LedgerReport { total: number; pending: number; min_n: number; caveat: string; by_verdict: Record<string, LedgerGroup> }

/** The ledger's verdict key for a look-up — mirrors lookup_ledger.report(): the gate it stopped at, else the verdict. */
export function ledgerKey(r: LookupResult | undefined): string | null {
  const d = r?.decision;
  if (!d) return null;
  return d.stopped_at ? `stopped at ${d.stopped_at}` : d.verdict || null;
}

export function ledgerLine(rep: LedgerReport | null | undefined, r: LookupResult | undefined): string | null {
  const key = ledgerKey(r);
  if (!rep || !key) return null;
  const g = rep.by_verdict?.[key];
  if (!g) return "first look-up with this verdict in the ledger";
  const c = g["+5d"];
  const head = `${g.count} look-up${g.count === 1 ? "" : "s"} got this verdict`;
  if (!c || !c.n_excess || c.mean_excess_pct === null) return `${head} · +5d outcomes not settled yet`;
  const sign = c.mean_excess_pct > 0 ? "+" : "";
  return `${head} · +5d vs SPY ${sign}${c.mean_excess_pct.toFixed(1)}% over ${c.n_excess}${c.thin ? ` (thin, under ${rep.min_n} — anecdote)` : ""}`;
}
