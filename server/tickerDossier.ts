// Ticker dossier (2026-10-29, human-asked) — the Node half of
// /api/ticker-dossier/:symbol. ticker_dossier.py (via the daemon, python3 -c
// fallback) builds the layered body; this module owns the pieces Node adds:
//   * strict symbol validation before anything reaches a command line;
//   * the per-symbol 5-minute cache + in-flight dedupe;
//   * the insider-transaction detail from the Form 4 archive
//     (server/edgarForm4.ts readFilingHistory, filtered by
//     issuerTradingSymbol). Form 4 buy-clustering FAILED gate 2 here, so this
//     is shown as RAW events with counts — never as a bullish/bearish signal.
// Named tickerDossier to stay clear of server/dossier.ts (the entity dossier
// behind /api/data/dossier).
import type { Form4Filing } from "./edgarForm4";

export const TICKER_DOSSIER_CACHE_MS = 5 * 60_000;
export const TICKER_DOSSIER_TIMEOUT_MS = 120_000;
export const INSIDER_WINDOW_DAYS = 30;

/** A stock symbol (1–6 letters) or an OCC option code, resolved to its underlying; null when invalid. */
export function parseDossierSymbol(raw: unknown): string | null {
  const s = String(raw ?? "").trim().toUpperCase();
  const m = /^([A-Z]{1,6})(\d{6}[CP]\d{8})?$/.exec(s);
  return m ? m[1] : null;
}

/** The exact subprocess fallback (the symbol is validated letters only). */
export function dossierSubprocessCmd(symbol: string): string {
  if (!/^[A-Z]{1,6}$/.test(symbol)) throw new Error("invalid symbol");
  return `python3 -c "import json, sys; from ticker_dossier import build; print(json.dumps(build(sys.argv[1]), default=str))" ${symbol}`;
}

export interface InsiderRow { filedAt: string | null; owner: string; role: string; kind: string; shares: number | null; price: number | null; date: string | null; url: string }
export interface InsiderSummary { windowDays: number; filings: number; buys: number; sells: number; buyShares: number; sellShares: number; rows: InsiderRow[]; source: string; note: string }

function roleOf(f: Form4Filing): string {
  const o = f.owners[0];
  if (!o) return "insider";
  if (o.officerTitle) return o.officerTitle;
  if (o.isDirector) return "director";
  if (o.isTenPercentOwner) return "10% owner";
  return o.isOfficer ? "officer" : "insider";
}

/** Open-market buys (P) and sales (S) for one issuer from archived Form 4s, newest first. */
export function insiderSummary(filings: Form4Filing[], symbol: string, nowMs: number, windowDays = INSIDER_WINDOW_DAYS): InsiderSummary {
  const since = new Date(nowMs - windowDays * 86_400_000).toISOString().slice(0, 10);
  const mine = filings.filter((f) => String(f.issuerTradingSymbol ?? "").toUpperCase() === symbol && String(f.filedAt ?? "").slice(0, 10) >= since);
  const rows: InsiderRow[] = [];
  let buys = 0, sells = 0, buyShares = 0, sellShares = 0;
  for (const f of mine) {
    for (const t of f.transactions) {
      if (t.kind !== "open_market_buy" && t.kind !== "open_market_sale") continue;
      if (t.kind === "open_market_buy") { buys += 1; buyShares += t.shares ?? 0; } else { sells += 1; sellShares += t.shares ?? 0; }
      rows.push({ filedAt: f.filedAt, owner: f.owners[0]?.name ?? "unknown", role: roleOf(f), kind: t.kind === "open_market_buy" ? "buy" : "sell", shares: t.shares, price: t.pricePerShare, date: t.transactionDate, url: f.indexUrl });
    }
  }
  rows.sort((a, b) => String(b.filedAt ?? "").localeCompare(String(a.filedAt ?? "")));
  return {
    windowDays, filings: mine.length, buys, sells, buyShares, sellShares, rows: rows.slice(0, 12),
    source: "SEC EDGAR Form 4 archive (server/edgarForm4.ts — the polled feed, not every filing ever made)",
    note: "Raw insider transactions. Form 4 buy-clustering failed ladder gate 2 in this system — not a signal.",
  };
}

type Body = Record<string, unknown>;
/** Per-symbol cache with in-flight dedupe; only successful bodies are cached. */
export class DossierCache {
  private hits = new Map<string, { at: number; body: Body }>();
  private inflight = new Map<string, Promise<Body>>();
  constructor(private ttlMs = TICKER_DOSSIER_CACHE_MS, private now: () => number = Date.now) {}
  async get(symbol: string, load: () => Promise<{ body: Body; ok: boolean }>): Promise<Body> {
    const hit = this.hits.get(symbol);
    if (hit && this.now() - hit.at < this.ttlMs) return { ...hit.body, cached: true, cachedAgeS: Math.round((this.now() - hit.at) / 1000) };
    let p = this.inflight.get(symbol);
    if (!p) {
      p = load().then(({ body, ok }) => {
        if (ok) this.hits.set(symbol, { at: this.now(), body });
        return body;
      }).finally(() => this.inflight.delete(symbol));
      this.inflight.set(symbol, p);
    }
    return p;
  }
}
