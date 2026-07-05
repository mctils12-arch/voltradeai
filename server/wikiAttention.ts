/**
 * wikiAttention.ts — Wikipedia pageviews attention proxy
 * (BUILD ORDER 5 #3, filed + probed 2026-07-05).
 *
 * THE PYTRENDS REPLACEMENT: Google Trends gate-1 FAILED (#215, upstream
 * abandoned). Wikimedia's pageviews REST API is keyless, stable, and
 * historical to 2015 — the free attention signal that survives.
 *
 * Source: wikimedia.org/api/rest_v1/metrics/pageviews/per-article
 * (en.wikipedia, all-access, agent=user — bot traffic excluded by the
 * agent filter). CC0/CC-BY data; attribution "Wikimedia pageviews API".
 * RATE LIMIT (observed live 2026-07-05): quick bursts 429 — the poller
 * spaces requests >=600ms and the seed stays small.
 *
 * Ticker->article map: datacore/wiki_articles.json, STATIC IMPORT —
 * the Docker image never copies datacore/ (frozen Dockerfile), so a
 * disk read would serve nothing in prod; bundling is the only path
 * that survives the image (entity-spine lesson, #226). Every seed pair
 * was hand-probed against the live API before inclusion; RIOT was
 * dropped (article renamed — the API does not follow redirects).
 *
 * HYPOTHESIS (gate-locked): attention spikes on company articles lead
 * volume/volatility 1-5d, most interesting on smaller names without
 * same-day news. v1 serves RAW daily views only — no z-scores or
 * "spike" claims until the archive has the trailing history to compute
 * them honestly and gate 1 (views vs known events) passes.
 *
 * Pageviews for "yesterday" finalize early UTC morning — the poller
 * fetches a 7-day window per article and archives complete days only.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";
import wikiArticles from "../datacore/wiki_articles.json";

export const ARTICLES: Record<string, string> = (wikiArticles as any).articles;
export const REQUEST_SPACING_MS = 600; // observed: bursts 429

export interface AttentionObs {
  date: string;      // YYYY-MM-DD (view day, UTC)
  ticker: string;
  article: string;
  views: number;
  rt: string;        // as-seen UTC date
}

const API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user";

/** One article's API payload -> observations. Header-driven on the
 *  documented items[] shape; malformed items dropped. */
export function parsePageviews(json: any, ticker: string, article: string, rt: string): AttentionObs[] {
  if (!json || !Array.isArray(json.items)) return [];
  const out: AttentionObs[] = [];
  for (const it of json.items) {
    const ts = String(it?.timestamp || "");
    const views = typeof it?.views === "number" ? it.views : null;
    if (!/^\d{10}$/.test(ts) || views == null) continue;
    out.push({
      date: `${ts.slice(0, 4)}-${ts.slice(4, 6)}-${ts.slice(6, 8)}`,
      ticker, article, views, rt,
    });
  }
  return out;
}

// ── Fetch ────────────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

/** Last poll cycle's outcome, exposed on the route while warming so a
 *  never-filling cache is diagnosable FROM PROD without log access
 *  (added 2026-07-05: attention stayed warming_up on prod after a
 *  clean deploy while every sibling stream served — the route itself
 *  must say why). */
export interface CycleStats {
  started: string;
  finished: string | null;
  ok_articles: number;
  err_articles: number;
  not_found: number;
  last_error: string | null;  // "TICKER -> status" or "TICKER: message" (no URLs/keys)
  obs: number;
}

let lastCycle: CycleStats | null = null;

export function lastAttentionCycle() {
  return lastCycle;
}

/** Fetch the last `windowDays` complete days for every seed article,
 *  politely spaced. A 404/no-data article is skipped (logged once) —
 *  absence is data, never faked. */
export async function fetchAttention(fetchImpl: FetchFn = fetch as any, nowMs?: number,
                                     windowDays = 7,
                                     spacingMs = REQUEST_SPACING_MS): Promise<AttentionObs[]> {
  const now = nowMs ?? Date.now();
  const rt = new Date(now).toISOString().slice(0, 10);
  // end = yesterday UTC (today is incomplete); start = end - windowDays
  const end = new Date(now - 86400_000).toISOString().slice(0, 10).replace(/-/g, "");
  const start = new Date(now - (windowDays + 1) * 86400_000).toISOString().slice(0, 10).replace(/-/g, "");
  const out: AttentionObs[] = [];
  const stats: CycleStats = {
    started: new Date().toISOString(), finished: null,
    ok_articles: 0, err_articles: 0, not_found: 0, last_error: null, obs: 0,
  };
  lastCycle = stats;
  for (const [ticker, article] of Object.entries(ARTICLES)) {
    const url = `${API}/${encodeURIComponent(article)}/daily/${start}00/${end}00`;
    try {
      const r = await fetchImpl(url, {
        headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
        signal: AbortSignal.timeout(20000) as any,
      });
      if (r.ok) {
        out.push(...parsePageviews(JSON.parse(await r.text()), ticker, article, rt));
        stats.ok_articles++;
      } else if (r.status === 404) {
        stats.not_found++;
      } else {
        stats.err_articles++;
        stats.last_error = `${ticker} -> ${r.status}`;
        console.error(`[datacore] wikiattention ${ticker} -> ${r.status}`);
      }
    } catch (e: any) {
      stats.err_articles++;
      stats.last_error = `${ticker}: ${e?.message || e}`;
      console.error(`[datacore] wikiattention ${ticker}:`, e?.message || e);
    }
    await new Promise((res) => setTimeout(res, spacingMs));
  }
  stats.finished = new Date().toISOString();
  stats.obs = out.length;
  return out;
}

// ── Archive (dedup date|ticker; day-files by VIEW date) ─────────────────────

const archivedKeys = new Set<string>();
let seeded = false;

function attDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "wikiattention");
}

const keyOf = (o: AttentionObs) => `${o.date}|${o.ticker}`;

function seedSeen(dir: string, nowMs: number): void {
  // seed window comfortably exceeds the fetch window; ~23 keys/day is tiny
  for (let i = 0; i < 20; i++) {
    const d = new Date(nowMs - i * 86400_000).toISOString().slice(0, 10);
    for (const fp of [path.join(dir, `${d}.jsonl`), path.join(dir, `${d}.jsonl.gz`)]) {
      let text: string | null = null;
      try {
        text = fp.endsWith(".gz")
          ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
          : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        try { archivedKeys.add(keyOf(JSON.parse(line))); } catch {}
      }
    }
  }
}

export function archiveAttention(obs: AttentionObs[], baseDir?: string, nowMs?: number): number {
  const dir = attDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) { seedSeen(dir, now); seeded = true; }
  const fresh = obs.filter((o) => !archivedKeys.has(keyOf(o)));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    // group by VIEW date so each day-file holds that day's panel
    const byDate = new Map<string, AttentionObs[]>();
    fresh.forEach((o) => {
      const arr = byDate.get(o.date) || [];
      arr.push(o);
      byDate.set(o.date, arr);
    });
    byDate.forEach((rows, date) => {
      fs.appendFileSync(path.join(dir, `${date}.jsonl`),
                        rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
    });
    fresh.forEach((o) => archivedKeys.add(keyOf(o)));
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] wikiattention archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldAttentionDays(baseDir?: string, nowMs?: number): number {
  const dir = attDir(baseDir);
  const now = nowMs ?? Date.now();
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
      // views for a day can be revised shortly after publish — gz at 4d
      if (now - Date.parse(f.slice(0, 10)) < 4 * 86400_000) continue;
      const fp = path.join(dir, f);
      fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
      fs.unlinkSync(fp);
      n++;
    }
  } catch {}
  return n;
}

// ── Cache + poll ────────────────────────────────────────────────────────────

export interface AttentionDay {
  date: string;
  tickers: Array<{ ticker: string; article: string; views: number }>;
}

let cache: { at: number; day: AttentionDay } | null = null;
let polling = false;

export function latestAttention() {
  return cache;
}

/** Latest complete day = newest date present for a MAJORITY of the seed
 *  (an in-progress publish day with 2 of 23 articles would otherwise
 *  masquerade as the panel). */
export function pickLatestCompleteDay(obs: AttentionObs[]): AttentionDay | null {
  if (!obs.length) return null;
  const byDate = new Map<string, AttentionObs[]>();
  obs.forEach((o) => {
    const arr = byDate.get(o.date) || [];
    arr.push(o);
    byDate.set(o.date, arr);
  });
  const need = Math.ceil(Object.keys(ARTICLES).length / 2);
  const dates = Array.from(byDate.keys()).sort().reverse();
  for (const d of dates) {
    const rows = byDate.get(d)!;
    if (rows.length >= need) {
      return {
        date: d,
        tickers: rows
          .map((r) => ({ ticker: r.ticker, article: r.article, views: r.views }))
          .sort((a, b) => b.views - a.views),
      };
    }
  }
  return null;
}

export async function refreshAttention(fetchImpl: FetchFn = fetch as any, nowMs?: number,
                                       spacingMs?: number): Promise<void> {
  try {
    const obs = await fetchAttention(fetchImpl, nowMs, 7, spacingMs);
    if (obs.length) {
      archiveAttention(obs, undefined, nowMs);
      const day = pickLatestCompleteDay(obs);
      if (day) cache = { at: Date.now(), day };
    }
    gzipOldAttentionDays(undefined, nowMs);
  } catch (e: any) {
    console.error("[datacore] wikiattention refresh:", e?.message || e);
  }
}

/** Daily source (yesterday finalizes early UTC) — 12h poll; dedup makes
 *  the second daily pass a near-no-op. Eager boot per KNOWN BROKEN #9. */
export function bootAttentionPoll(intervalMs = 12 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshAttention();
  setInterval(() => { refreshAttention(); }, intervalMs).unref?.();
}
