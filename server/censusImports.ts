/**
 * censusImports.ts — monthly US port-level import values incl. containerized
 * (BUILD ORDER 3 #4, unblocked 2026-07-05 when the human added
 * CENSUS_API_KEY; built key-gated on the fredMacro pattern so it activates
 * automatically wherever the key lands — Railway or session env).
 *
 * Census intltrade timeseries API: US government work, public domain;
 * attribution "U.S. Census Bureau (USA Trade Online / FT920)". The API
 * requires a free key on every request (probed 2026-07-05: keyless calls
 * redirect to missing_key).
 *
 * HYPOTHESIS (gate-locked in the build order): import value/TEU deltas
 * lead retail inventory cycles; pairs with our port-dwell analytics
 * (supply friction) for a two-sided port view.
 *
 * QUERY-SHAPE HONESTY: the exact porths parameter set could not be
 * live-verified this session (the key is not in this container's env —
 * fixed at start). The fetch therefore (a) tries documented variants in
 * order, (b) parses HEADER-DRIVEN (never positional), and (c) logs the
 * Census error body verbatim on failure so the next session fixes the
 * query from prod logs instead of guessing.
 *
 * LIVE-VERIFIED 2026-08-20 (this line said "PENDING" for 46 days): prod
 * /api/data/imports serves 1,059 rows over 3 months from QUERY_VARIANTS[0]
 * — the full column set, so no variant fallback is in play. Two shapes a
 * consumer must know about, both found in that payload and both handled in
 * client/src/lib/portImports.ts: Census ships its own national aggregate as
 * a row under port code "-" (excluding it is the caller's job), and the
 * per-port rows sum to that aggregate exactly, which makes it a free
 * integrity check on every read.
 *
 * Monthly source (~45d lag, FT920 schedule); daily poll is a cheap no-op
 * between releases (dedup by port|month).
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

export function censusEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(env.CENSUS_API_KEY);
}

export interface ImportObs {
  port: string;            // Census port code
  port_name: string | null;
  month: string;           // YYYY-MM
  gen_val: number | null;  // general imports value, USD
  cnt_val: number | null;  // containerized value, USD (null if variant lacks it)
  cnt_wgt: number | null;  // containerized vessel weight, kg (proxy for TEU flow)
  rt: string;              // as-seen UTC date
}

/** Query variants tried in order — first 200 wins; each failure's body is
 *  logged verbatim (Census errors are human-readable). */
export const QUERY_VARIANTS = [
  "get=PORT,PORT_NAME,GEN_VAL_MO,CNT_VAL_MO,CNT_WGT_MO",
  "get=PORT,PORT_NAME,GEN_VAL_MO",
];

const BASE_URL = "https://api.census.gov/data/timeseries/intltrade/imports/porths";

const num = (v: any): number | null => {
  if (v == null || v === "") return null;
  const n = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : null;
};

/** Census array-of-arrays -> records, HEADER-DRIVEN (column order is not
 *  contractual). Rows missing port or month are dropped. */
export function parseImports(json: any, rt: string): ImportObs[] {
  if (!Array.isArray(json) || json.length < 2 || !Array.isArray(json[0])) return [];
  const idx: Record<string, number> = {};
  json[0].forEach((h: any, i: number) => { idx[String(h).toUpperCase()] = i; });
  if (idx.PORT == null || idx.TIME == null) return [];
  const out: ImportObs[] = [];
  for (const row of json.slice(1)) {
    if (!Array.isArray(row)) continue;
    const port = row[idx.PORT];
    const month = row[idx.TIME];
    if (!port || typeof month !== "string" || !/^\d{4}-\d{2}$/.test(month)) continue;
    out.push({
      port: String(port),
      port_name: idx.PORT_NAME != null ? (row[idx.PORT_NAME] ?? null) : null,
      month,
      gen_val: idx.GEN_VAL_MO != null ? num(row[idx.GEN_VAL_MO]) : null,
      cnt_val: idx.CNT_VAL_MO != null ? num(row[idx.CNT_VAL_MO]) : null,
      cnt_wgt: idx.CNT_WGT_MO != null ? num(row[idx.CNT_WGT_MO]) : null,
      rt,
    });
  }
  return out;
}

// ── Fetch (variant fallback; key never logged) ──────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

function recentMonths(nowMs: number, n = 4): string[] {
  // FT920 lags ~45 days — ask for the last n months; absent months are
  // absent (empty result is a valid response before release day)
  const out: string[] = [];
  const d = new Date(nowMs);
  for (let i = 0; i < n; i++) {
    d.setUTCMonth(d.getUTCMonth() - 1);
    out.push(d.toISOString().slice(0, 7));
  }
  return out;
}

export async function fetchImports(
  fetchImpl: FetchFn = fetch as any,
  env: NodeJS.ProcessEnv = process.env,
  nowMs?: number,
): Promise<ImportObs[]> {
  const key = env.CENSUS_API_KEY || "";
  if (!key) return [];
  const now = nowMs ?? Date.now();
  const rt = new Date(now).toISOString().slice(0, 10);
  const out: ImportObs[] = [];
  for (const month of recentMonths(now)) {
    let got = false;
    for (const variant of QUERY_VARIANTS) {
      const url = `${BASE_URL}?${variant}&time=${month}&key=${encodeURIComponent(key)}`;
      try {
        const r = await fetchImpl(url, {
          headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
          signal: AbortSignal.timeout(25000) as any,
        });
        const body = await r.text();
        if (!r.ok) {
          // Census errors are readable — log them (never the key) so a
          // wrong query shape is fixable from prod logs, not guesswork
          console.error(`[datacore] census ${month} variant "${variant}" -> ${r.status}: ${body.slice(0, 200)}`);
          continue;
        }
        if (!body.trim()) { got = true; break; } // month not released yet — valid
        out.push(...parseImports(JSON.parse(body), rt));
        got = true;
        break;
      } catch (e: any) {
        console.error(`[datacore] census ${month}:`, e?.message || e);
      }
    }
    if (!got) continue; // all variants failed for this month — logged above
    await new Promise((res) => setTimeout(res, 250)); // polite spacing
  }
  return out;
}

// ── Archive (dedup port|month — FT920 revisions are rare; a revised value
//    appends as a new row with a new rt, vintage-style) ─────────────────────

const archivedKeys = new Set<string>();
let seeded = false;

function importsDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "censusimports");
}

const keyOf = (o: ImportObs) => `${o.port}|${o.month}|${o.gen_val}|${o.cnt_val}|${o.cnt_wgt}`;

function seedSeen(dir: string, nowMs: number): void {
  for (let i = 0; i < 40; i++) {
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

export function archiveImports(obs: ImportObs[], baseDir?: string, nowMs?: number): number {
  const dir = importsDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) { seedSeen(dir, now); seeded = true; }
  const fresh = obs.filter((o) => !archivedKeys.has(keyOf(o)));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const fp = path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`);
    fs.appendFileSync(fp, fresh.map((o) => JSON.stringify(o)).join("\n") + "\n");
    fresh.forEach((o) => archivedKeys.add(keyOf(o)));
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] censusimports archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldImportDays(baseDir?: string, nowMs?: number): number {
  const dir = importsDir(baseDir);
  const now = nowMs ?? Date.now();
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
      if (now - Date.parse(f.slice(0, 10)) < 2 * 86400_000) continue;
      const fp = path.join(dir, f);
      fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
      fs.unlinkSync(fp);
      n++;
    }
  } catch {}
  return n;
}

// ── Cache + poll ────────────────────────────────────────────────────────────

let cache: { at: number; imports: ImportObs[] } | null = null;
let polling = false;

export function latestImports() {
  return cache;
}

export async function refreshImportCache(fetchImpl: FetchFn = fetch as any,
                                         env: NodeJS.ProcessEnv = process.env, nowMs?: number): Promise<void> {
  try {
    const imports = await fetchImports(fetchImpl, env, nowMs);
    if (imports.length || !cache) cache = { at: Date.now(), imports };
    try { archiveImports(imports, undefined, nowMs); } catch {}
    try { gzipOldImportDays(undefined, nowMs); } catch {}
  } catch (e: any) {
    console.error("[datacore] censusimports refresh:", e?.message || e);
  }
}

/** Daily poll — the source is monthly (~45d lag); dedup makes off-days a
 *  no-op. No-ops entirely without CENSUS_API_KEY (awaiting_key honesty on
 *  the route). Eager boot per KNOWN BROKEN #9. */
export function bootCensusPoll(intervalMs = 24 * 60 * 60_000): void {
  if (polling || !censusEnabled()) return;
  polling = true;
  refreshImportCache();
  setInterval(() => { refreshImportCache(); }, intervalMs).unref?.();
}
