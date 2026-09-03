/**
 * fdicBanks.ts — FDIC bank failures event stream
 * (BUILD ORDER 6 #3 v1, filed + probed 2026-07-06).
 *
 * Source: FDIC Bank Data API — NOTE the host is api.fdic.gov; the
 * documented banks.data.fdic.gov host 301s there (probe finding,
 * recorded so the next session doesn't re-discover it). Keyless JSON;
 * /banks/failures covers 4,115 failures back to 1934, updated as they
 * occur (latest verified live: 2026-05-01). Fields verified live
 * 2026-07-06 with an explicit fields= list; FAILDATE arrives as
 * M/D/YYYY, QBFASSET/QBFDEP in $ thousands, COST (loss to the DIF)
 * null until estimated.
 *
 * v1 SCOPE — FAILURES ONLY (event-driven, the live kicker). The
 * quarterly financials snapshot (~4.6k banks, paginated, needs a
 * once-per-quarter claim) is the documented follow-up, its own PR.
 *
 * HYPOTHESIS (gate-locked in the build order — archive + RAW only):
 * deposit flight + failures at SMALL regional banks lead KRE and
 * listed small-cap banks — whales-can't-fish territory (EDGE
 * DOCTRINE #2). Gate 1 = cross-check a failure's QBFASSET/QBFDEP vs
 * the FDIC press release; gate 2 = failure events vs forward KRE /
 * regional-bank returns.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

const FAILURES_URL = "https://api.fdic.gov/banks/failures";
const FIELDS = "NAME,CERT,FAILDATE,CITYST,PSTALP,CHCLASS,RESTYPE,QBFASSET,QBFDEP,COST,SAVR";
/** Recent-window fetch — failures are rare (a handful per year lately). */
export const FAILURES_FETCH_LIMIT = 50;

export interface BankFailure {
  cert: number | null;       // FDIC certificate number (entity key)
  name: string;
  fail_date: string;         // YYYY-MM-DD (normalized from M/D/YYYY)
  city_state: string | null;
  state: string | null;      // PSTALP
  charter_class: string | null;
  restype: string | null;    // FAILURE | ASSISTANCE
  assets_k: number | null;   // QBFASSET, $ thousands as published
  deposits_k: number | null; // QBFDEP, $ thousands as published
  cost_k: number | null;     // estimated loss to the DIF, $ thousands (null until estimated)
  fund: string | null;       // SAVR (DIF etc.)
  rt: string;                // as-seen UTC date
}

const num = (v: any): number | null => {
  if (v == null || v === "") return null;
  const n = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : null;
};

/** "5/1/2026" -> "2026-05-01"; returns null on garbage. */
export function normalizeFailDate(v: any): string | null {
  const m = String(v ?? "").match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
  if (!m) return null;
  return `${m[3]}-${m[1].padStart(2, "0")}-${m[2].padStart(2, "0")}`;
}

/** FDIC envelope ({data:[{data:{...}}]}) -> BankFailures, newest first. */
export function parseFailures(json: any, rt: string): BankFailure[] {
  const data = json?.data;
  if (!Array.isArray(data)) return [];
  const out: BankFailure[] = [];
  for (const wrap of data) {
    const r = wrap?.data ?? wrap;
    if (!r || typeof r !== "object") continue;
    const failDate = normalizeFailDate(r.FAILDATE);
    const name = r.NAME;
    if (!failDate || !name) continue;
    out.push({
      cert: num(r.CERT),
      name: String(name),
      fail_date: failDate,
      city_state: r.CITYST != null ? String(r.CITYST) : null,
      state: r.PSTALP != null ? String(r.PSTALP) : null,
      charter_class: r.CHCLASS != null ? String(r.CHCLASS) : null,
      restype: r.RESTYPE != null ? String(r.RESTYPE) : null,
      assets_k: num(r.QBFASSET),
      deposits_k: num(r.QBFDEP),
      cost_k: num(r.COST),
      fund: r.SAVR != null ? String(r.SAVR) : null,
      rt,
    });
  }
  return out;
}

// ── Fetch ────────────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

export async function fetchRecentFailures(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<BankFailure[]> {
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  const url = `${FAILURES_URL}?limit=${FAILURES_FETCH_LIMIT}&sort_by=FAILDATE&sort_order=DESC&fields=${encodeURIComponent(FIELDS)}`;
  try {
    const r = await fetchImpl(url, {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(30000) as any,
    });
    if (!r.ok) {
      console.error(`[datacore] fdicfailures -> ${r.status}`);
      return [];
    }
    return parseFailures(JSON.parse(await r.text()), rt);
  } catch (e: any) {
    console.error("[datacore] fdicfailures:", e?.message || e);
    return [];
  }
}

/**
 * Historical failures since `sinceDate` ("YYYY-MM-DD"), for the
 * fdic_bank_failures GATE 2 (SIGNAL) event study (see fdicGate2.ts) —
 * independent of the day-file archive below, since FDIC's own API already
 * serves full history back to 1934 (module docstring). Reuses
 * fetchRecentFailures's DESC-sort/fields/host, just with a larger `limit`
 * and a client-side date filter rather than a server-side FAILDATE range
 * filter — this API's filter query syntax for date ranges was never
 * verified live, and a silently-wrong filter would return too few/zero
 * rows rather than erroring, which a client-side filter over a plain
 * larger fetch cannot do. Failures have run well under 10/year since
 * ~2013 (module docstring's 4,115-since-1934 count), so `limit=500`
 * comfortably covers everything back to well before any plausible GATE 2
 * window without needing the exact filter syntax.
 */
export async function fetchHistoricalFailures(
  sinceDate: string, fetchImpl: FetchFn = fetch as any, limit = 500,
): Promise<BankFailure[]> {
  const rt = new Date().toISOString().slice(0, 10);
  const url = `${FAILURES_URL}?limit=${limit}&sort_by=FAILDATE&sort_order=DESC&fields=${encodeURIComponent(FIELDS)}`;
  try {
    const r = await fetchImpl(url, {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(30000) as any,
    });
    if (!r.ok) {
      console.error(`[datacore] fdicfailures historical -> ${r.status}`);
      return [];
    }
    const all = parseFailures(JSON.parse(await r.text()), rt);
    return all.filter((f) => f.fail_date >= sinceDate);
  } catch (e: unknown) {
    console.error("[datacore] fdicfailures historical:", e instanceof Error ? e.message : e);
    return [];
  }
}

// ── Archive (event-identity dedup, day-file per FETCH date — FAA/CBP pattern) ─

const seenEvents = new Set<string>();   // cert|fail_date
let seeded = false;

const eventKey = (f: BankFailure) => `${f.cert ?? f.name}|${f.fail_date}`;

function failuresDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "fdicfailures");
}

function seedSeen(dir: string): void {
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!/^\d{4}-\d{2}-\d{2}\.jsonl(\.gz)?$/.test(f)) continue;
      const fp = path.join(dir, f);
      let text: string;
      try {
        text = f.endsWith(".gz")
          ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
          : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        try { seenEvents.add(eventKey(JSON.parse(line))); } catch {}
      }
    }
  } catch {}
}

/** Appends UNSEEN failures to today's day-file. Returns count written. */
export function archiveNewFailures(failures: BankFailure[], baseDir?: string, nowMs?: number): number {
  if (!failures.length) return 0;
  const dir = failuresDir(baseDir);
  if (!seeded) { seedSeen(dir); seeded = true; }
  const fresh = failures.filter((f) => !seenEvents.has(eventKey(f)));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const day = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
    fs.appendFileSync(path.join(dir, `${day}.jsonl`),
                      fresh.map((r) => JSON.stringify(r)).join("\n") + "\n");
    fresh.forEach((f) => seenEvents.add(eventKey(f)));
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] fdicfailures archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldFailureDays(baseDir?: string, nowMs?: number): number {
  const dir = failuresDir(baseDir);
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

let cache: { at: number; failures: BankFailure[] } | null = null;
let polling = false;

export function latestFailures() {
  return cache;
}

export async function refreshFailures(fetchImpl: FetchFn = fetch as any, nowMs?: number,
                                      baseDir?: string): Promise<void> {
  try {
    const failures = await fetchRecentFailures(fetchImpl, nowMs);
    if (failures.length) {
      archiveNewFailures(failures, baseDir, nowMs);
      cache = { at: Date.now(), failures };
    }
    gzipOldFailureDays(baseDir, nowMs);
  } catch (e: any) {
    console.error("[datacore] fdicfailures refresh:", e?.message || e);
  }
}

/** Event-driven source (failures announced Friday evenings ET when they
 *  happen) — 12h poll; identical fetches dedup to no-ops. Eager boot. */
export function bootFailuresPoll(intervalMs = 12 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshFailures();
  setInterval(() => { refreshFailures(); }, intervalMs).unref?.();
}
