/**
 * fmcsaOutOfService.ts — FMCSA Out-of-Service (OOS) orders for motor
 * carriers (EDGE DOCTRINE #1, filed + probed live 2026-09-28).
 *
 * Source: data.transportation.gov Socrata dataset `p2mt-9ige` ("OUT OF
 * SERVICE ORDERS") — keyless JSON, US government work (public domain).
 * Live-probed this session: `$select=count(*)` over the trailing 30 days
 * returned 2532 rows (~536/week), `status` distinct values
 * ACTIVE/INACTIVE/PENDING, most-recent rows dated same-day as the probe.
 * A SIBLING dataset on the same host, `AuthHist - All With History`
 * (9mw4-x3tu), was probed and REJECTED first: its own description reads
 * "last refreshed on 05/14/2026 and will no longer be updated" and its
 * own `max(orig_served_date)`/`max(disp_decided_date)` both cap at
 * 12/31/2025 — a frozen legacy extract, same "fake-fresh catalog
 * timestamp over dead content" shape `cboe_daily_stats` was declined for
 * (research/data_census.md). This dataset (`p2mt-9ige`) was independently
 * confirmed live: catalog `data_updated_at` and the live `max(oos_date)`
 * agree, both within 2 days of the probe.
 *
 * WHY THIS ROOT (EDGE DOCTRINE #2, fish where whales can't): OOS orders
 * hit almost exclusively small, non-public motor carriers — the
 * "New Entrant Revoked" reason code (the dominant recent reason live-
 * probed this session) specifically captures brand-new authorities
 * failing their first safety audit, i.e. small-carrier formation/failure
 * churn no equity-research desk tracks name-by-name. RAW OVERLAY per
 * CLAUDE.md (an enforcement-action log, not an interpreted reading) —
 * no ladder gating required to surface it; any claim that AGGREGATE OOS
 * order volume predicts trucking-sector returns is a SEPARATE, not yet
 * attempted, GATE 2 hypothesis (filed in research/open_questions.md),
 * discounted up front per REASONING STANDARD #4 as the same
 * "aggregate-volume vs. transport-sector-returns" shape the 2026-09-28
 * rail-carload GATE 2 test just rejected — a different underlying
 * mechanism (small-carrier exit vs. system-wide freight volume), but the
 * same family, so it earns no extra prior.
 *
 * CADENCE: dataset refreshes roughly daily; polled every 6h (change-only
 * dedup keeps quiet cycles nearly free, same discipline as cbpBorderWait).
 * WINDOW: only orders with `oos_date` inside the trailing LOOKBACK_DAYS
 * are queried each poll — a record whose `rescind_date` is added AFTER
 * its `oos_date` ages out of that window will not be re-observed. Stated
 * honestly as a known limitation of this v1, not fixed here (a full
 * historical backfill + full-dataset periodic re-scan is future work).
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";
import { resolveCacheItems } from "./cacheBackfill";

export interface OosOrderObs {
  dot_number: string;
  legal_name: string;
  oos_date: string;               // as published, "YYYY-MM-DD"
  oos_reason: string | null;
  status: string | null;          // ACTIVE | INACTIVE | PENDING, as published
  rescind_date: string | null;
  rt: string;                     // as-seen UTC timestamp
}

const str = (v: unknown): string | null => (v == null || v === "" ? null : String(v));

/** Socrata JSON rows -> typed observations. Rows missing dot_number or
 *  oos_date are dropped (neither is ever blank in the live feed, but a
 *  malformed row must never crash the poll — same defensive shape
 *  cbpBorderWait.ts's laneObs/parseBorderWaits use). */
export function parseOosOrders(json: unknown, rt: string): OosOrderObs[] {
  if (!Array.isArray(json)) return [];
  const out: OosOrderObs[] = [];
  for (const raw of json) {
    const rec = raw as Record<string, unknown> | null;
    if (!rec || !rec.dot_number || !rec.oos_date) continue;
    out.push({
      dot_number: String(rec.dot_number),
      legal_name: String(rec.legal_name || ""),
      oos_date: String(rec.oos_date).slice(0, 10),
      oos_reason: str(rec.oos_reason),
      status: str(rec.status),
      rescind_date: rec.rescind_date ? String(rec.rescind_date).slice(0, 10) : null,
      rt,
    });
  }
  return out;
}

// ── Fetch ────────────────────────────────────────────────────────────────────

type FetchFn = (
  url: string,
  init?: { headers?: Record<string, string>; signal?: AbortSignal },
) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

const DATASET = "p2mt-9ige";
const API = `https://data.transportation.gov/resource/${DATASET}.json`;
const LOOKBACK_DAYS = 45;
const PAGE_LIMIT = 5000; // ~536/week live-measured; 45d window is well under one page

export async function fetchOosOrders(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<OosOrderObs[] | null> {
  const now = nowMs ?? Date.now();
  const rt = new Date(now).toISOString();
  const cutoff = new Date(now - LOOKBACK_DAYS * 86400_000).toISOString().slice(0, 10);
  const url = `${API}?$where=oos_date>='${cutoff}'&$order=oos_date ASC&$limit=${PAGE_LIMIT}`;
  try {
    const r = await fetchImpl(url, {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(25000) as any,
    });
    if (!r.ok) {
      console.error(`[datacore] fmcsaoos -> ${r.status}`);
      return null;
    }
    return parseOosOrders(JSON.parse(await r.text()), rt);
  } catch (e: unknown) {
    console.error("[datacore] fmcsaoos:", e instanceof Error ? e.message : e);
    return null;
  }
}

// ── Archive (observation-identity dedup — only CHANGES land on disk) ────────

const archivedKeys = new Set<string>();
let seeded = false;

function oosDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "fmcsaoos");
}

const keyOf = (o: OosOrderObs) =>
  `${o.dot_number}|${o.oos_date}|${o.status}|${o.rescind_date}|${o.oos_reason}`;

function seedSeen(dir: string, nowMs: number): void {
  for (let i = 0; i < LOOKBACK_DAYS + 5; i++) {
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
        let key: string | null = null;
        try { key = keyOf(JSON.parse(line)); } catch { key = null; }
        if (key) archivedKeys.add(key);
      }
    }
  }
}

export function archiveOosOrders(obs: OosOrderObs[], baseDir?: string, nowMs?: number): number {
  const dir = oosDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) { seedSeen(dir, now); seeded = true; }
  const fresh = obs.filter((o) => !archivedKeys.has(keyOf(o)));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    fs.appendFileSync(path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`),
                      fresh.map((o) => JSON.stringify(o)).join("\n") + "\n");
    fresh.forEach((o) => archivedKeys.add(keyOf(o)));
    return fresh.length;
  } catch (e: unknown) {
    console.error("[datacore] fmcsaoos archive:", e instanceof Error ? e.message : e);
    return 0;
  }
}

export function gzipOldOosDays(baseDir?: string, nowMs?: number): number {
  const dir = oosDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!fs.existsSync(dir)) return 0;
  let n = 0;
  for (const f of fs.readdirSync(dir)) {
    if (!f.endsWith(".jsonl")) continue;
    if (now - Date.parse(f.slice(0, 10)) < 2 * 86400_000) continue;
    const fp = path.join(dir, f);
    fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
    fs.unlinkSync(fp);
    n++;
  }
  return n;
}

/** Reconstructs a snapshot from the change-only dedup archive: for every
 *  dot_number+oos_date identity seen in the lookback window, keeps the
 *  observation with the latest `rt` timestamp (last known state), same
 *  pattern as cbpBorderWait.ts's backfillBorderWaitsFromArchive (the
 *  cold-cache-no-disk-backfill fix, research/open_questions.md). */
export function backfillOosOrdersFromArchive(baseDir?: string, nowMs?: number, days = LOOKBACK_DAYS): OosOrderObs[] {
  const dir = oosDir(baseDir);
  const now = nowMs ?? Date.now();
  const latest = new Map<string, OosOrderObs>();
  for (let i = 0; i < days; i++) {
    const d = new Date(now - i * 86400_000).toISOString().slice(0, 10);
    for (const fp of [path.join(dir, `${d}.jsonl`), path.join(dir, `${d}.jsonl.gz`)]) {
      let text: string | null = null;
      try {
        text = fp.endsWith(".gz")
          ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
          : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        let o: OosOrderObs;
        try { o = JSON.parse(line); } catch { continue; }
        if (!o || !o.dot_number || !o.oos_date) continue;
        const identity = `${o.dot_number}|${o.oos_date}`;
        const prev = latest.get(identity);
        if (!prev || (o.rt || "") > (prev.rt || "")) latest.set(identity, o);
      }
    }
  }
  return Array.from(latest.values());
}

// ── Cache + poll ────────────────────────────────────────────────────────────

let cache: { at: number; obs: OosOrderObs[] } | null = null;
let polling = false;

export function latestOosOrders() {
  return cache;
}

export async function refreshOosOrders(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<void> {
  try {
    const obs = await fetchOosOrders(fetchImpl, nowMs);
    const next = resolveCacheItems(cache !== null, obs ?? [], () => backfillOosOrdersFromArchive(undefined, nowMs));
    if (next) cache = { at: Date.now(), obs: next };
    if (obs !== null) {
      archiveOosOrders(obs, undefined, nowMs);
      gzipOldOosDays(undefined, nowMs);
    }
  } catch (e: unknown) {
    console.error("[datacore] fmcsaoos refresh:", e instanceof Error ? e.message : e);
  }
}

export function _resetOosCacheForTests(): void {
  cache = null;
}

/** Dataset refreshes roughly daily (live-confirmed); 6h poll keeps the
 *  archive close to current without hammering a low-cadence source.
 *  Eager boot per KNOWN BROKEN #9's established convention. */
export function bootOosPoll(intervalMs = 6 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshOosOrders();
  setInterval(() => { refreshOosOrders(); }, intervalMs).unref?.();
}
