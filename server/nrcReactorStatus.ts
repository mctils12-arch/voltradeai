/**
 * nrcReactorStatus.ts — NRC daily Power Reactor Status Reports (EDGE
 * DOCTRINE axis (a); the last unbuilt CLAUDE.md-named-style item from the
 * NEW DATA ROOTS charter's own build order — signal_ladder.json's
 * `nrc_outage_reports` entry, filed 2026-07-03, never built until now).
 *
 * SOURCE: nrc.gov's own "Power Status" raw data text file — one
 * pipe-delimited file per calendar year, updated daily (NRC states
 * reactor status is collected 4am-8am ET), keyless, no signup, public
 * domain (US government work). Verified live this session via direct
 * curl (not assumed from training-data memory — NRC has restructured
 * this page before): `ReportDt|Unit|Power` header, one row per operating
 * unit per day, `Power` is percent of rated thermal power (0-100,
 * sometimes >100 briefly due to instrument rounding — NOT clamped here,
 * passed through verbatim per the "parse defensively, never silently
 * reshape a government number" precedent).
 *
 * HYPOTHESIS (gate 2, not attempted here — this module is DATA-layer
 * only): unplanned power drops at large single-operator units move
 * regional power prices and the operator's stock same-week; the join
 * partner is `datacore/powerplants/us_power_plants.json`'s 58 nuclear
 * rows (unit MW + operator), open_questions.md POWER-PLANT SIGNAL
 * HYPOTHESES entry. LADDER: gate 1 (this file's `matchToRegistry`) =
 * NRC report parse reconciles against the registry's unit set; gate 2 =
 * outage-day operator returns vs a same-universe random-entry baseline,
 * needs a quarter of archived history per the filed plan.
 *
 * NAME RECONCILIATION: NRC reports at UNIT granularity ("Beaver Valley
 * 1", "Beaver Valley 2"); the WRI GPPD registry (us_power_plants.json)
 * rows are at PLANT granularity, one row per site even when it hosts
 * multiple units ("Beaver Valley" covers both). `matchToRegistry` strips
 * a trailing unit number and applies a small hand-verified alias table
 * for the plant-name spelling differences between NRC's report and WRI's
 * registry (e.g. "Arkansas Nuclear 1" -> "Arkansas Nuclear One", "D.C.
 * Cook 1" -> "Donald C Cook") — every alias below was checked against
 * both source strings directly, not inferred. Two registry plants are
 * EXPECTED to have zero NRC match: Indian Point 2 and Indian Point 3
 * (both permanently shut down 2020/2021, so they legitimately never
 * appear in a live 2026 NRC status file) — this is the honest,
 * documented exception, not a matching-logic defect.
 *
 * Boundary: no imports from trading logic (SPINOUT-READY DATA LAYER).
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

const BASE_URL = "https://www.nrc.gov/reading-rm/doc-collections/event-status/reactor-status";

export interface ReactorStatusRow {
  date: string;   // YYYY-MM-DD
  unit: string;   // NRC unit name, verbatim
  power: number | null; // percent of rated thermal power
}

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

export function reactorStatusUrl(year: number): string {
  return `${BASE_URL}/${year}/${year}PowerStatus.txt`;
}

/** `ReportDt|Unit|Power` pipe-delimited, e.g. "8/3/2026 12:00:00 AM|Arkansas Nuclear 1|100".
 *  Defensive: skips malformed lines rather than throwing, matching the
 *  cropConditions/censusImports precedent of never letting one bad row
 *  kill the whole day's fetch. */
export function parseReactorStatus(text: string): ReactorStatusRow[] {
  const lines = text.split("\n");
  const out: ReactorStatusRow[] = [];
  for (const line of lines) {
    const parts = line.split("|");
    if (parts.length < 3) continue;
    const [dtRaw, unitRaw, powerRaw] = parts;
    if (dtRaw.trim() === "ReportDt") continue; // header
    const m = dtRaw.trim().match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})/);
    if (!m) continue;
    const [, mo, da, yr] = m;
    const date = `${yr}-${mo.padStart(2, "0")}-${da.padStart(2, "0")}`;
    const unit = unitRaw.trim();
    if (!unit) continue;
    const power = powerRaw.trim() === "" ? null : Number(powerRaw.trim());
    out.push({ date, unit, power: Number.isFinite(power as number) ? (power as number) : null });
  }
  return out;
}

export function latestDay(rows: ReactorStatusRow[]): string {
  return rows.reduce((mx, r) => (r.date > mx ? r.date : mx), "");
}

// ── Gate 1: unit -> registry plant-name reconciliation ──────────────────────

// Hand-verified 2026-08-04 directly against both the live NRC file and
// datacore/powerplants/us_power_plants.json's 58 nuclear rows (exact raw
// registry strings, copy-checked — not re-typed from memory). Only the
// genuine leftover mismatches after normalizePlantName + stripUnitSuffix
// need an entry; most of the 95 NRC units resolve without one. Keyed by
// the NRC name's normalized+unit-stripped form; value is the EXACT raw
// registry plant name (normalized the same way as the registry index at
// match time, so a fix to normalizePlantName can't silently desync this
// table from the index it's compared against).
const ALIASES: Record<string, string> = {
  "arkansas nuclear": "Arkansas Nuclear One",
  "dc cook": "Donald C Cook",
  hatch: "Edwin I Hatch",
  fitzpatrick: "James A Fitzpatrick",
  farley: "Joseph M Farley",
  robinson: "H B Robinson",
  "hope creek": "PSEG Hope Creek Generating Station",
  salem: "PSEG Salem Generating Station",
  ginna: "R E Ginna Nuclear Power Plant",
  "south texas": "South Texas Project",
  "saint lucie": "St Lucie",
  susquehanna: "TalenEnergy Susquehanna",
  summer: "V C Summer",
};

// Plants that legitimately never appear in a live NRC status file, so a
// non-match here is the EXPECTED, documented outcome, not a parser
// defect. Compared against the normalized registry name (see
// normalizePlantName). Verified live this session (2026-08-04), not
// assumed: Indian Point 2/3 permanently shut down 2020/2021 (Entergy);
// Duane Arnold ceased power 2020-08-10 (derecho damage) and has been in
// SAFSTOR decommissioning since — zero rows anywhere in the full 2026
// feed confirms this independently of the news search — NRC accepted a
// license-transfer application in 2026-01 for a targeted ~2028-2029
// restart, but it is not an operating reactor as of this session.
export const EXPECTED_REGISTRY_ONLY = new Set(["indian point 2", "indian point 3", "duane arnold"]);

const SUFFIX_WORDS = [
  "generating station", "nuclear station", "nuclear power plant",
  "nuclear plant", "nuclear facility", "power plant", "power station",
  "energy center", "generation station", "station", "plant", "facility",
  "nuclear",
];

export function normalizePlantName(raw: string): string {
  let s = raw.toLowerCase().trim().replace(/[.,]/g, "").replace(/-/g, " ").replace(/\s+/g, " ");
  for (const suf of SUFFIX_WORDS) {
    if (s.endsWith(" " + suf)) { s = s.slice(0, -(suf.length + 1)).trim(); break; }
  }
  return s;
}

/** Strips a trailing unit number ("beaver valley 1" -> "beaver valley")
 *  UNLESS the digit is part of the plant's own formal name (Waterford 3
 *  has no unit 1/2; matches the registry as-is either way since its
 *  normalized form already ends "waterford 3"). */
function stripUnitSuffix(s: string): string {
  return s.replace(/\s+\d+$/, "").trim();
}

export interface MatchResult {
  matched: number;
  unmatched: string[];
  expectedGaps: string[]; // registry plants correctly absent from the NRC file
  unexpectedRegistryGaps: string[]; // registry plants absent from the NRC file with NO documented reason — a real finding, never silently dropped
  matchRate: number;
}

/** Gate 1: does every NRC unit name resolve to a registry plant? Pure
 *  function — no network, no fs — takes the already-parsed unit names and
 *  the already-loaded registry plant names so it is trivially unit-testable
 *  and reusable by scripts/nrc_gate1_registry_match.ts against live data. */
export function matchToRegistry(nrcUnitNames: string[], registryPlantNames: string[]): MatchResult {
  const registryNorm = new Map<string, string>(); // normalized -> original
  for (const name of registryPlantNames) registryNorm.set(normalizePlantName(name), name);

  // Alias values are raw registry strings — normalize them through the
  // SAME function that built registryNorm, so the two can never drift
  // out of sync with each other.
  const aliasNorm = new Map<string, string>();
  for (const [k, v] of Object.entries(ALIASES)) aliasNorm.set(k, normalizePlantName(v));

  const candidatesFor = (unit: string): string[] => {
    const n = normalizePlantName(unit);
    const stripped = stripUnitSuffix(n);
    // Re-run suffix-word stripping after the unit number comes off — a
    // name like "River Bend Station 1" only exposes its "station" suffix
    // once the trailing "1" is gone (normalizePlantName's suffix check
    // only fires when the suffix word is what the string CURRENTLY ends
    // with).
    const doubleStripped = normalizePlantName(stripped);
    return [
      n, stripped, doubleStripped,
      aliasNorm.get(n) || "", aliasNorm.get(stripped) || "", aliasNorm.get(doubleStripped) || "",
    ].filter(Boolean);
  };

  const uniqueUnits = [...new Set(nrcUnitNames)];
  const unmatched: string[] = [];
  const matchedRegistryNorms = new Set<string>();
  let matched = 0;
  for (const unit of uniqueUnits) {
    const hitNorm = candidatesFor(unit).find((c) => registryNorm.has(c));
    if (hitNorm) { matched++; matchedRegistryNorms.add(hitNorm); }
    else unmatched.push(unit);
  }

  const allRegistryGaps = [...registryNorm.keys()].filter((rn) => !matchedRegistryNorms.has(rn));
  const expectedGaps = allRegistryGaps.filter((rn) => EXPECTED_REGISTRY_ONLY.has(rn));
  const unexpectedRegistryGaps = allRegistryGaps
    .filter((rn) => !EXPECTED_REGISTRY_ONLY.has(rn))
    .map((rn) => registryNorm.get(rn)!);

  return {
    matched,
    unmatched,
    expectedGaps,
    unexpectedRegistryGaps,
    matchRate: uniqueUnits.length ? matched / uniqueUnits.length : 0,
  };
}

// ── Fetch ─────────────────────────────────────────────────────────────────

export async function fetchReactorStatus(fetchImpl: FetchFn = fetch as any,
                                         nowMs?: number): Promise<ReactorStatusRow[]> {
  const year = new Date(nowMs ?? Date.now()).getUTCFullYear();
  try {
    const r = await fetchImpl(reactorStatusUrl(year), {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(30000) as any,
    });
    if (!r.ok) {
      console.error(`[datacore] nrcreactorstatus -> ${r.status}`);
      return [];
    }
    const rows = parseReactorStatus(await r.text());
    const latest = latestDay(rows);
    return latest ? rows.filter((row) => row.date === latest) : [];
  } catch (e: any) {
    console.error("[datacore] nrcreactorstatus fetch:", e?.message || e);
    return [];
  }
}

// ── Archive (event-identity dedup date|unit, day-file per fetch) ────────────

const seenRows = new Set<string>();
let seeded = false;

const rowKey = (r: ReactorStatusRow) => `${r.date}|${r.unit}`;

function statusDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "nrcreactorstatus");
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
        try { seenRows.add(rowKey(JSON.parse(line))); } catch {}
      }
    }
  } catch {}
}

export function archiveReactorStatus(rows: ReactorStatusRow[], baseDir?: string, nowMs?: number): number {
  if (!rows.length) return 0;
  const dir = statusDir(baseDir);
  if (!seeded) { seedSeen(dir); seeded = true; }
  const fresh = rows.filter((r) => !seenRows.has(rowKey(r)));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const day = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
    fs.appendFileSync(path.join(dir, `${day}.jsonl`),
                      fresh.map((r) => JSON.stringify(r)).join("\n") + "\n");
    fresh.forEach((r) => seenRows.add(rowKey(r)));
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] nrcreactorstatus archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldStatusDays(baseDir?: string, nowMs?: number): number {
  const dir = statusDir(baseDir);
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

// ── Cache + poll ─────────────────────────────────────────────────────────

let cache: { at: number; date: string; rows: ReactorStatusRow[] } | null = null;
let polling = false;

export function latestReactorStatus() {
  return cache;
}

export async function refreshReactorStatus(fetchImpl: FetchFn = fetch as any,
                                           nowMs?: number, baseDir?: string): Promise<void> {
  try {
    const rows = await fetchReactorStatus(fetchImpl, nowMs);
    if (rows.length) {
      archiveReactorStatus(rows, baseDir, nowMs);
      cache = { at: Date.now(), date: latestDay(rows), rows };
    }
    gzipOldStatusDays(baseDir, nowMs);
  } catch (e: any) {
    console.error("[datacore] nrcreactorstatus refresh:", e?.message || e);
  }
}

/** Source updates once/day (NRC collects 4am-8am ET); 6h poll makes
 *  off-hours a dedup no-op. Eager boot per KNOWN BROKEN #9's precedent
 *  (no visitor should mean no archive gap for a daily government feed
 *  either). */
export function bootReactorStatusPoll(intervalMs = 6 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshReactorStatus();
  setInterval(() => { refreshReactorStatus(); }, intervalMs).unref?.();
}
