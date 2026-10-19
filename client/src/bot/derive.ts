// Bot page v2 — stage health, issues and the day review, derived from real
// rows and fields only (pure, tested). Absence is reported as absence.
import { sortedNewestFirst, type AuditRow } from "./pageState";
import type { PanelId } from "./zoneModel";
import { fmtAge, fmtPct, fmtOcc, MINUS } from "./format";

export type Level = "ok" | "warn" | "bad" | "unk";
export type Box = "ok" | "late" | "miss" | "dead";
export interface StageRow { id: string; name: string; level: Level; value: string; boxes: Box[] }
export interface Issue {
  id: string; level: "L1" | "L2" | "L3"; title: string; condition: string; consequence: string; clears: string;
  action: { label: string; kind: "resume" | "stop" | "kill" | "focus"; target?: PanelId } | null; since: string | null;
}
export interface DeriveInput {
  audit: AuditRow[]; nowMs: number;
  killSwitch: boolean; active: boolean; marketOpen: boolean | null;
  equity: number | null; equityPeak: number | null; maxDrawdownPct: number | null;
  modelAgeHours: number | null; fallbackAtHours: number | null;
  constantFeatures: number | null; featureCount: number | null;
  openOrders: number | null; totalTrades: number | null;
  slotsAvailable: number | null; modelTypes: Record<string, number> | null;
}

const t = (r: AuditRow) => Date.parse(r.time);
const within = (r: AuditRow, now: number, ms: number) => Number.isFinite(t(r)) && now - t(r) <= ms;
const H = 3600_000;

export function todayEtKey(ms: number): string {
  return new Date(ms).toLocaleDateString("en-CA", { timeZone: "America/New_York" });
}
export function rowsToday(rows: AuditRow[], nowMs: number): AuditRow[] {
  const key = todayEtKey(nowMs);
  return rows.filter((r) => Number.isFinite(t(r)) && todayEtKey(t(r)) === key);
}

/** One box per expected tick, newest right: ok if a row landed in that
 *  bucket, miss otherwise; the newest bucket is "late" (not yet missed)
 *  while the current interval is still open. */
export function cadenceBoxes(times: number[], nowMs: number, intervalMs: number, n = 10): Box[] {
  if (!intervalMs || intervalMs <= 0) return [];
  const out: Box[] = [];
  for (let k = n - 1; k >= 0; k--) {
    const end = nowMs - k * intervalMs;
    const start = end - intervalMs;
    const hit = times.some((x) => x > start && x <= end);
    out.push(hit ? "ok" : k === 0 ? "late" : "miss");
  }
  return out;
}

export function deriveStages(i: DeriveInput): StageRow[] {
  const rows = sortedNewestFirst(i.audit);
  const now = i.nowMs;
  const first = (p: (r: AuditRow) => boolean) => rows.find(p) || null;
  const newest = rows[0] || null;
  const newestAge = newest ? now - t(newest) : null;

  const scanStarts = rows.filter((r) => r.type === "TIER2" && /^Starting scan/i.test(r.message)).map(t);
  const scanDone = first((r) => r.type === "TIER2" && /^Scanned\s/i.test(r.message));
  const intervalM = (() => { const r = first((x) => x.type === "TIER2" && /interval:\s*\d/i.test(x.message)); const m = r && /interval:\s*(\d+(?:\.\d+)?)\s*min/i.exec(r.message); return m ? Number(m[1]) : null; })();
  const intervalMs = intervalM ? intervalM * 60_000 : 5 * 60_000;

  const streamLive = first((r) => r.type === "STREAM" && /feed live/i.test(r.message));
  const streamDown = first((r) => r.type === "STREAM-DISCONNECT");
  const tickers = streamLive && /(\d+)\s+tickers/.exec(streamLive.message)?.[1];
  const feedsBad = !!(streamDown && (!streamLive || t(streamLive) < t(streamDown)) && now - t(streamDown) > 60_000 && i.marketOpen);

  const tier3 = rows.filter((r) => r.type === "TIER3" || r.type === "TIER3-ML-ERROR").map(t);
  const tier3Err = first((r) => r.type === "TIER3-ML-ERROR");
  const sizeBlocks = rows.filter((r) => r.type === "SIZE-BLOCK" && within(r, now, 2 * H));
  const posWarn = first((r) => r.type === "POS-WARN" && within(r, now, 2 * H));
  const shutdowns = rowsToday(rows, now).filter((r) => r.type === "SHUTDOWN" && /orders cancelled/i.test(r.message));
  const stalls = rowsToday(rows, now).filter((r) => r.type === "EVENTLOOP-LAG");

  const dd = i.equity !== null && i.equityPeak ? (i.equity / i.equityPeak - 1) * 100 : null;
  const killAt = i.maxDrawdownPct ?? null;
  const ddLevel: Level = i.killSwitch ? "bad" : dd !== null && killAt !== null && dd <= -(killAt - 2) ? "warn" : dd === null ? "unk" : "ok";

  const modelLevel: Level = i.modelAgeHours === null ? "unk" : i.fallbackAtHours !== null && i.modelAgeHours > i.fallbackAtHours ? "bad" : i.fallbackAtHours !== null && i.modelAgeHours > i.fallbackAtHours - 4 ? "warn" : tier3Err && within(tier3Err, now, 2 * H) ? "warn" : "ok";
  const fallbackCount = i.modelTypes ? Object.entries(i.modelTypes).filter(([k]) => k.startsWith("fallback")).reduce((a, [, v]) => a + v, 0) : 0;

  const constLevel: Level = i.constantFeatures === null ? "unk" : i.constantFeatures >= 11 ? "bad" : i.constantFeatures >= 5 ? "warn" : "ok";

  return [
    { id: "loop", name: "loop", level: newestAge === null ? "unk" : newestAge > 15 * 60_000 && i.marketOpen ? "bad" : newestAge > 10 * 60_000 ? "warn" : "ok", value: newest ? `${fmtAge(newestAge)} · ${i.active ? "active" : "paused"}` : "no rows", boxes: cadenceBoxes(rows.map(t), now, 5 * 60_000) },
    { id: "feeds", name: "feeds", level: feedsBad ? "bad" : streamLive ? "ok" : "unk", value: feedsBad ? `feed down ${fmtAge(now - t(streamDown!))}` : streamLive ? `WS ${tickers ?? "?"} · ${fmtAge(now - t(streamLive))}` : "no stream row", boxes: [] },
    { id: "scans", name: "scans", level: !scanDone ? "unk" : now - t(scanDone) > 2 * intervalMs && i.marketOpen ? "warn" : "ok", value: scanDone ? `${intervalM ?? "?"}m · ${/,\s+(\d+)\s+trade/.exec(scanDone.message)?.[1] ?? "?"} cand.` : "no scan yet", boxes: cadenceBoxes(scanStarts, now, intervalMs) },
    { id: "features", name: "features", level: constLevel, value: i.constantFeatures === null ? "not reported" : `${i.constantFeatures} / ${i.featureCount ?? "?"} constant`, boxes: [] },
    { id: "model", name: "model", level: modelLevel, value: i.modelAgeHours === null ? "age unknown" : `${i.modelAgeHours.toFixed(1)} / ${i.fallbackAtHours ?? "?"} h${fallbackCount ? ` · ${fallbackCount} fb` : ""}`, boxes: cadenceBoxes(tier3, now, H) },
    { id: "rules", name: "rules", level: i.slotsAvailable === null ? "unk" : i.slotsAvailable <= 0 ? "warn" : "ok", value: i.slotsAvailable === null ? "slots unknown" : `${i.slotsAvailable} stock slots`, boxes: [] },
    { id: "sizing", name: "sizing", level: sizeBlocks.length >= 3 ? "bad" : sizeBlocks.length ? "warn" : "ok", value: sizeBlocks.length ? `${sizeBlocks.length} refused · 2 h` : "none refused · 2 h", boxes: [] },
    { id: "orders", name: "orders", level: shutdowns.length ? "warn" : "ok", value: `${i.openOrders ?? "?"} open · ${shutdowns.length} deploy${shutdowns.length === 1 ? "" : "s"}`, boxes: [] },
    { id: "exits", name: "exits", level: posWarn ? "warn" : "ok", value: posWarn ? `${fmtOcc(posWarn.message.split(":")[0])} ${/down\s+(-?[\d.]+%)/.exec(posWarn.message)?.[1] ?? ""}`.trim() : "no warnings · 2 h", boxes: [] },
    { id: "risk", name: "risk", level: ddLevel, value: dd === null ? "equity unknown" : `${fmtPct(dd, 2)} · kill ${MINUS}${killAt ?? "?"}%${i.killSwitch ? " · latched" : ""}`, boxes: [] },
    { id: "state", name: "state", level: i.equityPeak ? "ok" : "unk", value: i.equityPeak ? `peak $${Math.round(i.equityPeak).toLocaleString("en-US")} · ${stalls.length} stall${stalls.length === 1 ? "" : "s"}` : "peak unknown", boxes: [] },
    { id: "measure", name: "measure", level: i.totalTrades === null ? "unk" : i.totalTrades < 30 ? "warn" : "ok", value: i.totalTrades === null ? "not reported" : `n ${i.totalTrades} · pnl at trigger`, boxes: [] },
  ];
}

export function deriveIssues(i: DeriveInput): Issue[] {
  const rows = sortedNewestFirst(i.audit);
  const now = i.nowMs;
  const out: Issue[] = [];
  const first = (p: (r: AuditRow) => boolean) => rows.find(p) || null;

  if (i.killSwitch) {
    const k = first((r) => r.type === "DRAWDOWN-KILL");
    const ev = first((r) => r.type === "KILL-AUTO-RESUME-EVAL");
    out.push({ id: "kill", level: "L3", title: "Kill switch latched", condition: k ? k.message.replace(/^MAX DRAWDOWN KILL SWITCH:\s*/i, "") : "kill switch set", consequence: "no entries · no exits of any kind · open orders cancelled", clears: ev ? ev.message.replace(/^EVALUATED\s*/i, "").slice(0, 160) : "auto-resume evaluates hourly", action: { label: "Resume", kind: "resume" }, since: k?.time ?? null });
  }
  if (i.modelAgeHours !== null && i.fallbackAtHours !== null && i.modelAgeHours > i.fallbackAtHours) {
    out.push({ id: "model", level: "L3", title: "Model on the rules fallback", condition: `model ${i.modelAgeHours.toFixed(1)} h old · line ${i.fallbackAtHours} h`, consequence: "every score since is rules-only; the ML column is empty", clears: "a retrain writes a new model (hourly Tier 3 check)", action: { label: "open model", kind: "focus", target: "model" }, since: null });
  }
  const newest = rows[0];
  if (newest && i.marketOpen && now - Date.parse(newest.time) > 2 * H) {
    out.push({ id: "dark", level: "L3", title: "Loop dark", condition: `no audit row for ${fmtAge(now - Date.parse(newest.time))} during market hours`, consequence: "nothing is scanning, monitoring or exiting", clears: "a new audit row", action: null, since: newest.time });
  }
  const fails = rows.filter((r) => (r.type === "TIER2-FAIL" || r.type === "TIER2-ERROR") && within(r, now, H));
  if (fails.length >= 3) out.push({ id: "scanner", level: "L2", title: "Scans failing", condition: `${fails.length} scan failures in the last hour`, consequence: "no new candidates", clears: "a scan completes", action: { label: "open stages", kind: "focus", target: "stages" }, since: fails[0].time });
  const sb = rows.filter((r) => r.type === "SIZE-BLOCK" && within(r, now, 24 * H));
  if (sb.length) {
    const buckets = new Set(sb.map((r) => /\[(\w+)\]/.exec(r.message)?.[1]).filter(Boolean));
    out.push({ id: "sizing", level: sb.length >= 5 ? "L3" : "L2", title: "Candidates refused by sizing", condition: `${sb.length} refusal${sb.length === 1 ? "" : "s"} · 24 h${buckets.size ? ` · ${[...buckets].join(", ")}` : ""}`, consequence: "candidates that passed score and sector were not placed", clears: "the bucket's live expectancy turns positive, or a prior reset", action: { label: "open decision", kind: "focus", target: "state" }, since: sb[sb.length - 1].time });
  }
  if (i.constantFeatures !== null && i.constantFeatures >= 5) {
    out.push({ id: "features", level: "L2", title: "Model inputs constant", condition: `${i.constantFeatures} / ${i.featureCount ?? "?"} features never move at serve time`, consequence: "the model is scored on a distribution it did not train on", clears: "each source wired (one PR per feature)", action: { label: "open model", kind: "focus", target: "model" }, since: null });
  }
  const sd = first((r) => r.type === "STREAM-DISCONNECT");
  const sl = first((r) => r.type === "STREAM" && /feed live/i.test(r.message));
  if (sd && (!sl || Date.parse(sl.time) < Date.parse(sd.time)) && now - Date.parse(sd.time) > 60_000 && i.marketOpen) {
    out.push({ id: "stream", level: "L3", title: "Real-time feed down", condition: `disconnected ${fmtAge(now - Date.parse(sd.time))}`, consequence: "stops starve — the exit engine sees no bars", clears: "STREAM feed live", action: null, since: sd.time });
  }
  const shut = rowsToday(rows, now).filter((r) => r.type === "SHUTDOWN" && /orders cancelled/i.test(r.message));
  if (shut.length) out.push({ id: "deploy", level: "L1", title: "Deploys cancelled orders", condition: `${shut.length} shutdown${shut.length === 1 ? "" : "s"} today`, consequence: "working orders were cancelled at each restart", clears: "—", action: null, since: shut[0].time });
  if (i.totalTrades !== null && i.totalTrades < 30) out.push({ id: "n", level: "L1", title: "Edge measured on a thin sample", condition: `n ${i.totalTrades} closed trades · pnl at trigger, not fill`, consequence: "expectancy is an upper bound", clears: "n ≥ 30 with fill-based P&L", action: null, since: null });
  const order = { L3: 0, L2: 1, L1: 2 };
  return out.sort((a, b) => order[a.level] - order[b.level]);
}

export interface DayReview {
  scans: number; candidatesMax: number; trades: number; deploys: number; stalls: number; stallMaxMs: number | null;
  sizingBlocks: number; posWarns: number; retrains: number; freshChecks: number; killAt: string | null; emails: number;
}
export function deriveDayReview(audit: AuditRow[], nowMs: number): DayReview {
  const rows = rowsToday(sortedNewestFirst(audit), nowMs);
  const scans = rows.filter((r) => r.type === "TIER2" && /^Scanned\s/i.test(r.message));
  const cands = scans.map((r) => Number(/,\s+(\d+)\s+trade/.exec(r.message)?.[1] ?? 0));
  const stalls = rows.filter((r) => r.type === "EVENTLOOP-LAG").map((r) => Number(/~(\d+)ms/.exec(r.message)?.[1] ?? 0));
  const kill = rows.find((r) => r.type === "DRAWDOWN-KILL");
  return {
    scans: scans.length, candidatesMax: cands.length ? Math.max(...cands) : 0,
    trades: rows.filter((r) => /^(TRADE|ETF-TRADE|OPTIONS-TRADE|MORNING-TRADE)$/.test(r.type)).length,
    deploys: rows.filter((r) => r.type === "SHUTDOWN" && /orders cancelled/i.test(r.message)).length,
    stalls: stalls.length, stallMaxMs: stalls.length ? Math.max(...stalls) : null,
    sizingBlocks: rows.filter((r) => r.type === "SIZE-BLOCK").length,
    posWarns: rows.filter((r) => r.type === "POS-WARN").length,
    retrains: rows.filter((r) => r.type === "TIER3" && /retrain complete/i.test(r.message)).length,
    freshChecks: rows.filter((r) => r.type === "TIER3" && /fresh/i.test(r.message)).length,
    killAt: kill ? kill.time : null,
    emails: rows.filter((r) => r.type === "EMAIL").length,
  };
}
