/**
 * Hermetic tests for visitHistory.ts (the persisted daily visit ledger).
 * Temp files only; the clock is injected so day rollover is deterministic.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import http from "node:http";
import express from "express";
import {
  createVisitHistory, recordDailyVisit, visitHistorySnapshot, _useVisitHistory,
  DAY_ROUTE_CAP, DAY_UNIQUES_CAP, VISIT_HISTORY_DAYS,
} from "./visitHistory";
import { requestLog } from "./requestLog";

const T0 = Date.parse("2026-10-13T09:00:00Z");
const HOUR = 3_600_000;
const tmpFile = () => path.join(fs.mkdtempSync(path.join(os.tmpdir(), "vt-visits-")), "site_visits.json");
const clock = (start: number) => { let t = start; return { now: () => t, set: (v: number) => { t = v; } }; };

test("counts page loads, unique visitors and the browser subset", () => {
  const c = clock(T0);
  const v = createVisitHistory(tmpFile(), c.now);
  v.record("1.1.1.1", "browser", "/");
  v.record("1.1.1.1", "browser", "/app");      // same person, second page
  v.record("2.2.2.2", "bot", "/");
  v.record(undefined, "cli", "/");              // no IP -> one "anon" visitor
  const d = v.snapshot().days["2026-10-13"];
  assert.deepEqual({ views: d.views, uniques: d.uniques, browserViews: d.browserViews, browserUniques: d.browserUniques },
    { views: 4, uniques: 3, browserViews: 2, browserUniques: 1 });
  assert.deepEqual(d.routes, { "/": 3, "/app": 1 });
});

test("a same-day restart keeps deduping (the day's salt + hashes persist); no raw IP ever reaches the file", () => {
  const file = tmpFile();
  const c = clock(T0);
  const a = createVisitHistory(file, c.now);
  a.record("10.0.0.7", "browser", "/");
  assert.equal(a.flush(), true);
  assert.equal(a.flush(), false, "nothing new -> no write");
  assert.ok(!fs.existsSync(`${file}.tmp`), "atomic write leaves no temp file");
  const text = fs.readFileSync(file, "utf8");
  assert.ok(!text.includes("10.0.0.7"), "raw IP must not be persisted");

  c.set(T0 + 2 * HOUR);
  const b = createVisitHistory(file, c.now);    // the redeploy
  b.record("10.0.0.7", "browser", "/data");
  b.record("10.0.0.8", "browser", "/");
  const d = b.snapshot().days["2026-10-13"];
  assert.equal(d.views, 3);
  assert.equal(d.uniques, 2, "the returning visitor is not counted twice across the restart");
  assert.equal(d.browserUniques, 2);
});

test("UTC rollover drops the salt and hashes: past days keep counts only, visitors can't be linked across days", () => {
  const file = tmpFile();
  const c = clock(T0);
  const v = createVisitHistory(file, c.now);
  v.record("10.0.0.7", "browser", "/");
  v.flush();
  const salt1 = JSON.parse(fs.readFileSync(file, "utf8")).today.salt;

  c.set(Date.parse("2026-10-14T00:30:00Z"));
  v.record("10.0.0.7", "browser", "/");
  v.flush();
  const saved = JSON.parse(fs.readFileSync(file, "utf8"));
  assert.equal(saved.today.day, "2026-10-14");
  assert.notEqual(saved.today.salt, salt1, "a fresh salt every day");
  assert.equal(saved.today.hashes.length, 1, "only today's hashes are kept");
  assert.equal(saved.days["2026-10-13"].uniques, 1);
  assert.equal(saved.days["2026-10-14"].uniques, 1, "the same IP is a new visitor on a new day");
  assert.ok(!JSON.stringify(saved.days).includes(salt1));

  // A restart on a LATER day than the saved salt drops it at the next flush.
  c.set(Date.parse("2026-10-16T08:00:00Z"));
  const w = createVisitHistory(file, c.now);
  assert.equal(w.flush(), true);
  assert.equal(JSON.parse(fs.readFileSync(file, "utf8")).today, null);
});

test("route tally is capped and attacker-chosen paths can't reach Object.prototype", () => {
  const v = createVisitHistory(tmpFile(), clock(T0).now);
  for (const r of ["/__proto__", "__proto__", "constructor", "prototype"]) v.record("9.9.9.9", "browser", r);
  for (let i = 0; i < DAY_ROUTE_CAP + 5; i++) v.record("9.9.9.9", "browser", `/p${i}`);
  const routes = v.snapshot().days["2026-10-13"].routes;
  assert.ok(Object.keys(routes).length <= DAY_ROUTE_CAP + 1, "cap + the (other) bucket");
  assert.ok(routes["(other)"] >= 5 + 3, "overflow and prototype-named paths land in (other)");
  assert.equal(({} as Record<string, unknown>).polluted, undefined);
  assert.equal(typeof routes["/__proto__"], "number");
});

test("unique visitors stop at the cap and the day says so", () => {
  const v = createVisitHistory(tmpFile(), clock(T0).now);
  for (let i = 0; i <= DAY_UNIQUES_CAP; i++) v.record(`10.${(i >> 16) & 255}.${(i >> 8) & 255}.${i & 255}`, "bot", "/");
  const d = v.snapshot().days["2026-10-13"];
  assert.equal(d.uniques, DAY_UNIQUES_CAP);
  assert.equal(d.uniquesCapped, true);
  assert.equal(d.views, DAY_UNIQUES_CAP + 1, "views are never capped");
});

test("an unreadable file is moved aside (not overwritten) and a new ledger starts", () => {
  const file = tmpFile();
  fs.writeFileSync(file, "{ not json");
  const errors: unknown[] = [];
  const orig = console.error;
  console.error = (...a: unknown[]) => { errors.push(a); };
  try {
    const v = createVisitHistory(file, clock(T0).now);
    v.record("1.2.3.4", "browser", "/");
    v.flush();
  } finally { console.error = orig; }
  const aside = fs.readdirSync(path.dirname(file)).filter((f) => f.includes(".unreadable-"));
  assert.equal(aside.length, 1);
  assert.equal(fs.readFileSync(path.join(path.dirname(file), aside[0]), "utf8"), "{ not json", "the old bytes survive");
  assert.equal(JSON.parse(fs.readFileSync(file, "utf8")).days["2026-10-13"].views, 1);
  assert.ok(errors.length >= 1, "loud, never silent");
});

test("retention keeps the newest VISIT_HISTORY_DAYS days", () => {
  const c = clock(Date.parse("2025-01-01T12:00:00Z"));
  const v = createVisitHistory(tmpFile(), c.now);
  for (let i = 0; i < VISIT_HISTORY_DAYS + 10; i++) {
    c.set(Date.parse("2025-01-01T12:00:00Z") + i * 86_400_000);
    v.record("1.1.1.1", "browser", "/");
  }
  const days = Object.keys(v.snapshot().days).sort();
  assert.ok(days.length <= VISIT_HISTORY_DAYS + 1);
  assert.equal(days[days.length - 1], new Date(c.now()).toISOString().slice(0, 10), "today is kept");
  assert.ok(!days.includes("2025-01-01"), "the oldest day is dropped");
});

test("requestLog feeds the ledger for page views only; inert until a ledger is set", async () => {
  recordDailyVisit("1.1.1.1", "browser", "/");     // no ledger -> no-op, no throw
  assert.equal(visitHistorySnapshot(), null);
  const v = createVisitHistory(tmpFile());
  _useVisitHistory(v);
  const app = express();
  app.use(requestLog());
  app.get("/data", (_q, r) => r.send("page"));
  app.get("/api/x", (_q, r) => r.json({ ok: 1 }));
  const srv = http.createServer(app);
  await new Promise<void>((r) => srv.listen(0, "127.0.0.1", () => r()));
  const port = (srv.address() as { port: number }).port;
  try {
    await fetch(`http://127.0.0.1:${port}/data`, { headers: { "user-agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120" } });
    await fetch(`http://127.0.0.1:${port}/data`, { headers: { "user-agent": "curl/8.0" } });
    await fetch(`http://127.0.0.1:${port}/api/x`);
    await new Promise((r) => setTimeout(r, 30));     // let finish handlers run
    const days = Object.values(visitHistorySnapshot()!.days);
    assert.equal(days.length, 1);
    assert.equal(days[0].views, 2, "the API call is not a page view");
    assert.equal(days[0].browserViews, 1);
    assert.equal(days[0].uniques, 1, "same client address -> one visitor");
  } finally {
    _useVisitHistory(null);
    await new Promise((r) => srv.close(r));
  }
});
