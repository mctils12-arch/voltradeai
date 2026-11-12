// githubGate2 — weekly in-process job + read-only status for the pre-registered
// GATE 2 test of github_org_engineering_momentum (scripts/github_activity_gate2.ts's rule).
// Run: npx tsx --test server/githubGate2.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  githubGate2Flag, inGithubGate2Window, decideGithubGate2, evaluateGithubGate2, fetchAlpacaCloses, barsWindow, carryFirstSufficientRun,
  forwardCloseReturn, nearestTradingDayOnOrBefore, runGithubGate2, githubGate2Status, readArchiveSeries,
  MIN_CLUSTERS, MIN_PANEL_FOR_WEEK, GITHUB_GATE2_TIMEOUT_MS, GITHUB_GATE2_STALE_DAYS, GITHUB_GATE2_MIN_FREE_BYTES,
  GITHUB_GATE2_WINDOW_START_MIN, GITHUB_GATE2_WINDOW_END_MIN, type OrgWeek, type CloseSeries, type GithubGate2Evaluation,
} from "./githubGate2";
import { inEarningsQoqWindow, EARNINGS_QOQ_WINDOW_END_MIN } from "./earningsQoq";
import { inLazyPricesWindow } from "./lazyPrices";
import { WINDOW_START_MIN as PANEL_WINDOW_START_MIN } from "./pricePanel";
import { WATCHLIST, archiveGithubActivity, _resetGithubActivityForTests, type GithubActivityRecord } from "./githubOrgActivity";
import { JOB_REGISTRY } from "./jobHeartbeat";
import { resolveAlpacaFeed } from "./alpacaFeed";
import { sanitizeDiag, DIAG_PROBES } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));
// 2026-11-15 is a Sunday — the ladder's registered not_before date; 00:30Z Mon = 19:30 ET Sun (EST)
const SUN_IN = Date.parse("2026-11-16T00:30:00Z");
const DAY = 86_400_000;
const base = { enabled: true, latest: null, lastAttemptWeek: "", codeVersion: "1.0.1", freeBytes: 5e9 as number | null };

test("flag: env present wins; registry enabledWhen agrees; registry covers the budget", () => {
  const job = JOB_REGISTRY.find((j) => j.id === "github_gate2_weekly");
  assert.ok(job && job.kind === "interval" && job.source === "server/githubGate2.ts");
  assert.ok(job!.maxRunMs >= GITHUB_GATE2_TIMEOUT_MS, "registry maxRunMs must cover the job budget");
  for (const [v, on] of [[undefined, true], ["1", true], ["0", false], ["off", false], ["", false]] as const) {
    const env = v === undefined ? {} : { VOLTRADE_GITHUB_GATE2: v };
    assert.equal(githubGate2Flag(env).on, on);
    assert.equal(job!.enabledWhen!(env), on);
  }
});

test("window is Sunday 19:00-20:00 ET only — after earningsQoq (18-19) and lazyPrices (16-18), done before the 20:15 panel build", () => {
  assert.equal(inGithubGate2Window(SUN_IN), true);
  assert.equal(inGithubGate2Window(Date.parse("2026-11-15T23:30:00Z")), false);   // 18:30 ET (earningsQoq's window)
  assert.equal(inGithubGate2Window(Date.parse("2026-11-16T01:05:00Z")), false);   // 20:05 ET
  assert.equal(inGithubGate2Window(Date.parse("2026-11-15T00:30:00Z")), false);   // Saturday 19:30 ET
  assert.equal(inEarningsQoqWindow(SUN_IN), false, "disjoint from earningsQoq");
  assert.equal(inLazyPricesWindow(SUN_IN), false, "disjoint from lazyPrices");
  assert.ok(GITHUB_GATE2_WINDOW_START_MIN >= EARNINGS_QOQ_WINDOW_END_MIN, "starts no earlier than earningsQoq ends");
  assert.ok(GITHUB_GATE2_WINDOW_END_MIN * 60_000 + GITHUB_GATE2_TIMEOUT_MS <= PANEL_WINDOW_START_MIN * 60_000, "latest start + budget ends before the panel build");
  // the first run lands on the registered date: a Sunday tick on 2026-11-15 decides "run"
  assert.equal(decideGithubGate2(SUN_IN, base).action, "run");
});

test("decide: idle outside/disabled/already-attempted; skip on low disk or a fresh same-version result; run when stale or code changed", () => {
  assert.equal(decideGithubGate2(SUN_IN, { ...base, enabled: false }).action, "idle");
  assert.equal(decideGithubGate2(Date.parse("2026-11-15T00:30:00Z"), base).action, "idle");
  const run = decideGithubGate2(SUN_IN, base);
  assert.equal(run.action, "run");
  assert.equal(decideGithubGate2(SUN_IN, { ...base, lastAttemptWeek: (run as { week: string }).week }).action, "idle");
  assert.equal(decideGithubGate2(SUN_IN, { ...base, freeBytes: GITHUB_GATE2_MIN_FREE_BYTES - 1 }).action, "skip");
  assert.equal(decideGithubGate2(SUN_IN, { ...base, freeBytes: 1_000_000_000 }).action, "run", "typical live free space (~1 GB) must not block");
  assert.equal(decideGithubGate2(SUN_IN, { ...base, freeBytes: null }).action, "run");
  const fresh = { code_version: "1.0.1", generated_at: new Date(SUN_IN - 2 * DAY).toISOString() };
  assert.equal(decideGithubGate2(SUN_IN, { ...base, latest: fresh }).action, "skip");
  assert.equal(decideGithubGate2(SUN_IN, { ...base, latest: { ...fresh, code_version: "1.0.0" } }).action, "run");
  const weekOld = { ...fresh, generated_at: new Date(SUN_IN - (GITHUB_GATE2_STALE_DAYS + 1) * DAY).toISOString() };
  assert.equal(decideGithubGate2(SUN_IN, { ...base, latest: weekOld }).action, "run", "a weekly cadence re-runs last Sunday's result");
});

// ── synthetic panel: N weeks, 15 tickers, a planted TOP>MID>BOTTOM structure ──

const TICKERS = WATCHLIST.map((o) => o.ticker);

/** Trading days (Mon-Fri) from a Monday start, `n` of them. */
function tradingDays(startYmd: string, n: number): string[] {
  const out: string[] = [];
  let t = Date.parse(`${startYmd}T00:00:00Z`);
  while (out.length < n) {
    const dow = new Date(t).getUTCDay();
    if (dow >= 1 && dow <= 5) out.push(new Date(t).toISOString().slice(0, 10));
    t += DAY;
  }
  return out;
}

/** `weeks` archived Mon-Sun weeks from 2026-07-20; ticker i's total moves so
 *  that in week w the ranking is a deterministic rotation (every ticker lands
 *  in every bucket over the sample). */
function syntheticArchive(weeks: number): Map<string, OrgWeek[]> {
  const out = new Map<string, OrgWeek[]>();
  const mon0 = Date.parse("2026-07-20T00:00:00Z");
  TICKERS.forEach((ticker, i) => {
    const rows: OrgWeek[] = [];
    for (let w = 0; w < weeks; w++) {
      const ws = new Date(mon0 + w * 7 * DAY).toISOString().slice(0, 10);
      const we = new Date(mon0 + (w * 7 + 6) * DAY).toISOString().slice(0, 10);
      // growth factor cycles with (i + w) so each week has a distinct ordering
      const rank = (i + w) % 15;
      const total = Math.round(1000 * Math.pow(1.01, rank * (w % 2 === 0 ? 1 : -1)) * (1 + 0.001 * w));
      rows.push({ ticker, weekStart: ws, weekEnd: we, total });
    }
    out.set(ticker, rows);
  });
  return out;
}

/** Prices: ticker in that week's TOP bucket gains `edge` over +5d, BOTTOM loses it, MID flat. */
function plantedPrices(archive: Map<string, OrgWeek[]>, edge: number): Map<string, CloseSeries> {
  const dates = tradingDays("2026-07-13", 160);
  const deltas = new Map<string, Map<string, number>>(); // ticker -> weekEnd -> pct
  for (const [ticker, weeks] of archive) {
    const m = new Map<string, number>();
    for (let i = 1; i < weeks.length; i++) m.set(weeks[i].weekEnd, (weeks[i].total! - weeks[i - 1].total!) / weeks[i - 1].total!);
    deltas.set(ticker, m);
  }
  const out = new Map<string, CloseSeries>();
  for (const ticker of archive.keys()) {
    const closes: number[] = [];
    let px = 100;
    for (let k = 0; k < dates.length; k++) {
      // week W ends Sunday S; its entry is the Friday before S and its +5d exit the Friday after,
      // so W's planted edge is spread over the five bars Mon..Fri AFTER S (each bar's move is
      // keyed to the Sunday that precedes it)
      if (k > 0) px *= 1 + drift(ticker, dates[k], deltas);
      closes.push(px);
    }
    out.set(ticker, { dates, closes });
  }
  return out;

  function drift(ticker: string, d: string, dl: Map<string, Map<string, number>>): number {
    // rank this ticker's delta for the week whose weekEnd is the Sunday BEFORE d
    const dt = new Date(`${d}T00:00:00Z`);
    const dow = dt.getUTCDay(); // 1..5
    const sunday = new Date(dt.getTime() - dow * DAY).toISOString().slice(0, 10);
    const mine = dl.get(ticker)?.get(sunday);
    if (mine == null) return 0;
    const all = Array.from(dl.values()).map((m) => m.get(sunday)).filter((x): x is number => x != null).sort((a, b) => b - a);
    const third = Math.floor(all.length / 3);
    const rank = all.indexOf(mine);
    // spread the week's +5d edge evenly over the following week's 5 bars
    if (rank < third) return edge / 5;
    if (rank >= all.length - third) return -edge / 5;
    return 0;
  }
}

test("evaluate: WAITING below MIN_CLUSTERS, never PASS/FAIL; a 15-week planted edge renders PASS; a reversed edge renders FAIL", () => {
  const few = syntheticArchive(10);
  const evFew = evaluateGithubGate2(few, plantedPrices(few, 0.05));
  assert.equal(evFew.verdict, "WAITING");
  assert.ok(evFew.n_clusters < MIN_CLUSTERS);
  assert.ok(evFew.clusterTest, "exploratory statistics are still reported while WAITING");

  const full = syntheticArchive(MIN_CLUSTERS + 2); // +1 week consumed by the first delta, +1 unrealized at the series end
  const evPass = evaluateGithubGate2(full, plantedPrices(full, 0.05));
  assert.ok(evPass.n_clusters >= MIN_CLUSTERS, `n_clusters ${evPass.n_clusters}`);
  assert.equal(evPass.ready, true);
  assert.equal(evPass.orderingHolds, true);
  assert.equal(evPass.survivesAtCrit005, true);
  assert.equal(evPass.verdict, "PASS");
  assert.equal(evPass.panel.watchlist, 15);
  assert.equal(evPass.panel.tickers_priced, 15);

  const evFail = evaluateGithubGate2(full, plantedPrices(full, -0.05));
  assert.equal(evFail.verdict, "FAIL");
  assert.equal(evFail.orderingHolds, false);
  assert.ok(evFail.clusterTest!.t < 0);

  // significant but with the ordering broken (MID above TOP) is still FAIL — both clauses gate
  const noise = plantedPrices(full, 0.0);
  const evNoise = evaluateGithubGate2(full, noise);
  assert.equal(evNoise.verdict, "FAIL");
  assert.equal(evNoise.survivesAtCrit005, false, "a zero-edge panel must not read significant");
});

test("evaluate: a week with fewer than MIN_PANEL_FOR_WEEK valid deltas is dropped; an unpriced ticker scores null, never zero", () => {
  const arch = syntheticArchive(6);
  // knock 4 tickers out of week index 3 (null totals) -> 11 valid deltas that week (< 12) -> dropped; the NEXT week also loses those 4 priors
  for (const t of TICKERS.slice(0, 4)) arch.get(t)![3].total = null;
  const prices = plantedPrices(arch, 0.02);
  const ev = evaluateGithubGate2(arch, prices);
  assert.ok(ev.panel.dropped_weeks >= 1);
  const dropped = arch.get(TICKERS[5])![3].weekEnd;
  assert.ok(!ev.results.some((r) => r.weekEnd === dropped), "the thin week must not be bucketed unevenly");
  prices.delete(TICKERS[7]);
  const ev2 = evaluateGithubGate2(arch, prices);
  assert.equal(ev2.panel.tickers_priced, 14);
  assert.ok(ev2.results.every((r) => r.topMean5 === null || Number.isFinite(r.topMean5)));
});

test("forward return: exact entry date, +h bars, null when the window is not yet in the past (no lookahead)", () => {
  const s: CloseSeries = { dates: ["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09"], closes: [100, 101, 102, 103, 104, 110] };
  assert.equal(nearestTradingDayOnOrBefore(s, "2026-11-08"), "2026-11-06", "a Sunday weekEnd enters on the Friday");
  assert.equal(nearestTradingDayOnOrBefore(s, "2026-11-01"), null);
  assert.ok(Math.abs(forwardCloseReturn(s, "2026-11-02", 5)! - 0.10) < 1e-12);
  assert.equal(forwardCloseReturn(s, "2026-11-03", 5), null, "+5 bars past the series end is unrealized");
  assert.equal(forwardCloseReturn(s, "2026-11-07", 1), null, "a non-trading entry date is not snapped silently");
});

test("fetchAlpacaCloses: multi-symbol adjustment=all request on the resolved feed, follows page tokens, builds ascending series, omits empty symbols", async () => {
  const urls: string[] = [];
  const get = async (url: string) => {
    urls.push(url);
    if (!url.includes("page_token")) {
      return { bars: { MDB: [{ t: "2026-11-04T05:00:00Z", c: 250.5 }, { t: "2026-11-03T05:00:00Z", c: 249 }], NET: [{ t: "bad", c: 1 }, { t: "2026-11-03T05:00:00Z", c: "x" }] }, next_page_token: "p2" };
    }
    return { bars: { MDB: [{ t: "2026-11-05T05:00:00Z", c: 252 }], DDOG: [{ t: "2026-11-03T05:00:00Z", c: 120 }] }, next_page_token: null };
  };
  const out = await fetchAlpacaCloses(get, "https://data.example", ["MDB", "NET", "DDOG"], "2026-07-13", "2026-11-14", resolveAlpacaFeed(undefined));
  assert.equal(urls.length, 2);
  const u = new URL(urls[0]);
  assert.equal(u.pathname, "/v2/stocks/bars");
  assert.equal(u.searchParams.get("symbols"), "MDB,NET,DDOG");
  assert.equal(u.searchParams.get("feed"), "delayed_sip", "the subscription-permitted default, never a hardcoded sip");
  assert.equal(u.searchParams.get("adjustment"), "all");
  assert.equal(u.searchParams.get("timeframe"), "1Day");
  assert.equal(u.searchParams.get("start"), "2026-07-13");
  assert.equal(u.searchParams.get("end"), "2026-11-14");
  assert.ok(urls[1].includes("page_token=p2"));
  assert.deepEqual(out.get("MDB"), { dates: ["2026-11-03", "2026-11-04", "2026-11-05"], closes: [249, 250.5, 252] });
  assert.deepEqual(out.get("DDOG"), { dates: ["2026-11-03"], closes: [120] });
  assert.equal(out.has("NET"), false, "a symbol with no valid bar is omitted, not zero-filled");
});

test("barsWindow: a week before the earliest archived weekEnd through yesterday (ET); null on an empty archive", () => {
  const arch = syntheticArchive(3);
  const w = barsWindow(arch, Date.parse("2026-11-16T00:30:00Z")); // Sun 19:30 ET
  assert.equal(w!.start, "2026-07-19");
  assert.equal(w!.end, "2026-11-14", "Saturday: yesterday in ET, so Friday's bar is final and the free-SIP 15-minute rule can never bite");
  assert.equal(barsWindow(new Map(), Date.now()), null);
});

test("first_sufficient_run: never created by WAITING, created by the first PASS/FAIL, then carried forward UNCHANGED", () => {
  const waiting = { verdict: "WAITING", n_clusters: 10, clusterTest: null, orderingHolds: false } as unknown as GithubGate2Evaluation;
  const fail = { verdict: "FAIL", n_clusters: 15, clusterTest: { t: -0.4, df: 14 }, orderingHolds: false } as unknown as GithubGate2Evaluation;
  const pass = { verdict: "PASS", n_clusters: 16, clusterTest: { t: 2.9, df: 15 }, orderingHolds: true } as unknown as GithubGate2Evaluation;
  assert.equal(carryFirstSufficientRun(null, waiting, "2026-11-08T00:00:00Z", "1.0.1"), null);
  const first = carryFirstSufficientRun(null, fail, "2026-11-16T00:31:00Z", "1.0.1");
  assert.deepEqual(first, { generated_at: "2026-11-16T00:31:00Z", code_version: "1.0.1", verdict: "FAIL", n_clusters: 15, t: -0.4, df: 14, orderingHolds: false });
  const later = carryFirstSufficientRun({ first_sufficient_run: first }, pass, "2026-11-23T00:31:00Z", "1.0.2");
  assert.deepEqual(later, first, "a later PASS does not overwrite the registered verdict");
});

test("run: reads the real archive format, writes latest.json + last_run.json with statistics only (no price), skips without Alpaca keys, never throws", async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "gg2-"));
  try {
    _resetGithubActivityForTests();
    // 17 archived weeks in the archiver's own on-disk format (archiveGithubActivity), base dir = tmp/archive
    const archDir = path.join(tmp, "archive");
    const synth = syntheticArchive(MIN_CLUSTERS + 2);
    const recs: GithubActivityRecord[] = [];
    for (const [ticker, weeks] of synth) {
      for (const w of weeks) {
        recs.push({ t: "org_week", key: `${w.weekStart}|${ticker}`, ticker, org: ticker.toLowerCase(), weekStart: w.weekStart, weekEnd: w.weekEnd, mergedPRs: Math.floor(w.total! / 2), commits: w.total! - Math.floor(w.total! / 2), uniqueActorsSample: null, actorSampleCapped: false, rt: w.weekEnd });
      }
    }
    assert.equal(archiveGithubActivity(recs, archDir, Date.parse("2026-11-10T00:00:00Z")), recs.length);
    const read = readArchiveSeries(archDir);
    assert.equal(read.size, 15);
    assert.equal(read.get("MDB")![0].total, synth.get("MDB")![0].total);

    const root = path.join(tmp, "github_gate2");
    const noKeys = await runGithubGate2({ env: {}, root, archiveBaseDir: archDir, now: () => SUN_IN });
    assert.equal(noKeys.status, "skipped");
    assert.match(noKeys.detail, /ALPACA_KEY/);
    assert.ok(!fs.existsSync(path.join(root, "latest.json")));

    const prices = plantedPrices(synth, 0.05);
    const urls: string[] = [];
    const get = async (url: string) => {
      urls.push(url);
      const bars: Record<string, Array<{ t: string; c: number }>> = {};
      for (const [t, s] of prices) bars[t] = s.dates.map((d, i) => ({ t: `${d}T05:00:00Z`, c: s.closes[i] }));
      return { bars, next_page_token: null };
    };
    const env = { ALPACA_KEY: "k", ALPACA_SECRET: "s" };
    const r1 = await runGithubGate2({ env, root, archiveBaseDir: archDir, get, dataBase: "https://data.example", now: () => SUN_IN, codeVersion: "1.0.1" });
    assert.equal(r1.status, "ok", r1.detail);
    assert.match(r1.detail, /verdict PASS/);
    assert.equal(urls.length, 1);
    assert.ok(urls[0].startsWith("https://data.example/v2/stocks/bars?"));
    assert.equal(new URL(urls[0]).searchParams.get("feed"), "delayed_sip", "no ALPACA_DATA_FEED => the resolver's default");
    assert.equal(new URL(urls[0]).searchParams.get("adjustment"), "all");
    const latest = JSON.parse(fs.readFileSync(path.join(root, "latest.json"), "utf8"));
    assert.equal(latest.verdict, "PASS");
    assert.equal(latest.first_sufficient_run.verdict, "PASS");
    assert.equal(latest.first_sufficient_run.code_version, "1.0.1");
    assert.equal(latest.per_week.length, latest.n_clusters + latest.per_week.filter((r: { spread5: number | null }) => r.spread5 == null).length);
    const text = fs.readFileSync(path.join(root, "latest.json"), "utf8");
    for (const px of ["\"closes\"", "\"dates\"", ":100,", "105.0"]) assert.ok(!text.includes(px), `no price series in the result file (${px})`);
    const lastRun = JSON.parse(fs.readFileSync(path.join(root, "last_run.json"), "utf8"));
    assert.equal(lastRun.status, "ok");
    assert.equal(lastRun.verdict, "PASS");

    // a second run with the opposite edge is informational: first_sufficient_run is carried forward
    const reversed = plantedPrices(synth, -0.05);
    const get2 = async () => {
      const bars: Record<string, Array<{ t: string; c: number }>> = {};
      for (const [t, s] of reversed) bars[t] = s.dates.map((d, i) => ({ t: `${d}T05:00:00Z`, c: s.closes[i] }));
      return { bars, next_page_token: null };
    };
    const urls2: string[] = [];
    const get2Logged = async (url: string) => { urls2.push(url); return get2(); };
    const r2 = await runGithubGate2({ env: { ...env, ALPACA_DATA_FEED: "sip" }, root, archiveBaseDir: archDir, get: get2Logged, now: () => SUN_IN + 7 * DAY, codeVersion: "1.0.2" });
    assert.match(r2.detail, /verdict FAIL/);
    assert.equal(new URL(urls2[0]).searchParams.get("feed"), "sip", "ALPACA_DATA_FEED overrides the feed, as everywhere else");
    const latest2 = JSON.parse(fs.readFileSync(path.join(root, "latest.json"), "utf8"));
    assert.equal(latest2.verdict, "FAIL");
    assert.equal(latest2.first_sufficient_run.verdict, "PASS", "the registered verdict is the FIRST sufficient run");
    assert.equal(latest2.first_sufficient_run.code_version, "1.0.1");

    // a fetch failure is an error outcome, not a throw, and leaves latest.json intact
    const boom = async () => { throw new Error("alpaca HTTP 500"); };
    const r3 = await runGithubGate2({ env, root, archiveBaseDir: archDir, get: boom, now: () => SUN_IN + 14 * DAY });
    assert.equal(r3.status, "error");
    assert.match(r3.detail, /HTTP 500/);
    assert.equal(JSON.parse(fs.readFileSync(path.join(root, "latest.json"), "utf8")).verdict, "FAIL");
    assert.equal(JSON.parse(fs.readFileSync(path.join(root, "last_run.json"), "utf8")).status, "error");

    // status: headline blocks only, survives sanitizeDiag, carries the registered date
    const st = githubGate2Status({ DATA_DIR: tmp }, root);
    assert.equal(st.registered_not_before, "2026-11-15");
    const l = st.latest as Record<string, unknown>;
    assert.equal(l.verdict, "FAIL");
    assert.equal((l.first_sufficient_run as { verdict: string }).verdict, "PASS");
    assert.ok(Array.isArray(l.per_week));
    assert.deepEqual(sanitizeDiag(st), st);
    const empty = githubGate2Status({ DATA_DIR: tmp }, path.join(tmp, "nothing"));
    assert.equal(empty.latest, null);
    assert.equal(empty.last_run, null);
  } finally {
    _resetGithubActivityForTests();
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("wiring: whitelisted probe, read-only case in bot.ts, booted from routes.ts, no subprocess/daemon/order path, never ticks on boot; the script re-exports the same rule", () => {
  assert.ok((DIAG_PROBES as readonly string[]).includes("github_gate2"));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const i = bot.indexOf('case "github_gate2": {');
  assert.ok(i > 0);
  const block = bot.slice(i, bot.indexOf("}", bot.indexOf("return res.json", i)) + 1);
  assert.ok(block.includes("sanitizeDiag(githubGate2Status())"));
  assert.ok(!/execPython|pythonRpc|beat\(|runGithubGate2/.test(block));
  assert.ok(fs.readFileSync(path.join(here, "routes.ts"), "utf8").includes("bootGithubGate2();"));
  const src = fs.readFileSync(path.join(here, "githubGate2.ts"), "utf8");
  const boot = src.slice(src.indexOf("export function bootGithubGate2"));
  assert.ok(!/^\s*(void\s+)?tick\(\);/m.test(boot));
  assert.ok(boot.includes("setInterval(") && boot.includes(".unref"));
  assert.ok(!/execPythonSerialized|pythonRunning|pythonRpc|execGroupKill|child_process/.test(src));
  assert.ok(!src.includes("/orders") && !/method:\s*"(POST|PUT|PATCH|DELETE)"/.test(src));
  assert.ok(src.includes('beat("github_gate2_weekly", "start")') && src.includes('beat("github_gate2_weekly", "finish"'));
  // the session script imports the rule from here rather than carrying a second copy
  const script = fs.readFileSync(path.join(here, "..", "scripts", "github_activity_gate2.ts"), "utf8");
  assert.ok(script.includes('from "../server/githubGate2"'));
  assert.ok(script.includes("evaluateGithubGate2("));
  assert.ok(!/^export function (computeWeeklyDeltas|bucketWeek|toWeeklyTotal)\b/m.test(script), "no second copy of the pre-registered rule");
  assert.equal(MIN_PANEL_FOR_WEEK, 12);
  assert.equal(MIN_CLUSTERS, 15);
});
