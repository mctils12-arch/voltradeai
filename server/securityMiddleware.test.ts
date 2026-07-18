import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import express from "express";

import {
  securityHeaders,
  buildCsp,
  cspEnforceEnabled,
  CSP_SOURCES,
} from "./securityHeaders";
import {
  requestLog,
  hashIp,
  classifyUa,
  isPageView,
  recentRequests,
  analyticsSnapshot,
  _resetRequestLog,
} from "./requestLog";
import {
  globalRateLimit,
  createStrictAuthLimiter,
  authEventSummary,
  recentAuthEvents,
  _resetRateLimit,
} from "./rateLimit";
import {
  antiScraping,
  registerHoneypots,
  scoreFeatures,
  evaluate,
  antiScrapingSnapshot,
  HONEYPOT_PATHS,
  AS_CONFIG,
  _resetAntiScraping,
  _ipStoreSize,
} from "./antiScraping";
import { registerRobots, buildRobotsTxt } from "./robots";
import { registerAdminStats } from "./adminStats";
import { registerTerms, termsAsText } from "./terms";

const here = path.dirname(fileURLToPath(import.meta.url));

// Spin up a throwaway express server and return {port, close}.
async function serve(configure: (app: express.Express) => void) {
  const app = express();
  app.set("trust proxy", true);
  configure(app);
  const srv = app.listen(0);
  await new Promise((r) => srv.once("listening", r));
  const port = (srv.address() as any).port;
  return { port, close: () => new Promise((r) => srv.close(() => r(null))) };
}

// ── 1. SECURITY HEADERS ─────────────────────────────────────────────────────
test("security headers: present, report-only by default", async () => {
  const s = await serve((app) => {
    app.use(securityHeaders({} as any));
    app.get("/x", (_q, r) => r.send("ok"));
  });
  try {
    const res = await fetch(`http://127.0.0.1:${s.port}/x`);
    assert.equal(res.headers.get("x-content-type-options"), "nosniff");
    assert.equal(res.headers.get("x-frame-options"), "DENY");
    assert.equal(res.headers.get("referrer-policy"), "strict-origin-when-cross-origin");
    // Report-only by default (SECURITY_CSP_ENFORCE unset)
    assert.ok(res.headers.get("content-security-policy-report-only"), "expected report-only CSP");
    assert.equal(res.headers.get("content-security-policy"), null, "enforcing CSP must be absent by default");
  } finally {
    await s.close();
  }
});

test("security headers: enforce switch flips to enforcing CSP", async () => {
  assert.equal(cspEnforceEnabled({ SECURITY_CSP_ENFORCE: "true" } as any), true);
  assert.equal(cspEnforceEnabled({} as any), false);
  const s = await serve((app) => {
    app.use(securityHeaders({ SECURITY_CSP_ENFORCE: "1" } as any));
    app.get("/x", (_q, r) => r.send("ok"));
  });
  try {
    const res = await fetch(`http://127.0.0.1:${s.port}/x`);
    assert.ok(res.headers.get("content-security-policy"), "expected enforcing CSP");
    assert.equal(res.headers.get("content-security-policy-report-only"), null);
  } finally {
    await s.close();
  }
});

test("CSP allows the sources the map needs", () => {
  const csp = buildCsp();
  // MapLibre worker bundles load as blob:; topojson/d3 CDNs; fonts.
  assert.ok(csp.includes("worker-src 'self' blob:"));
  assert.ok(csp.includes("blob:"), "blob workers");
  assert.ok(csp.includes("https://cdn.jsdelivr.net"));
  assert.ok(csp.includes("https://cdnjs.cloudflare.com"));
  assert.ok(csp.includes("https://fonts.googleapis.com"));
  assert.ok(csp.includes("https://fonts.gstatic.com"));
  assert.ok(csp.includes("img-src") && CSP_SOURCES.img.includes("https:"), "broad img for tiles");
  assert.ok(csp.includes("connect-src") && CSP_SOURCES.connect.includes("https:"), "broad connect for data feeds");
  assert.ok(csp.includes("frame-ancestors 'none'"));
});

// ── 2. RATE LIMITING ────────────────────────────────────────────────────────
test("global rate limiter blocks after N with 429", async () => {
  _resetRateLimit();
  const tiny = { default: { windowMs: 60_000, max: 3 }, api: { windowMs: 60_000, max: 3 }, tiles: { windowMs: 60_000, max: 3 } };
  const s = await serve((app) => {
    app.use(globalRateLimit(() => 1000, tiny));
    app.get("/x", (_q, r) => r.send("ok"));
  });
  try {
    for (let i = 0; i < 3; i++) {
      const ok = await fetch(`http://127.0.0.1:${s.port}/x`);
      assert.equal(ok.status, 200, `req ${i} should pass`);
    }
    const blocked = await fetch(`http://127.0.0.1:${s.port}/x`);
    assert.equal(blocked.status, 429, "4th request over limit blocked");
    assert.ok(blocked.headers.get("retry-after"));
  } finally {
    await s.close();
  }
});

test("strict auth limiter: blocks after N, generic message, incremental backoff", async () => {
  _resetRateLimit();
  let now = 0;
  const strict = createStrictAuthLimiter(
    { windowMs: 60_000, maxAttempts: 2, baseLockMs: 1000, maxLockMs: 100_000 },
    () => now,
  );
  const s = await serve((app) => {
    app.use("/api/auth/login", strict.middleware);
    app.post("/api/auth/login", (_q, r) => r.status(401).json({ error: "Invalid credentials" }));
  });
  const login = () => fetch(`http://127.0.0.1:${s.port}/api/auth/login`, { method: "POST" });
  try {
    // 2 allowed attempts (real auth failure passes through).
    let r1 = await login();
    assert.equal(r1.status, 401);
    let r2 = await login();
    assert.equal(r2.status, 401);
    // 3rd trips lockout — generic message, NOT "Invalid credentials".
    let r3 = await login();
    assert.equal(r3.status, 429);
    const b3 = await r3.json();
    assert.equal(b3.error, "Too many attempts. Please try again later.");
    assert.ok(!JSON.stringify(b3).includes("Invalid credentials"), "must not leak credential result");
    const retry1 = Number(r3.headers.get("retry-after"));
    // Advance clock past the first lockout, trip again → longer backoff.
    now = 2000;
    let r4 = await login();
    assert.equal(r4.status, 429);
    const retry2 = Number(r4.headers.get("retry-after"));
    assert.ok(retry2 > retry1, `backoff should grow: ${retry1} -> ${retry2}`);
    // auth events recorded (lockout/ratelimited + failures).
    const sum = authEventSummary();
    assert.ok(sum.total >= 4);
    assert.ok(sum.ratelimited + sum.lockout >= 1, "lockout event recorded");
  } finally {
    await s.close();
  }
});

test("strict auth limiter records success and clears counters", async () => {
  _resetRateLimit();
  const strict = createStrictAuthLimiter({ windowMs: 60_000, maxAttempts: 2, baseLockMs: 1000, maxLockMs: 1000 }, () => 0);
  const s = await serve((app) => {
    app.use("/api/auth/login", strict.middleware);
    app.post("/api/auth/login", (_q, r) => r.status(200).json({ ok: true }));
  });
  try {
    for (let i = 0; i < 5; i++) {
      const r = await fetch(`http://127.0.0.1:${s.port}/api/auth/login`, { method: "POST" });
      assert.equal(r.status, 200, "success clears the counter so no lockout");
    }
    const events = recentAuthEvents();
    assert.ok(events.some((e) => e.kind === "success"));
    // IPs in events are hashed (no dotted-quad).
    assert.ok(events.every((e) => !/\d+\.\d+\.\d+\.\d+/.test(e.ipHash)));
  } finally {
    await s.close();
  }
});

// ── 3. ANTI-SCRAPING ────────────────────────────────────────────────────────
test("scorer flags velocity, headless UA, and honeypot", () => {
  // velocity
  assert.ok(scoreFeatures({ uaClass: "browser", missingUa: false, missingAccept: false, velocity: AS_CONFIG.velocityHard, breadth: 0, honeypot: false }) > 0);
  // headless UA
  assert.ok(scoreFeatures({ uaClass: "headless", missingUa: false, missingAccept: false, velocity: 0, breadth: 0, honeypot: false }) >= 60);
  // honeypot dominates
  assert.equal(scoreFeatures({ uaClass: "browser", missingUa: false, missingAccept: false, velocity: 0, breadth: 0, honeypot: true }), AS_CONFIG.honeypotScore);
  // a normal browser page-load-shaped request scores 0
  assert.equal(scoreFeatures({ uaClass: "browser", missingUa: false, missingAccept: false, velocity: 30, breadth: 20, honeypot: false }), 0);
});

test("anti-scraping walks log -> throttle -> block ladder on BEHAVIORAL evidence (catalogue sweep)", () => {
  _resetAntiScraping();
  // A real scraper signature: a cli client sweeping many distinct /api
  // endpoints at high velocity in one window. Behavioral overage accumulates
  // → allow first, then throttle, then block with the offender persisted.
  const verdicts: string[] = [];
  for (let i = 0; i < 400; i++) {
    const r = evaluate(
      { path: `/api/endpoint-${i}`, headers: { "user-agent": "curl/8.0", accept: "*/*" }, ip: "9.9.9.9" },
      1000 + i, // 400 hits in ~0.4s — far over velocitySoft within one window
    );
    verdicts.push(r.verdict);
    if (r.verdict === "block") break;
  }
  assert.equal(verdicts[0], "allow", "first request only logs");
  assert.ok(verdicts.includes("throttle"), "sustained sweep gets throttled before blocking");
  assert.equal(verdicts[verdicts.length - 1], "block", "sustained sweep ends blocked");
  const snap = antiScrapingSnapshot(2000);
  assert.ok(snap.activeOffenders >= 1, "offender persisted with TTL");
});

// ── Parent-review regressions (2026-07-18): static client signals must not
// compound. The original scorer accumulated the +60 headless / +20 cli UA
// score PER REQUEST, blocking any headless browser after ~3 API calls (our
// own Playwright harness) and any curl/script API customer after ~10 — the
// exact legitimate clients of the /api/v1 product. ──
test("REGRESSION: headless browser at human map-load rate is never throttled", () => {
  _resetAntiScraping();
  let now = 10_000;
  for (let i = 0; i < 80; i++) {
    const r = evaluate(
      { path: `/api/data/layer-${i % 50}`, headers: { "user-agent": "Mozilla/5.0 HeadlessChrome/120", accept: "application/json" }, ip: "7.7.7.7" },
      now,
    );
    assert.equal(r.verdict, "allow", `headless at human rate must stay allowed (req ${i}, score ${r.score})`);
    now += 250; // 4 req/s — a busy multi-layer /data hydration
  }
});

test("REGRESSION: cli API customer at modest sustained rate stays allowed", () => {
  _resetAntiScraping();
  let now = 50_000;
  for (let i = 0; i < 120; i++) {
    const r = evaluate(
      { path: "/api/v1/signals", headers: { "user-agent": "python-requests/2.32", accept: "application/json" }, ip: "6.6.6.6" },
      now,
    );
    assert.equal(r.verdict, "allow", `scripted API customer must stay allowed (req ${i}, score ${r.score})`);
    now += 1000; // 1 req/s, all day long — a paying customer's poller
  }
});

test("ip store hygiene: idle records are swept; blocked offenders survive the sweep", () => {
  _resetAntiScraping();
  evaluate({ path: "/api/a", headers: { "user-agent": "Mozilla/5.0", accept: "*/*" }, ip: "1.1.1.1" }, 1000);
  evaluate({ path: HONEYPOT_PATHS[0], headers: { "user-agent": "curl/8.0" }, ip: "2.2.2.2" }, 1000); // blocked offender
  assert.equal(_ipStoreSize(), 2);
  // 200s later another IP arrives → sweep runs: idle 1.1.1.1 dropped, the
  // still-blocked offender kept.
  evaluate({ path: "/api/b", headers: { "user-agent": "Mozilla/5.0", accept: "*/*" }, ip: "3.3.3.3" }, 201_000);
  assert.equal(_ipStoreSize(), 2, "idle record swept; offender + newcomer remain");
});

test("honeypot hit blocks immediately and persists offender", async () => {
  _resetAntiScraping();
  const s = await serve((app) => {
    app.use(antiScraping(() => 5000));
    registerHoneypots(app, () => 5000);
    app.get("/api/data", (_q, r) => r.json({ ok: true }));
  });
  try {
    const good = await fetch(`http://127.0.0.1:${s.port}/api/data`, { headers: { accept: "application/json", "user-agent": "Mozilla/5.0" } });
    assert.equal(good.status, 200, "normal API request passes");
    const trap = await fetch(`http://127.0.0.1:${s.port}${HONEYPOT_PATHS[0]}`);
    assert.equal(trap.status, 403, "honeypot hit blocked");
    const snap = antiScrapingSnapshot(5000);
    assert.ok(snap.totalHoneypotHits >= 1);
    assert.ok(snap.activeOffenders >= 1);
  } finally {
    await s.close();
  }
});

test("anti-scraping ignores tiles/static (map-heavy traffic is exempt)", async () => {
  _resetAntiScraping();
  let n = 0;
  const s = await serve((app) => {
    app.use(antiScraping(() => 1000));
    app.get("/tiles/:z", (_q, r) => r.send("tile"));
  });
  try {
    for (let i = 0; i < 50; i++) {
      const r = await fetch(`http://127.0.0.1:${s.port}/tiles/${i}`, { headers: { "user-agent": "HeadlessChrome/120" } });
      assert.equal(r.status, 200, "tile requests never scored, even headless");
    }
  } finally {
    await s.close();
  }
});

// ── 4. ROBOTS.TXT ───────────────────────────────────────────────────────────
test("robots.txt disallows /api and honeypots", async () => {
  const txt = buildRobotsTxt();
  assert.ok(txt.includes("Disallow: /api/"));
  for (const h of HONEYPOT_PATHS) assert.ok(txt.includes(`Disallow: ${h}`), `${h} disallowed`);
  const s = await serve((app) => registerRobots(app));
  try {
    const res = await fetch(`http://127.0.0.1:${s.port}/robots.txt`);
    assert.equal(res.status, 200);
    assert.ok((res.headers.get("content-type") || "").includes("text/plain"));
    const body = await res.text();
    assert.ok(body.includes("Disallow: /api/"));
  } finally {
    await s.close();
  }
});

// ── 5. REQUEST LOG + ANALYTICS ──────────────────────────────────────────────
test("request log: structured records, IP hashed (no raw IP stored)", async () => {
  _resetRequestLog();
  assert.notEqual(hashIp("1.2.3.4"), "1.2.3.4");
  assert.equal(hashIp("1.2.3.4"), hashIp("1.2.3.4")); // deterministic in-process
  assert.equal(classifyUa("HeadlessChrome/120"), "headless");
  assert.equal(classifyUa("curl/8.0"), "cli");
  assert.equal(classifyUa(""), "empty");
  assert.equal(isPageView("GET", "/data"), true);
  assert.equal(isPageView("GET", "/api/x"), false);
  assert.equal(isPageView("GET", "/foo.png"), false);
  const s = await serve((app) => {
    app.use(requestLog());
    app.get("/data", (_q, r) => r.send("page"));
    app.get("/api/x", (_q, r) => r.json({ ok: 1 }));
  });
  try {
    await fetch(`http://127.0.0.1:${s.port}/data`);
    await fetch(`http://127.0.0.1:${s.port}/api/x`);
    await new Promise((r) => setTimeout(r, 30)); // let finish handlers run
    const recs = recentRequests();
    assert.ok(recs.length >= 2);
    for (const rec of recs) {
      assert.equal(typeof rec.latencyMs, "number");
      assert.ok(typeof rec.route === "string");
      // No raw dotted-quad / IPv6 loopback leaked into the record.
      assert.ok(!/\d+\.\d+\.\d+\.\d+/.test(rec.ipHash) && rec.ipHash !== "::1" && rec.ipHash !== "::ffff:127.0.0.1");
      assert.match(rec.ipHash, /^[a-f0-9]{1,16}$|^anon$/);
    }
    const snap = analyticsSnapshot();
    assert.ok(snap.totalPageViews >= 1, "the /data page view counted");
    assert.ok(snap.topRoutes.some((r) => r.route === "/data"));
  } finally {
    await s.close();
  }
});

// ── 6. ADMIN STATS ──────────────────────────────────────────────────────────
test("admin stats: 404 without token env, 401 wrong token, 200 with token", async () => {
  const s = await serve((app) => registerAdminStats(app));
  try {
    // No ADMIN_STATS_TOKEN → 404 (invisible).
    delete process.env.ADMIN_STATS_TOKEN;
    let r = await fetch(`http://127.0.0.1:${s.port}/api/_admin/security-stats`);
    assert.equal(r.status, 404);
    // Token set but wrong → 401.
    process.env.ADMIN_STATS_TOKEN = "sekret";
    r = await fetch(`http://127.0.0.1:${s.port}/api/_admin/security-stats`, { headers: { "x-admin-token": "nope" } });
    assert.equal(r.status, 401);
    // Correct token → 200 with snapshot.
    r = await fetch(`http://127.0.0.1:${s.port}/api/_admin/security-stats`, { headers: { "x-admin-token": "sekret" } });
    assert.equal(r.status, 200);
    const body = await r.json();
    assert.ok(body.antiScraping && body.authEvents);
    // analytics endpoint too
    r = await fetch(`http://127.0.0.1:${s.port}/api/_admin/analytics?token=sekret`);
    assert.equal(r.status, 200);
    const a = await r.json();
    assert.ok(a.analytics);
  } finally {
    delete process.env.ADMIN_STATS_TOKEN;
    await s.close();
  }
});

// ── 7. TERMS OF SERVICE ─────────────────────────────────────────────────────
test("terms content prohibits scraping and credits upstream sources", async () => {
  const txt = termsAsText().toLowerCase();
  assert.ok(txt.includes("scrap"), "prohibits scraping");
  assert.ok(txt.includes("harvest") || txt.includes("bulk"), "prohibits bulk harvesting");
  assert.ok(txt.includes("as is"), "data as is");
  assert.ok(txt.includes("openstreetmap") && txt.includes("odbl"), "OSM/ODbL attribution");
  assert.ok(txt.includes("noaa") && txt.includes("adsb.lol"), "upstream attribution");
  const s = await serve((app) => registerTerms(app));
  try {
    const j = await (await fetch(`http://127.0.0.1:${s.port}/api/legal/terms`)).json();
    assert.ok(Array.isArray(j.sections) && j.sections.length > 0);
    const html = await (await fetch(`http://127.0.0.1:${s.port}/terms`)).text();
    assert.ok(html.includes("Terms of Service"));
  } finally {
    await s.close();
  }
});

// ── 8. MOUNT-ORDER PIN (security before routes, like compression.test.ts) ────
test("index.ts mounts security middleware before routes register", () => {
  const src = fs.readFileSync(path.join(here, "index.ts"), "utf8");
  const routesAt = src.indexOf("registerRoutes(");
  assert.ok(routesAt > -1);
  for (const marker of [
    "app.use(securityHeaders())",
    "app.use(requestLog())",
    "app.use(globalRateLimit())",
    "app.use(antiScraping())",
    "registerRobots(app)",
    "registerAdminStats(app)",
    "registerTerms(app)",
  ]) {
    const at = src.indexOf(marker);
    assert.ok(at > -1, `${marker} must be mounted`);
    assert.ok(at < routesAt, `${marker} must mount before registerRoutes`);
  }
  // strict auth limiter is mounted on the real auth paths.
  assert.ok(src.includes('app.use("/api/auth/login"'), "strict limiter on login path");
  assert.ok(src.includes('app.use("/api/auth/register"'), "strict limiter on register path");
});
