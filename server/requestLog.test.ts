/**
 * Route attribution in requestLog.ts under the production mount order.
 *
 * Regression (2026-10-14): static.ts serves the SPA with
 * `app.use("/{*path}", ...)`. A use()-mounted handler has no req.route, and
 * Express trims the matched mount path out of req.url before calling it, so
 * req.path read "/" for every SPA page. Every page load landed under "/" in
 * the page-view counters and the persisted visit ledger, and asset misses
 * (/favicon.ico) and unknown /api paths that fell through to the SPA counted
 * as page views of "/".
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import express from "express";
import { requestLog, recentRequests, analyticsSnapshot, _resetRequestLog } from "./requestLog";

async function withServer(app: express.Express, fn: (base: string) => Promise<void>) {
  const srv = http.createServer(app);
  await new Promise<void>((r) => srv.listen(0, "127.0.0.1", () => r()));
  const port = (srv.address() as { port: number }).port;
  try {
    await fn(`http://127.0.0.1:${port}`);
  } finally {
    await new Promise((r) => srv.close(r));
  }
}

const settle = () => new Promise((r) => setTimeout(r, 30)); // let finish handlers run

test("SPA fallback mounted with app.use records the real path, not '/'", async () => {
  _resetRequestLog();
  const app = express();
  app.use(requestLog());
  app.get("/api/ok", (_q, r) => r.json({ ok: 1 }));
  app.use("/{*path}", (_q, r) => { r.send("<html></html>"); }); // as in static.ts
  await withServer(app, async (base) => {
    await fetch(`${base}/developers?utm_source=x`);
    await fetch(`${base}/data`);
    await fetch(`${base}/favicon.ico`);
    await fetch(`${base}/api/nope`);
    await fetch(`${base}/api/ok`);
    await settle();
  });
  assert.deepEqual(
    recentRequests().map((x) => x.route),
    ["/developers", "/data", "/favicon.ico", "/api/nope", "/api/ok"],
  );
  const snap = analyticsSnapshot();
  assert.equal(snap.totalPageViews, 2, "only /developers and /data are page views");
  assert.deepEqual(
    snap.topRoutes.map((r) => r.route).sort(),
    ["/data", "/developers"],
  );
});

test("matched routes keep their pattern (no path-param cardinality); long paths are capped", async () => {
  _resetRequestLog();
  const app = express();
  app.use(requestLog());
  const router = express.Router();
  router.get("/stock/:ticker", (_q, r) => { r.send("page"); });
  app.use("/app", router);
  app.use("/{*path}", (_q, r) => { r.send("<html></html>"); });
  await withServer(app, async (base) => {
    await fetch(`${base}/app/stock/AAPL`);
    await fetch(`${base}/${"a".repeat(500)}`);
    await settle();
  });
  const routes = recentRequests().map((x) => x.route);
  assert.equal(routes[0], "/app/stock/:ticker");
  assert.ok(routes[1].length <= 120, `route capped, got ${routes[1].length} chars`);
  assert.ok(routes[1].startsWith("/aaaa"));
});
