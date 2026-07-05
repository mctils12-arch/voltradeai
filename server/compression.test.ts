import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import express from "express";
import compression from "compression";

const here = path.dirname(fileURLToPath(import.meta.url));

// [REPAIR 2026-07-05] /data load perf: Railway's edge does not gzip —
// without app-level compression an aircraft snapshot ships ~0.8MB raw.
// Pin (1) the wiring in index.ts and (2) that compression actually
// gzips a JSON payload end-to-end through a real express server.

test("index.ts mounts compression() before routes register", () => {
  const src = fs.readFileSync(path.join(here, "index.ts"), "utf8");
  const at = src.indexOf("app.use(compression())");
  assert.ok(at > -1, "compression middleware not mounted");
  assert.ok(at < src.indexOf("registerRoutes("), "compression must mount before routes");
  const pkg = JSON.parse(fs.readFileSync(path.join(here, "..", "package.json"), "utf8"));
  assert.ok(pkg.dependencies.compression, "compression must be a runtime dependency");
});

test("compression gzips JSON responses end-to-end (and honors accept-encoding)", async () => {
  const app = express();
  app.use(compression());
  const big = { rows: Array.from({ length: 2000 }, (_, i) => ({ i, v: "x".repeat(50) })) };
  app.get("/big.json", (_req, res) => res.json(big));
  const srv = app.listen(0);
  try {
    const port = (srv.address() as any).port;
    const r = await fetch(`http://127.0.0.1:${port}/big.json`, { headers: { "Accept-Encoding": "gzip" } });
    assert.equal(r.headers.get("content-encoding"), "gzip");
    const body = await r.json(); // undici transparently decompresses
    assert.equal(body.rows.length, 2000);
  } finally {
    srv.close();
  }
});
