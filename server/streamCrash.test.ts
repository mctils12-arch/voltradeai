// [REPAIR 2026-11-05] KEEP THE SYSTEM ALIVE: a fetch body that dies mid-stream
// must fail ONE refresh/response, never the whole process.
//
// Live: two FATAL-EXCEPTION exits, 2026-11-04 05:03:27Z and 2026-11-05
// 01:17:05Z, each ~5-6 min after a boot, both
//   "TypeError: terminated @ at Fetch.onAborted (node:internal/deps/undici/...)"
// — undici errors a response body stream with TypeError("terminated") when
// the connection dies mid-body (socket reset, body timeout). Two server paths
// fed a fetch body into `.pipe()`, which never listens for errors on its
// SOURCE: dtccSwaps.streamDtccZip (the boot-eager 130-167 MB S3 zip, in flight
// at the 01:17 crash — its crash-guard marker kept DTCC warming_up 5 h later)
// and the /tiles-r2 proxy. An 'error' with no listener is an uncaught
// exception, and the FATAL handler exits so Railway restarts the container.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import http from "node:http";
import { fileURLToPath } from "node:url";
import { streamDtccZip, dtccUrl } from "./dtccSwaps";
import { pipeWebBody } from "./streamSafety";

const here = path.dirname(fileURLToPath(import.meta.url));

/** Records uncaught exceptions while `fn` runs (a listener keeps the test
 *  process alive so the assertion can report it instead of the runner dying). */
async function uncaughtDuring(fn: () => Promise<unknown>, settleMs = 100): Promise<Error[]> {
  const seen: Error[] = [];
  const onUncaught = (e: Error) => { seen.push(e); };
  process.on("uncaughtException", onUncaught);
  try {
    // bounded: on the old code the failing promise never settled (the process died instead)
    await Promise.race([fn().catch(() => { /* asserted by the caller */ }), new Promise((r) => setTimeout(r, 5_000))]);
    await new Promise((r) => setTimeout(r, settleMs));
  } finally {
    process.off("uncaughtException", onUncaught);
  }
  return seen;
}

function buildZip(csvText: string): Buffer {
  const nameBuf = Buffer.from("SEC_CUMULATIVE_EQUITIES_2026_11_04.csv", "utf8");
  const dataBuf = Buffer.from(csvText, "utf8");
  const compressed = zlib.deflateRawSync(dataBuf);
  const header = Buffer.alloc(30);
  header.writeUInt32LE(0x04034b50, 0);
  header.writeUInt16LE(20, 4);
  header.writeUInt16LE(0, 6);
  header.writeUInt16LE(8, 8);
  header.writeUInt32LE(compressed.length, 18);
  header.writeUInt32LE(dataBuf.length, 22);
  header.writeUInt16LE(nameBuf.length, 26);
  header.writeUInt16LE(0, 28);
  return Buffer.concat([header, nameBuf, compressed]);
}

/** A body that drips `bytes` in 8 KB chunks, then dies the way undici's
 *  Fetch.onAborted kills it: mid-member, after the reader is past the zip
 *  header (an error during the header read was always an ordinary rejection). */
function dyingBody(bytes: Buffer, onCancel?: () => void): ReadableStream<Uint8Array> {
  let off = 0;
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      await new Promise((r) => setTimeout(r, 2));
      if (off < bytes.length) {
        const end = Math.min(bytes.length, off + 8192);
        controller.enqueue(new Uint8Array(bytes.subarray(off, end)));
        off = end;
        return;
      }
      controller.error(new TypeError("terminated"));
    },
    cancel() { onCancel?.(); },
  });
}

function bigCsv(rows: number): string {
  const out = ["Dissemination Identifier,Underlier ID-Leg 1,Notional amount-Leg 1"];
  // pseudo-random text so deflate cannot shrink it to nothing
  let x = 12345;
  for (let i = 0; i < rows; i++) { x = (x * 1103515245 + 12345) & 0x7fffffff; out.push(`${i},US${x.toString(36)},${x % 100000}`); }
  return out.join("\n") + "\n";
}

test("DTCC zip: the connection dying mid-body rejects the refresh, never crashes the process", async () => {
  const zip = buildZip(bigCsv(20_000));
  const cut = Math.floor(zip.length / 2);       // well past the header: inflate is mid-member
  assert.ok(cut > 64 * 1024, "fixture is big enough to die mid-member");
  const fetchImpl = async () => ({ ok: true, status: 200, body: dyingBody(zip.subarray(0, cut)) });
  let rejection: unknown = null;
  const uncaught = await uncaughtDuring(async () => {
    try { await streamDtccZip(dtccUrl("2026_11_04"), () => {}, fetchImpl as any); }
    catch (e) { rejection = e; }
  });
  assert.deepEqual(uncaught.map((e) => String(e)), [], "no uncaught exception: the old .pipe() left the zip source with no 'error' listener");
  assert.ok(rejection instanceof Error && /terminated/.test(String(rejection)),
    `the refresh's own promise carries the network failure (got ${String(rejection)})`);
});

test("DTCC zip: a member read to its end releases the HTTP body (trailing central directory never left half-read)", async () => {
  const csv = bigCsv(50);
  const zip = Buffer.concat([buildZip(csv), Buffer.alloc(4096, 0x50)]); // central-directory stand-in
  let cancelled = false;
  let sent = false;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (!sent) { sent = true; controller.enqueue(new Uint8Array(zip)); return; }
      // a real S3 body keeps the socket open until it is read or cancelled
    },
    cancel() { cancelled = true; },
  });
  const r = await streamDtccZip(dtccUrl("2026_11_04"), () => {}, (async () => ({ ok: true, status: 200, body })) as any);
  assert.equal(r.ok, true);
  assert.equal(r.lines, 51);
  await new Promise((res) => setTimeout(res, 50));
  assert.equal(cancelled, true, "the response body is cancelled once the member is read, so undici frees the connection");
});

async function withServer(handler: http.RequestListener, fn: (port: number) => Promise<void>): Promise<void> {
  const server = http.createServer(handler);
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  try { await fn((server.address() as any).port); }
  finally { await new Promise<void>((r) => server.close(() => r())); }
}

test("proxy: an upstream body that dies mid-response ends that response, never the process", async () => {
  let done: Error | null | undefined;
  const uncaught = await uncaughtDuring(() => withServer((_req, res) => {
    res.writeHead(200, { "content-type": "application/octet-stream" });
    pipeWebBody(dyingBody(Buffer.alloc(64 * 1024, 7)), res, (err) => { done = err; });
  }, async (port) => {
    await new Promise<void>((resolve) => {
      http.get({ port, host: "127.0.0.1", path: "/" }, (r) => {
        r.on("data", () => {});
        r.on("error", () => resolve());   // the server cut the response short
        r.on("close", () => resolve());
      }).on("error", () => resolve());
    });
  }), 150);
  assert.deepEqual(uncaught.map((e) => String(e)), [], "no uncaught exception");
  assert.ok(done instanceof Error && /terminated/.test(String(done)), `onDone hears the upstream failure (got ${String(done)})`);
});

test("proxy: a client that leaves mid-response cancels the upstream body at once (no 300 s undici stall)", async () => {
  let cancelled = false;
  const upstream = new ReadableStream<Uint8Array>({
    pull(controller) { controller.enqueue(new Uint8Array(256 * 1024)); }, // endless, like a big range
    cancel() { cancelled = true; },
  });
  const uncaught = await uncaughtDuring(() => withServer((_req, res) => {
    res.writeHead(200, { "content-type": "application/octet-stream" });
    pipeWebBody(upstream, res);
  }, async (port) => {
    await new Promise<void>((resolve) => {
      const req = http.get({ port, host: "127.0.0.1", path: "/" }, (r) => {
        r.once("data", () => { req.destroy(); setTimeout(resolve, 100); }); // the visitor closes the tab
      });
      req.on("error", () => {});
    });
  }), 150);
  assert.deepEqual(uncaught.map((e) => String(e)), []);
  assert.equal(cancelled, true, "the upstream fetch body is cancelled when the client goes away");
});

test("ratchet: no server module pipes a fetch body with .pipe() (its source would have no error listener)", () => {
  const offenders: string[] = [];
  for (const f of fs.readdirSync(here)) {
    if (!f.endsWith(".ts") || f.endsWith(".test.ts")) continue;
    const src = fs.readFileSync(path.join(here, f), "utf8");
    if (/Readable\.fromWeb\([^)]*\)[^;\n]*\.pipe\(/.test(src)) offenders.push(f);
  }
  assert.deepEqual(offenders, [], "use pipeWebBody (server/streamSafety.ts) or stream.pipeline");
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  const i = routes.indexOf('app.get("/tiles-r2/:name"');
  assert.ok(i > 0);
  const handler = routes.slice(i, routes.indexOf("app.get(", i + 10));
  assert.ok(handler.includes("pipeWebBody(up.body"), "the tile proxy streams through pipeWebBody");
  assert.ok(!/\.pipe\(res\)/.test(handler), "and never .pipe(res)");
  const dtcc = fs.readFileSync(path.join(here, "dtccSwaps.ts"), "utf8");
  assert.ok(!/bounded\.pipe\(/.test(dtcc), "the DTCC zip member is piped with stream.pipeline, not .pipe()");
});
