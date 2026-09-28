import { test } from "node:test";
import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  signV4, sha256Hex, EMPTY_SHA256, encodeS3Path, encodeRfc3986, canonicalQueryString, amzDateOf,
  r2ConfigFromEnv, createR2Client, parseListObjectsV2, deleteObjectsXml, R2_NOT_CONFIGURED,
  type R2Config,
} from "./r2Client";

// ── SigV4: AWS published test vectors ────────────────────────────────────────
// aws-sig-v4-test-suite "get-vanilla" (service/us-east-1, AKIDEXAMPLE) and the
// four worked S3 examples from the AWS "Signature Calculations for the
// Authorization Header" S3 docs (examplebucket, AKIAIOSFODNN7EXAMPLE).
const S3_SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";

test("SigV4 get-vanilla (aws-sig-v4-test-suite) reproduces the published canonical hash + signature", () => {
  const out = signV4({
    method: "GET", host: "example.amazonaws.com", canonicalPath: "/",
    headers: { "X-Amz-Date": "20150830T123600Z" }, payloadHash: EMPTY_SHA256,
    accessKeyId: "AKIDEXAMPLE", secretAccessKey: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
    region: "us-east-1", service: "service", amzDate: "20150830T123600Z",
  });
  assert.equal(out.canonicalRequest,
    "GET\n/\n\nhost:example.amazonaws.com\nx-amz-date:20150830T123600Z\n\nhost;x-amz-date\n" + EMPTY_SHA256);
  assert.equal(sha256Hex(out.canonicalRequest), "bb579772317eb040ac9ed261061d46c1f17a8133879d6129b6e1c25292927e63");
  assert.equal(out.stringToSign,
    "AWS4-HMAC-SHA256\n20150830T123600Z\n20150830/us-east-1/service/aws4_request\nbb579772317eb040ac9ed261061d46c1f17a8133879d6129b6e1c25292927e63");
  assert.equal(out.signature, "5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31");
  assert.equal(out.authorization,
    "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, SignedHeaders=host;x-amz-date, Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31");
});

test("SigV4 S3 GET Object example (Range header) — published signature", () => {
  const out = signV4({
    method: "GET", host: "examplebucket.s3.amazonaws.com", canonicalPath: encodeS3Path("/test.txt"),
    headers: { range: "bytes=0-9", "x-amz-content-sha256": EMPTY_SHA256, "x-amz-date": "20130524T000000Z" },
    payloadHash: EMPTY_SHA256, accessKeyId: "AKIAIOSFODNN7EXAMPLE", secretAccessKey: S3_SECRET,
    region: "us-east-1", service: "s3", amzDate: "20130524T000000Z",
  });
  assert.equal(out.signedHeaders, "host;range;x-amz-content-sha256;x-amz-date");
  assert.equal(out.signature, "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41");
});

test("SigV4 S3 PUT Object example — '$' in the key is encoded once, published signature", () => {
  const ph = sha256Hex("Welcome to Amazon S3.");
  assert.equal(ph, "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072");
  const out = signV4({
    method: "PUT", host: "examplebucket.s3.amazonaws.com", canonicalPath: encodeS3Path("/test$file.text"),
    headers: {
      date: "Fri, 24 May 2013 00:00:00 GMT", "x-amz-content-sha256": ph,
      "x-amz-date": "20130524T000000Z", "x-amz-storage-class": "REDUCED_REDUNDANCY",
    },
    payloadHash: ph, accessKeyId: "AKIAIOSFODNN7EXAMPLE", secretAccessKey: S3_SECRET,
    region: "us-east-1", service: "s3", amzDate: "20130524T000000Z",
  });
  assert.match(out.canonicalRequest, /^PUT\n\/test%24file\.text\n/);
  assert.equal(out.signature, "98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd");
});

test("SigV4 S3 GET Bucket (list, query params sorted) + GET lifecycle (empty-valued subresource) — published signatures", () => {
  const common = {
    method: "GET", host: "examplebucket.s3.amazonaws.com", canonicalPath: "/",
    headers: { "x-amz-content-sha256": EMPTY_SHA256, "x-amz-date": "20130524T000000Z" },
    payloadHash: EMPTY_SHA256, accessKeyId: "AKIAIOSFODNN7EXAMPLE", secretAccessKey: S3_SECRET,
    region: "us-east-1", service: "s3", amzDate: "20130524T000000Z",
  };
  // deliberately out of order: the canonical query must sort them
  const list = signV4({ ...common, query: [["prefix", "J"], ["max-keys", "2"]] });
  assert.match(list.canonicalRequest, /\nmax-keys=2&prefix=J\n/);
  assert.equal(list.signature, "34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7");
  const lc = signV4({ ...common, query: [["lifecycle", ""]] });
  assert.match(lc.canonicalRequest, /\nlifecycle=\n/);
  assert.equal(lc.signature, "fea454ca298b7da1c68078a5d1bdbfbbe0d65c699e0f91ac7a200a0136783543");
});

test("encoding helpers: RFC3986 unreserved set, '/' kept in paths, amz date format", () => {
  assert.equal(encodeRfc3986("a b!'()*~._-"), "a%20b%21%27%28%29%2A~._-");
  assert.equal(encodeS3Path("/bucket/archive/aircraft/2026-09-20/13.jsonl.gz"), "/bucket/archive/aircraft/2026-09-20/13.jsonl.gz");
  assert.equal(encodeS3Path("/b/a key+x"), "/b/a%20key%2Bx");
  assert.equal(canonicalQueryString([["b", "2"], ["a", "x y"], ["a", "1"]]), "a=1&a=x%20y&b=2");
  assert.equal(amzDateOf(new Date("2026-09-28T12:34:56.789Z")), "20260928T123456Z");
});

// ── configuration gating ─────────────────────────────────────────────────────
const FULL_ENV = {
  R2_ACCOUNT_ID: "0123456789abcdef0123456789abcdef",
  R2_ACCESS_KEY_ID: "AKID", R2_SECRET_ACCESS_KEY: "SECRET", R2_ARCHIVE_BUCKET: "voltrade-archive",
};

test("config: active only when all four env vars exist; shape-checked account id + bucket", () => {
  assert.ok(r2ConfigFromEnv(FULL_ENV as any));
  assert.equal(r2ConfigFromEnv(FULL_ENV as any)!.host, "0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com");
  for (const k of Object.keys(FULL_ENV)) {
    const env: any = { ...FULL_ENV };
    delete env[k];
    assert.equal(r2ConfigFromEnv(env), null, `missing ${k} => not configured`);
  }
  // the public tiles env alone never activates the archive client
  assert.equal(r2ConfigFromEnv({ R2_ACCESS_KEY_ID: "a", R2_SECRET_ACCESS_KEY: "b", R2_PUBLIC_URL: "https://pub-x.r2.dev" } as any), null);
  assert.equal(r2ConfigFromEnv({ ...FULL_ENV, R2_ACCOUNT_ID: "evil.com/x" } as any), null, "host injection rejected");
  assert.equal(r2ConfigFromEnv({ ...FULL_ENV, R2_ARCHIVE_BUCKET: "Bad_Bucket" } as any), null);
});

test("not configured: every call is a clean no-op that never touches the network", async () => {
  let calls = 0;
  const c = createR2Client(null, { fetchImpl: (async () => { calls++; throw new Error("no"); }) as any });
  assert.equal(c.configured, false);
  const put = await c.putObject("k", "x");
  assert.equal(put.ok, false);
  assert.equal(put.notConfigured, true);
  assert.equal(put.error, R2_NOT_CONFIGURED);
  assert.equal((await c.headObject("k")).exists, false);
  assert.equal((await c.getObject("k")).notConfigured, true);
  assert.equal((await c.listObjects("p")).objects.length, 0);
  assert.equal((await c.deleteObjects(["a"])).notConfigured, true);
  assert.equal((await c.getObjectToFile("k", "/nonexistent/x")).notConfigured, true);
  assert.equal(calls, 0);
});

// ── request construction with a fake fetch ───────────────────────────────────
interface Captured { url: string; method: string; headers: Record<string, string>; body?: Buffer }

function fakeFetch(responder: (req: Captured) => Response | Promise<Response>, log: Captured[]) {
  return (async (url: any, init: any = {}) => {
    const headers: Record<string, string> = {};
    for (const [k, v] of Object.entries(init.headers || {})) headers[k.toLowerCase()] = String(v);
    const req: Captured = { url: String(url), method: init.method || "GET", headers, body: init.body ? Buffer.from(init.body) : undefined };
    log.push(req);
    if (init.signal?.aborted) throw new Error("aborted");
    return responder(req);
  }) as typeof fetch;
}

const CFG: R2Config = r2ConfigFromEnv(FULL_ENV as any)!;
const FIXED_NOW = () => new Date("2026-09-28T12:00:00Z");

/** re-derive the signature the server would compute from what was sent */
function verifySigned(req: Captured, payload: Buffer | undefined) {
  const u = new URL(req.url);
  const query: Array<[string, string]> = [];
  u.searchParams.forEach((v, k) => query.push([k, v]));
  const signedNames = /SignedHeaders=([^,]+)/.exec(req.headers.authorization)![1].split(";");
  const headers: Record<string, string> = {};
  for (const n of signedNames) if (n !== "host") headers[n] = req.headers[n];
  const expect = signV4({
    method: req.method, host: u.host, canonicalPath: u.pathname, query, headers,
    payloadHash: req.headers["x-amz-content-sha256"], accessKeyId: CFG.accessKeyId,
    secretAccessKey: CFG.secretAccessKey, region: "auto", service: "s3", amzDate: req.headers["x-amz-date"],
  });
  assert.equal(req.headers.authorization, expect.authorization, "server-side re-derivation matches");
  if (payload) assert.equal(req.headers["x-amz-content-sha256"], sha256Hex(payload));
}

test("putObject: path-style URL on the account endpoint, region auto / service s3, signed payload hash", async () => {
  const log: Captured[] = [];
  const c = createR2Client(CFG, { fetchImpl: fakeFetch(() => new Response(null, { status: 200, headers: { etag: '"abc"' } }), log), now: FIXED_NOW });
  const body = Buffer.from("hello archive");
  const r = await c.putObject("archive/aircraft/2026-09-20/13.jsonl.gz", body, "application/gzip");
  assert.equal(r.ok, true);
  assert.equal(r.etag, '"abc"');
  assert.equal(log.length, 1);
  const req = log[0];
  assert.equal(req.method, "PUT");
  assert.equal(req.url, "https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com/voltrade-archive/archive/aircraft/2026-09-20/13.jsonl.gz");
  assert.equal(req.headers["content-type"], "application/gzip");
  assert.equal(req.headers["content-encoding"], undefined, "never Content-Encoding (fetch would auto-gunzip on GET)");
  assert.equal(req.headers["x-amz-date"], "20260928T120000Z");
  assert.match(req.headers.authorization, /^AWS4-HMAC-SHA256 Credential=AKID\/20260928\/auto\/s3\/aws4_request, SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date, Signature=[0-9a-f]{64}$/);
  assert.deepEqual(req.body, body);
  verifySigned(req, body);
});

test("retries: 503 then success retries with backoff and a fresh signature; 403 is never retried", async () => {
  const log: Captured[] = [];
  const sleeps: number[] = [];
  let n = 0;
  const c = createR2Client(CFG, {
    fetchImpl: fakeFetch(() => (++n === 1 ? new Response("busy", { status: 503 }) : new Response(null, { status: 200 })), log),
    sleep: async (ms) => { sleeps.push(ms); }, now: FIXED_NOW,
  });
  const r = await c.putObject("k", "x");
  assert.equal(r.ok, true);
  assert.equal(r.attempts, 2);
  assert.equal(sleeps.length, 1);
  assert.ok(sleeps[0] >= 400 && sleeps[0] < 500);

  const log2: Captured[] = [];
  const c2 = createR2Client(CFG, {
    fetchImpl: fakeFetch(() => new Response("<Error><Code>AccessDenied</Code><Message>nope</Message></Error>", { status: 403 }), log2),
    sleep: async () => {},
  });
  const r2 = await c2.putObject("k", "x");
  assert.equal(r2.ok, false);
  assert.equal(r2.status, 403);
  assert.match(r2.error || "", /AccessDenied: nope/);
  assert.equal(log2.length, 1, "4xx is not retried");
});

test("retries are limited: persistent network errors give up after maxRetries+1 attempts", async () => {
  let calls = 0;
  const c = createR2Client(CFG, {
    fetchImpl: (async () => { calls++; throw new Error("ECONNRESET"); }) as any,
    sleep: async () => {}, maxRetries: 2,
  });
  const r = await c.headObject("k");
  assert.equal(r.ok, false);
  assert.equal(calls, 3);
  assert.match(r.error || "", /ECONNRESET/);
});

test("timeout: a hung request is aborted by the per-attempt timeout", async () => {
  const c = createR2Client(CFG, {
    fetchImpl: ((_u: any, init: any) => new Promise((_res, rej) => {
      init.signal.addEventListener("abort", () => rej(new Error("The operation was aborted")));
    })) as any,
    sleep: async () => {}, maxRetries: 0, timeoutMs: 30,
  });
  const r = await c.getObject("k");
  assert.equal(r.ok, false);
  assert.match(r.error || "", /timeout after 30ms/);
});

test("headObject: 404 = ok/exists:false; 200 reports size from content-length", async () => {
  const log: Captured[] = [];
  const c = createR2Client(CFG, {
    fetchImpl: fakeFetch((req) => (req.url.endsWith("/missing")
      ? new Response(null, { status: 404 })
      : new Response(null, { status: 200, headers: { "content-length": "1234", etag: '"e"' } })), log),
  });
  const miss = await c.headObject("missing");
  assert.deepEqual([miss.ok, miss.exists], [true, false]);
  const hit = await c.headObject("present");
  assert.deepEqual([hit.ok, hit.exists, hit.size], [true, true, 1234]);
  assert.equal(log[1].method, "HEAD");
  verifySigned(log[1], undefined);
});

test("getObject is bounded by maxBytes; getObjectToFile writes atomically and cleans up on overflow", async () => {
  const payload = crypto.randomBytes(5000);
  const c = createR2Client(CFG, { fetchImpl: fakeFetch(() => new Response(payload, { status: 200 }), []) });
  const ok = await c.getObject("k", { maxBytes: 10_000 });
  assert.equal(ok.ok, true);
  assert.deepEqual(ok.body, payload);
  const tooBig = await c.getObject("k", { maxBytes: 1000 });
  assert.equal(tooBig.ok, false);
  assert.match(tooBig.error || "", /exceeded 1000 bytes/);

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-r2get-"));
  const dest = path.join(dir, "sub", "hour.jsonl.gz");
  const f = await c.getObjectToFile("k", dest, { maxBytes: 10_000 });
  assert.equal(f.ok, true);
  assert.equal(f.bytes, 5000);
  assert.deepEqual(fs.readFileSync(dest), payload);
  const dest2 = path.join(dir, "sub", "big.jsonl.gz");
  const f2 = await c.getObjectToFile("k", dest2, { maxBytes: 100 });
  assert.equal(f2.ok, false);
  assert.equal(fs.existsSync(dest2), false);
  assert.deepEqual(fs.readdirSync(path.join(dir, "sub")), ["hour.jsonl.gz"], "no .part leftovers");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("listObjects: ListObjectsV2 pagination via continuation-token, prefix/delimiter signed", async () => {
  const log: Captured[] = [];
  const page1 = `<?xml version="1.0"?><ListBucketResult><IsTruncated>true</IsTruncated>
    <Contents><Key>archive/aircraft/2026-09-01/00.jsonl.gz</Key><Size>10</Size><ETag>&quot;a&quot;</ETag></Contents>
    <NextContinuationToken>tok+/=1</NextContinuationToken></ListBucketResult>`;
  const page2 = `<ListBucketResult><IsTruncated>false</IsTruncated>
    <Contents><Key>archive/aircraft/2026-09-01/01.jsonl.gz</Key><Size>20</Size></Contents></ListBucketResult>`;
  const c = createR2Client(CFG, {
    fetchImpl: fakeFetch((req) => new Response(new URL(req.url).searchParams.get("continuation-token") ? page2 : page1, { status: 200 }), log),
  });
  const r = await c.listObjects("archive/aircraft/");
  assert.equal(r.ok, true);
  assert.equal(r.pages, 2);
  assert.deepEqual(r.objects.map((o) => [o.key, o.size]), [
    ["archive/aircraft/2026-09-01/00.jsonl.gz", 10], ["archive/aircraft/2026-09-01/01.jsonl.gz", 20]]);
  assert.equal(r.objects[0].etag, '"a"');
  assert.equal(new URL(log[0].url).pathname, "/voltrade-archive");
  assert.equal(new URL(log[0].url).searchParams.get("list-type"), "2");
  assert.equal(new URL(log[1].url).searchParams.get("continuation-token"), "tok+/=1");
  verifySigned(log[0], undefined);
  verifySigned(log[1], undefined);
});

test("parseListObjectsV2 reads CommonPrefixes and decodes XML entities", () => {
  const p = parseListObjectsV2(`<ListBucketResult><IsTruncated>false</IsTruncated>
    <CommonPrefixes><Prefix>archive/aircraft/2026-09-01/</Prefix></CommonPrefixes>
    <CommonPrefixes><Prefix>archive/a&amp;b/</Prefix></CommonPrefixes></ListBucketResult>`);
  assert.deepEqual(p.prefixes, ["archive/aircraft/2026-09-01/", "archive/a&b/"]);
  assert.equal(p.truncated, false);
});

test("deleteObjects: POST ?delete with Quiet XML + Content-MD5, per-key errors surfaced, batches of 1000", async () => {
  const log: Captured[] = [];
  const c = createR2Client(CFG, {
    fetchImpl: fakeFetch((req) => {
      const xml = req.body!.toString();
      const bad = xml.includes("<Key>k-3</Key>")
        ? "<Error><Key>k-3</Key><Code>AccessDenied</Code><Message>no</Message></Error>" : "";
      return new Response(`<DeleteResult>${bad}</DeleteResult>`, { status: 200 });
    }, log),
  });
  const keys = Array.from({ length: 1500 }, (_, i) => `k-${i}`);
  const r = await c.deleteObjects(keys);
  assert.equal(log.length, 2, "1000 + 500");
  assert.equal(r.requested, 1500);
  assert.deepEqual(r.errors, [{ key: "k-3", code: "AccessDenied", message: "no" }]);
  assert.equal(r.ok, false);
  const req = log[0];
  assert.equal(req.method, "POST");
  assert.match(req.url, /\/voltrade-archive\?delete$/);
  assert.equal(req.headers["content-md5"], crypto.createHash("md5").update(req.body!).digest("base64"));
  assert.match(req.body!.toString(), /^<\?xml version="1\.0" encoding="UTF-8"\?><Delete xmlns="http:\/\/s3\.amazonaws\.com\/doc\/2006-03-01\/"><Quiet>true<\/Quiet><Object><Key>k-0<\/Key><\/Object>/);
  verifySigned(req, req.body);
  assert.equal(deleteObjectsXml(["a&b"]).includes("<Key>a&amp;b</Key>"), true);
});

test("deleteObject: DELETE on the key; absent key (404) is not an error", async () => {
  const log: Captured[] = [];
  const c = createR2Client(CFG, { fetchImpl: fakeFetch(() => new Response(null, { status: 204 }), log) });
  assert.equal((await c.deleteObject("archive/x")).ok, true);
  assert.equal(log[0].method, "DELETE");
  const c404 = createR2Client(CFG, { fetchImpl: fakeFetch(() => new Response(null, { status: 404 }), []) });
  assert.equal((await c404.deleteObject("gone")).ok, true);
});

test("caller abort signal stops the request without retrying", async () => {
  const ac = new AbortController();
  ac.abort();
  let calls = 0;
  const c = createR2Client(CFG, { fetchImpl: (async () => { calls++; return new Response(null); }) as any, sleep: async () => {} });
  const r = await c.getObject("k", { signal: ac.signal });
  assert.equal(r.ok, false);
  assert.equal(r.error, "aborted");
  assert.equal(calls, 0);
});
