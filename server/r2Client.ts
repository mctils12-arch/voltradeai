// r2Client.ts — minimal S3-compatible client for Cloudflare R2 (FLIGHT
// PROGRAM cold tier, 2026-09-28). Hand-rolled AWS Signature Version 4 on
// node:crypto only — no SDK dependency (the AWS SDK is ~MBs of bundle for
// five calls). Path-style addressing against the account endpoint:
//
//   https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com/<bucket>/<key>
//   region "auto", service "s3"
//
// ACTIVE ONLY when R2_ACCOUNT_ID + R2_ACCESS_KEY_ID + R2_SECRET_ACCESS_KEY +
// R2_ARCHIVE_BUCKET are ALL set. Otherwise every call is a clean no-op that
// returns { ok:false, notConfigured:true } without touching the network —
// the position archive keeps its existing local-only behavior.
//
// The archive bucket is a SEPARATE, PRIVATE bucket (R2_ARCHIVE_BUCKET). It is
// never the public map-tiles bucket that /tiles-r2 reads via R2_PUBLIC_URL:
// raw position logs are not a public download.
//
// Robustness: per-attempt timeout, limited retries with exponential backoff
// + jitter on network errors / 429 / 5xx (never on other 4xx), a fresh
// signature per attempt (x-amz-date moves), bounded downloads (maxBytes) and
// abortable requests (caller AbortSignal).
//
// NEVER set Content-Encoding: gzip on archive objects: Node's fetch (undici)
// transparently DECOMPRESSES a Content-Encoding: gzip response, so a later
// GET would hand back plain bytes whose length no longer matches HEAD's
// Content-Length. Archive objects are opaque .jsonl.gz blobs stored with
// Content-Type application/gzip; the reader gunzips them itself.

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

// ── SigV4 primitives (pure; pinned by the AWS published test vectors) ───────

export const EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

export function sha256Hex(data: string | Buffer): string {
  return crypto.createHash("sha256").update(data).digest("hex");
}

function hmac(key: string | Buffer, data: string): Buffer {
  return crypto.createHmac("sha256", key).update(data, "utf8").digest();
}

/** RFC 3986 percent-encoding (SigV4's UriEncode): unreserved A-Za-z0-9-_.~
 *  stay literal, everything else (incl. !'()*) is %XX with uppercase hex. */
export function encodeRfc3986(s: string): string {
  return encodeURIComponent(s).replace(/[!'()*]/g, (c) => "%" + c.charCodeAt(0).toString(16).toUpperCase());
}

/** S3 canonical URI: each path segment RFC3986-encoded ONCE, '/' kept (S3
 *  does not double-encode and does not normalize dot segments). */
export function encodeS3Path(rawPath: string): string {
  return rawPath.split("/").map(encodeRfc3986).join("/");
}

/** Canonical query string: encode each key/value, sort by key then value,
 *  empty values rendered "key=" (e.g. the ?delete subresource). */
export function canonicalQueryString(query: Array<[string, string]>): string {
  return query
    .map(([k, v]) => [encodeRfc3986(k), encodeRfc3986(v)] as [string, string])
    .sort((a, b) => (a[0] === b[0] ? (a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0) : a[0] < b[0] ? -1 : 1))
    .map(([k, v]) => `${k}=${v}`)
    .join("&");
}

/** YYYYMMDDTHHMMSSZ from a Date (SigV4's x-amz-date). */
export function amzDateOf(d: Date): string {
  return d.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}/, "");
}

export interface SigV4Input {
  method: string;
  host: string;
  /** ALREADY-ENCODED canonical path (use encodeS3Path on a raw key path). */
  canonicalPath: string;
  query?: Array<[string, string]>;
  /** every header to sign EXCEPT host (added from `host`); must include
   *  x-amz-date. Names are case-insensitive. */
  headers: Record<string, string>;
  payloadHash: string;
  accessKeyId: string;
  secretAccessKey: string;
  region: string;
  service: string;
  amzDate: string;
}

export interface SigV4Output {
  authorization: string;
  signature: string;
  signedHeaders: string;
  canonicalRequest: string;
  stringToSign: string;
  credentialScope: string;
}

export function signV4(inp: SigV4Input): SigV4Output {
  const hdrs: Record<string, string> = { host: inp.host };
  for (const [k, v] of Object.entries(inp.headers)) {
    hdrs[k.toLowerCase()] = String(v).trim().replace(/\s+/g, " ");
  }
  const names = Object.keys(hdrs).sort();
  const canonicalHeaders = names.map((n) => `${n}:${hdrs[n]}\n`).join("");
  const signedHeaders = names.join(";");
  const canonicalRequest = [
    inp.method.toUpperCase(),
    inp.canonicalPath,
    canonicalQueryString(inp.query || []),
    canonicalHeaders,
    signedHeaders,
    inp.payloadHash,
  ].join("\n");
  const dateStamp = inp.amzDate.slice(0, 8);
  const credentialScope = `${dateStamp}/${inp.region}/${inp.service}/aws4_request`;
  const stringToSign = [
    "AWS4-HMAC-SHA256",
    inp.amzDate,
    credentialScope,
    sha256Hex(canonicalRequest),
  ].join("\n");
  const kDate = hmac("AWS4" + inp.secretAccessKey, dateStamp);
  const kRegion = hmac(kDate, inp.region);
  const kService = hmac(kRegion, inp.service);
  const kSigning = hmac(kService, "aws4_request");
  const signature = crypto.createHmac("sha256", kSigning).update(stringToSign, "utf8").digest("hex");
  const authorization =
    `AWS4-HMAC-SHA256 Credential=${inp.accessKeyId}/${credentialScope}, ` +
    `SignedHeaders=${signedHeaders}, Signature=${signature}`;
  return { authorization, signature, signedHeaders, canonicalRequest, stringToSign, credentialScope };
}

// ── configuration ─────────────────────────────────────────────────────────

export interface R2Config {
  accountId: string;
  accessKeyId: string;
  secretAccessKey: string;
  bucket: string;
  /** <accountId>.r2.cloudflarestorage.com */
  host: string;
}

export const R2_NOT_CONFIGURED =
  "R2 archive not configured (needs R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ARCHIVE_BUCKET)";

/** Read the four env vars. Any missing -> null (client becomes a no-op).
 *  The account id and bucket are shape-checked so a typo can never steer
 *  the signed request at a different host. */
export function r2ConfigFromEnv(env: NodeJS.ProcessEnv = process.env): R2Config | null {
  const accountId = (env.R2_ACCOUNT_ID || "").trim();
  const accessKeyId = (env.R2_ACCESS_KEY_ID || "").trim();
  const secretAccessKey = (env.R2_SECRET_ACCESS_KEY || "").trim();
  const bucket = (env.R2_ARCHIVE_BUCKET || "").trim();
  if (!accountId || !accessKeyId || !secretAccessKey || !bucket) return null;
  if (!/^[a-zA-Z0-9]{8,64}$/.test(accountId)) return null;
  if (!/^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$/.test(bucket)) return null;
  return { accountId, accessKeyId, secretAccessKey, bucket, host: `${accountId}.r2.cloudflarestorage.com` };
}

// ── client ───────────────────────────────────────────────────────────────

export interface R2Base {
  ok: boolean;
  notConfigured?: boolean;
  status?: number;
  error?: string;
  attempts?: number;
}
export interface R2PutResult extends R2Base { etag?: string }
export interface R2HeadResult extends R2Base { exists: boolean; size?: number; etag?: string; lastModified?: string }
export interface R2GetResult extends R2Base { body?: Buffer }
export interface R2GetFileResult extends R2Base { bytes?: number }
export interface R2ListObject { key: string; size: number; etag?: string; lastModified?: string }
export interface R2ListResult extends R2Base { objects: R2ListObject[]; prefixes: string[]; truncated: boolean; pages: number }
export interface R2DeleteResult extends R2Base { requested: number; errors: Array<{ key: string; code: string; message: string }> }

export interface R2ClientDeps {
  fetchImpl?: typeof fetch;
  now?: () => Date;
  sleep?: (ms: number) => Promise<void>;
  /** per-attempt timeout (default 30s) */
  timeoutMs?: number;
  /** retries AFTER the first attempt (default 2 -> 3 attempts max) */
  maxRetries?: number;
  /** backoff base ms (default 400; attempt n waits base*3^n + jitter) */
  backoffBaseMs?: number;
}

export interface R2Client {
  readonly configured: boolean;
  putObject(key: string, body: Buffer | string, contentType?: string, contentEncoding?: string,
            opts?: { signal?: AbortSignal; payloadSha256?: string }): Promise<R2PutResult>;
  getObject(key: string, opts?: { maxBytes?: number; signal?: AbortSignal }): Promise<R2GetResult>;
  getObjectToFile(key: string, destPath: string, opts?: { maxBytes?: number; signal?: AbortSignal }): Promise<R2GetFileResult>;
  headObject(key: string, opts?: { signal?: AbortSignal }): Promise<R2HeadResult>;
  listObjects(prefix: string, opts?: { delimiter?: string; maxPages?: number; signal?: AbortSignal }): Promise<R2ListResult>;
  deleteObjects(keys: string[], opts?: { signal?: AbortSignal }): Promise<R2DeleteResult>;
  deleteObject(key: string, opts?: { signal?: AbortSignal }): Promise<R2Base>;
}

/** default cap on a single GET (bytes) — an hour of global aircraft fixes is
 *  single-digit MB gzipped; anything near this is not an archive hour. */
export const R2_DEFAULT_MAX_GET_BYTES = 256 * 1024 * 1024;
const RETRYABLE = new Set([408, 429, 500, 502, 503, 504]);

function xmlDecode(s: string): string {
  return s.replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">").replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n))).replace(/&amp;/g, "&");
}
function xmlEscape(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&apos;");
}
function tag(xml: string, name: string): string | undefined {
  const m = new RegExp(`<${name}>([\\s\\S]*?)</${name}>`).exec(xml);
  return m ? xmlDecode(m[1]) : undefined;
}
function blocks(xml: string, name: string): string[] {
  const out: string[] = [];
  const re = new RegExp(`<${name}>([\\s\\S]*?)</${name}>`, "g");
  let m: RegExpExecArray | null;
  while ((m = re.exec(xml))) out.push(m[1]);
  return out;
}

/** Parse one ListObjectsV2 page. Exported for tests. */
export function parseListObjectsV2(xml: string): { objects: R2ListObject[]; prefixes: string[]; truncated: boolean; nextToken?: string } {
  const objects = blocks(xml, "Contents").map((b) => ({
    key: tag(b, "Key") || "",
    size: Number(tag(b, "Size") || 0),
    etag: tag(b, "ETag"),
    lastModified: tag(b, "LastModified"),
  })).filter((o) => o.key);
  const prefixes = blocks(xml, "CommonPrefixes").map((b) => tag(b, "Prefix") || "").filter(Boolean);
  return {
    objects, prefixes,
    truncated: (tag(xml, "IsTruncated") || "").trim() === "true",
    nextToken: tag(xml, "NextContinuationToken"),
  };
}

/** Body for DeleteObjects (Quiet: only failures come back). Exported for tests. */
export function deleteObjectsXml(keys: string[]): string {
  return '<?xml version="1.0" encoding="UTF-8"?>' +
    '<Delete xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Quiet>true</Quiet>' +
    keys.map((k) => `<Object><Key>${xmlEscape(k)}</Key></Object>`).join("") +
    "</Delete>";
}

function errFromXml(status: number, text: string): string {
  const code = tag(text, "Code");
  const msg = tag(text, "Message");
  return `HTTP ${status}${code ? ` ${code}` : ""}${msg ? `: ${msg}` : ""}`;
}

function anySignal(signals: Array<AbortSignal | undefined>): AbortSignal | undefined {
  const list = signals.filter((s): s is AbortSignal => !!s);
  if (list.length <= 1) return list[0];
  return (AbortSignal as any).any ? (AbortSignal as any).any(list) : list[0];
}

export function createR2Client(cfg: R2Config | null, deps: R2ClientDeps = {}): R2Client {
  const fetchImpl = deps.fetchImpl ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
  const now = deps.now ?? (() => new Date());
  const sleep = deps.sleep ?? ((ms: number) => new Promise<void>((r) => setTimeout(r, ms)));
  const timeoutMs = deps.timeoutMs ?? 30_000;
  const maxRetries = deps.maxRetries ?? 2;
  const backoffBase = deps.backoffBaseMs ?? 400;

  const notConfigured = { ok: false, notConfigured: true, error: R2_NOT_CONFIGURED } as const;

  interface SendReq {
    method: string;
    key?: string;                       // raw object key ("" = bucket root)
    query?: Array<[string, string]>;
    headers?: Record<string, string>;
    body?: Buffer;
    payloadHash?: string;
    signal?: AbortSignal;
  }

  /**
   * Signed request with retries; `onRes` consumes the final Response (any
   * status) INSIDE the attempt's timeout window, so a stalled body read is
   * aborted too. The timeout is a REF'd timer cleared when the attempt ends
   * (AbortSignal.timeout's timer is unref'd — a hung request could then
   * outlive a process with nothing else scheduled).
   */
  async function send<T>(req: SendReq, onRes: (res: Response, attempts: number) => Promise<T>,
                          onErr: (error: string, attempts: number) => T): Promise<T> {
    const c = cfg!;
    const rawPath = `/${c.bucket}${req.key ? "/" + req.key : ""}`;
    const canonicalPath = encodeS3Path(rawPath);
    const query = req.query || [];
    // URL query mirrors the canonical encoding; an empty value is sent as
    // the bare key (?delete) — the SDK convention; it signs as "delete=".
    const qs = query.length
      ? "?" + query.map(([k, v]) => (v === "" ? encodeRfc3986(k) : `${encodeRfc3986(k)}=${encodeRfc3986(v)}`)).join("&")
      : "";
    const url = `https://${c.host}${canonicalPath}${qs}`;
    const payloadHash = req.payloadHash ?? (req.body ? sha256Hex(req.body) : EMPTY_SHA256);
    let lastErr = "";
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      if (req.signal?.aborted) return onErr("aborted", attempt);
      const amzDate = amzDateOf(now());
      const headers: Record<string, string> = {
        ...(req.headers || {}),
        "x-amz-date": amzDate,
        "x-amz-content-sha256": payloadHash,
      };
      const sig = signV4({
        method: req.method, host: c.host, canonicalPath, query, headers, payloadHash,
        accessKeyId: c.accessKeyId, secretAccessKey: c.secretAccessKey,
        region: "auto", service: "s3", amzDate,
      });
      const ac = new AbortController();
      let timedOut = false;
      const timer = setTimeout(() => { timedOut = true; ac.abort(); }, timeoutMs);
      try {
        const res = await fetchImpl(url, {
          method: req.method,
          headers: { ...headers, authorization: sig.authorization },
          body: req.body as any,
          signal: anySignal([ac.signal, req.signal]),
        });
        if (RETRYABLE.has(res.status) && attempt < maxRetries) {
          lastErr = `HTTP ${res.status}`;
          try { await res.body?.cancel(); } catch {}
        } else {
          return await onRes(res, attempt + 1);
        }
      } catch (e: any) {
        if (req.signal?.aborted) return onErr("aborted", attempt + 1);
        lastErr = timedOut ? `timeout after ${timeoutMs}ms` : (e?.message || String(e));
        if (attempt >= maxRetries) return onErr(lastErr, attempt + 1);
      } finally {
        clearTimeout(timer);
      }
      await sleep(backoffBase * Math.pow(3, attempt) + Math.floor(Math.random() * 100));
    }
    return onErr(lastErr || "request failed", maxRetries + 1);
  }

  async function readBounded(res: Response, maxBytes: number, sink?: (chunk: Buffer) => Promise<void>):
      Promise<{ ok: true; chunks: Buffer[]; bytes: number } | { ok: false; error: string }> {
    const declared = Number(res.headers.get("content-length"));
    if (Number.isFinite(declared) && declared > maxBytes) {
      try { await res.body?.cancel(); } catch {}
      return { ok: false, error: `object too large (${declared} > ${maxBytes} bytes)` };
    }
    const chunks: Buffer[] = [];
    let bytes = 0;
    if (!res.body) return { ok: true, chunks, bytes };
    const reader = (res.body as any).getReader();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        const b = Buffer.from(value);
        bytes += b.length;
        if (bytes > maxBytes) {
          try { await reader.cancel(); } catch {}
          return { ok: false, error: `object exceeded ${maxBytes} bytes` };
        }
        if (sink) await sink(b); else chunks.push(b);
      }
    } catch (e: any) {
      return { ok: false, error: e?.message || "body read failed" };
    }
    return { ok: true, chunks, bytes };
  }

  const drain = async (res: Response) => { try { await res.body?.cancel(); } catch {} };
  const failText = async (res: Response) => errFromXml(res.status, await res.text().catch(() => ""));

  return {
    configured: !!cfg,

    async putObject(key, body, contentType = "application/octet-stream", contentEncoding, opts = {}) {
      if (!cfg) return { ...notConfigured };
      const buf = typeof body === "string" ? Buffer.from(body, "utf8") : body;
      const headers: Record<string, string> = { "content-type": contentType };
      if (contentEncoding) headers["content-encoding"] = contentEncoding;
      return send<R2PutResult>(
        { method: "PUT", key, headers, body: buf, payloadHash: opts.payloadSha256, signal: opts.signal },
        async (res, attempts) => {
          if (!res.ok) return { ok: false, status: res.status, error: await failText(res), attempts };
          await drain(res);
          return { ok: true, status: res.status, etag: res.headers.get("etag") || undefined, attempts };
        },
        (error, attempts) => ({ ok: false, error, attempts }),
      );
    },

    async headObject(key, opts = {}) {
      if (!cfg) return { ...notConfigured, exists: false };
      return send<R2HeadResult>(
        { method: "HEAD", key, signal: opts.signal },
        async (res, attempts) => {
          await drain(res);
          if (res.status === 404) return { ok: true, exists: false, status: 404, attempts };
          if (!res.ok) return { ok: false, exists: false, status: res.status, error: `HTTP ${res.status}`, attempts };
          const len = Number(res.headers.get("content-length"));
          return {
            ok: true, exists: true, status: res.status, attempts,
            size: Number.isFinite(len) ? len : undefined,
            etag: res.headers.get("etag") || undefined,
            lastModified: res.headers.get("last-modified") || undefined,
          };
        },
        (error, attempts) => ({ ok: false, exists: false, error, attempts }),
      );
    },

    async getObject(key, opts = {}) {
      if (!cfg) return { ...notConfigured };
      return send<R2GetResult>(
        { method: "GET", key, signal: opts.signal },
        async (res, attempts) => {
          if (!res.ok) return { ok: false, status: res.status, error: await failText(res), attempts };
          const body = await readBounded(res, opts.maxBytes ?? R2_DEFAULT_MAX_GET_BYTES);
          if (!body.ok) return { ok: false, status: res.status, error: body.error, attempts };
          return { ok: true, status: res.status, body: Buffer.concat(body.chunks), attempts };
        },
        (error, attempts) => ({ ok: false, error, attempts }),
      );
    },

    async getObjectToFile(key, destPath, opts = {}) {
      if (!cfg) return { ...notConfigured };
      return send<R2GetFileResult>(
        { method: "GET", key, signal: opts.signal },
        async (res, attempts) => {
          if (!res.ok) return { ok: false, status: res.status, error: await failText(res), attempts };
          // stream to a temp sibling, rename on success: a reader never sees
          // a half-written file, a failure leaves nothing behind
          const tmp = `${destPath}.part-${process.pid}-${Math.random().toString(36).slice(2)}`;
          await fs.promises.mkdir(path.dirname(destPath), { recursive: true });
          const fh = await fs.promises.open(tmp, "w");
          let out: Awaited<ReturnType<typeof readBounded>>;
          try {
            out = await readBounded(res, opts.maxBytes ?? R2_DEFAULT_MAX_GET_BYTES, async (chunk) => { await fh.write(chunk); });
          } finally {
            await fh.close().catch(() => {});
          }
          if (!out.ok) {
            await fs.promises.unlink(tmp).catch(() => {});
            return { ok: false, status: res.status, error: out.error, attempts };
          }
          await fs.promises.rename(tmp, destPath);
          return { ok: true, status: res.status, bytes: out.bytes, attempts };
        },
        (error, attempts) => ({ ok: false, error, attempts }),
      );
    },

    async listObjects(prefix, opts = {}) {
      const empty = { objects: [], prefixes: [], truncated: false, pages: 0 };
      if (!cfg) return { ...notConfigured, ...empty };
      const maxPages = opts.maxPages ?? 20;
      const objects: R2ListObject[] = [];
      const prefixes: string[] = [];
      let token: string | undefined;
      let pages = 0;
      let attempts = 0;
      for (;;) {
        const query: Array<[string, string]> = [["list-type", "2"], ["prefix", prefix]];
        if (opts.delimiter) query.push(["delimiter", opts.delimiter]);
        if (token) query.push(["continuation-token", token]);
        const page = await send<{ ok: true; xml: string } | { ok: false; status?: number; error: string }>(
          { method: "GET", query, signal: opts.signal },
          async (res, a) => {
            attempts += a;
            const text = await res.text().catch(() => "");
            return res.ok ? { ok: true, xml: text } : { ok: false, status: res.status, error: errFromXml(res.status, text) };
          },
          (error, a) => { attempts += a; return { ok: false, error }; },
        );
        if (!page.ok) return { ok: false, status: page.status, error: page.error, attempts, objects, prefixes, truncated: true, pages };
        const p = parseListObjectsV2(page.xml);
        pages++;
        objects.push(...p.objects);
        prefixes.push(...p.prefixes);
        if (!p.truncated || !p.nextToken) return { ok: true, status: 200, attempts, objects, prefixes, truncated: false, pages };
        if (pages >= maxPages) return { ok: true, status: 200, attempts, objects, prefixes, truncated: true, pages };
        token = p.nextToken;
      }
    },

    async deleteObjects(keys, opts = {}) {
      if (!cfg) return { ...notConfigured, requested: keys.length, errors: [] };
      const errors: Array<{ key: string; code: string; message: string }> = [];
      let attempts = 0;
      for (let i = 0; i < keys.length; i += 1000) {
        const batch = keys.slice(i, i + 1000);
        const xml = Buffer.from(deleteObjectsXml(batch), "utf8");
        const md5 = crypto.createHash("md5").update(xml).digest("base64");
        const r = await send<{ ok: true; xml: string } | { ok: false; status?: number; error: string }>(
          {
            method: "POST", query: [["delete", ""]], body: xml,
            headers: { "content-type": "application/xml", "content-md5": md5 },
            signal: opts.signal,
          },
          async (res, a) => {
            attempts += a;
            const text = await res.text().catch(() => "");
            return res.ok ? { ok: true, xml: text } : { ok: false, status: res.status, error: errFromXml(res.status, text) };
          },
          (error, a) => { attempts += a; return { ok: false, error }; },
        );
        if (!r.ok) return { ok: false, status: r.status, error: r.error, attempts, requested: keys.length, errors };
        for (const b of blocks(r.xml, "Error")) {
          errors.push({ key: tag(b, "Key") || "", code: tag(b, "Code") || "", message: tag(b, "Message") || "" });
        }
      }
      return { ok: errors.length === 0, status: 200, attempts, requested: keys.length, errors,
               error: errors.length ? `${errors.length} key(s) failed to delete` : undefined };
    },

    async deleteObject(key, opts = {}) {
      if (!cfg) return { ...notConfigured };
      return send<R2Base>(
        { method: "DELETE", key, signal: opts.signal },
        async (res, attempts) => {
          // S3/R2 answer 204 for a delete; deleting an absent key is not an error
          if (!res.ok && res.status !== 404) return { ok: false, status: res.status, error: await failText(res), attempts };
          await drain(res);
          return { ok: true, status: res.status, attempts };
        },
        (error, attempts) => ({ ok: false, error, attempts }),
      );
    },
  };
}
