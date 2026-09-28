import { test } from "node:test";
import assert from "node:assert/strict";
import zlib from "node:zlib";
import {
  swimProductConfigFromEnv, swimEnvVarNames, swimMissingEnv, swimProductsEnvStatus, redactSecrets,
  startSwimProduct, swimProductStatus, payloadToString, _resetSwimConnectorForTests, SWIM_PRODUCTS,
} from "./swimConnector";

const SFDPS_ENV = {
  SWIM_SFDPS_URL: "tcps://ems1.swim.faa.gov:55443", SWIM_SFDPS_VPN: "FDPS",
  SWIM_SFDPS_QUEUE: "someone.example.com.FDPS.8f42ca91-0000.OUT",
  SWIM_USER: "someone@example.com", SWIM_PASSWORD: "hunter2-secret",
} as unknown as NodeJS.ProcessEnv;

// ── env ─────────────────────────────────────────────────────────────────────

test("config: shared SCDS login is the fallback, product-specific credentials override it", () => {
  assert.deepEqual(swimProductConfigFromEnv("SWIM_SFDPS", SFDPS_ENV), {
    url: "tcps://ems1.swim.faa.gov:55443", vpn: "FDPS", queue: "someone.example.com.FDPS.8f42ca91-0000.OUT",
    user: "someone@example.com", password: "hunter2-secret",
  });
  const own = swimProductConfigFromEnv("SWIM_SFDPS", { ...SFDPS_ENV, SWIM_SFDPS_USER: "sfdps-user", SWIM_SFDPS_PASSWORD: "sfdps-pw" })!;
  assert.equal(own.user, "sfdps-user");
  assert.equal(own.password, "sfdps-pw");
});

test("config: URL, VPN, QUEUE and a resolved user+password are all required", () => {
  assert.equal(swimProductConfigFromEnv("SWIM_SFDPS", {} as NodeJS.ProcessEnv), null);
  for (const k of ["SWIM_SFDPS_URL", "SWIM_SFDPS_VPN", "SWIM_SFDPS_QUEUE", "SWIM_USER", "SWIM_PASSWORD"]) {
    assert.equal(swimProductConfigFromEnv("SWIM_SFDPS", { ...SFDPS_ENV, [k]: " " }), null, `${k} missing must unconfigure`);
  }
  assert.deepEqual(swimMissingEnv("SWIM_SFDPS", { ...SFDPS_ENV, SWIM_SFDPS_VPN: "", SWIM_PASSWORD: "" }),
    ["SWIM_SFDPS_VPN", "SWIM_SFDPS_PASSWORD|SWIM_PASSWORD"]);
  assert.deepEqual(swimEnvVarNames("SWIM_SFDPS"),
    ["SWIM_SFDPS_URL", "SWIM_SFDPS_VPN", "SWIM_SFDPS_QUEUE", "SWIM_SFDPS_USER|SWIM_USER", "SWIM_SFDPS_PASSWORD|SWIM_PASSWORD"]);
});

test("swimProductsEnvStatus: every product reported by name only — never a value", () => {
  const env = {
    ...SFDPS_ENV,
    SWIM_NOTAM_URL: "tcps://ems2.swim.faa.gov:55443", SWIM_NOTAM_VPN: "AIM_FNS", SWIM_NOTAM_QUEUE: "x.AIM_FNS.y.OUT",
    SWIM_TFMS_URL: "tcps://ems1.swim.faa.gov:55443",
  } as unknown as NodeJS.ProcessEnv;
  const s = swimProductsEnvStatus(env);
  assert.deepEqual(Object.keys(s), [...SWIM_PRODUCTS]);
  assert.equal(s.SFDPS.configured, true);
  assert.equal(s.NOTAM.configured, true);
  assert.equal(s.TFMS.configured, false);
  assert.deepEqual(s.TFMS.missing, ["SWIM_TFMS_VPN", "SWIM_TFMS_QUEUE"]);
  assert.equal(s.ITWS.configured, false);
  const json = JSON.stringify(s);
  assert.ok(!json.includes("hunter2") && !json.includes("someone@"), "no secrets or usernames in the payload");
});

test("redactSecrets: credentials and password-shaped properties are scrubbed", () => {
  assert.equal(redactSecrets("login failed for someone@example.com with hunter2-secret", ["hunter2-secret", "someone@example.com"]),
    "login failed for *** with ***");
  assert.equal(redactSecrets(`props {"password":"abc123", "url":"x"}`, []), `props {"password":"***", "url":"x"}`);
  assert.equal(redactSecrets("password=abc pwd: xyz", []), "password=*** pwd: ***");
});

// ── lifecycle ───────────────────────────────────────────────────────────────

test("unconfigured product: zero cost — nothing imported, status says so", async () => {
  _resetSwimConnectorForTests();
  let imported = 0;
  const h = await startSwimProduct({
    product: "SFDPS", envPrefix: "SWIM_SFDPS", env: {} as NodeJS.ProcessEnv,
    onPayload: () => {}, importer: async () => { imported++; return {}; },
  });
  assert.equal(imported, 0);
  const s = h.status();
  assert.equal(s.configured, false);
  assert.equal(s.solclientAvailable, null);
});

test("configured but solclientjs missing: logged ONCE per product, never throws, no secrets", async () => {
  _resetSwimConnectorForTests();
  const logs: string[] = [];
  const opts = {
    product: "SFDPS", envPrefix: "SWIM_SFDPS", env: SFDPS_ENV, onPayload: () => {}, log: (m: string) => logs.push(m),
    importer: async () => { throw new Error("Cannot find module 'solclientjs'"); },
  };
  const h = await startSwimProduct(opts);
  await startSwimProduct(opts);
  assert.equal(h.status().configured, true);
  assert.equal(h.status().solclientAvailable, false);
  assert.equal(h.status().connected, false);
  assert.equal(logs.length, 1);
  assert.match(logs[0], /SWIM configured but solclientjs not installed/);
  assert.ok(!logs[0].includes("hunter2"));
});

/** a fake of the solclientjs surface the connector uses */
type Handler = (x?: unknown) => void;
interface FakeRec {
  sessionArgs: Record<string, unknown> | null;
  consumerArgs: { queueDescriptor?: unknown; acknowledgeMode?: unknown; windowSize?: unknown } | null;
  inits: number; disposed: number;
}
function fakeSolace(opts: { failConnect?: string } = {}) {
  const h: Record<string, Handler> = {};
  const ch: Record<string, Handler> = {};
  const rec: FakeRec = { sessionArgs: null, consumerArgs: null, inits: 0, disposed: 0 };
  const mod = {
    SolclientFactoryProperties: class { profile: unknown; },
    SolclientFactoryProfiles: { version10: "v10" },
    SolclientFactory: {
      init: () => { rec.inits++; },
      createSession: (args: Record<string, unknown>) => {
        rec.sessionArgs = args;
        return {
          on: (ev: string | number, fn: Handler) => { h[String(ev)] = fn; },
          connect: () => { if (opts.failConnect) h.FAIL({ message: opts.failConnect }); else h.UP(); },
          dispose: () => { rec.disposed++; },
          createMessageConsumer: (cargs: Record<string, unknown>) => {
            rec.consumerArgs = cargs;
            return { on: (ev: string | number, fn: Handler) => { ch[String(ev)] = fn; }, connect: () => ch.CUP() };
          },
        };
      },
    },
    SessionEventCode: { UP_NOTICE: "UP", CONNECT_FAILED_ERROR: "FAIL", DISCONNECTED: "DISC", DOWN_ERROR: "DOWN" },
    QueueType: { QUEUE: "QUEUE" },
    MessageConsumerAcknowledgeMode: { CLIENT: "CLIENT" },
    MessageConsumerEventName: { UP: "CUP", CONNECT_FAILED_ERROR: "CFAIL", DOWN: "CDOWN", DOWN_ERROR: "CDOWNERR", MESSAGE: "MSG" },
  };
  return { mod, h, ch, rec };
}

test("consumer: binds the durable QUEUE with CLIENT ack + flow window; drains in slices and acks after processing", async () => {
  _resetSwimConnectorForTests();
  const f = fakeSolace();
  const seen: string[] = [];
  const scheduled: Array<() => void> = [];
  const h = await startSwimProduct({
    product: "SFDPS", envPrefix: "SWIM_SFDPS", env: SFDPS_ENV, maxInFlight: 8,
    onPayload: (p) => { seen.push(p); if (p === "boom") throw new Error("bad payload"); },
    importer: async () => ({ default: f.mod }), setTimer: () => 0, log: () => {},
    schedule: (fn) => { scheduled.push(fn); },
  });
  assert.equal(f.rec.inits, 1);
  assert.deepEqual(f.rec.sessionArgs, {
    url: "tcps://ems1.swim.faa.gov:55443", vpnName: "FDPS", userName: "someone@example.com", password: "hunter2-secret",
    connectRetries: 3, reconnectRetries: 20, reconnectRetryWaitInMsecs: 3000,
  });
  assert.deepEqual(f.rec.consumerArgs?.queueDescriptor, { name: "someone.example.com.FDPS.8f42ca91-0000.OUT", type: "QUEUE" });
  assert.equal(f.rec.consumerArgs?.acknowledgeMode, "CLIENT");
  assert.equal(f.rec.consumerArgs?.windowSize, 8);
  assert.equal(h.status().connected, true);

  let acked = 0;
  const msg = (body: object) => ({ ...body, acknowledge: () => { acked++; } });
  f.ch.MSG(msg({ getBinaryAttachment: () => new Uint8Array(zlib.gzipSync(Buffer.from("<a>gz</a>"))) }));
  f.ch.MSG(msg({ getSdtContainer: () => ({ getValue: () => "<b>text</b>" }) }));
  f.ch.MSG(msg({ getBinaryAttachment: () => "boom" }));
  assert.equal(seen.length, 0, "nothing processed inside the Solace callback");
  assert.equal(acked, 0, "nothing acked before processing");
  assert.equal(scheduled.length, 1, "one drain scheduled for the burst");
  scheduled.shift()!();
  assert.deepEqual(seen, ["<a>gz</a>", "<b>text</b>", "boom"]);
  assert.equal(acked, 3, "every message acked after processing (the bad one too — counted, not redelivered forever)");
  const s = h.status();
  assert.equal(s.messagesReceived, 3);
  assert.equal(s.messagesProcessed, 2);
  assert.equal(s.processErrors, 1);
  assert.equal(s.queueDepth, 0);
  h.stop();
  assert.equal(swimProductStatus("SFDPS").connected, false);
});

test("consumer: local overflow beyond 2x the window is dropped + counted, never silent", async () => {
  _resetSwimConnectorForTests();
  const f = fakeSolace();
  const h = await startSwimProduct({
    product: "SFDPS", envPrefix: "SWIM_SFDPS", env: SFDPS_ENV, maxInFlight: 2,
    onPayload: () => {}, importer: async () => f.mod, setTimer: () => 0, log: () => {}, schedule: () => {},
  });
  for (let i = 0; i < 6; i++) f.ch.MSG({ getBinaryAttachment: () => `<m${i}/>`, acknowledge: () => {} });
  assert.equal(h.status().queueDepth, 4);
  assert.equal(h.status().droppedOverflow, 2);
});

test("connect failure: retried with backoff; error text is redacted of credentials", async () => {
  _resetSwimConnectorForTests();
  const f = fakeSolace({ failConnect: "auth failed for someone@example.com password=hunter2-secret" });
  const logs: string[] = [];
  const timers: number[] = [];
  const h = await startSwimProduct({
    product: "SFDPS", envPrefix: "SWIM_SFDPS", env: SFDPS_ENV, onPayload: () => {},
    importer: async () => f.mod, setTimer: (_fn, ms) => { timers.push(ms); return 0; }, log: (m) => logs.push(m),
  });
  const s = h.status();
  assert.equal(s.connected, false);
  assert.equal(s.reconnects, 1);
  assert.deepEqual(timers, [30_000]);
  assert.ok(s.lastError && !s.lastError.includes("hunter2") && !s.lastError.includes("someone@"), s.lastError!);
  assert.ok(logs.every((l) => !l.includes("hunter2") && !l.includes("someone@")));
  assert.equal(f.rec.disposed, 1, "failed session disposed before retry");
});

test("payloadToString: JMS TextMessage SDT header before the XML is stripped", () => {
  assert.equal(payloadToString({ getBinaryAttachment: () => "\u001c\u0000\u0010<x/>" }), "<x/>");
  assert.equal(payloadToString({ getBinaryAttachment: () => "<x/>" }), "<x/>");
  assert.equal(payloadToString({}), null);
});
