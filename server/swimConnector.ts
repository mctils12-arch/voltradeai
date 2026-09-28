// FLIGHT PROGRAM (2026-09-28) — reusable FAA SWIM (SCDS) queue connector.
//
// One module for every SCDS product the human subscribed to. Each product is a
// durable Solace queue named "<user>.<PRODUCT>.<uuid>.OUT" (SFDPS: "...FDPS...",
// TFMS, STDDS, NOTAM: "...AIM_FNS..."), configured per product plus ONE shared
// SCDS login (product-specific credentials override the shared ones):
//   SWIM_<P>_URL  SWIM_<P>_VPN  SWIM_<P>_QUEUE        (per product)
//   SWIM_<P>_USER ?? SWIM_USER   SWIM_<P>_PASSWORD ?? SWIM_PASSWORD
// with P in SFDPS | TFMS | STDDS | NOTAM | ITWS. A product is "configured" when
// URL + VPN + QUEUE + a resolved user + password all exist. The message VPN is
// per product (FDPS, TFMS, STDDS, AIM_FNS), so it is an env var, never assumed.
// Credentials never appear in logs, status payloads or error strings: every
// error text is redacted against the resolved user/password first (solclientjs
// errors can echo session properties).
//
// Contract with a product adapter: onPayload(text) is called once per message,
// OFF the Solace callback, from a drain loop that yields to the event loop
// between slices (drainBudgetMs) — SFDPS nationwide runs hundreds of msgs/sec
// at peak and must never block the trading loop in the same process.
// Backpressure: CLIENT acknowledgement + a consumer windowSize of maxInFlight
// means the broker stops delivering while maxInFlight messages are unacked; a
// message is acked only after its payload was processed (or dropped when the
// local queue overflows — counted, never silent).
//
// Zero cost when unconfigured: no import, no socket, no timer. The Solace JS
// client ("solclientjs") is loaded by dynamic import inside try/catch and is
// deliberately NOT a dependency of this repo; a missing module is logged ONCE
// per product and shown on the product's status.

import zlib from "zlib";

/** the SCDS products with an env block (only SFDPS gets a consumer today) */
export const SWIM_PRODUCTS = ["SFDPS", "TFMS", "STDDS", "NOTAM", "ITWS"] as const;
export const SWIM_SHARED_USER = "SWIM_USER";
export const SWIM_SHARED_PASSWORD = "SWIM_PASSWORD";

export interface SwimProductConfig { url: string; vpn: string; user: string; password: string; queue: string }

/** env var names a product reads, for ops display ("A|B" = A, else B) */
export function swimEnvVarNames(prefix: string): string[] {
  return [
    `${prefix}_URL`, `${prefix}_VPN`, `${prefix}_QUEUE`,
    `${prefix}_USER|${SWIM_SHARED_USER}`, `${prefix}_PASSWORD|${SWIM_SHARED_PASSWORD}`,
  ];
}

const envVal = (env: NodeJS.ProcessEnv, k: string) => (env[k] || "").trim();

/** Which of a product's settings are missing (names only — never values). */
export function swimMissingEnv(prefix: string, env: NodeJS.ProcessEnv = process.env): string[] {
  const missing: string[] = [];
  for (const s of ["URL", "VPN", "QUEUE"]) if (!envVal(env, `${prefix}_${s}`)) missing.push(`${prefix}_${s}`);
  if (!envVal(env, `${prefix}_USER`) && !envVal(env, SWIM_SHARED_USER)) missing.push(`${prefix}_USER|${SWIM_SHARED_USER}`);
  if (!envVal(env, `${prefix}_PASSWORD`) && !envVal(env, SWIM_SHARED_PASSWORD)) missing.push(`${prefix}_PASSWORD|${SWIM_SHARED_PASSWORD}`);
  return missing;
}

/** config for a product, or null unless URL+VPN+QUEUE and a resolved
 *  user+password (product-specific, else shared) all exist */
export function swimProductConfigFromEnv(prefix: string, env: NodeJS.ProcessEnv = process.env): SwimProductConfig | null {
  if (swimMissingEnv(prefix, env).length) return null;
  return {
    url: envVal(env, `${prefix}_URL`),
    vpn: envVal(env, `${prefix}_VPN`),
    queue: envVal(env, `${prefix}_QUEUE`),
    user: envVal(env, `${prefix}_USER`) || envVal(env, SWIM_SHARED_USER),
    password: envVal(env, `${prefix}_PASSWORD`) || envVal(env, SWIM_SHARED_PASSWORD),
  };
}

/** Per-product env readiness for ops (configured + missing names; no values,
 *  no connection attempted). */
export function swimProductsEnvStatus(env: NodeJS.ProcessEnv = process.env): Record<string, { configured: boolean; missing: string[] }> {
  const out: Record<string, { configured: boolean; missing: string[] }> = {};
  for (const p of SWIM_PRODUCTS) {
    const missing = swimMissingEnv(`SWIM_${p}`, env);
    out[p] = { configured: missing.length === 0, missing };
  }
  return out;
}

/** Strip credentials (and anything shaped like a password property) from a
 *  string before it is logged or stored in a status payload. */
export function redactSecrets(text: string, secrets: Array<string | null | undefined>): string {
  let s = String(text ?? "");
  for (const x of secrets) {
    if (x && x.length >= 3) s = s.split(x).join("***");
  }
  return s.replace(/(pass(?:word)?|pwd|secret)(["']?\s*[:=]\s*["']?)[^\s,"'}]+/gi, "$1$2***");
}

export interface SwimProductStatus {
  product: string;
  envPrefix: string;
  configured: boolean;
  /** null = not attempted (unconfigured) */
  solclientAvailable: boolean | null;
  connected: boolean;
  messagesReceived: number;
  messagesProcessed: number;
  processErrors: number;
  droppedOverflow: number;
  /** ack / dispose / payload-read failures inside the client library */
  clientErrors: number;
  /** messages with no readable body in any section (acked, never silent) */
  unreadablePayloads: number;
  /** structural description (never content) of the latest unreadable message */
  lastUnreadableShape: string | null;
  queueDepth: number;
  reconnects: number;
  lastMessageAt: number | null;
  lastError: string | null;
}

// ── the slice of the solclientjs API this connector uses ────────────────────
// Typed locally (the package is loaded dynamically and is not a dependency);
// names follow solclientjs 10.x.
type SolaceHandler = (e?: unknown) => void;
interface SolaceEmitter { on(event: string | number, fn: SolaceHandler): void }
interface SolaceConsumer extends SolaceEmitter { connect(): void }
interface SolaceSession extends SolaceEmitter {
  connect(): void;
  dispose?(): void;
  createMessageConsumer(props: Record<string, unknown>): SolaceConsumer;
}
export interface SolaceModule {
  SolclientFactoryProperties?: new () => { profile?: unknown };
  SolclientFactoryProfiles?: { version10?: unknown };
  SolclientFactory: { init(props: unknown): void; createSession(props: Record<string, unknown>): SolaceSession };
  SessionEventCode?: Record<string, string | number>;
  MessageConsumerEventName?: Record<string, string | number>;
  QueueType?: Record<string, unknown>;
  MessageConsumerAcknowledgeMode?: Record<string, unknown>;
}
interface SolaceMessage {
  getType?(): unknown;
  /** XML content section, UTF-8 decoded — where Solace JMS puts a TextMessage
   *  body when the connection factory's "XML payload" option is on (the FAA
   *  SCDS brokers: live 2026-09-28, 39k SFDPS messages had ONLY this section) */
  getXmlContentDecoded?(): unknown;
  /** the same section as a latin1 "binary string" (older API shape) */
  getXmlContent?(): unknown;
  getSdtContainer?(): { getValue?(): unknown } | null | undefined;
  getBinaryAttachment?(): unknown;
  acknowledge?(): void;
}

const registry = new Map<string, SwimProductStatus>();
const initedFactories = new WeakSet<object>();
const warnedMissing = new Set<string>();

/** best human-readable text from whatever an API threw or emitted */
export function errText(e: unknown): string {
  if (e instanceof Error) return e.message || e.name;
  if (typeof e === "string") return e;
  if (e && typeof e === "object") {
    const o = e as { message?: unknown; infoStr?: unknown; reason?: unknown };
    for (const v of [o.message, o.infoStr, o.reason]) if (typeof v === "string" && v) return v;
  }
  return "unknown error";
}

function freshStatus(product: string, envPrefix: string): SwimProductStatus {
  return {
    product, envPrefix, configured: false, solclientAvailable: null, connected: false,
    messagesReceived: 0, messagesProcessed: 0, processErrors: 0, droppedOverflow: 0, clientErrors: 0,
    unreadablePayloads: 0, lastUnreadableShape: null,
    queueDepth: 0, reconnects: 0, lastMessageAt: null, lastError: null,
  };
}

/** Structural description of a message for diagnostics — which body
 *  sections exist and their sizes, never any content. */
export function messageShape(message: unknown): string {
  const m = asSolaceMessage(message);
  const size = (v: unknown): string =>
    v == null ? String(v) : typeof v === "string" ? `str${v.length}` : v instanceof Uint8Array ? `bytes${v.length}` : typeof v;
  const read = (fn?: () => unknown): string => {
    if (!fn) return "absent";
    try { return size(fn.call(m)); } catch (e) { return `threw:${errText(e).slice(0, 40)}`; }
  };
  return `type=${read(m.getType)} xmlDecoded=${read(m.getXmlContentDecoded)} xml=${read(m.getXmlContent)} ` +
    `sdt=${read(m.getSdtContainer)} bin=${read(m.getBinaryAttachment)}`;
}

/** status snapshot for a product (an unconfigured default when never started) */
export function swimProductStatus(product: string, envPrefix = `SWIM_${product}`): SwimProductStatus {
  return { ...(registry.get(product) ?? freshStatus(product, envPrefix)) };
}

function asSolaceMessage(m: unknown): SolaceMessage {
  return (m && typeof m === "object" ? m : {}) as SolaceMessage;
}

/** Solace payloads arrive in the XML content section (JMS TextMessage with
 *  the XML-payload option — the FAA SCDS case), as a text (SDT string)
 *  container, or as a binary attachment (latin1 "binary string", Uint8Array
 *  or Buffer), possibly gzipped. Returns null when no section is readable. */
export function payloadToString(message: unknown): string | null {
  const m = asSolaceMessage(message);
  let xml: unknown = null;
  try { xml = m.getXmlContentDecoded?.(); } catch { xml = null; }
  if (typeof xml === "string" && xml) return xml;
  try { xml = m.getXmlContent?.(); } catch { xml = null; }
  if (typeof xml === "string" && xml) return Buffer.from(xml, "latin1").toString("utf8");
  let text: unknown = null;
  try { text = m.getSdtContainer?.()?.getValue?.(); } catch { text = null; }
  if (typeof text === "string" && text) return text;
  let bin: unknown = null;
  try { bin = m.getBinaryAttachment?.(); } catch { bin = null; }
  if (bin == null) return null;
  let buf: Buffer;
  if (typeof bin === "string") buf = Buffer.from(bin, "latin1");
  else if (bin instanceof Uint8Array) buf = Buffer.from(bin);
  else return null;
  if (buf.length > 2 && buf[0] === 0x1f && buf[1] === 0x8b) {
    try { buf = zlib.gunzipSync(buf); } catch { return null; }
  }
  let s = buf.toString("utf8");
  // a JMS TextMessage read through getBinaryAttachment() carries a short SDT
  // header before the text: start at the XML
  const lt = s.indexOf("<");
  if (lt > 0 && lt < 64) s = s.slice(lt);
  return s;
}

export interface SwimProductOptions {
  product: string;          // "SFDPS"
  envPrefix: string;        // "SWIM_SFDPS"
  env?: NodeJS.ProcessEnv;
  /** called once per message from the drain loop; throwing is counted, never fatal */
  onPayload: (payload: string, now: number) => void;
  /** broker flow window = max unacked messages in flight (default 255) */
  maxInFlight?: number;
  /** max ms of payload processing per event-loop turn (default 8) */
  drainBudgetMs?: number;
  /** module loader — tests inject a fake Solace module */
  importer?: (name: string) => Promise<unknown>;
  log?: (msg: string) => void;
  /** reconnect timer (tests pass a stub) */
  setTimer?: (fn: () => void, ms: number) => unknown;
  /** next-turn scheduler for the drain loop (default setImmediate) */
  schedule?: (fn: () => void) => void;
  now?: () => number;
}

export interface SwimConnectorHandle {
  status(): SwimProductStatus;
  /** tear down the session (tests; graceful shutdown) */
  stop(): void;
}

const noopHandle = (product: string, envPrefix: string): SwimConnectorHandle => ({
  status: () => swimProductStatus(product, envPrefix),
  stop: () => { /* nothing was started */ },
});

function asSolaceModule(mod: unknown): SolaceModule | null {
  const m = (mod && typeof mod === "object" ? mod : null) as { default?: unknown } | null;
  const cand = (m?.default && typeof m.default === "object" ? m.default : m) as Partial<SolaceModule> | null;
  return cand && cand.SolclientFactory && typeof cand.SolclientFactory.createSession === "function"
    ? (cand as SolaceModule) : null;
}

/** Start a product consumer when (and only when) its env is complete.
 *  Never throws. */
export async function startSwimProduct(o: SwimProductOptions): Promise<SwimConnectorHandle> {
  const status = registry.get(o.product) ?? freshStatus(o.product, o.envPrefix);
  registry.set(o.product, status);
  const cfg = swimProductConfigFromEnv(o.envPrefix, o.env ?? process.env);
  status.configured = !!cfg;
  if (!cfg) return noopHandle(o.product, o.envPrefix); // zero cost: nothing imported/scheduled

  const log = o.log ?? ((m: string) => console.warn(m));
  const now = o.now ?? Date.now;
  const importer = o.importer ?? ((name: string) => import(/* @vite-ignore */ name) as Promise<unknown>);
  const secrets = [cfg.password, cfg.user];
  let solace: SolaceModule | null = null;
  try {
    // the name travels through a variable so bundlers never try to resolve an
    // intentionally-absent dependency at build time
    const modName = "solclientjs";
    solace = asSolaceModule(await importer(modName));
    if (!solace) throw new Error("solclientjs export shape not recognized");
    status.solclientAvailable = true;
  } catch (e) {
    status.solclientAvailable = false;
    status.lastError = redactSecrets(`solclientjs unavailable: ${errText(e)}`, secrets).slice(0, 200);
    if (!warnedMissing.has(o.product)) {
      warnedMissing.add(o.product);
      log(`[swim] SWIM configured but solclientjs not installed — ${o.product} consumer stays off (${o.envPrefix}_* set; install solclientjs to enable)`);
    }
    return noopHandle(o.product, o.envPrefix);
  }
  const sol = solace;

  const setTimer = o.setTimer ?? ((fn: () => void, ms: number) => { const t = setTimeout(fn, ms); t.unref?.(); return t; });
  const schedule = o.schedule ?? ((fn: () => void) => { setImmediate(fn); });
  const maxInFlight = Math.max(1, o.maxInFlight ?? 255);
  const budget = Math.max(1, o.drainBudgetMs ?? 8);
  const queue: Array<{ payload: string | null; msg: SolaceMessage }> = [];
  let draining = false;
  let stopped = false;
  let failures = 0;
  let session: SolaceSession | null = null;

  const clientError = (what: string, e: unknown) => {
    status.clientErrors++;
    status.lastError = redactSecrets(`${what}: ${errText(e)}`, secrets).slice(0, 200);
  };
  const ack = (msg: SolaceMessage) => {
    try { msg.acknowledge?.(); } catch (e) { clientError("ack", e); }
  };
  const disposeSession = () => {
    try { session?.dispose?.(); } catch (e) { clientError("dispose", e); }
    session = null;
  };
  const drain = () => {
    draining = false;
    const t0 = Date.now();
    while (queue.length && Date.now() - t0 < budget) {
      const { payload, msg } = queue.shift()!;
      if (payload) {
        try { o.onPayload(payload, now()); status.messagesProcessed++; } catch (e) {
          status.processErrors++;
          status.lastError = redactSecrets(`process: ${errText(e)}`, secrets).slice(0, 200);
        }
      } else {
        // acked either way (an unread message would block the queue), but
        // counted and described so a reader gap is never silent again
        status.unreadablePayloads++;
        try { status.lastUnreadableShape = messageShape(msg); } catch (e) { clientError("shape", e); }
      }
      ack(msg);
    }
    status.queueDepth = queue.length;
    if (queue.length && !stopped) { draining = true; schedule(drain); }
  };
  const enqueue = (raw?: unknown) => {
    const msg = asSolaceMessage(raw);
    status.messagesReceived++;
    status.lastMessageAt = now();
    if (queue.length >= maxInFlight * 2) { // the window should prevent this; never silent if it doesn't
      status.droppedOverflow++;
      ack(msg);
      return;
    }
    let payload: string | null = null;
    try { payload = payloadToString(msg); } catch (e) { clientError("payload", e); }
    queue.push({ payload, msg });
    status.queueDepth = queue.length;
    if (!draining) { draining = true; schedule(drain); }
  };

  const connect = () => {
    if (stopped) return;
    let retried = false; // FAILED + DISCONNECTED can both fire: one retry per session
    const retry = (why: string) => {
      if (retried || stopped) return;
      retried = true;
      disposeSession();
      status.connected = false;
      const safe = redactSecrets(why, secrets).slice(0, 300);
      status.lastError = safe;
      failures++;
      status.reconnects++;
      const wait = Math.min(15 * 60_000, 30_000 * 2 ** Math.min(5, failures - 1));
      log(`[swim] ${o.product}: ${safe} — retry in ${Math.round(wait / 1000)}s`);
      setTimer(connect, wait);
    };
    try {
      if (!initedFactories.has(sol)) { // SolclientFactory.init is once per process
        const profile = sol.SolclientFactoryProfiles?.version10;
        let fp: { profile?: unknown } = { profile };
        if (typeof sol.SolclientFactoryProperties === "function") { fp = new sol.SolclientFactoryProperties(); fp.profile = profile; }
        sol.SolclientFactory.init(fp);
        initedFactories.add(sol);
      }
      // SMF over TLS (tcps://) is supported by solclientjs under Node; the
      // API's own reconnect handles blips, our backoff handles exhaustion
      const s = sol.SolclientFactory.createSession({
        url: cfg.url, vpnName: cfg.vpn, userName: cfg.user, password: cfg.password,
        connectRetries: 3, reconnectRetries: 20, reconnectRetryWaitInMsecs: 3000,
      });
      session = s;
      const E = sol.SessionEventCode || {};
      const C = sol.MessageConsumerEventName || {};
      s.on(E.UP_NOTICE, () => {
        try {
          const consumer = s.createMessageConsumer({
            queueDescriptor: { name: cfg.queue, type: sol.QueueType?.QUEUE },
            acknowledgeMode: sol.MessageConsumerAcknowledgeMode?.CLIENT,
            windowSize: maxInFlight,
            createIfMissing: false,
          });
          consumer.on(C.UP, () => { status.connected = true; status.lastError = null; failures = 0; });
          consumer.on(C.CONNECT_FAILED_ERROR, (e) => retry(`queue bind failed: ${errText(e)}`));
          if (C.DOWN_ERROR != null) consumer.on(C.DOWN_ERROR, (e) => retry(`consumer down: ${errText(e)}`));
          consumer.on(C.DOWN, () => { status.connected = false; });
          consumer.on(C.MESSAGE, enqueue);
          consumer.connect();
        } catch (e) { retry(`consumer: ${errText(e)}`); }
      });
      s.on(E.CONNECT_FAILED_ERROR, (e) => retry(`session connect failed: ${errText(e)}`));
      if (E.DOWN_ERROR != null) s.on(E.DOWN_ERROR, (e) => retry(`session down: ${errText(e)}`));
      s.on(E.DISCONNECTED, () => retry("session disconnected"));
      s.connect();
    } catch (e) {
      retry(`connect: ${errText(e)}`);
    }
  };
  connect();

  return {
    status: () => swimProductStatus(o.product, o.envPrefix),
    stop: () => {
      stopped = true;
      status.connected = false;
      disposeSession();
    },
  };
}

/** test seam */
export function _resetSwimConnectorForTests(): void {
  registry.clear();
  warnedMissing.clear();
}
