// One-shot GP fetch+parse worker — PERF session #2 (SCALE queue item 3).
//
// WHY: fetchGp on the main thread does res.text() over the ~2.4 MB
// CelesTrak `active` CSV payload + parseGpCsv of ~16k records — a 150-500 ms
// freeze exactly when the user toggles the satellites layer on. Same fix
// shape as ./satcatWorker (E4-2, v1.0.324): fetch + parse in a worker,
// the main thread receives already-parsed records via structured clone
// (~tens of ms) and keeps ALL its existing flow (session cache, resilient
// retry, abort-on-timeout — the caller terminates the worker on abort).

import { fetchGp, type GpRecord } from './tle.js';

/** main -> worker: fetch + parse a GP group. */
export interface GpFetchMessage {
  type: 'fetch';
  group?: string;
}
export type GpWorkerInbound = GpFetchMessage;

/** worker -> main: parsed GP records (possibly empty — the CALLER owns the
 *  empty-means-failure rule, mirroring the satcatWorker contract). */
export interface GpRowsMessage {
  type: 'rows';
  rows: GpRecord[];
}
export interface GpErrorMessage {
  type: 'error';
  message: string;
}
export type GpWorkerOutbound = GpRowsMessage | GpErrorMessage;

export type GpPostFn = (msg: GpWorkerOutbound) => void;

export interface GpWorkerDeps {
  fetchGpImpl?: (group: string) => Promise<GpRecord[]>;
}

export interface GpWorkerRuntime {
  handle(msg: GpWorkerInbound): Promise<void>;
}

/** Create the worker's message handler with the fetch injected (testable). */
export function createGpWorker(post: GpPostFn, deps: GpWorkerDeps = {}): GpWorkerRuntime {
  const fetchImpl = deps.fetchGpImpl ?? ((group: string) => fetchGp(group));
  return {
    async handle(msg: GpWorkerInbound): Promise<void> {
      if (!msg || msg.type !== 'fetch') return; // unknown: ignore (forward-compat)
      try {
        const rows = await fetchImpl(msg.group ?? 'active');
        post({ type: 'rows', rows });
      } catch (e) {
        post({ type: 'error', message: e instanceof Error ? e.message : 'GP fetch failed' });
      }
    },
  };
}

// --- auto-install when running as a real dedicated worker (the satWorker /
// satcatWorker detection pattern; inert in Node tests and on the main thread).
interface DedicatedScopeLike {
  postMessage(message: unknown): void;
  onmessage: ((ev: { data: GpWorkerInbound }) => void) | null;
}

export function installGpWorker(scope: DedicatedScopeLike, deps: GpWorkerDeps = {}): GpWorkerRuntime {
  const runtime = createGpWorker((msg) => scope.postMessage(msg), deps);
  scope.onmessage = (ev) => { void runtime.handle(ev.data); };
  return runtime;
}

{
  const g = globalThis as unknown as {
    DedicatedWorkerGlobalScope?: unknown;
    self?: unknown;
  };
  const Ctor = g.DedicatedWorkerGlobalScope as (new () => unknown) | undefined;
  if (typeof Ctor === 'function' && g.self instanceof Ctor) {
    installGpWorker(g.self as unknown as DedicatedScopeLike);
  }
}
