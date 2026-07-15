// One-shot SATCAT fetch+parse worker — EARTH TWIN E4-2 (perf slice).
//
// WHY: parseSatcat over the ~6 MB / ~63k-row satcat.csv measured ~280-340 ms
// of SYNCHRONOUS main-thread time on a desktop-class CPU (~1 s+ on mobile),
// landing exactly when the user has just enabled the satellites layer and is
// interacting with the globe (session #1 review, measured). Fetch + parse in
// a worker; the main thread receives already-parsed records via structured
// clone (~tens of ms) and builds its lookup index in frame-sized chunks
// (buildNoradIndex in ./identity).
//
// Mirrors ./satWorker's testability pattern: the logic lives in a pure
// factory with injected deps (hermetic under `npx tsx --test`); the worker
// runtime is a thin auto-installed wrapper.

import { fetchSatcat, type SatcatRecord } from './tle.js';

/** main -> worker: fetch + parse the catalog. */
export interface SatcatFetchMessage {
  type: 'fetch';
}
export type SatcatWorkerInbound = SatcatFetchMessage;

/** worker -> main: parsed rows (possibly empty — the CALLER owns the
 *  empty-means-failure honesty rule; the worker just reports what it got). */
export interface SatcatRowsMessage {
  type: 'rows';
  rows: SatcatRecord[];
}
/** worker -> main: the fetch/parse itself failed. */
export interface SatcatErrorMessage {
  type: 'error';
  message: string;
}
export type SatcatWorkerOutbound = SatcatRowsMessage | SatcatErrorMessage;

export type SatcatPostFn = (msg: SatcatWorkerOutbound) => void;

export interface SatcatWorkerDeps {
  fetchSatcatImpl?: () => Promise<SatcatRecord[]>;
}

export interface SatcatWorkerRuntime {
  handle(msg: SatcatWorkerInbound): Promise<void>;
}

/** Create the worker's message handler with the fetch injected (testable). */
export function createSatcatWorker(
  post: SatcatPostFn,
  deps: SatcatWorkerDeps = {},
): SatcatWorkerRuntime {
  const fetchImpl = deps.fetchSatcatImpl ?? fetchSatcat;
  return {
    async handle(msg: SatcatWorkerInbound): Promise<void> {
      if (!msg || msg.type !== 'fetch') return; // unknown: ignore (forward-compat)
      try {
        const rows = await fetchImpl();
        post({ type: 'rows', rows });
      } catch (e) {
        post({ type: 'error', message: e instanceof Error ? e.message : 'SATCAT fetch failed' });
      }
    },
  };
}

// --- auto-install when running as a real dedicated worker (satWorker's
// detection pattern: only a true DedicatedWorkerGlobalScope self passes;
// stays inert in Node tests and on the browser main thread). ---
interface DedicatedScopeLike {
  postMessage(message: unknown): void;
  onmessage: ((ev: { data: SatcatWorkerInbound }) => void) | null;
}

export function installSatcatWorker(
  scope: DedicatedScopeLike,
  deps: SatcatWorkerDeps = {},
): SatcatWorkerRuntime {
  const runtime = createSatcatWorker((msg) => scope.postMessage(msg), deps);
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
    installSatcatWorker(g.self as unknown as DedicatedScopeLike);
  }
}
