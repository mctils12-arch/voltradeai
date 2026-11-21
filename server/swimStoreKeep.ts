// swimStoreKeep.ts — the FAA traffic-management programs, airport
// configurations and flight restrictions kept across a server restart.
//
// WHY ([PIPELINE] 2026-11-21, open_questions.md 2026-11-20 item (t6)). Both
// stores (swimTfms.TfmsStore, swimNotam.NotamStore) live in memory and were
// refilled only as the FAA re-sent each message. Seen on production
// 2026-11-20 23:57Z, two minutes after a deploy: the traffic-management store
// held 48 rows (351 an hour earlier), the flight-restriction store 0, and
// Teterboro's ground stop of 21:26Z was not on its airport card. The
// container restarts at every deploy (16 to 24 times a day in November), so
// the pages built on these stores under-reported for up to an hour each time.
//
// WHAT. Once a minute the rows that are still in force are written to ONE
// small file on the volume (written beside, then renamed over, so a kill
// mid-write leaves the previous file). At boot the file is read back before
// the feed's first message is handled. A row already held wins over a saved
// one; a row past its end is dropped. Nothing is written when no message
// arrived since the last save.
//
// WHAT THIS CANNOT KNOW. Messages the FAA sent while the server was down are
// still missed: a program that started in that gap is absent until re-sent,
// and one cancelled in that gap stays listed until its published end. Up to
// one minute of rows before a restart is not in the file. The routes say
// what was carried over (`kept`), so a page can say it too.
//
// Not kept: per-flight amendments (off by default, LADD-gated when on), the
// flight-plan store (its own disk tier, swimSfdps.ts), counters and samples.
//
// RAW plumbing for two RAW feeds: nothing here is read by any trading path.
import fs from "fs";
import path from "path";
import { archiveBaseDir } from "./datacoreArchive";
import type { TfmsStore, TfmsSnapshot } from "./swimTfms";
import type { NotamStore, TfrRecord } from "./swimNotam";

export const SWIM_KEEP_VERSION = 1;
export const SWIM_KEEP_INTERVAL_MS = 60_000;
/** a serialised file over this size is not written (the two stores are capped
 *  at 2,000 rows each; measured sizes are far below) */
export const SWIM_KEEP_MAX_BYTES = 8 * 1024 * 1024;
/** a file older than this is not read back: every default lifetime in it has
 *  run out or is about to, and week-old lists would read as current */
export const SWIM_KEEP_MAX_AGE_MS = 24 * 3600_000;

/** VOLTRADE_SWIM_STORE_KEEP=off|0|false|no turns saving and restoring off. */
export function swimKeepEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  const v = String(env.VOLTRADE_SWIM_STORE_KEEP ?? "").trim().toLowerCase();
  return !(v === "off" || v === "0" || v === "false" || v === "no");
}

/** The one file, on the volume beside the archive folder (not inside it: it
 *  is a cache of live state, replaced each minute, never history). */
export function swimKeepFile(baseDir?: string): string {
  return path.join(baseDir ?? path.dirname(archiveBaseDir()), "swim_active_stores.json");
}

interface KeepDoc {
  v: number;
  saved_at: number;
  tfms: TfmsSnapshot;
  tfrs: TfrRecord[];
}

/** why a restore put nothing back; a closed list, never free text */
export type RestoreCause = "ok" | "no_file" | "unreadable" | "wrong_version" | "too_old";
export type SaveCause = "ok" | "unchanged" | "too_large" | "write_failed";

export interface SwimKeepRestored {
  /** when this process read the file */
  at: number;
  /** when the previous process wrote it; null when nothing was read */
  saved_at: number | null;
  cause: RestoreCause;
  programs: number;
  configs: number;
  tfrs: number;
  /** saved rows not put back: ended, too old or malformed */
  dropped: number;
}

/** Read the file into the two stores. Never throws. */
export function restoreSwimStores(tfms: TfmsStore, notam: NotamStore, o: { file?: string; now?: number } = {}): SwimKeepRestored {
  const now = o.now ?? Date.now();
  const none = (cause: RestoreCause, savedAt: number | null = null): SwimKeepRestored =>
    ({ at: now, saved_at: savedAt, cause, programs: 0, configs: 0, tfrs: 0, dropped: 0 });
  let text: string;
  try { text = fs.readFileSync(o.file ?? swimKeepFile(), "utf8"); } catch (e) {
    return none((e as NodeJS.ErrnoException)?.code === "ENOENT" ? "no_file" : "unreadable");
  }
  let doc: KeepDoc;
  try { doc = JSON.parse(text) as KeepDoc; } catch { return none("unreadable"); }
  if (!doc || typeof doc !== "object" || !Number.isFinite(doc.saved_at)) return none("unreadable");
  if (doc.v !== SWIM_KEEP_VERSION) return none("wrong_version", doc.saved_at);
  if (now - doc.saved_at > SWIM_KEEP_MAX_AGE_MS || doc.saved_at > now + 60_000) return none("too_old", doc.saved_at);
  const a = tfms.restore(doc.tfms, now);
  const b = notam.restore(doc.tfrs, now);
  return { at: now, saved_at: doc.saved_at, cause: "ok", programs: a.programs, configs: a.configs, tfrs: b.tfrs, dropped: a.dropped + b.dropped };
}

/** What changes when a message is ingested: the save is skipped while it stands still. */
function ingestMark(tfms: TfmsStore, notam: NotamStore): string {
  return `${tfms.parsedMessages}:${notam.notams}`;
}

export interface SwimKeeper {
  /** restore once, then save every `intervalMs`; a second call does nothing */
  start(): SwimKeepRestored | null;
  /** one save now (the timer's own step; exposed for tests and shutdown) */
  saveNow(now?: number): Promise<SaveCause>;
  stop(): void;
  view(): SwimKeepView;
}

/** What the two public routes say about the keeping. Counts and times only. */
export interface SwimKeepView {
  enabled: boolean;
  interval_s: number;
  restored: SwimKeepRestored | null;
  last_saved_at: number | null;
  last_save: SaveCause | null;
  saves: number;
  save_failures: number;
  last_bytes: number | null;
}

export function createSwimKeeper(tfms: TfmsStore, notam: NotamStore, o: {
  file?: string;
  intervalMs?: number;
  enabled?: boolean;
  now?: () => number;
  log?: (m: string) => void;
} = {}): SwimKeeper {
  const file = o.file ?? swimKeepFile();
  const enabled = o.enabled ?? swimKeepEnabled();
  const clock = o.now ?? Date.now;
  const log = o.log ?? ((m: string) => console.log(m));
  let timer: ReturnType<typeof setInterval> | null = null;
  let started = false;
  let restored: SwimKeepRestored | null = null;
  let mark: string | null = null;
  let busy = false;
  const st = { lastSavedAt: null as number | null, lastSave: null as SaveCause | null, saves: 0, failures: 0, bytes: null as number | null };

  async function saveNow(nowIn?: number): Promise<SaveCause> {
    if (!enabled || busy) return "unchanged";
    const now = nowIn ?? clock();
    const m = ingestMark(tfms, notam);
    if (m === mark) return (st.lastSave = "unchanged");
    busy = true;
    try {
      const doc: KeepDoc = { v: SWIM_KEEP_VERSION, saved_at: now, tfms: tfms.snapshot(now), tfrs: notam.snapshot(now) };
      const body = JSON.stringify(doc);
      if (body.length > SWIM_KEEP_MAX_BYTES) { st.failures++; return (st.lastSave = "too_large"); }
      const tmp = `${file}.${process.pid}.tmp`;
      await fs.promises.mkdir(path.dirname(file), { recursive: true });
      await fs.promises.writeFile(tmp, body);
      await fs.promises.rename(tmp, file);
      mark = m;
      st.lastSavedAt = now; st.saves++; st.bytes = body.length;
      return (st.lastSave = "ok");
    } catch (e) {
      st.failures++;
      // the first failure and every 60th are logged: a full volume must be seen, not flood the log
      if (st.failures === 1 || st.failures % 60 === 0) log(`[swimStoreKeep] save failed (${st.failures}): ${(e as NodeJS.ErrnoException)?.code ?? "error"}`);
      return (st.lastSave = "write_failed");
    } finally {
      busy = false;
    }
  }

  return {
    start() {
      if (started || !enabled) return restored;
      started = true;
      restored = restoreSwimStores(tfms, notam, { file, now: clock() });
      // what was just read back is what the file holds: no rewrite until a message arrives
      mark = ingestMark(tfms, notam);
      log(`[swimStoreKeep] restore ${restored.cause}: ${restored.programs} programs, ${restored.configs} airport configurations, ${restored.tfrs} flight restrictions put back, ${restored.dropped} dropped`);
      timer = setInterval(() => { void saveNow(); }, o.intervalMs ?? SWIM_KEEP_INTERVAL_MS);
      timer.unref?.();
      return restored;
    },
    saveNow,
    stop() { if (timer) clearInterval(timer); timer = null; },
    view() {
      return {
        enabled, interval_s: Math.round((o.intervalMs ?? SWIM_KEEP_INTERVAL_MS) / 1000), restored,
        last_saved_at: st.lastSavedAt, last_save: st.lastSave, saves: st.saves, save_failures: st.failures, last_bytes: st.bytes,
      };
    },
  };
}
