// bootWatch.test — the page half of the boot marker's ownership rule, driven
// through a fake window/document: what each page event does to the marker.
import test from "node:test";
import assert from "node:assert/strict";
import { watchBootMarker } from "./bootWatch.ts";
import { bootBegin, bootComplete, armBoot, heartbeat, shouldRunSafe, mark, KEYS, type Storage } from "./blackbox.ts";

function mem(): Storage {
  const m = new Map<string, string>();
  return { getItem: (k) => (m.has(k) ? m.get(k)! : null), setItem: (k, v) => void m.set(k, v), removeItem: (k) => void m.delete(k) };
}

function page(store: Storage, opts: { mapMounted?: () => boolean } = {}) {
  const on: Record<string, Array<() => void>> = {};
  const timers: Array<{ fn: () => void; ms: number }> = [];
  const doc = {
    visibilityState: "visible",
    addEventListener: (t: "visibilitychange", fn: () => void) => { (on[t] ||= []).push(fn); },
  };
  const win = {
    addEventListener: (t: "pagehide", fn: () => void) => { (on[t] ||= []).push(fn); },
    setTimeout: (fn: () => void, ms: number) => { timers.push({ fn, ms }); return timers.length; },
  };
  watchBootMarker({ store, mapMounted: opts.mapMounted ?? (() => false), windowMs: 20_000, win, doc });
  return {
    fire: (t: string) => (on[t] || []).forEach((f) => f()),
    hide: () => { doc.visibilityState = "hidden"; (on.visibilitychange || []).forEach((f) => f()); },
    show: () => { doc.visibilityState = "visible"; (on.visibilitychange || []).forEach((f) => f()); },
    timers,
  };
}

test("leaving ANY page gracefully disarms the marker (it was only handled while the map was mounted)", () => {
  const s = mem();
  bootBegin(s, 0, "b1");                         // pricing page: armed by the module
  const p = page(s);
  p.fire("pagehide");
  assert.equal(s.getItem(KEYS.IN_PROGRESS), null);
  assert.equal(shouldRunSafe(bootBegin(s, 60_000, "b2")), false);
});

test("a page sent to the background with no map mounted stands down; coming back does not re-arm it", () => {
  const s = mem();
  bootBegin(s, 0, "b1");
  const p = page(s);
  p.show();
  assert.ok(s.getItem(KEYS.IN_PROGRESS), "becoming visible is not a reason to stand down");
  p.hide();
  assert.equal(s.getItem(KEYS.IN_PROGRESS), null);
  p.show();
  assert.equal(s.getItem(KEYS.IN_PROGRESS), null);
  // the tab is discarded here with no pagehide: the next visit is clean
  const r2 = bootBegin(s, 60_000, "b2");
  assert.equal(r2.prevCrashed, false);
  assert.equal(shouldRunSafe(r2), false);
});

test("a page that lives the whole start-up window with no map mounted stands down, on ONE timer of that length", () => {
  const s = mem();
  bootBegin(s, 0, "b1");
  const p = page(s);
  assert.deepEqual(p.timers.map((t) => t.ms), [20_000]);
  assert.ok(s.getItem(KEYS.IN_PROGRESS), "still armed inside the window: the map may be about to mount");
  p.timers[0].fn();
  assert.equal(s.getItem(KEYS.IN_PROGRESS), null);
});

test("while the map IS mounted nothing stands down: hidden or not, a death during its start is still a start-up death", () => {
  const s = mem();
  bootBegin(s, 0, "b1");
  let mounted = true;
  const p = page(s, { mapMounted: () => mounted });
  mark(s, 0, 900, "map-create");
  p.hide();
  p.timers[0].fn();
  assert.ok(s.getItem(KEYS.IN_PROGRESS), "the marker must survive: this is the case it exists for");
  // ...renderer killed
  const r2 = bootBegin(s, 60_000, "b2");
  assert.equal(r2.prevCrashed, true);
  assert.ok(shouldRunSafe(r2));
  // and once the map is unmounted the same page is an ordinary page again
  mounted = false;
  const p2 = page(s, { mapMounted: () => mounted });
  p2.hide();
  assert.equal(s.getItem(KEYS.IN_PROGRESS), null);
});

test("standing down a healthy session's page does not disturb its heartbeat record or invent a debt", () => {
  const s = mem();
  bootBegin(s, 0, "b1");
  armBoot(s, 0, "b1");
  bootComplete(s);
  heartbeat(s, 30_000, "healthy");
  const p = page(s);                             // (the map was unmounted without its cleanup having run yet)
  p.hide();
  assert.ok(s.getItem(KEYS.HEARTBEAT), "the late-death record is the heartbeat's business, not the stand-down's");
  assert.equal(s.getItem(KEYS.OWED), null);
});
