// BUG 4 regression: every listener attachLayerInteractions adds must be removed
// by the returned detach(), with the SAME handler reference (map.off is a no-op
// otherwise) — so toggling a layer off/on never stacks handlers.
// Run: npx tsx --test client/src/lib/mapInteractions.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { attachLayerInteractions, type InteractiveMap } from "./mapInteractions.ts";

function fakeMap() {
  const on: Array<[string, string, (e: any) => void]> = [];
  const off: Array<[string, string, (e: any) => void]> = [];
  const map: InteractiveMap = {
    on: (t, l, h) => { on.push([t, l, h]); },
    off: (t, l, h) => { off.push([t, l, h]); },
    getCanvas: () => ({ style: { cursor: "" } }),
  };
  return { map, on, off };
}

// net live listeners = attached minus detached, matched by (type, layer, handler ref)
function net(on: any[], off: any[]) {
  const key = (x: any[]) => `${x[0]}|${x[1]}|${x[2] === undefined ? "?" : "fn" + x.__i}`;
  // match by identity: for each off, remove one matching on
  const live = [...on];
  for (const o of off) {
    const idx = live.findIndex((a) => a[0] === o[0] && a[1] === o[1] && a[2] === o[2]);
    if (idx >= 0) live.splice(idx, 1);
  }
  void key;
  return live;
}

test("attaches click+enter+leave per layer, detach removes exactly those (same refs)", () => {
  const { map, on, off } = fakeMap();
  const onClick = () => {};
  const detach = attachLayerInteractions(map, "layer-x", onClick);
  assert.equal(on.length, 3, "click + mouseenter + mouseleave attached");
  assert.deepEqual(on.map((x) => x[0]).sort(), ["click", "mouseenter", "mouseleave"]);
  // since 2026-07-21 the click handler is a claim-guard WRAPPER (aircraft
  // click precedence), so it is deliberately NOT the caller's raw ref —
  // what matters is that detach removes the SAME wrapper (net() below)
  assert.ok(on.some((x) => x[0] === "click" && typeof x[2] === "function"), "a click handler is attached");
  detach();
  assert.equal(off.length, 3, "detach removed all three");
  assert.equal(net(on, off).length, 0, "no live listeners remain after detach (same handler refs)");
});

test("multiple layers: all bound and all detached", () => {
  const { map, on, off } = fakeMap();
  const detach = attachLayerInteractions(map, ["a", "b"], () => {});
  assert.equal(on.length, 6, "3 handlers x 2 layers");
  detach();
  assert.equal(net(on, off).length, 0, "every listener removed");
});

test("simulated toggle cycles do not stack: net live stays 3 per enable", () => {
  const { map, on, off } = fakeMap();
  // 3 off/on cycles: attach, detach, attach, detach, attach (end enabled)
  let detach = attachLayerInteractions(map, "L", () => {});
  for (let i = 0; i < 2; i++) { detach(); detach = attachLayerInteractions(map, "L", () => {}); }
  // one more detach+attach to be at "enabled" with a single live set
  detach(); detach = attachLayerInteractions(map, "L", () => {});
  const live = net(on, off);
  assert.equal(live.length, 3, "exactly one live click+enter+leave set, never stacked");
  assert.equal(live.filter((x) => x[0] === "click").length, 1, "exactly ONE live click handler");
});

test("aircraft click claim: a claimed event never reaches the ground handler; unclaimed passes (2026-07-21)", () => {
  const handlers: Record<string, Array<(e: any) => void>> = {};
  const map = {
    on(t: string, id: string, h: (e: any) => void) { (handlers[`${t}:${id}`] ??= []).push(h); },
    off() {},
    getCanvas() { return { style: { cursor: "" } }; },
  };
  let clicks = 0;
  attachLayerInteractions(map as any, "sites-layer", () => { clicks++; });
  const fire = (e: any) => handlers["click:sites-layer"].forEach((h) => h(e));
  fire({ originalEvent: {} });
  assert.equal(clicks, 1, "unclaimed click reaches the ground handler");
  fire({ originalEvent: { __vtAirClaim: true } });
  assert.equal(clicks, 1, "a plane-claimed click never opens the ground card");
  fire({}); // no originalEvent (programmatic) — passes through
  assert.equal(clicks, 2);
});
