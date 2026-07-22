// siteImagery.test.ts — DATACORE MAXIMUS Phase 3b: latest-Sentinel-2-chip
// merge into /api/data/sites. Pure function, no IO.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mergeSiteImagery } from "./siteImagery";

const SITES = [
  { id: "cushing_hub", name: "Cushing Oil Hub", category: "tank_farm" },
  { id: "port_la", name: "Port of LA", category: "port" },
  { id: "port_lb", name: "Port of Long Beach", category: "port" },
];

const CHIP = {
  file: "/imagery/sites/cushing_hub.jpg",
  scene: "S2B_14SPE_20260707_0_L2A",
  date: "2026-07-07",
  cloud_pct: 17.1,
  bbox: [-96.77, 35.94, -96.75, 35.96] as [number, number, number, number],
  attribution: "Contains modified Copernicus Sentinel data",
  pulled_at: "2026-07-22T13:19:34Z",
};

test("attaches imagery only to sites present in the chip index", () => {
  const out = mergeSiteImagery(SITES, { cushing_hub: CHIP });
  assert.equal((out[0] as any).imagery, CHIP);
  assert.equal((out[1] as any).imagery, undefined);
  assert.equal((out[2] as any).imagery, undefined);
});

test("a site never pulled yet gets no fabricated placeholder", () => {
  const out = mergeSiteImagery(SITES, {});
  for (const s of out) assert.equal((s as any).imagery, undefined);
});

test("never mutates the input site objects", () => {
  const original = SITES[0];
  const out = mergeSiteImagery(SITES, { cushing_hub: CHIP });
  assert.notEqual(out[0], original);
  assert.equal((original as any).imagery, undefined);
});

test("all other site fields pass through unchanged", () => {
  const out = mergeSiteImagery(SITES, { port_la: CHIP });
  assert.equal((out[1] as any).name, "Port of LA");
  assert.equal((out[1] as any).category, "port");
});
