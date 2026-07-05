// Port dwell analytics — synthetic-archive tests (hermetic tmpdir; same JSONL
// format datacoreArchive writes, same harness pattern as shadowFleet.test.ts).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  portsFromSites, assignPort, detectVisits, computePortDwell, PortDef,
} from "./portDwell";

const here = path.dirname(fileURLToPath(import.meta.url));
const NOW = Date.UTC(2026, 6, 4, 12, 0, 0);
const NOW_SEC = Math.floor(NOW / 1000);

const LA: PortDef = { id: "port_la", name: "Port of Los Angeles", lat: 33.74, lon: -118.272, radius_km: 5 };
const LB: PortDef = { id: "port_lb", name: "Port of Long Beach", lat: 33.7515, lon: -118.213, radius_km: 5 };

function writeArchive(base: string, points: Array<Record<string, any>>) {
  const dir = path.join(base, "vessels");
  fs.mkdirSync(dir, { recursive: true });
  const byHour = new Map<string, string[]>();
  for (const p of points) {
    const d = new Date(p.t * 1000).toISOString();
    const f = `${d.slice(0, 10)}-${d.slice(11, 13)}.jsonl`;
    if (!byHour.has(f)) byHour.set(f, []);
    byHour.get(f)!.push(JSON.stringify(p));
  }
  for (const [f, lines] of byHour) fs.writeFileSync(path.join(dir, f), lines.join("\n") + "\n");
}

const t = (hoursAgo: number) => Math.floor((NOW - hoursAgo * 3600_000) / 1000);

// Moored-ship track inside a fence: n points, `stepH` apart, ~0 speed.
const moored = (mmsi: string, name: string, port: PortDef, fromH: number, toH: number, stepH = 1, v = 0.2) => {
  const pts: Array<Record<string, any>> = [];
  for (let h = fromH; h >= toH; h -= stepH) {
    pts.push({ t: t(h), i: mmsi, c: name, la: port.lat + 0.005, lo: port.lon + 0.005, v });
  }
  return pts;
};

test("visit detection: moored ship in fence = one completed visit with correct dwell", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-dwell-"));
  writeArchive(base, moored("101", "BOXSHIP", LA, 40, 20)); // 20h call, left 20h ago
  const s = computePortDwell([LA, LB], 168, base, NOW);
  const la = s.ports.find((p) => p.id === "port_la")!;
  assert.equal(la.visits_completed, 1);
  assert.equal(la.in_port_now, 0);
  assert.ok(Math.abs((la.dwell_median_h ?? 0) - 20) < 0.2, `median ${la.dwell_median_h} ≠ ~20h`);
});

test("right-censoring: ship still in port counts as in_port_now, not in the dwell distribution", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-dwell-rc-"));
  writeArchive(base, moored("102", "STAYER", LA, 30, 1)); // last point 1h ago -> ongoing
  const s = computePortDwell([LA, LB], 168, base, NOW);
  const la = s.ports.find((p) => p.id === "port_la")!;
  assert.equal(la.in_port_now, 1);
  assert.equal(la.visits_completed, 0);
  assert.equal(la.dwell_median_h, null);
});

test("overlapping fences: nearest-port assignment separates LA from Long Beach", () => {
  // Point at LB's exact terminal is inside both 5km fences at the midline edge —
  // nearest wins.
  assert.equal(assignPort(LB.lat, LB.lon, [LA, LB]), "port_lb");
  assert.equal(assignPort(LA.lat, LA.lon, [LA, LB]), "port_la");
  assert.equal(assignPort(0, 0, [LA, LB]), null);
});

test("speed filter: harbor craft with high median SOG is not a port call; brief transits filtered by span/points", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-dwell-tug-"));
  writeArchive(base, [
    ...moored("103", "TUG", LA, 30, 20, 1, 7),          // 10h in fence but median 7kts
    { t: t(10), i: "104", c: "PASSER", la: LA.lat, lo: LA.lon, v: 12 },
    { t: t(9.8), i: "104", c: "PASSER", la: LA.lat + 0.01, lo: LA.lon + 0.01, v: 12 },
  ]);
  const s = computePortDwell([LA, LB], 168, base, NOW);
  const la = s.ports.find((p) => p.id === "port_la")!;
  assert.equal(la.visits_completed + la.in_port_now, 0);
});

test("anomaly honesty: 3x-median flags only appear at >=10 completed visits", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-dwell-anom-"));
  // 9 normal completed calls (~10h) + 1 monster (40h): 10 total -> anomaly may fire
  const pts: Array<Record<string, any>> = [];
  for (let i = 0; i < 9; i++) {
    const start = 160 - i * 14;
    pts.push(...moored(`20${i}`, `SHIP${i}`, LA, start, start - 10, 2));
  }
  pts.push(...moored("299", "MONSTER", LA, 130, 90, 2)); // 40h dwell, done 90h ago
  writeArchive(base, pts);
  const s = computePortDwell([LA], 168, base, NOW);
  const la = s.ports[0];
  assert.equal(la.visits_completed, 10);
  assert.equal(la.anomaly_count, 1);
  assert.equal(la.anomaly_examples[0].mmsi, "299");
  assert.ok(la.anomaly_examples[0].dwell_h >= 3 * (la.dwell_median_h ?? Infinity));

  // Same monster with only 3 normal calls -> suppressed (thin median)
  const base2 = fs.mkdtempSync(path.join(os.tmpdir(), "vt-dwell-anom2-"));
  const pts2: Array<Record<string, any>> = [];
  for (let i = 0; i < 3; i++) {
    const start = 90 - i * 14;
    pts2.push(...moored(`30${i}`, `S${i}`, LA, start, start - 10, 2));
  }
  pts2.push(...moored("399", "MONSTER2", LA, 160, 120, 2));
  writeArchive(base2, pts2);
  const s2 = computePortDwell([LA], 168, base2, NOW);
  assert.equal(s2.ports[0].anomaly_count, 0);
});

test("coverage gap inside a call splits the visit (dwell is a lower bound, never inflated)", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-dwell-gap-"));
  writeArchive(base, [
    ...moored("105", "GAPPY", LA, 60, 50), // 10h
    // 8h archive hole (> maxGapHours 6) while presumably still moored
    ...moored("105", "GAPPY", LA, 42, 32), // 10h
  ]);
  const s = computePortDwell([LA], 168, base, NOW);
  assert.equal(s.ports[0].visits_completed, 2);
  assert.ok((s.ports[0].dwell_max_h ?? 0) < 12, "gap must split, not bridge, the visit");
});

test("wiring pinned: ports come from the verified sites registry; route + layer registered; caveat honest", () => {
  const sites = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "sites", "strategic_sites.json"), "utf8"));
  const ports = portsFromSites(sites.sites);
  assert.equal(ports.length, 9, "expected the 9 imagery-verified ports");
  assert.ok(ports.every((p) => p.radius_km > 0 && Number.isFinite(p.lat) && Number.isFinite(p.lon)));

  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("/api/data/portdwell"), "route missing");
  const layers = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "layers.json"), "utf8"));
  const l = layers.layers.find((x: any) => x.id === "portdwell");
  assert.ok(l && l.kind === "raw", "layers.json entry missing/not raw");
  assert.ok(/lower bound|gate/i.test(l.description), "user-facing description must carry the honesty caveats");

  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-dwell-cav-"));
  const s = computePortDwell(ports, 168, base, NOW);
  assert.ok(s.caveat.includes("lower bounds"), "caveat must state the lower-bound property");
  assert.ok(s.caveat.includes("gate 2"), "caveat must state the signal gate");
});

test("RATCHET [REPAIR 2026-07-05]: async dwell stats identical to sync (event-loop defect, 4th site)", async () => {
  const { computePortDwellAsync } = await import("./portDwell");
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-dwell-"));
  writeArchive(base, moored("555", "BOXSHIP", LA, 40, 20));
  const a = await computePortDwellAsync([LA, LB], 168, base, NOW);
  const s = computePortDwell([LA, LB], 168, base, NOW);
  assert.deepEqual(a, s, "async streaming dwell stats must equal the sync scan");
});
