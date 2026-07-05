// CBP border-wait battery (BUILD ORDER 5 #5): live-shape parse with
// locale-independent keying, absent-lane honesty, change-only dedup.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  parseBorderWaits, fetchBorderWaits, archiveBorderWaits,
  gzipOldBorderWaitDays, refreshBorderWaits, latestBorderWaits,
} from "./cbpBorderWait";

// Trimmed from the LIVE response captured 2026-07-05 — including the
// Spanish localization our probe egress received (status strings are
// archived verbatim; parsing never matches on them).
const REC = {
  port_number: "230401", border: "Frontera mexicana", port_name: "Laredo",
  crossing_name: "Bridge I", date: "7/5/2026", time: "15:07:51", port_status: "Abierto",
  commercial_vehicle_lanes: {
    standard_lanes: { update_time: "En 2:00 pm CDT", operational_status: "demora", delay_minutes: "45", lanes_open: "3" },
    FAST_lanes: { update_time: "", operational_status: "N/A", delay_minutes: "", lanes_open: "" },
  },
  passenger_vehicle_lanes: {
    standard_lanes: { update_time: "", operational_status: "Carriles cerrados", delay_minutes: "", lanes_open: "" },
  },
};

test("parseBorderWaits: locale-independent fields parsed, localized strings verbatim", () => {
  const obs = parseBorderWaits([REC], "2026-07-05T20:10:00Z");
  assert.equal(obs.length, 3, "three lane classes flattened");
  const cv = obs.find((o) => o.lane === "commercial_standard")!;
  assert.equal(cv.delay_min, 45);
  assert.equal(cv.lanes_open, 3);
  assert.equal(cv.status, "demora", "status archived verbatim, never translated");
  assert.equal(cv.port_number, "230401");
  const fast = obs.find((o) => o.lane === "commercial_FAST")!;
  assert.equal(fast.delay_min, null, "empty delay is null, never zero");
  assert.deepEqual(parseBorderWaits(null, "x"), []);
  assert.deepEqual(parseBorderWaits([{ no_port: true }], "x"), []);
});

test("absent lane blocks contribute nothing (many crossings have no commercial lanes)", () => {
  const pedOnly = { port_number: "1", port_name: "X", crossing_name: "Y", border: "B",
                    passenger_vehicle_lanes: { standard_lanes: { operational_status: "open", delay_minutes: "5", lanes_open: "2" } } };
  const obs = parseBorderWaits([pedOnly], "x");
  assert.equal(obs.length, 1);
  assert.equal(obs[0].lane, "passenger_standard");
});

test("archive: change-only dedup — identical snapshot re-poll writes nothing; a delay change appends", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "bwt-"));
  const now = Date.parse("2026-07-05T20:10:00Z");
  assert.equal(archiveBorderWaits(parseBorderWaits([REC], "2026-07-05T20:10:00Z"), base, now), 3);
  assert.equal(archiveBorderWaits(parseBorderWaits([REC], "2026-07-05T21:10:00Z"), base, now), 0,
    "unchanged waits re-polled an hour later never re-archive");
  const worse = JSON.parse(JSON.stringify(REC));
  worse.commercial_vehicle_lanes.standard_lanes.delay_minutes = "90";
  assert.equal(archiveBorderWaits(parseBorderWaits([worse], "2026-07-05T22:10:00Z"), base, now), 1,
    "only the changed lane appends");
  assert.equal(gzipOldBorderWaitDays(base, now + 3 * 86400_000), 1);
});

test("refresh: transport error keeps the last snapshot", async () => {
  const ok = async () => ({ ok: true, status: 200, text: async () => JSON.stringify([REC]) });
  await refreshBorderWaits(ok as any, Date.parse("2026-07-05T20:10:00Z"));
  const hit = latestBorderWaits();
  assert.equal(hit!.obs.length, 3);
  const err = async () => ({ ok: false, status: 503, text: async () => "" });
  await refreshBorderWaits(err as any);
  assert.equal(latestBorderWaits(), hit);
  assert.equal(await fetchBorderWaits(err as any), null);
});
