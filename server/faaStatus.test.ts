// FAA status battery (BUILD ORDER 5 #4): live-shape XML parse, event
// dedup semantics (persisting program archives once, changed numbers
// append), empty-NAS honesty, gz lifecycle.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  parseFaaStatus, fetchFaaStatus, archiveFaaEvents, gzipOldFaaDays,
  refreshFaaStatus, latestFaaStatus,
} from "./faaStatus";

// Trimmed from the LIVE response captured 2026-07-05 (thunderstorm day).
const XML = `<AIRPORT_STATUS_INFORMATION><Update_Time>Sun Jul 5 20:08:26 2026 GMT</Update_Time>
<Delay_type><Name>Ground Stop Programs</Name><Ground_Stop_List>
<Program><ARPT>DCA</ARPT><Reason>thunderstorms</Reason><End_Time>5:00 pm EDT</End_Time></Program>
</Ground_Stop_List></Delay_type>
<Delay_type><Name>Ground Delay Programs</Name><Ground_Delay_List>
<Ground_Delay><ARPT>JFK</ARPT><Reason>thunderstorms</Reason><Avg>2 hours and 30 minutes</Avg><Max>7 hours and 35 minutes</Max></Ground_Delay>
</Ground_Delay_List></Delay_type>
<Delay_type><Name>General Arrival/Departure Delay Info</Name><Arrival_Departure_Delay_List>
<Delay><ARPT>LGA</ARPT><Reason>TM Initiatives:SWAP:WX</Reason><Arrival_Departure Type="Departure"><Min>31 minutes</Min><Max>45 minutes</Max><Trend>Increasing</Trend></Arrival_Departure></Delay>
</Arrival_Departure_Delay_List></Delay_type>
<Delay_type><Name>Airport Closures</Name><Airport_Closure_List>
<Airport><ARPT>ASE</ARPT><Reason>snow removal</Reason><Reopen>6:00 am MST</Reopen></Airport>
</Airport_Closure_List></Delay_type></AIRPORT_STATUS_INFORMATION>`;

test("parseFaaStatus: all four event families from the live shape", () => {
  const ev = parseFaaStatus(XML, "2026-07-05T20:10:00Z");
  assert.equal(ev.length, 4);
  const gs = ev.find((e) => e.type === "ground_stop")!;
  assert.deepEqual([gs.airport, gs.reason, gs.end_time], ["DCA", "thunderstorms", "5:00 pm EDT"]);
  const gd = ev.find((e) => e.type === "ground_delay")!;
  assert.deepEqual([gd.airport, gd.avg, gd.max], ["JFK", "2 hours and 30 minutes", "7 hours and 35 minutes"]);
  const d = ev.find((e) => e.type === "delay")!;
  assert.deepEqual([d.airport, d.direction, d.trend, d.min], ["LGA", "Departure", "Increasing", "31 minutes"]);
  const c = ev.find((e) => e.type === "closure")!;
  assert.deepEqual([c.airport, c.reopen], ["ASE", "6:00 am MST"]);
  assert.ok(ev.every((e) => e.update_time.includes("Jul 5")));
  assert.deepEqual(parseFaaStatus("", "x"), []);
  assert.deepEqual(parseFaaStatus("<html>maintenance</html>", "x"), [], "non-feed body = empty");
});

test("empty NAS (no programs) is a real state, not an error", async () => {
  const empty = `<AIRPORT_STATUS_INFORMATION><Update_Time>Sun Jul 5 22:00:00 2026 GMT</Update_Time></AIRPORT_STATUS_INFORMATION>`;
  const ok = async () => ({ ok: true, status: 200, text: async () => empty });
  await refreshFaaStatus(ok as any, Date.parse("2026-07-05T22:00:00Z"));
  const hit = latestFaaStatus();
  assert.ok(hit, "empty snapshot still cached");
  assert.equal(hit!.events.length, 0);
  const err = async () => ({ ok: false, status: 503, text: async () => "" });
  await refreshFaaStatus(err as any);
  assert.equal(latestFaaStatus(), hit, "transport error keeps the last snapshot");
});

test("archive: persisting program archives ONCE; changed numbers append as new state", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "faa-"));
  const now = Date.parse("2026-07-05T20:10:00Z");
  const ev = parseFaaStatus(XML, "2026-07-05T20:10:00Z");
  assert.equal(archiveFaaEvents(ev, base, now), 4);
  // same programs re-polled 15 min later — identical identities, nothing new
  const again = parseFaaStatus(XML, "2026-07-05T20:25:00Z");
  assert.equal(archiveFaaEvents(again, base, now), 0);
  // the JFK GDP worsens — new identity, appends
  const worse = parseFaaStatus(XML.replace("2 hours and 30 minutes", "3 hours and 10 minutes"), "2026-07-05T20:40:00Z");
  assert.equal(archiveFaaEvents(worse, base, now), 1, "only the changed program appends");
  assert.equal(gzipOldFaaDays(base, now + 3 * 86400_000), 1);
});

test("fetchFaaStatus: non-200 -> null (transport), 200 -> parsed", async () => {
  assert.equal(await fetchFaaStatus((async () => ({ ok: false, status: 500, text: async () => "" })) as any), null);
  const ev = await fetchFaaStatus((async () => ({ ok: true, status: 200, text: async () => XML })) as any);
  assert.equal(ev!.length, 4);
});
