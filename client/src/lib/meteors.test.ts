import { test } from "node:test";
import assert from "node:assert/strict";
import {
  meteorSeverity, compassPoint, meteorIconSize, meteorStreak,
  meteorRegion, meteorCoverageLinks, siteLocalTime,
} from "./meteors";

test("severity ramp matches the approved mock: blue < 0.1, amber to 1, red ≥ 1 kt", () => {
  assert.equal(meteorSeverity(0.05).key, "minor");
  assert.equal(meteorSeverity(0.05).color, "#7cc4ff");
  assert.equal(meteorSeverity(0.5).key, "moderate");
  assert.equal(meteorSeverity(1.6).key, "major");
  assert.equal(meteorSeverity(1.6).color, "#ff5a6e");
  assert.match(meteorSeverity(2).label, /Hiroshima/);
});

test("compass wording", () => {
  assert.equal(compassPoint(0), "N");
  assert.equal(compassPoint(347), "NNW");
  assert.equal(compassPoint(90), "E");
  assert.equal(compassPoint(359), "N");
  assert.equal(compassPoint(-13), "NNW");
});

test("icon size is log-bounded — Chelyabinsk-class cannot dwarf the map", () => {
  assert.ok(meteorIconSize(0) >= 0.3 && meteorIconSize(0) < 0.4);
  assert.ok(meteorIconSize(65.8) > meteorIconSize(3));
  assert.ok(meteorIconSize(10_000) <= 1.05);
});

test("streak: tail behind the travel direction, never invented without a heading", () => {
  assert.equal(meteorStreak(30.9, 131.8, null, 19.6), null);
  const s = meteorStreak(30.9, 131.8, 347.2, 19.6)!;
  const [tail, head] = s;
  assert.deepEqual(head, [131.8, 30.9], "head = the burst point");
  // traveling NNW ⇒ arrived from SSE ⇒ tail is SOUTH and slightly EAST
  assert.ok(tail[1] < 30.9, "tail south of the burst");
  assert.ok(tail[0] > 131.8, "tail east of the burst");
  const s2 = meteorStreak(89.5, 0, 90, 20)!;
  assert.ok(s2[0][1] <= 89, "polar tail clamps inside the map");
});

test("region keywords from the tz database; open ocean honestly yields none", () => {
  assert.equal(meteorRegion(30.9, 131.8), "Tokyo");
  assert.equal(meteorRegion(42.0, -70.5), "New York");
  assert.equal(meteorRegion(0, -30), null, "nautical Etc zone → no fake region");
  assert.equal(meteorRegion(-30, -140), "Gambier", "remote Pacific still inside a real zone — usable keyword");
});

test("coverage links: AMS ±1-day window, searches carry date + region, CNEOS provenance", () => {
  const links = meteorCoverageLinks("2025-08-19 14:08:48", 30.9, 131.8);
  assert.equal(links.length, 4);
  const ams = links[0];
  assert.match(ams.href, /amsmeteors\.org\/fireballs\/\?start_date=2025-08-18&end_date=2025-08-20/);
  assert.match(links[1].href, /news\.google\.com\/search\?q=meteor%20fireball%20Tokyo%202025-08-19/);
  assert.match(links[2].href, /youtube\.com\/results\?search_query=/);
  assert.match(links[3].href, /cneos\.jpl\.nasa\.gov/);
  // ocean event: searches run date-only (no fabricated region)
  const ocean = meteorCoverageLinks("2026-08-01 17:43:48", 0, -30);
  assert.ok(ocean[1].href.includes("2026-08-01"));
});

test("site local time via the tz database; nautical zones omit rather than fake", () => {
  // 2025-08-19 14:08 UTC at 30.9N 131.8E = 23:08 JST (the mock's card line)
  const t = Math.floor(Date.parse("2025-08-19T14:08:48Z") / 1000);
  const s = siteLocalTime(t, 30.9, 131.8);
  assert.ok(s && s.includes("23:08"), String(s));
  assert.equal(siteLocalTime(t, 0, -30), null);
});
