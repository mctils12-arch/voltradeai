import { test } from "node:test";
import assert from "node:assert/strict";
import {
  meteorSeverity, compassPoint, meteorIconSize, meteorStreak,
  meteorRegion, meteorCoverageLinks, meteorCoverageVerdict,
  naturalDate, naturalMonthYear, nightAtSite, fmtBlastAlt, fmtEntrySpeed,
  siteLocalTime,
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

test("links v2 — land + recent: AMS ±1 day, loose-worded videos, News, CNEOS", () => {
  const NOW = Date.parse("2025-09-01T00:00:00Z");
  const links = meteorCoverageLinks("2025-08-19 14:08:48", 30.9, 131.8, NOW);
  assert.equal(links.length, 4);
  assert.match(links[0].href, /amsmeteors\.org\/fireballs\/\?start_date=2025-08-18&end_date=2025-08-20/);
  // video search uses the LOOSE month-year wording titles actually use
  assert.match(links[1].href, /youtube\.com\/results\?search_query=meteor%20Tokyo%20August%202025/);
  // recent event → Google News still has it
  assert.match(links[2].href, /news\.google\.com\/search\?q=meteor%20fireball%20Tokyo%20August%2019%202025/);
  assert.match(links[3].href, /cneos\.jpl\.nasa\.gov/);
});

test("links v2 — old events skip the decayed news index for the general web (the 2016 report)", () => {
  const NOW = Date.parse("2026-08-13T00:00:00Z");
  const links = meteorCoverageLinks("2016-08-05 18:02:00", 30.9, 131.8, NOW);
  const text = links[2];
  assert.match(text.href, /www\.google\.com\/search\?q=meteor%20fireball%20Tokyo%20August%205%202016/);
  assert.match(text.label, /news indexes decay/);
  assert.ok(!links.some((l) => l.href.includes("news.google.com")), "no dead News link for a 2016 event");
});

test("links v2 — ocean events collapse to the record + one 'anyway' search, never dead chips", () => {
  const ocean = meteorCoverageLinks("2016-08-05 18:02:00", 0, -30);
  assert.equal(ocean.length, 2);
  assert.match(ocean[0].href, /cneos\.jpl\.nasa\.gov/);
  assert.match(ocean[1].label, /anyway/);
  assert.match(ocean[1].href, /www\.google\.com\/search\?q=meteor%20fireball%20August%205%202016/);
});

test("naturalDate: human wording, never the machine form that returned nothing", () => {
  assert.equal(naturalDate("2016-08-05 18:02:00"), "August 5 2016");
  assert.equal(naturalMonthYear("2016-08-05 18:02:00"), "August 2016");
});

test("nightAtSite: plain solar test — 18:02 UTC is mid-day at 174°W, night at 131°E", () => {
  const t = Math.floor(Date.parse("2016-08-05T18:02:00Z") / 1000);
  assert.equal(nightAtSite(t, -174.4), false, "~6:26 solar AM edge...actually daytime side");
  assert.equal(nightAtSite(t, 131.8), true, "~2:49 solar next-day = dark");
});

test("verdict: ocean unlikely, land night likely, land day possible", () => {
  const t = Math.floor(Date.parse("2025-08-19T14:08:48Z") / 1000);
  assert.equal(meteorCoverageVerdict(t, 0, -30).key, "unlikely");
  const jp = meteorCoverageVerdict(t, 30.9, 131.8);
  assert.equal(jp.key, "likely");
  assert.match(jp.label, /Tokyo/);
  const day = meteorCoverageVerdict(t, 42.0, -70.5); // 14:08 UTC ≈ 09:26 solar at 70.5W
  assert.equal(day.key, "possible");
});

test("compact stat formats fit the chip row (the 106299… truncation fix)", () => {
  assert.equal(fmtBlastAlt(32.4, "imperial"), "106k ft");
  assert.equal(fmtBlastAlt(32.4, "metric"), "32 km");
  assert.equal(fmtBlastAlt(null, "imperial"), "—");
  assert.equal(fmtEntrySpeed(19.6, "imperial"), "44k mph");
  assert.equal(fmtEntrySpeed(19.6, "metric"), "20 km/s");
  assert.equal(fmtEntrySpeed(null, "metric"), "—");
});

test("site local time via the tz database; nautical zones omit rather than fake", () => {
  // 2025-08-19 14:08 UTC at 30.9N 131.8E = 23:08 JST (the mock's card line)
  const t = Math.floor(Date.parse("2025-08-19T14:08:48Z") / 1000);
  const s = siteLocalTime(t, 30.9, 131.8);
  assert.ok(s && s.includes("23:08"), String(s));
  assert.equal(siteLocalTime(t, 0, -30), null);
});
