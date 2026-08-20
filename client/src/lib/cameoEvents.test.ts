import { test } from "node:test";
import assert from "node:assert/strict";
import {
  decodeCode,
  goldsteinAudit,
  groupByArticle,
  hostOf,
  eventDistanceKm,
  haversineKm,
  siteInfo,
  typeSummary,
  CAMEO_GOLDSTEIN,
  CAMEO_EVENT_LABEL,
  type GdeltEventRow,
} from "./cameoEvents";

// Live rows from /api/data/facility-events on 2026-08-20, trimmed but not
// reshaped: two rows of one article at port_la (downtown LA and Boyle Heights,
// both matched to the port), two rows of another at port_nynj, and one
// single-row article at port_houston.
const LIVE: GdeltEventRow[] = [
  { id: "1319205286", day: "20260820", code: "192", root: "19", gold: -9.5, tone: -4.5631, mentions: 7, lat: 34.0339, lon: -118.205, site: "port_la", url: "https://laist.com/news/climate-environment/lineage-faces-key-deadline-today", rt: "2026-08-20T19:13:56.905Z" },
  { id: "1319205287", day: "20260820", code: "192", root: "19", gold: -9.5, tone: -4.5631, mentions: 3, lat: 34.0522, lon: -118.244, site: "port_la", url: "https://laist.com/news/climate-environment/lineage-faces-key-deadline-today", rt: "2026-08-20T19:13:56.905Z" },
  { id: "1319205425", day: "20260820", code: "173", root: "17", gold: -5, tone: -3.6585, mentions: 3, lat: 40.314, lon: -74.5089, site: "port_nynj", url: "https://www.dailymail.com/news/article-16068005/scots-fraudster.html", rt: "2026-08-20T19:13:56.905Z" },
  { id: "1319205426", day: "20260820", code: "173", root: "17", gold: -5, tone: -3.6585, mentions: 1, lat: 40.314, lon: -74.5089, site: "port_nynj", url: "https://www.dailymail.com/news/article-16068005/scots-fraudster.html", rt: "2026-08-20T19:13:56.905Z" },
  { id: "1319208125", day: "20260820", code: "172", root: "17", gold: -5, tone: -7.6376, mentions: 10, lat: 29.7052, lon: -95.1238, site: "port_houston", url: "https://www.houstonpublicmedia.org/articles/police/2026/08/20/559941/deer-park.html", rt: "2026-08-20T19:28:56.910Z" },
];

// ── the decode table is a lookup, never an inference ─────────────────────────

test("known codes decode to the published label, root label and Goldstein constant", () => {
  const d = decodeCode("1712", "17");
  assert.equal(d.label, "Destroy property");
  assert.equal(d.rootLabel, "Coerce");
  assert.equal(d.goldstein, -9.2);
  assert.equal(decodeCode("1431", "14").label, "Conduct strike or boycott for leadership change");
  assert.equal(decodeCode("192", "19").rootLabel, "Fight");
});

test("an unknown code is null, never guessed from its root", () => {
  // The one failure mode a decode table may not have: inventing a plausible
  // label for a code upstream added after this table was transcribed.
  const d = decodeCode("1799", "17");
  assert.equal(d.label, null);
  assert.equal(d.goldstein, null);
  assert.equal(d.rootLabel, "Coerce"); // the root IS known — that much is real
  assert.equal(decodeCode("", "").label, null);
  assert.equal(decodeCode(null).rootLabel, null);
});

test("the root falls back to the code's own prefix only when the row omits it", () => {
  assert.equal(decodeCode("173", "17").root, "17");
  assert.equal(decodeCode("173", null).root, "17");
  // A row whose root disagrees with its code is reported as sent, not silently
  // corrected — a disagreement is information about the feed.
  assert.equal(decodeCode("173", "19").root, "19");
});

test("Goldstein differs between codes sharing a root, so it is a real per-code table", () => {
  assert.notEqual(CAMEO_GOLDSTEIN["1712"], CAMEO_GOLDSTEIN["173"]);
  assert.equal(CAMEO_GOLDSTEIN["1712"], -9.2);
  assert.equal(CAMEO_GOLDSTEIN["173"], -5);
});

test("every code with a label also has a Goldstein value and vice versa", () => {
  for (const c of Object.keys(CAMEO_EVENT_LABEL)) assert.ok(c in CAMEO_GOLDSTEIN, `no Goldstein for ${c}`);
  for (const c of Object.keys(CAMEO_GOLDSTEIN)) assert.ok(c in CAMEO_EVENT_LABEL, `no label for ${c}`);
});

// ── the Goldstein audit — the page's visible integrity check ─────────────────

test("live rows match the published constant for their code, every one", () => {
  const a = goldsteinAudit(LIVE);
  assert.equal(a.checked, 5);
  assert.equal(a.matched, 5);
  assert.deepEqual(a.mismatches, []);
  assert.deepEqual(a.unknownCodes, []);
  assert.deepEqual(a.varyingCodes, []);
});

test("a row whose value contradicts the published table is reported, not averaged away", () => {
  const tampered = [...LIVE, { ...LIVE[0], id: "x1", gold: -2.5 }];
  const a = goldsteinAudit(tampered);
  assert.equal(a.checked, 6);
  assert.equal(a.matched, 5);
  assert.deepEqual(a.mismatches, [{ id: "x1", code: "192", sent: -2.5, published: -9.5 }]);
  // The same row also falsifies "constant of the code" — both are surfaced.
  assert.deepEqual(a.varyingCodes, ["192"]);
});

test("an unknown code is counted as unknown rather than as a mismatch", () => {
  const a = goldsteinAudit([...LIVE, { ...LIVE[0], id: "x2", code: "1799", root: "17", gold: -1 }]);
  assert.deepEqual(a.unknownCodes, ["1799"]);
  assert.equal(a.checked, 5); // unchanged — nothing to check it against
  assert.deepEqual(a.mismatches, []);
});

test("null and non-finite Goldstein values are skipped, not read as 0", () => {
  const a = goldsteinAudit([{ ...LIVE[0], id: "x3", gold: null }, { ...LIVE[0], id: "x4", gold: Number.NaN }]);
  assert.equal(a.checked, 0);
  assert.deepEqual(a.mismatches, []);
});

// ── rows are not incidents ───────────────────────────────────────────────────

test("rows sharing an article collapse into one incident carrying its row count", () => {
  const inc = groupByArticle(LIVE);
  assert.equal(LIVE.length, 5);
  assert.equal(inc.length, 3); // 5 rows, 3 articles — the whole point
  const houston = inc.find((i) => i.host === "houstonpublicmedia.org")!;
  assert.equal(houston.rows, 1);
  assert.equal(houston.maxMentions, 10);
  const laist = inc.find((i) => i.host === "laist.com")!;
  assert.equal(laist.rows, 2);
  // GDELT counts mentions per EVENT, so the two rows differ (7 and 3); the
  // incident reports the max and the page says so.
  assert.equal(laist.maxMentions, 7);
});

test("incidents sort by mentions, then rows, then key — a total order", () => {
  const inc = groupByArticle(LIVE);
  assert.deepEqual(inc.map((i) => i.maxMentions), [10, 7, 3]);
  // Identical input must produce an identical order (no Map-iteration drift).
  assert.deepEqual(groupByArticle(LIVE).map((i) => i.key), inc.map((i) => i.key));
});

test("tone is kept only when an article's rows agree on it", () => {
  const inc = groupByArticle(LIVE);
  assert.equal(inc.find((i) => i.host === "laist.com")!.tone, -4.5631);
  const conflicted = groupByArticle([LIVE[0], { ...LIVE[1], tone: 9 }]);
  assert.equal(conflicted.length, 1);
  assert.equal(conflicted[0].tone, null); // never an average of two claims
});

test("a row with no article URL becomes its own incident rather than merging with other URL-less rows", () => {
  const inc = groupByArticle([
    { ...LIVE[0], id: "a", url: null },
    { ...LIVE[1], id: "b", url: null },
  ]);
  assert.equal(inc.length, 2);
  assert.deepEqual(inc.map((i) => i.host), [null, null]);
});

test("hostOf strips www and survives a malformed URL", () => {
  assert.equal(hostOf("https://www.dailymail.com/news/x.html"), "dailymail.com");
  assert.equal(hostOf("http://laist.com/a"), "laist.com");
  assert.equal(hostOf("not a url"), null);
  assert.equal(hostOf(null), null);
  assert.equal(hostOf(""), null);
});

// ── "near" is a measured number, not a word ──────────────────────────────────

test("the matched facility resolves from our own site catalogue", () => {
  const s = siteInfo("port_la");
  assert.ok(s && Number.isFinite(s.lat) && Number.isFinite(s.lon));
  assert.equal(siteInfo("no_such_site"), null);
  assert.equal(siteInfo(null), null);
});

test("haversine is 0 at a point and matches a known separation", () => {
  assert.equal(haversineKm(34, -118, 34, -118), 0);
  // LA -> NYC, ~3936 km great circle.
  const d = haversineKm(34.0522, -118.2437, 40.7128, -74.006);
  assert.ok(Math.abs(d - 3936) < 15, `got ${d}`);
});

test("a downtown-LA row matched to the Port of LA is tens of km away, and says so", () => {
  // This is the caveat made quantitative: the +/-0.5 degree ingest box means a
  // row can be a metro away from the facility it is filed under.
  const d = eventDistanceKm(LIVE[1]);
  assert.ok(d != null && d > 10, `expected a double-digit separation, got ${d}`);
  assert.ok(d! < 80, `a matched row cannot exceed the ingest box diagonal, got ${d}`);
});

test("an unknown site or unusable coordinates give null, never 0", () => {
  // 0 would render as "at the facility" — the single most misleading value here.
  assert.equal(eventDistanceKm({ ...LIVE[0], site: "no_such_site" }), null);
  assert.equal(eventDistanceKm({ ...LIVE[0], lat: Number.NaN }), null);
  assert.equal(eventDistanceKm({ ...LIVE[0], lon: undefined as unknown as number }), null);
});

test("an incident reports its nearest row's distance and one entry per matched site", () => {
  const inc = groupByArticle(LIVE);
  const laist = inc.find((i) => i.host === "laist.com")!;
  assert.equal(laist.sites.length, 1);
  assert.equal(laist.sites[0].id, "port_la");
  assert.ok(laist.sites[0].name.length > "port_la".length); // catalogue name, not the id
  assert.ok(laist.nearestKm != null && laist.nearestKm <= eventDistanceKm(LIVE[1])!);
});

// ── the type reference table ─────────────────────────────────────────────────

test("typeSummary is one row per code, most frequent first, carrying the constant", () => {
  const t = typeSummary(LIVE);
  assert.deepEqual(t.map((r) => [r.code, r.count]), [["173", 2], ["192", 2], ["172", 1]]);
  assert.equal(t[0].label, "Arrest, detain, or charge with legal action");
  assert.equal(t[0].goldstein, -5);
  assert.equal(t.find((r) => r.code === "192")!.rootLabel, "Fight");
});

test("empty input degrades to empty results everywhere rather than throwing", () => {
  assert.deepEqual(groupByArticle([]), []);
  assert.deepEqual(typeSummary([]), []);
  const a = goldsteinAudit([]);
  assert.equal(a.checked, 0);
  assert.deepEqual(a.unknownCodes, []);
});
