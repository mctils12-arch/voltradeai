// Find plane + the tracked/watchlist chips in datamap.tsx (owner report
// 2026-11-12, N843S: the Find box searched only the current viewport and
// answered `no "N843S" in the current snapshot` for a plane that was
// airborne elsewhere or heard without a position). Source pins on the
// wiring; the wording and decisions are tested in
// client/src/lib/air/planeFind.test.ts.
// Run: npx tsx --test client/src/pages/datamap.planeFind.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(here, "datamap.tsx"), "utf8");

function between(start: string, end: string): string {
  const a = src.indexOf(start);
  assert.ok(a >= 0, `not found: ${start}`);
  const b = src.indexOf(end, a + start.length);
  assert.ok(b > a, `not found after it: ${end}`);
  return src.slice(a, b);
}
const findCb = between("const findPlaneCb = useCallback(", "const openLiveAircraftRow = ");
const trackedPanel = between("function TrackedPlanesPanel(", "function LegendIcon(");
const legend = between("const LegendPanel = memo(function LegendPanel(", "{enabled.timezones && (");

test("Find: the instant viewport search still runs first, then the server is asked for the plane anywhere", () => {
  const viewport = findCb.indexOf("airPayloadRef.current");
  const server = findCb.indexOf("fetchPlaneFind(Q, { signal: ac.signal })");
  assert.ok(viewport >= 0 && server > viewport, "viewport rows first, the server lookup only on a miss");
  assert.match(findCb, /if \(!hit\) \{[\s\S]*fetchPlaneFind\(/);
  assert.match(findCb, /return findPendingNote\(Q\);/, "the note says 'checking' at once — the lookup never blocks the paint");
  assert.doesNotMatch(src, /pan\/zoom to search elsewhere/, "the viewport-only miss message is gone");
});

test("Find: each state acts honestly — live opens the normal card, last seen opens the archive card, heard only says so", () => {
  assert.match(findCb, /const d = decidePlaneFind\(Q, ans, Date\.now\(\)\);\s*setNote\(d\.note\);/);
  assert.match(findCb, /if \(d\.kind === "live"\) openLiveAircraftRowRef\.current\(d\.row\);/, "the same path a plane click / watchlist tap uses");
  assert.match(findCb, /else if \(d\.kind === "last_seen" && d\.hex\) openWatchedCb\(\{ hex: d\.hex, reg: d\.reg \}\);/);
  // a heard plane has no position: nothing in the server branch flies the map or opens a card for it
  const serverBranch = findCb.slice(findCb.indexOf("if (!hit) {"), findCb.indexOf("return findPendingNote(Q);"));
  assert.doesNotMatch(serverBranch, /easeTo|flyTo|setDetail|airClickRef/);
  assert.doesNotMatch(serverBranch, /"heard"/);
});

test("Find: a newer search aborts a pending one, and a late answer is dropped", () => {
  assert.match(findCb, /findAbortRef\.current\?\.abort\(\);/);
  assert.match(findCb, /if \(ac\.signal\.aborted\) return;/);
});

test("Law I: the find is a user action — not wired to any map event", () => {
  assert.equal(src.split("fetchPlaneFind(").length - 1, 1, "one call site: the Find box's callback");
  assert.doesNotMatch(findCb, /\.on\(\s*["'](zoomend|moveend|idle|move|rotate|pitch|zoom|dragend)["']/);
  assert.match(legend, /if \(e\.key !== "Enter"\) return;[\s\S]*setAirSearchMiss\(onFindPlane\(q, setAirSearchMiss\)\);/);
  assert.match(src, /onFindPlane: \(q: string, setNote: \(note: string \| null\) => void\) => string \| null;/);
});

test("watchlist chips: the third honest state, marked only where the state is known", () => {
  assert.match(legend, /● broadcasting now, ◐ transponder heard \(no position yet\), ○ archive only/);
  assert.match(legend, /const signal = trackedSignalFor\(p\.hex, Date\.now\(\)\);/);
  assert.match(legend, /\{signal \? `\$\{SIGNAL_MARK\[signal\]\} \$\{label\}` : label\}/, "no mark for a plane whose state is unknown");
  assert.match(legend, /subscribeTrackedSignals\(\(\) => setSignalTick\(\(n\) => n \+ 1\)\)/);
});

test("tracked planes panel: chips show live / heard / dark from the registry's own rows and share them with the legend", () => {
  assert.match(trackedPanel, /\{trackedChipText\(p, Date\.now\(\)\)\}/);
  assert.match(trackedPanel, /data-signal=\{planeSignal\(p, Date\.now\(\)\)\}/);
  assert.match(trackedPanel, /● position under 2 min old · ◐ transponder heard, no position yet · ○ no signal now/);
  assert.match(trackedPanel, /setRows\(d\.planes \|\| \[\]\);[\s\S]{0,160}publishTrackedSignals\(d\.planes \|\| \[\]\);/, "the panel's own poll feeds the legend — one request, two readers");
});

test("legend: with the tracked panel closed (a phone) the watchlist marks still load — abortable, hidden-tab gated, torn down", () => {
  assert.match(legend, /if \(!airOn \|\| !legendOpen \|\| !hasWatch\) return;/, "no request without an open aircraft legend and a watchlist");
  assert.match(legend, /if \(!document\.hidden\) void refreshTrackedSignals\(\{ signal: ac\.signal \}\);/);
  assert.match(legend, /return \(\) => \{ window\.clearInterval\(t\); ac\.abort\(\); \};/);
});

test("watched-plane card: a plane heard without a position says so, and drops the 'usually on the ground' guess", () => {
  const card = between("const openWatchedCb = useCallback(", "const fetchDossier = async (");
  assert.match(card, /planeSignal\(tracked, Date\.now\(\)\) === "heard"/);
  assert.match(card, /heardSentence\(label, tracked\.heard, Date\.now\(\), null\)/);
  assert.match(card, /\(heardNow \? "" : "Usually on the ground without ADS-B, or out of receiver coverage\. "\)/);
});
