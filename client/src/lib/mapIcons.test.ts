// EPA CAMD utilization helpers (datamap.tsx "plant_operations" layer,
// server/epaCamd.ts's ground-truth stream). Run:
// npx tsx --test client/src/lib/mapIcons.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { camdQuarterHours, camdUtilizationPct, camdUtilizationColor, volcanoAlertColor, ironOreStatusColor, ironSteelTechColor, chemicalFeedstockColor, lngShipyardCountryTier, summarizeFurnaceUnits, registerIcons, iconDataURL, iconNames, ICON_CANVAS_2D } from "./mapIcons.ts";

test("camdQuarterHours: real calendar length per quarter, not a fixed 91-day assumption", () => {
  assert.equal(camdQuarterHours(2026, 1), 90 * 24); // Jan(31)+Feb(28, non-leap)+Mar(31)
  assert.equal(camdQuarterHours(2024, 1), 91 * 24); // 2024 is a leap year — Feb has 29
  assert.equal(camdQuarterHours(2026, 2), 91 * 24); // Apr(30)+May(31)+Jun(30)
  assert.equal(camdQuarterHours(2026, 3), 92 * 24); // Jul(31)+Aug(31)+Sep(30)
  assert.equal(camdQuarterHours(2026, 4), 92 * 24); // Oct(31)+Nov(30)+Dec(31)
});

test("camdUtilizationPct: sumOpTime / (unitCount x quarter hours), null on missing inputs", () => {
  const hours = camdQuarterHours(2026, 2); // 91 * 24 = 2184
  assert.ok(Math.abs((camdUtilizationPct(hours, 1, 2026, 2) ?? 0) - 1) < 1e-9, "one unit operating every possible hour = 100%");
  assert.ok(Math.abs((camdUtilizationPct(hours * 2, 2, 2026, 2) ?? 0) - 1) < 1e-9, "two units, same per-unit rate, still normalizes to 100%");
  assert.equal(camdUtilizationPct(null, 3, 2026, 2), null, "no sumOpTime — no fabricated pct");
  assert.equal(camdUtilizationPct(100, null, 2026, 2), null, "no unitCount — no fabricated pct");
  assert.equal(camdUtilizationPct(100, 0, 2026, 2), null, "zero units — never divide by zero");
});

test("camdUtilizationColor: low-to-high blue-to-red bands, null tints as the lowest band", () => {
  assert.equal(camdUtilizationColor(null), "#4d9fff");
  assert.equal(camdUtilizationColor(0.1), "#4d9fff");
  assert.equal(camdUtilizationColor(0.25), "#8bc34a");
  assert.equal(camdUtilizationColor(0.49), "#8bc34a");
  assert.equal(camdUtilizationColor(0.5), "#ffd23f");
  assert.equal(camdUtilizationColor(0.74), "#ffd23f");
  assert.equal(camdUtilizationColor(0.75), "#ff3b3b");
  assert.equal(camdUtilizationColor(1.4), "#ff3b3b", "readings above 100% (data quirks) still clamp to the top band, not crash");
});

test("volcanoAlertColor: USGS's own color_code vocabulary maps directly, unrecognized/missing falls back to neutral gray", () => {
  assert.equal(volcanoAlertColor("RED"), "#ff3b3b");
  assert.equal(volcanoAlertColor("ORANGE"), "#ff8c42");
  assert.equal(volcanoAlertColor("YELLOW"), "#ffd23f");
  assert.equal(volcanoAlertColor("GREEN"), "#8bc34a");
  assert.equal(volcanoAlertColor("orange"), "#ff8c42", "case-insensitive — feed values are uppercase but never assumed");
  assert.equal(volcanoAlertColor(null), "#9aa5b1");
  assert.equal(volcanoAlertColor(undefined), "#9aa5b1");
  assert.equal(volcanoAlertColor("UNKNOWN"), "#9aa5b1", "never guesses a severity color for an unrecognized code");
});

test("ironOreStatusColor: GEM's 7 lowercase Operating status buckets map directly, unrecognized/missing falls back to gray", () => {
  assert.equal(ironOreStatusColor("operating"), "#4ade80");
  assert.equal(ironOreStatusColor("proposed"), "#a78bfa");
  assert.equal(ironOreStatusColor("mothballed"), "#78716c");
  assert.equal(ironOreStatusColor("retired"), "#64748b");
  assert.equal(ironOreStatusColor("shelved"), "#fbbf24");
  assert.equal(ironOreStatusColor("cancelled"), "#f87171");
  assert.equal(ironOreStatusColor("unknown"), "#94a3b8");
  assert.equal(ironOreStatusColor(null), "#94a3b8");
  assert.equal(ironOreStatusColor(undefined), "#94a3b8");
  assert.equal(ironOreStatusColor("not-a-real-bucket"), "#94a3b8", "never guesses a lifecycle color for an unrecognized status");
});

test("ironSteelTechColor: GEM's 5 production-technology buckets map directly, unrecognized/missing falls back to the 'other' gray", () => {
  assert.equal(ironSteelTechColor("bf_bof"), "#f97316");
  assert.equal(ironSteelTechColor("dri"), "#22d3ee");
  assert.equal(ironSteelTechColor("eaf"), "#4ade80");
  assert.equal(ironSteelTechColor("if"), "#a78bfa");
  assert.equal(ironSteelTechColor("other"), "#94a3b8");
  assert.equal(ironSteelTechColor(null), "#94a3b8");
  assert.equal(ironSteelTechColor(undefined), "#94a3b8");
  assert.equal(ironSteelTechColor("not-a-real-bucket"), "#94a3b8", "never guesses a technology color for an unrecognized bucket");
});

test("chemicalFeedstockColor: GEM chemicals' 6 feedstock-family buckets map directly, unrecognized/missing falls back to the 'other' gray", () => {
  assert.equal(chemicalFeedstockColor("coal"), "#f87171");
  assert.equal(chemicalFeedstockColor("natural_gas"), "#fbbf24");
  assert.equal(chemicalFeedstockColor("petroleum"), "#f97316");
  assert.equal(chemicalFeedstockColor("ngl"), "#a78bfa");
  assert.equal(chemicalFeedstockColor("low_carbon"), "#4ade80");
  assert.equal(chemicalFeedstockColor("other"), "#94a3b8");
  assert.equal(chemicalFeedstockColor(null), "#94a3b8");
  assert.equal(chemicalFeedstockColor(undefined), "#94a3b8");
  assert.equal(chemicalFeedstockColor("not-a-real-bucket"), "#94a3b8", "never guesses a feedstock color for an unrecognized bucket");
});

test("summarizeFurnaceUnits: no units returns [], never a fabricated summary", () => {
  assert.deepEqual(summarizeFurnaceUnits(null), []);
  assert.deepEqual(summarizeFurnaceUnits(undefined), []);
  assert.deepEqual(summarizeFurnaceUnits([]), []);
});

test("summarizeFurnaceUnits: groups by furnace type, one line per type, with a status breakdown and capacity sum", () => {
  const lines = summarizeFurnaceUnits([
    { furnaceType: "eaf", status: "operating", capacityTtpa: 1100 },
    { furnaceType: "eaf", status: "operating", capacityTtpa: 900 },
    { furnaceType: "bof", status: "retired", capacityTtpa: 1950 },
  ]);
  assert.equal(lines.length, 2);
  assert.equal(lines[0], "2x EAF (2 operating), 2,000 ttpa");
  assert.equal(lines[1], "1x BOF (1 retired), 1,950 ttpa");
});

test("summarizeFurnaceUnits: mixed statuses within one furnace type are broken out, never collapsed", () => {
  const lines = summarizeFurnaceUnits([
    { furnaceType: "eaf", status: "operating", capacityTtpa: 1100 },
    { furnaceType: "eaf", status: "retired", capacityTtpa: null },
  ]);
  assert.equal(lines.length, 1);
  assert.ok(lines[0].includes("1 operating"));
  assert.ok(lines[0].includes("1 retired"));
  assert.ok(lines[0].endsWith("(capacity known for 1/2)"), "capacity sum must flag itself as partial, never silently omit the unknown unit");
});

test("summarizeFurnaceUnits: a furnace type with no unit stating a capacity reports no capacity figure at all (never a fabricated 0)", () => {
  const lines = summarizeFurnaceUnits([{ furnaceType: "induction", status: "announced", capacityTtpa: null }]);
  assert.equal(lines[0], "1x Induction furnace (1 announced)");
});

test("lngShipyardCountryTier: the top-3 shipbuilding nations get their own tier, every other country pools to 'other'", () => {
  assert.equal(lngShipyardCountryTier("South Korea"), "south_korea");
  assert.equal(lngShipyardCountryTier("China"), "china");
  assert.equal(lngShipyardCountryTier("Japan"), "japan");
  assert.equal(lngShipyardCountryTier("Russia"), "other");
  assert.equal(lngShipyardCountryTier("France"), "other");
  assert.equal(lngShipyardCountryTier(null), "other");
  assert.equal(lngShipyardCountryTier(undefined), "other");
});

// ── icon rasterization cost (map speed, 2026-11-19) ─────────────────────────
// The load trace showed the icon registry as two ~1 s main-thread tasks: every
// shape was drawn on a GPU-backed canvas and read back (a synchronous round
// trip to the GPU process per shape), once for the map and again for the
// legend. These pin the two properties that removed it: every icon canvas is
// CPU-backed, and a shape is drawn once per session whoever asks for it.
type CanvasRecord = { kind: string | null; settings: { willReadFrequently?: boolean } | undefined; readbacks: number };
type FakeImage = { width: number; height: number; data: Uint8ClampedArray };
type FakeCanvas = { width: number; height: number; getContext(kind: string, settings?: CanvasRecord["settings"]): unknown; toDataURL(): string };
function fakeCanvasDocument() {
  const canvases: CanvasRecord[] = [];
  const doc = {
    createElement(tag: string) {
      assert.equal(tag, "canvas");
      const rec: CanvasRecord = { kind: null, settings: undefined, readbacks: 0 };
      canvases.push(rec);
      const canvas = { width: 0, height: 0 } as FakeCanvas;
      canvas.getContext = (kind, settings) => {
        rec.kind = kind; rec.settings = settings;
        // any drawing call is accepted; only the read-backs are observed
        return new Proxy({} as Record<string, unknown>, {
          get(target, prop) {
            if (prop === "getImageData") return (_x: number, _y: number, w: number, h: number) => { rec.readbacks++; return { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }; };
            if (prop in target) return target[prop as string];
            return () => undefined;
          },
          set(target, prop, value) { target[prop as string] = value; return true; },
        });
      };
      canvas.toDataURL = () => { rec.readbacks++; return `data:image/png;base64,canvas${canvases.indexOf(rec)}`; };
      return canvas;
    },
  };
  return { doc, canvases };
}

test("icon canvases: every one is CPU-backed (willReadFrequently), and a shape is drawn once for map + legend", () => {
  assert.deepEqual(ICON_CANVAS_2D, { willReadFrequently: true });
  const { doc, canvases } = fakeCanvasDocument();
  const g = globalThis as unknown as { document?: unknown };
  const prevDoc = g.document;
  g.document = doc;
  try {
    const names = iconNames();
    assert.ok(names.length >= 50, `the registry holds the map's shapes (${names.length})`);
    const added = new Map<string, { img: FakeImage; o: unknown }>();
    const map = { hasImage: (n: string) => added.has(n), addImage: (n: string, img: FakeImage, o: unknown) => { added.set(n, { img, o }); } };
    registerIcons(map);
    assert.deepEqual([...added.keys()].sort(), [...names].sort(), "every shape is registered");
    for (const { o } of added.values()) assert.deepEqual(o, { sdf: true });
    const drawn = canvases.length;
    assert.equal(drawn, names.length, "one canvas per shape, no more");
    assert.ok(canvases.every((c) => c.kind === "2d" && c.settings?.willReadFrequently === true),
      "a GPU-backed icon canvas makes each read-back a synchronous GPU round trip");

    // a second map (the landing globe, a re-registration after a lost context) redraws nothing
    const added2 = new Map<string, FakeImage>();
    registerIcons({ hasImage: (n: string) => added2.has(n), addImage: (n: string, img: FakeImage) => { added2.set(n, img); } });
    assert.equal(canvases.length, drawn, "a second registration reuses the session's rasters");
    assert.equal(added2.get("vt-jet"), added.get("vt-jet")?.img, "same pixels for every consumer");

    // the legend tints the SAME raster: two small canvases per (shape, color), never a redraw of the shape
    const url = iconDataURL("vt-jet", "#4d9fff");
    assert.match(url, /^data:image\/png/);
    assert.equal(canvases.length, drawn + 2, "tint canvas + scaled output canvas, the shape is not drawn again");
    assert.ok(canvases.slice(drawn).every((c) => c.settings?.willReadFrequently === true), "legend canvases are CPU-backed too");
    assert.equal(iconDataURL("vt-jet", "#4d9fff"), url);
    assert.equal(canvases.length, drawn + 2, "a repeated legend render is a cache hit");
    assert.equal(iconDataURL("vt-not-a-shape", "#fff"), "", "an unknown shape renders nothing rather than throwing");
  } finally {
    if (prevDoc === undefined) delete g.document; else g.document = prevDoc;
  }
});
