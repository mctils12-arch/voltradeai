import { test } from "node:test";
import assert from "node:assert/strict";
import { eventWindowRatio, checkRedirect } from "./wikiattention_gate1";

test("eventWindowRatio: peak is max(event day, event day + 1), baseline excludes a 1-day buffer on each side", () => {
  const series: Record<string, number> = {
    "2026-07-01": 100, "2026-07-02": 105, "2026-07-03": 95,
    "2026-07-04": 110, // buffer day before event, excluded from baseline
    "2026-07-05": 300, // event day
    "2026-07-06": 500, // event day + 1 — this is the real peak
    "2026-07-07": 120, // buffer day after event, excluded from baseline
    "2026-07-08": 90, "2026-07-09": 108, "2026-07-10": 102,
  };
  const r = eventWindowRatio(series, "2026-07-05")!;
  assert.equal(r.peakDate, "2026-07-06");
  assert.equal(r.peakViews, 500);
  assert.equal(r.baselineDays, 6, "excludes event-1..event+2, keeps the other 6");
  assert.equal(r.baselineMedian, 101, "median of [90,95,100,102,105,108]");
  assert.equal(r.ratio, 500 / 101);
});

test("eventWindowRatio: honest null when the event date isn't in the series (never fabricates a ratio)", () => {
  assert.equal(eventWindowRatio({ "2026-07-01": 100 }, "2026-07-05"), null);
  assert.equal(eventWindowRatio({}, "2026-07-05"), null);
});

test("eventWindowRatio: baseline of zero degrades to Infinity, not a divide-by-zero NaN or a fabricated finite ratio", () => {
  const series = { "2026-07-01": 0, "2026-07-02": 0, "2026-07-05": 10 };
  const r = eventWindowRatio(series, "2026-07-05")!;
  assert.equal(r.ratio, Infinity);
});

test("checkRedirect: flags a title that resolves to a different page (the PLTR/AMC/SMCI defect shape)", async () => {
  const fake = async () => ({
    ok: true, status: 200,
    text: async () => JSON.stringify({
      query: { redirects: [{ from: "AMC Entertainment", to: "AMC Theatres" }],
               pages: { "1": { title: "AMC Theatres" } } },
    }),
  });
  const resolved = await checkRedirect(fake as any, "AMC_Entertainment");
  assert.equal(resolved, "AMC Theatres");
});

test("checkRedirect: a title that resolves to itself (spaces-for-underscores normalized) is healthy, not flagged", async () => {
  const fake = async () => ({
    ok: true, status: 200,
    text: async () => JSON.stringify({
      query: { pages: { "1": { title: "Nvidia" } } },
    }),
  });
  assert.equal(await checkRedirect(fake as any, "Nvidia"), null);
});

test("checkRedirect: a transient fetch failure is not evidence of a redirect (never flags on inconclusive data)", async () => {
  const fake = async () => ({ ok: false, status: 503, text: async () => "" });
  assert.equal(await checkRedirect(fake as any, "Nvidia"), null);
});
