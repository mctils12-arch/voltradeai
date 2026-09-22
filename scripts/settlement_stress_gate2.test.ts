import { test } from "node:test";
import assert from "node:assert/strict";
import {
  isForeignListing, collapseEpisodes, assessReadiness,
  MAX_EPISODE_GAP_DAYS, MIN_DOMESTIC_EPISODES, type CompositeRow,
} from "./settlement_stress_gate2";

function row(date: string, symbol: string, name: string, persistence_days: number, composite_score = 1): CompositeRow {
  return {
    date, symbol, name, persistence_days,
    ftd_qty: 1000, ftd_delta: 100, short_ratio: 0.5, short_vol_percentile: 50,
    composite_score,
  };
}

test("isForeignListing recognizes FINRA's real ADR/ordinary-share naming conventions", () => {
  assert.ok(isForeignListing("Compagnie Financiere Richemont Unsponsored ADR (Switzerland)"));
  assert.ok(isForeignListing("Alkane Resources Ltd. Ordinary Shares (Australia)"));
  assert.ok(isForeignListing("FRESENIUS SE & CO KGAA SPONSORED ADR (Germany)"));
  assert.ok(isForeignListing("Bayer Aktiengesellschaft American Depositary Shares (Each repstg one Bayer AG ordinary share of no par value)"));
});

test("isForeignListing does not flag a plain US common stock", () => {
  assert.ok(!isForeignListing("NetBrands Corp. Common Stock"));
  assert.ok(!isForeignListing("Apple Inc."));
});

test("collapseEpisodes merges consecutive observations of the same symbol into one episode", () => {
  const rows = [
    row("2026-07-21", "ALKEF", "Alkane Resources Ltd. Ordinary Shares (Australia)", 11),
    row("2026-07-22", "ALKEF", "Alkane Resources Ltd. Ordinary Shares (Australia)", 12),
    row("2026-07-27", "ALKEF", "Alkane Resources Ltd. Ordinary Shares (Australia)", 15),
  ];
  const episodes = collapseEpisodes(rows);
  assert.equal(episodes.length, 1);
  assert.equal(episodes[0].symbol, "ALKEF");
  assert.equal(episodes[0].observations, 3);
  assert.equal(episodes[0].startDate, "2026-07-21");
  assert.equal(episodes[0].endDate, "2026-07-27");
  assert.equal(episodes[0].foreign, true);
});

test("collapseEpisodes splits into two episodes when the gap exceeds MAX_EPISODE_GAP_DAYS", () => {
  const farDay = (1 + MAX_EPISODE_GAP_DAYS + 1).toString().padStart(2, "0"); // one day past the gap limit
  const rows = [
    row("2026-07-01", "AAAA", "AAAA Corp Common Stock", 1),
    row(`2026-07-${farDay}`, "AAAA", "AAAA Corp Common Stock", 1),
  ];
  const episodes = collapseEpisodes(rows);
  assert.equal(episodes.length, 2);
});

test("collapseEpisodes keeps one episode when the gap is exactly MAX_EPISODE_GAP_DAYS", () => {
  const atLimitDay = (1 + MAX_EPISODE_GAP_DAYS).toString().padStart(2, "0");
  const rows = [
    row("2026-07-01", "AAAA", "AAAA Corp Common Stock", 1),
    row(`2026-07-${atLimitDay}`, "AAAA", "AAAA Corp Common Stock", 1),
  ];
  const episodes = collapseEpisodes(rows);
  assert.equal(episodes.length, 1);
});

test("collapseEpisodes keeps two distinct symbols as two episodes even on the same date", () => {
  const rows = [
    row("2026-08-20", "WEGZY", "Weg S.A. Sponsored ADR (Brazil)", 1),
    row("2026-08-20", "TGOPY", "3i Group Plc Unsponsored ADR (UK)", 1),
  ];
  const episodes = collapseEpisodes(rows);
  assert.equal(episodes.length, 2);
  assert.deepEqual(episodes.map((e) => e.symbol).sort(), ["TGOPY", "WEGZY"]);
});

test("collapseEpisodes reports peakAbsScore as the max absolute composite_score across the episode", () => {
  const rows = [
    row("2026-08-01", "NBND", "NetBrands Corp. Common Stock", 1, 4.14),
    row("2026-08-02", "NBND", "NetBrands Corp. Common Stock", 2, -90.01),
    row("2026-08-03", "NBND", "NetBrands Corp. Common Stock", 3, 36),
  ];
  const episodes = collapseEpisodes(rows);
  assert.equal(episodes.length, 1);
  assert.equal(episodes[0].entryScore, 4.14);
  assert.equal(episodes[0].peakAbsScore, 90.01);
});

test("collapseEpisodes handles unsorted input identically to sorted input", () => {
  const sorted = [
    row("2026-07-21", "ALKEF", "Alkane Resources Ltd. Ordinary Shares (Australia)", 11),
    row("2026-07-22", "ALKEF", "Alkane Resources Ltd. Ordinary Shares (Australia)", 12),
  ];
  const shuffled = [sorted[1], sorted[0]];
  assert.deepEqual(collapseEpisodes(sorted), collapseEpisodes(shuffled));
});

test("assessReadiness excludes foreign-listing episodes from the domestic count", () => {
  const episodes = collapseEpisodes([
    row("2026-01-01", "NBND", "NetBrands Corp. Common Stock", 1),
    row("2026-01-01", "ALKEF", "Alkane Resources Ltd. Ordinary Shares (Australia)", 1),
  ]);
  const r = assessReadiness(episodes);
  assert.equal(r.domestic, 1);
  assert.equal(r.foreign, 1);
  assert.equal(r.ready, false);
});

test("assessReadiness is READY only once domestic episodes reach MIN_DOMESTIC_EPISODES", () => {
  const domesticRows = Array.from({ length: MIN_DOMESTIC_EPISODES }, (_, i) =>
    row(`2026-01-${(i % 28 + 1).toString().padStart(2, "0")}`, `SYM${i}`, `SYM${i} Corp Common Stock`, 1));
  const justBelow = collapseEpisodes(domesticRows.slice(0, -1));
  const atThreshold = collapseEpisodes(domesticRows);
  assert.equal(assessReadiness(justBelow).ready, false);
  assert.equal(assessReadiness(atThreshold).ready, true);
});
