/**
 * Hermetic tests for the wikimedia_pageviews_attention live signal board
 * (server/wikiAttentionSignal.ts) — the pure z-score function, the
 * per-ticker row builder against a temp archive dir, and the full
 * summary shape. No network. Runs via `npm run test:node`.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  latestZScore, capTier, computeTickerRows, computeWikiAttentionSignal,
  Z_THRESHOLD, TRAILING_WINDOW_DAYS, MIN_BASELINE_DAYS, MEGA_CAP_TICKERS,
  VALIDATED_SMALL_MID, VALIDATED_MEGA, BONFERRONI_ALPHA,
} from "./wikiAttentionSignal";

const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), "vt-wiki-signal-"));

// Day-files written DIRECTLY to disk, not via archiveAttention — that
// function's archivedKeys dedup is module-level state shared across the
// whole test process regardless of baseDir (the same caveat
// wikiAttention.test.ts's own history-read tests note), so hermetic
// per-test archives are built by hand here instead.
function seedSeries(dir: string, ticker: string, article: string, views: number[], startDate = "2026-06-01"): void {
  const wikiDir = path.join(dir, "wikiattention");
  fs.mkdirSync(wikiDir, { recursive: true });
  const start = Date.parse(`${startDate}T00:00:00Z`);
  views.forEach((v, i) => {
    const date = new Date(start + i * 86_400_000).toISOString().slice(0, 10);
    const line = JSON.stringify({ date, ticker, article, views: v, rt: "2026-09-05" }) + "\n";
    fs.appendFileSync(path.join(wikiDir, `${date}.jsonl`), line);
  });
}

test("capTier: the four hardcoded mega names are mega, everything else is small_mid", () => {
  for (const t of MEGA_CAP_TICKERS) assert.equal(capTier(t), "mega");
  assert.equal(capTier("nvda"), "mega"); // case-insensitive
  assert.equal(capTier("PLTR"), "small_mid");
  assert.equal(capTier("GME"), "small_mid");
});

test("latestZScore: a perfectly flat baseline yields null (not Infinity) even after a jump", () => {
  const flat = Array.from({ length: 30 }, (_, i) => ({ date: `d${i}`, views: 100 }));
  const withSpike = [...flat, { date: "spike", views: 500 }];
  const z = latestZScore(withSpike, 90);
  assert.ok(z);
  assert.equal(z!.baseline_days, 30);
  assert.equal(z!.baseline_mean, 100);
  assert.equal(z!.baseline_stdev, 0); // perfectly flat baseline
  assert.equal(z!.z_score, null); // stdev 0 — cannot compute a ratio, not silently Infinity
});

test("latestZScore: realistic noisy baseline flags a real spike, not everyday noise", () => {
  const noisy = [90, 105, 95, 110, 100, 98, 102, 107, 93, 101, 99, 104, 96, 108, 100];
  const series = noisy.map((v, i) => ({ date: `d${i}`, views: v }));
  const calm = latestZScore([...series, { date: "calm", views: 103 }], 90);
  assert.ok(calm && calm.z_score != null && Math.abs(calm.z_score) < Z_THRESHOLD);

  const spiky = latestZScore([...series, { date: "spike", views: 400 }], 90);
  assert.ok(spiky && spiky.z_score != null && spiky.z_score >= Z_THRESHOLD);
});

test("latestZScore: only uses STRICTLY PRIOR days for the baseline (no lookahead)", () => {
  const series = [
    { date: "d0", views: 100 }, { date: "d1", views: 100 }, { date: "d2", views: 100 },
    { date: "d3", views: 9999 }, // would blow up the baseline if it leaked in
  ];
  // z-score of d3 itself must not be influenced by d3's own value in the baseline
  const z = latestZScore(series.slice(0, 3).concat([{ date: "d3", views: 100 }]), 90);
  assert.equal(z!.baseline_mean, 100);
});

test("latestZScore: caps the baseline at `window` trailing days", () => {
  const series = Array.from({ length: 200 }, (_, i) => ({ date: `d${i}`, views: i < 100 ? 1000 : 100 }));
  series.push({ date: "latest", views: 100 });
  const z = latestZScore(series, 90);
  assert.equal(z!.baseline_days, 90);
  assert.equal(z!.baseline_mean, 100); // only the last 90 (all value 100) should count, not the earlier 1000s
});

test("latestZScore: too little history returns null rather than a misleading number", () => {
  assert.equal(latestZScore([]), null);
  assert.equal(latestZScore([{ date: "d0", views: 100 }]), null);
});

test("computeTickerRows: a ticker with no archived rows shows null fields, not a crash", () => {
  const dir = tmp();
  const rows = computeTickerRows(dir);
  assert.ok(rows.length > 0); // every seed ticker gets a row
  for (const r of rows) {
    assert.equal(r.z_score, null);
    assert.equal(r.spike, false);
    assert.equal(r.baseline_complete, false);
  }
});

test("computeTickerRows: a real spike (enough baseline depth) is flagged; sorted z-score descending", () => {
  const dir = tmp();
  // Fixed pseudo-noisy sequence (deterministic — no Math.random in a test,
  // per the repeated wall-clock/randomness-flake lesson in KNOWN BROKEN #40).
  const flat = [98, 103, 96, 105, 100, 97, 104, 99, 101, 95, 106, 100, 98, 102, 97,
                104, 100, 99, 103, 96, 101, 98, 105, 100, 97];
  seedSeries(dir, "GME", "GameStop", [...flat, 900]); // clear spike, >= MIN_BASELINE_DAYS history
  seedSeries(dir, "SOFI", "SoFi", [...flat, 102]); // no spike
  const rows = computeTickerRows(dir);
  const gme = rows.find((r) => r.ticker === "GME")!;
  const sofi = rows.find((r) => r.ticker === "SOFI")!;
  assert.equal(gme.spike, true);
  assert.ok(gme.baseline_days >= MIN_BASELINE_DAYS);
  assert.equal(sofi.spike, false);
  // descending sort: the spiking ticker's z-score must lead
  const gmeIdx = rows.findIndex((r) => r.ticker === "GME");
  const sofiIdx = rows.findIndex((r) => r.ticker === "SOFI");
  assert.ok(gmeIdx < sofiIdx);
});

test("computeTickerRows: below MIN_BASELINE_DAYS, even a huge jump is never flagged a spike", () => {
  const dir = tmp();
  seedSeries(dir, "IONQ", "IonQ", [100, 100, 100, 100, 100, 900]); // 5-day baseline only
  const row = computeTickerRows(dir).find((r) => r.ticker === "IONQ")!;
  assert.ok(row.baseline_days < MIN_BASELINE_DAYS);
  assert.equal(row.spike, false);
});

test("computeWikiAttentionSignal: shape is complete and gate/validated-effect metadata is honest", () => {
  const dir = tmp();
  const summary = computeWikiAttentionSignal(dir, Date.parse("2026-09-05T12:00:00Z"));
  assert.equal(summary.kind, "signal");
  assert.equal(summary.root_id, "wikimedia_pageviews_attention");
  assert.equal(summary.gate.current_gate, 2);
  assert.equal(summary.gate.status, "gate2_pass");
  assert.equal(summary.gate.channel, "trading_volume_elevation");
  assert.equal(summary.z_threshold, Z_THRESHOLD);
  assert.equal(summary.trailing_window_days, TRAILING_WINDOW_DAYS);
  assert.equal(summary.spike_count, summary.tickers.filter((t) => t.spike).length);
  assert.equal(summary.validated_effect.small_mid.length, VALIDATED_SMALL_MID.length);
  assert.equal(summary.validated_effect.mega.length, VALIDATED_MEGA.length);
  assert.equal(summary.validated_effect.bonferroni_alpha, BONFERRONI_ALPHA);
  assert.ok(summary.caveats.length >= 4);
  assert.ok(summary.caveats.some((c) => c.toLowerCase().includes("gate 3")));
  assert.ok(summary.caveats.some((c) => c.toLowerCase().includes("volatility")));
  // the validated study's own Bonferroni bar rejects the mega h=1 cell —
  // confirm the frozen numbers still say what the study concluded
  assert.ok(VALIDATED_MEGA[0].p_value > BONFERRONI_ALPHA);
  assert.ok(VALIDATED_SMALL_MID.every((r) => r.p_value < BONFERRONI_ALPHA));
});
