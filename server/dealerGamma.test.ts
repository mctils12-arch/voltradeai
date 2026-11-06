// Edge P1 dealer-gamma filter, Node side: the GAMMA-* audit rows from a scan's
// `dealer_gamma` field, the two file readers, the read-only `gamma` diag probe
// body (no raw GEX anywhere), and the probe's registration.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  dealerGammaAuditRows, dealerGammaProbeBody, readDealerGammaDecisions, readDealerGammaState,
  dealerGammaDataDir, DEALER_GAMMA_STATE_FILE, DEALER_GAMMA_DECISIONS_FILE, DEALER_GAMMA_AUDIT_ROWS_MAX,
  DEALER_GAMMA_TAIL_BYTES,
} from "./dealerGamma";
import { DIAG_PROBES } from "./diag";
import type { SqueezeReport } from "./squeezeGex";

const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), "dgamma-"));

const SKIP = {
  ts: "2026-11-06T10:01:00-05:00", day: "2026-11-06", ticker: "NVDA", source: "options_scanner.get_options_trades",
  setup: "earnings_iv_crush", strategy: "iron_condor", structure: "short_two_sided", action: "skip", enforced: true,
  rank252: 0.09, threshold: 0.2, asOf: "2026-11-05",
  reason: "dealer gamma low (rank 0.09): realized vol tends to exceed implied, no new short premium (SqueezeMetrics GEX estimate as of 2026-11-05)",
  new: true,
};

test("audit rows: one per NEW skip / would-skip, plain-English message, malformed and old ones ignored", () => {
  const rows = dealerGammaAuditRows({
    state: { rank252: 0.09 },
    decisions: [
      SKIP,
      { ...SKIP, ticker: "AMD", action: "would_skip", setup: "csp_core", source: "tiered_strategy.tier1_csp_core",
        reason: "dealer gamma low (rank 0.09): would skip this short put, but it is logged only" },
      { ...SKIP, ticker: "OLD", new: false },
      { ...SKIP, action: "allow" },
      { ...SKIP, ticker: "" },
      "junk", null,
    ],
  });
  assert.deepEqual(rows.map((r) => r.type), ["GAMMA-SKIP", "GAMMA-WOULD-SKIP"]);
  assert.equal(rows[0].message, `NVDA earnings_iv_crush via options_scanner.get_options_trades: ${SKIP.reason}`);
  assert.match(rows[1].message, /^AMD csp_core via tiered_strategy\.tier1_csp_core: dealer gamma low \(rank 0\.09\): would skip/);
});

test("audit rows: a field error is surfaced, junk input is empty, rows are capped", () => {
  assert.deepEqual(dealerGammaAuditRows({ error: "RuntimeError: boom", decisions: [] }).map((r) => r.type), ["GAMMA-ERROR"]);
  assert.match(dealerGammaAuditRows({ error: "RuntimeError: boom" })[0].message, /fails open/);
  for (const bad of [null, undefined, "x", 3, [SKIP]]) assert.deepEqual(dealerGammaAuditRows(bad), []);
  const many = Array.from({ length: 50 }, (_, i) => ({ ...SKIP, ticker: `T${i}` }));
  assert.equal(dealerGammaAuditRows({ decisions: many }).length, DEALER_GAMMA_AUDIT_ROWS_MAX);
});

test("readers: state file, decisions tail (partial first line and torn lines skipped), absent files", () => {
  const d = tmp();
  assert.equal(readDealerGammaState(d), null);
  assert.deepEqual(readDealerGammaDecisions(d), []);
  fs.writeFileSync(path.join(d, DEALER_GAMMA_STATE_FILE), JSON.stringify({ asOf: "2026-11-05", rank252: 0.09, label: "dealer gamma: low (rank 0.09)" }));
  assert.equal(readDealerGammaState(d)?.rank252, 0.09);
  const lines = Array.from({ length: 5 }, (_, i) => JSON.stringify({ ...SKIP, ticker: `T${i}` }));
  fs.writeFileSync(path.join(d, DEALER_GAMMA_DECISIONS_FILE), [lines[0], "{torn", ...lines.slice(1)].join("\n") + "\n");
  assert.deepEqual(readDealerGammaDecisions(d).map((x) => x.ticker), ["T0", "T1", "T2", "T3", "T4"]);
  assert.deepEqual(readDealerGammaDecisions(d, 2).map((x) => x.ticker), ["T3", "T4"]);
  // a file larger than the tail window: only whole lines from the tail come back
  const big = Array.from({ length: Math.ceil((DEALER_GAMMA_TAIL_BYTES * 1.5) / 400) },
    (_, i) => JSON.stringify({ ...SKIP, ticker: `B${i}`, reason: "r".repeat(200) }));
  fs.writeFileSync(path.join(d, DEALER_GAMMA_DECISIONS_FILE), big.join("\n") + "\n");
  const tail = readDealerGammaDecisions(d, 10_000);
  assert.ok(tail.length > 0 && tail.length < big.length);
  assert.equal(tail[tail.length - 1].ticker, `B${big.length - 1}`);
  assert.ok(tail.every((x) => typeof x.ticker === "string" && x.ticker.startsWith("B")));
});

test("probe body: shape, label fallback, per-day counts, feed subset, no raw GEX", () => {
  // 2026-11-06: the sub-object was `archive` (the squeeze_gex archive's freshness); there is no
  // archive any more, so it is `feed` (the in-memory read's status) — same five keys, same values.
  const feed: SqueezeReport = {
    probe: "squeeze_gex", stored: false, rows: 3903, first_date: "2011-05-02", latest_date: "2026-11-05", latest_age_days: 1,
    latest_gex_negative: false, latest_rank: 0.0916, rank_rows: 3652, fetched_at: "2026-11-06T05:00:12Z",
    last_refresh_ok: true, last_error: null, archive_cleanup: null, note: "n",
  };
  const state = { asOf: "2026-11-05", rank252: 0.0916, lowGamma: true, active: true, stale: false, ageSessions: 1,
    threshold: 0.2, label: "dealer gamma: low (rank 0.09) — no new condors / strangles; cash-secured puts and put spreads logged only",
    computedAt: "2026-11-06T10:00:00-05:00", gexNegative: false };
  const decisions = [SKIP, { ...SKIP, ticker: "AMD", action: "would_skip" }, { ...SKIP, day: "2026-11-05" }, { ...SKIP, action: "allow" }];
  const b = dealerGammaProbeBody(state, decisions, feed, Date.parse("2026-11-06T15:30:00Z"));
  assert.equal(b.probe, "gamma");
  assert.equal(b.label, state.label);
  assert.equal(b.state_age_minutes, 30);
  assert.deepEqual(b.counts_by_day, { "2026-11-06": { skip: 1, would_skip: 1 }, "2026-11-05": { skip: 1, would_skip: 0 } });
  assert.deepEqual(b.feed, { rows: 3903, latest_date: "2026-11-05", latest_rank: 0.0916, latest_age_days: 1, last_refresh_ok: true });
  assert.ok(!/"gex"\s*:/.test(JSON.stringify(b)), "no raw gex value in the probe");
  const empty = dealerGammaProbeBody(null, [], null);
  assert.match(empty.label, /no state yet/);
  assert.equal(empty.state_age_minutes, null);
  assert.equal(empty.feed, null);
});

test("probe registration and data dir rule", () => {
  assert.ok((DIAG_PROBES as readonly string[]).includes("gamma"));
  assert.equal(dealerGammaDataDir({ DATA_DIR: "/srv/v" } as NodeJS.ProcessEnv), "/srv/v");
  assert.equal(dealerGammaDataDir({} as NodeJS.ProcessEnv), fs.existsSync("/data") ? "/data/voltrade" : "/tmp");
});
