// Tests for the paper-account drawdown-kill AUTO-RESUME (human-directed
// 2026-09-28, KNOWN BROKEN #42/#43). The properties pinned here are the
// safety contract: an owner/unknown latch never clears, a non-paper process
// never clears, invalid reads never count, the fast path needs cooldown +
// debounce, and an explained loss waits for the paper re-baseline window.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  AUTO_RESUME,
  allPaperAlpacaUrls,
  autoResumeEligibility,
  autoResumeEnabled,
  classifyLatchReason,
  evaluateAutoResume,
  fillsWindowStartMs,
  isPaperAlpacaUrl,
  parseFills,
  parseKillSwitchRecord,
  parsePositions,
  realizedPnlFromFills,
  reconcileDrawdown,
  summarizePythonSync,
  type AutoResumeDecision,
  type AutoResumeInput,
} from "./killSwitchAutoResume";

const PAPER = "https://paper-api.alpaca.markets";
const PEAK = 110727.04;
const MIN = 60_000;
const t = (iso: string) => Date.parse(iso);

// A quiet book: 3 ETF longs + one short put, unrealized ~ -$279 in total —
// the shape every session on KNOWN BROKEN #42 actually observed.
function book(unrealizedTotal = -279) {
  return [
    { symbol: "QQQ", qty: "51", side: "long", avg_entry_price: "560.10", unrealized_pl: String(unrealizedTotal + 100), asset_class: "us_equity" },
    { symbol: "SMH", qty: "20", side: "long", avg_entry_price: "290.00", unrealized_pl: "-60", asset_class: "us_equity" },
    { symbol: "VXUS", qty: "133", side: "long", avg_entry_price: "70.00", unrealized_pl: "-40", asset_class: "us_equity" },
    { symbol: "HPE261016P00015000", qty: "-1", side: "short", avg_entry_price: "0.50", unrealized_pl: "0", asset_class: "us_option" },
  ];
}

function input(over: Partial<AutoResumeInput> = {}): AutoResumeInput {
  return {
    nowMs: t("2026-09-29T14:00:00Z"),          // Tue 10:00 ET, market open
    paper: true,
    killSwitch: true,
    latchReason: "drawdown-kill (tier1)",
    latchedAtMs: t("2026-09-10T03:12:26Z"),
    firstObservedAtMs: t("2026-09-29T13:30:00Z"),
    lastAutoResumeAtMs: null,
    marketOpen: true,
    account: { equity: "101094.07", status: "ACTIVE", trading_blocked: false, account_blocked: false },
    positions: book(),
    fills: { rows: [], truncated: false },
    equityPeak: PEAK,
    maxDrawdownPct: -10,
    ...over,
  };
}

/** Runs n evaluations 45s apart, threading the streak the way bot.ts does. */
function run(n: number, over: Partial<AutoResumeInput> | ((i: number) => Partial<AutoResumeInput>) = {}): AutoResumeDecision[] {
  const out: AutoResumeDecision[] = [];
  let prev = { consecutiveOk: 0 };
  for (let i = 0; i < n; i++) {
    const o = typeof over === "function" ? over(i) : over;
    const baseNow = o.nowMs ?? t("2026-09-29T14:00:00Z");
    const d = evaluateAutoResume(input({ ...o, nowMs: baseNow + i * 45_000 }), prev);
    out.push(d);
    prev = { consecutiveOk: d.consecutiveOk };
  }
  return out;
}

// ── paper gate ─────────────────────────────────────────────────────────────

test("paper detection: only the exact https paper host counts", () => {
  assert.equal(isPaperAlpacaUrl(PAPER), true);
  assert.equal(isPaperAlpacaUrl(PAPER + "/"), true);
  assert.equal(isPaperAlpacaUrl("https://api.alpaca.markets"), false, "LIVE endpoint");
  assert.equal(isPaperAlpacaUrl("http://paper-api.alpaca.markets"), false, "plain http");
  assert.equal(isPaperAlpacaUrl("https://paper-api.alpaca.markets.evil.example"), false, "lookalike host");
  assert.equal(isPaperAlpacaUrl(""), false);
  assert.equal(isPaperAlpacaUrl(undefined), false);
  assert.equal(isPaperAlpacaUrl("not a url"), false);
  assert.equal(allPaperAlpacaUrls([PAPER, PAPER]), true);
  assert.equal(allPaperAlpacaUrls([PAPER, "https://api.alpaca.markets"]), false, "one live path (e.g. ALPACA_BASE_URL env) disqualifies the process");
  assert.equal(allPaperAlpacaUrls([]), false, "nothing to verify is not verifiably paper");
});

test("non-paper process never auto-resumes, however perfect the anomaly", () => {
  const ds = run(20, { paper: false });
  for (const d of ds) {
    assert.equal(d.eligible, false);
    assert.equal(d.outcome, "INELIGIBLE");
    assert.equal(d.resume, false);
    assert.equal(d.consecutiveOk, 0);
  }
});

// ── latch reason gate ──────────────────────────────────────────────────────

test("latch reasons: both drawdown-kill sites are eligible, owner/unknown are not", () => {
  assert.equal(classifyLatchReason("drawdown-kill (tier1)"), "DRAWDOWN_KILL");
  assert.equal(classifyLatchReason("drawdown-kill (account route)"), "DRAWDOWN_KILL");
  assert.equal(classifyLatchReason("owner toggle ON"), "OWNER_MANUAL");
  assert.equal(classifyLatchReason(null), "UNKNOWN");
  assert.equal(classifyLatchReason("deploy_gate_smoke: DD-HALT latched"), "UNKNOWN");
  assert.equal(classifyLatchReason("auto-resume: DATA_ANOMALY"), "UNKNOWN");
});

test("an owner-set manual kill NEVER auto-clears — not after 30 days, not on a perfect anomaly", () => {
  const ds = run(40, { latchReason: "owner toggle ON", latchedAtMs: t("2026-08-01T00:00:00Z") });
  for (const d of ds) {
    assert.equal(d.eligible, false);
    assert.equal(d.resume, false);
    assert.match(d.reason, /owner/);
  }
});

test("a latch with an unknown or missing reason is treated like a manual kill", () => {
  for (const latchReason of [null, "", "something else"]) {
    const ds = run(10, { latchReason });
    assert.ok(ds.every((d) => !d.eligible && !d.resume), `reason ${JSON.stringify(latchReason)} must never clear`);
  }
});

test("eligibility short-circuit: not latched / not paper / not drawdown -> no broker reads needed", () => {
  assert.equal(autoResumeEligibility({ paper: true, killSwitch: false, latchReason: null }).outcome, "NOT_LATCHED");
  assert.equal(autoResumeEligibility({ paper: false, killSwitch: true, latchReason: "drawdown-kill (tier1)" }).eligible, false);
  assert.equal(autoResumeEligibility({ paper: true, killSwitch: true, latchReason: "drawdown-kill (tier1)" }).eligible, true);
});

// ── persisted file format ──────────────────────────────────────────────────

test("kill-file parse is backward compatible with every historical shape", () => {
  assert.deepEqual(parseKillSwitchRecord({ killSwitch: true }), { killSwitch: true, reason: null, savedAtMs: null, lastAutoResumeAtMs: null });
  const r16 = parseKillSwitchRecord({ killSwitch: true, reason: "drawdown-kill (tier1)", savedAt: "2026-09-10T03:12:26.354Z" });
  assert.equal(r16?.reason, "drawdown-kill (tier1)");
  assert.equal(r16?.savedAtMs, t("2026-09-10T03:12:26.354Z"));
  const cur = parseKillSwitchRecord({ killSwitch: false, reason: "auto-resume: DATA_ANOMALY", savedAt: "2026-09-29T14:02:15Z", lastAutoResumeAt: "2026-09-29T14:02:15Z" });
  assert.equal(cur?.lastAutoResumeAtMs, t("2026-09-29T14:02:15Z"));
  assert.equal(parseKillSwitchRecord({ killSwitch: "true" }), null, "same acceptance rule loadKillSwitch always had");
  assert.equal(parseKillSwitchRecord(null), null);
  assert.equal(parseKillSwitchRecord({ killSwitch: true, savedAt: "garbage" })?.savedAtMs, null);
});

// ── the anomaly fast path: cooldown + debounce ─────────────────────────────

test("DATA_ANOMALY clears only after the cooldown AND 3 consecutive OK evaluations", () => {
  const latchedAtMs = t("2026-09-29T14:00:00Z");
  // inside the cooldown: basis holds, condition does not
  const early = run(6, { latchedAtMs, nowMs: latchedAtMs + 5 * MIN });
  for (const d of early) {
    assert.equal(d.classification, "DATA_ANOMALY");
    assert.equal(d.basis, "DATA_ANOMALY");
    assert.equal(d.conditionOk, false, "cooldown not yet elapsed");
    assert.equal(d.resume, false);
    assert.match(d.reason, /cooldown/);
  }
  // after the cooldown: resume on exactly the 3rd consecutive evaluation
  const later = run(3, { latchedAtMs, nowMs: latchedAtMs + (AUTO_RESUME.COOLDOWN_MINUTES + 1) * MIN });
  assert.deepEqual(later.map((d) => d.consecutiveOk), [1, 2, 3]);
  assert.deepEqual(later.map((d) => d.resume), [false, false, true]);
  assert.equal(later[2].rebaselinePeakTo, 101094.07, "anomaly re-baselines the peak to current validated equity");
  assert.equal(later[0].rebaselinePeakTo, null, "no rebaseline before the resume itself");
});

test("invalid reads never count and reset the streak", () => {
  const bad: Array<[string, Partial<AutoResumeInput>]> = [
    ["zero equity", { account: { equity: "0", status: "ACTIVE" } }],
    ["garbage equity", { account: { equity: "NaN", status: "ACTIVE" } }],
    ["account fetch failed", { account: null }],
    ["trading blocked", { account: { equity: "101094.07", status: "ACTIVE", trading_blocked: true } }],
    ["account not ACTIVE", { account: { equity: "101094.07", status: "ACCOUNT_UPDATED" } }],
    ["positions fetch failed", { positions: null }],
    ["positions garbage", { positions: { message: "rate limited" } }],
    ["fills fetch failed", { fills: null }],
    ["fills garbage", { fills: { rows: "oops", truncated: false } }],
    ["clock failed", { marketOpen: null }],
  ];
  for (const [label, over] of bad) {
    const ds = run(5, (i) => (i === 2 ? over : {}));
    assert.deepEqual(ds.map((d) => d.consecutiveOk), [1, 2, 0, 1, 2], `${label}: the bad read must reset, not pause, the streak`);
    assert.ok(ds.every((d) => !d.resume), `${label}: no resume inside the window`);
    assert.equal(ds[2].outcome, "INVALID_READ", label);
    assert.equal(ds[2].conditionOk, false, label);
  }
});

test("market closed: evaluations do not count (tier-1 cadence) and the streak resets", () => {
  const ds = run(4, (i) => (i === 2 ? { marketOpen: false } : {}));
  assert.deepEqual(ds.map((d) => d.consecutiveOk), [1, 2, 0, 1]);
  assert.equal(ds[2].outcome, "MARKET_CLOSED");
});

// ── real loss: stays halted until the paper re-baseline window ─────────────

test("an explained (real) loss stays halted until PAPER_REBASE_AFTER_TRADING_DAYS market days, then re-baselines", () => {
  const latchedAtMs = t("2026-09-14T13:40:00Z"); // Mon 09:40 ET
  const realLoss = { latchedAtMs, positions: book(-6000) }; // explains ~62% of a ~$9.6K drop
  // Mon 11:00 ET — cooldown passed, 1.3 market hours
  const mon = run(10, { ...realLoss, nowMs: t("2026-09-14T15:00:00Z") });
  // Tue 15:00 ET — ~11.8 market hours, still short of 13
  const tue = run(10, { ...realLoss, nowMs: t("2026-09-15T19:00:00Z") });
  for (const d of [...mon, ...tue]) {
    assert.equal(d.classification, "REAL_LOSS");
    assert.equal(d.basis, null);
    assert.equal(d.consecutiveOk, 0);
    assert.equal(d.resume, false);
  }
  // Wed 10:00 ET — >= 13 market hours: paper re-baseline
  const wed = run(3, { ...realLoss, nowMs: t("2026-09-16T14:00:00Z") });
  assert.ok(wed.every((d) => d.basis === "PAPER_REBASE"));
  assert.deepEqual(wed.map((d) => d.resume), [false, false, true]);
  assert.equal(wed[2].rebaselinePeakTo, 101094.07);
  assert.ok(wed[2].numbers!.marketHoursSinceLatch >= 13);
});

test("RECOVERED: back above threshold + hysteresis resumes WITHOUT touching the (real) peak", () => {
  // -7.0% with a real, fully explained loss
  const eq = (PEAK * 0.93).toFixed(2);
  const ds = run(3, { account: { equity: eq, status: "ACTIVE" }, positions: book(-7000), latchedAtMs: t("2026-09-29T13:00:00Z") });
  assert.ok(ds.every((d) => d.basis === "RECOVERED"));
  assert.equal(ds[2].resume, true);
  assert.equal(ds[2].rebaselinePeakTo, null, "recovery keeps the real peak — the kill stays armed at -10% from it");
});

test("hysteresis: at or below kill+2pp is NOT recovered", () => {
  // clean decimals so -8.0% is exactly the boundary (strictly-above semantics)
  for (const equity of ["92000", "91500", "90500"]) { // -8.0%, -8.5%, -9.5%
    const d = evaluateAutoResume(input({
      equityPeak: 100_000,
      account: { equity, status: "ACTIVE" },
      positions: book(-8000),
      latchedAtMs: t("2026-09-29T13:00:00Z"),
    }), { consecutiveOk: 0 });
    assert.notEqual(d.basis, "RECOVERED", `equity ${equity} vs peak 100000`);
    assert.equal(d.resume, false);
  }
  const justAbove = evaluateAutoResume(input({
    equityPeak: 100_000, account: { equity: "92010", status: "ACTIVE" },
    positions: book(-8000), latchedAtMs: t("2026-09-29T13:00:00Z"),
  }), { consecutiveOk: 0 });
  assert.equal(justAbove.basis, "RECOVERED", "-7.99% is above the -8% resume line");
});

test("UNRECONCILED (truncated fill history) can never take the anomaly fast path", () => {
  const d = evaluateAutoResume(input({
    latchedAtMs: t("2026-09-29T13:00:00Z"),
    fills: { rows: [], truncated: true },
  }), { consecutiveOk: 0 });
  assert.equal(d.classification, "UNRECONCILED");
  assert.equal(d.basis, null, "not recovered, not anomaly, rebase window not reached");
});

test("anti-flap: a latch within 24h of the previous auto-resume cannot use the anomaly fast path", () => {
  const latchedAtMs = t("2026-09-29T13:00:00Z");
  const d = evaluateAutoResume(input({ latchedAtMs, lastAutoResumeAtMs: latchedAtMs - 2 * 3_600_000 }), { consecutiveOk: 0 });
  assert.equal(d.classification, "DATA_ANOMALY");
  assert.equal(d.basis, null);
  assert.match(d.reason, /anti|fast path disabled/);
  const old = evaluateAutoResume(input({ latchedAtMs, lastAutoResumeAtMs: latchedAtMs - 48 * 3_600_000 }), { consecutiveOk: 0 });
  assert.equal(old.basis, "DATA_ANOMALY");
});

test("missing latch timestamp: cooldown anchors to when this process first saw the latch", () => {
  const now = t("2026-09-29T14:00:00Z");
  const fresh = evaluateAutoResume(input({ latchedAtMs: null, firstObservedAtMs: now - 10 * MIN, nowMs: now }), { consecutiveOk: 0 });
  assert.equal(fresh.conditionOk, false);
  const aged = evaluateAutoResume(input({ latchedAtMs: null, firstObservedAtMs: now - 45 * MIN, nowMs: now }), { consecutiveOk: 0 });
  assert.equal(aged.conditionOk, true);
});

// ── today's live state (2026-09-28 brief) ──────────────────────────────────

test("live scenario: latched since 09-10, -8.7% vs peak 110727.04, quiet book -> DATA_ANOMALY resume + rebaseline", () => {
  const ds = run(3);
  assert.equal(ds[0].classification, "DATA_ANOMALY");
  assert.equal(ds[0].numbers!.drawdownPct, -8.7);
  assert.ok(ds[0].numbers!.explainedRatio! < 0.05);
  assert.deepEqual(ds.map((d) => d.resume), [false, false, true]);
  assert.equal(ds[2].basis, "DATA_ANOMALY");
  assert.equal(ds[2].rebaselinePeakTo, 101094.07);
});

test("live scenario, pessimistic book: an explained drop still resumes via the elapsed paper re-baseline", () => {
  const ds = run(3, { positions: book(-4000) });
  assert.equal(ds[0].classification, "REAL_LOSS");
  assert.ok(ds.every((d) => d.basis === "PAPER_REBASE"), "18 days latched >> 2 market days");
  assert.equal(ds[2].resume, true);
  assert.equal(ds[2].rebaselinePeakTo, 101094.07);
});

// ── reconciliation arithmetic ──────────────────────────────────────────────

test("realized P&L: round trip inside the window is exact", () => {
  const fills = parseFills([
    { symbol: "GLD", side: "buy", qty: "20", price: "300", transaction_time: "2026-09-02T14:00:00Z" },
    { symbol: "GLD", side: "sell", qty: "20", price: "297.5", transaction_time: "2026-09-03T14:00:00Z" },
  ])!;
  const r = realizedPnlFromFills(fills.rows, []);
  assert.equal(r.realized, -50);
  assert.equal(r.unpricedCloses, 0);
});

test("realized P&L: trimming a still-held position uses its current avg entry; options x100; shorts sign-correct", () => {
  const positions = parsePositions([
    { symbol: "QQQ", qty: "36", side: "long", avg_entry_price: "500", unrealized_pl: "10" },
  ])!;
  const fills = parseFills([
    { symbol: "QQQ", side: "sell", qty: "15", price: "510", transaction_time: "2026-09-05T15:00:00Z" },
    { symbol: "BAC261016P00057500", side: "sell_short", qty: "1", price: "1.20", transaction_time: "2026-09-04T15:00:00Z" },
    { symbol: "BAC261016P00057500", side: "buy", qty: "1", price: "0.40", transaction_time: "2026-09-08T15:00:00Z" },
  ])!;
  const r = realizedPnlFromFills(fills.rows, positions);
  // QQQ: 15 * (510-500) = 150 ; short put: (1.20-0.40) * 1 * 100 = 80
  assert.equal(r.realized, 230);
  assert.equal(r.unpricedCloses, 0);
});

test("realized P&L: a close against a pre-window basis is counted as unpriced, never guessed", () => {
  const fills = parseFills([
    { symbol: "FCEL", side: "sell", qty: "70", price: "4.10", transaction_time: "2026-09-05T15:00:00Z" },
  ])!;
  const r = realizedPnlFromFills(fills.rows, []);
  assert.equal(r.realized, 0);
  assert.equal(r.unpricedCloses, 70);
  const rec = reconcileDrawdown({ equity: 100000, peak: 110000, positions: [], fills: fills.rows, fillsTruncated: false, malformedFills: 0 });
  assert.equal(rec.classification, "UNRECONCILED", "unknown basis cannot prove an anomaly");
});

test("reconciliation reproduces the 2026-09-09 incident numbers as a DATA_ANOMALY", () => {
  // reconstructed -$414.82 against a reported -$12,059.74 drop
  const positions = parsePositions([{ symbol: "QQQ", qty: "51", side: "long", avg_entry_price: "560", unrealized_pl: "-414.82" }])!;
  const rec = reconcileDrawdown({ equity: 90748, peak: 90748 + 12059.74, positions, fills: [], fillsTruncated: false, malformedFills: 0 });
  assert.equal(rec.classification, "DATA_ANOMALY");
  assert.ok(Math.abs(rec.explainedRatio! - 0.0344) < 0.001);
  // and a genuinely explained drop is not
  const real = parsePositions([{ symbol: "QQQ", qty: "51", side: "long", avg_entry_price: "560", unrealized_pl: "-9000" }])!;
  assert.equal(reconcileDrawdown({ equity: 90748, peak: 102807.74, positions: real, fills: [], fillsTruncated: false, malformedFills: 0 }).classification, "REAL_LOSS");
});

test("gains offset losses in the explained figure (net, clipped at zero)", () => {
  const positions = parsePositions([
    { symbol: "A", qty: "10", side: "long", avg_entry_price: "10", unrealized_pl: "3000" },
    { symbol: "B", qty: "10", side: "long", avg_entry_price: "10", unrealized_pl: "-1000" },
  ])!;
  const rec = reconcileDrawdown({ equity: 95000, peak: 105000, positions, fills: [], fillsTruncated: false, malformedFills: 0 });
  assert.equal(rec.explainedLoss, 0);
  assert.equal(rec.classification, "DATA_ANOMALY");
});

test("fills window is bounded to FILLS_LOOKBACK_DAYS before the latch", () => {
  const latch = t("2026-09-10T03:12:26Z");
  assert.equal(fillsWindowStartMs(latch), latch - AUTO_RESUME.FILLS_LOOKBACK_DAYS * 86_400_000);
});

test("constants the audit trail depends on are the documented values", () => {
  assert.equal(AUTO_RESUME.COOLDOWN_MINUTES, 30);
  assert.equal(AUTO_RESUME.CONSECUTIVE_OK_REQUIRED, 3);
  assert.equal(AUTO_RESUME.ANOMALY_EXPLAINED_RATIO, 0.25);
  assert.equal(AUTO_RESUME.RECOVERY_HYSTERESIS_PP, 2);
  assert.equal(AUTO_RESUME.PAPER_REBASE_AFTER_TRADING_DAYS, 2);
  assert.ok(Object.isFrozen(AUTO_RESUME));
});

test("python-sync summary: applied report, failure, and malformed shapes all produce an honest line", () => {
  const ok = summarizePythonSync(true, {
    applied: true, basis: "DATA_ANOMALY",
    bot_engine_dd: { before: { peak_equity: 110727.04, halted: true }, after: { peak_equity: 101094.07, halted: false }, rebaselined: true },
    risk_kill_switch: { killed: false, reset: false, manual_kill_file: false },
    shared_peak: { peak_equity: 110727.04, drawdown_pct_vs_equity: -8.7 },
    notes: ["voltrade_peak_equity.json left at its stale peak"],
  }, "daemon");
  assert.match(ok, /applied/);
  assert.match(ok, /halted=true peak=110727.04 -> 101094.07 \(re-baselined, halt cleared\)/);
  assert.match(ok, /never lowered/);
  assert.match(ok, /stale peak/);
  const manual = summarizePythonSync(true, { applied: true, bot_engine_dd: {}, risk_kill_switch: { killed: true, manual_kill_file: true } }, "daemon");
  assert.match(manual, /MANUAL kill file present/);
  const failed = summarizePythonSync(false, { error: "daemon paper_resume_sync failed: boom" }, "none");
  assert.match(failed, /sync FAILED \(via none: daemon paper_resume_sync failed: boom\)/);
  assert.match(summarizePythonSync(true, null, "subprocess"), /sync FAILED/);
  assert.match(summarizePythonSync(true, "garbage", "subprocess"), /sync FAILED/);
});

test("env rollback lever: VOLTRADE_KILL_AUTO_RESUME=off restores owner-only clearing", () => {
  assert.equal(autoResumeEnabled({}), true, "unset = enabled (the human's directive)");
  assert.equal(autoResumeEnabled({ VOLTRADE_KILL_AUTO_RESUME: "on" }), true);
  for (const v of ["off", "OFF", "0", "false", "disabled", "no", " off "]) {
    assert.equal(autoResumeEnabled({ VOLTRADE_KILL_AUTO_RESUME: v }), false, v);
  }
  const ds = run(10, { enabled: false });
  for (const d of ds) {
    assert.equal(d.eligible, false);
    assert.equal(d.resume, false);
    assert.match(d.reason, /VOLTRADE_KILL_AUTO_RESUME/);
  }
});
