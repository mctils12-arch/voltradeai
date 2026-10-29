// Brain page (2026-10-29) — the pure shaping in server/brain.ts: status from
// env/config, short- and long-term shaping, the "not yet measured" honesty,
// the agreement flag, the registry <-> research/brain_inventory.md sync, every
// quoted long-term figure still present in its source file, and the route +
// daemon wiring.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { planPythonExec } from "./pythonExec";
import {
  BRAIN_COMPONENTS, LONG_EVIDENCE, LONG_COMBINATIONS, BRAIN_CONFIG_MIRROR, BRAIN_EVIDENCE_CMD, BRAIN_RULE,
  agreementOf, brainBody, deriveStatus, flagRaw, parseNamedSet, parseOptionsGates, readSwitches, shortTermFor, longTermFor,
  SELECTION_GUARDS, type EvidencePayload, type BrainComponent,
} from "./brain";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, "..");
const inventory = fs.readFileSync(path.join(root, "research", "brain_inventory.md"), "utf8");
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
const daemon = fs.readFileSync(path.join(root, "voltrade_daemon.py"), "utf8");
const comp = (id: string): BrainComponent => {
  const c = BRAIN_COMPONENTS.find((x) => x.id === id);
  assert.ok(c, `component ${id} missing`);
  return c;
};
const status = (id: string, env: NodeJS.ProcessEnv, cfg: Record<string, unknown> | null = null) =>
  deriveStatus(comp(id).status, readSwitches(env, cfg));

test("registry: unique ids; every kind present; every entry is tied to the inventory or to its own doc", () => {
  const ids = BRAIN_COMPONENTS.map((c) => c.id);
  assert.equal(new Set(ids).size, ids.length);
  for (const k of ["input", "rule", "model", "sleeve"]) assert.ok(BRAIN_COMPONENTS.some((c) => c.kind === k), k);
  for (const c of BRAIN_COMPONENTS) {
    if (c.inventory_ref) assert.ok(inventory.includes(c.inventory_ref), `${c.id}: inventory row "${c.inventory_ref}" not in research/brain_inventory.md`);
    else assert.ok(c.ref_file && fs.existsSync(path.join(root, c.ref_file)), `${c.id}: no inventory row and no existing ref_file`);
    assert.ok(c.does.length > 10, `${c.id} needs a one-line description`);
  }
});

test("registry: the inventory's score adjustments, ML, floors and halts all have a component (doc -> registry)", () => {
  for (const row of ["Headline sentiment", "Alt composite", "Fundamentals (rating", "Reddit sentiment / buzz", "LightGBM global model",
    "Floor basket (QQQ .50", "Defensive GLD floor", "Third leg: SVXY", "QQQ put convexity overlay", "Node drawdown kill", "strategyWeights"]) {
    assert.ok(BRAIN_COMPONENTS.some((c) => c.inventory_ref === row), `inventory row "${row}" has no Brain component`);
  }
});

test("long-term evidence: every quoted figure is still in its source file (doc drift fails CI)", () => {
  for (const it of [...Object.values(LONG_EVIDENCE), ...LONG_COMBINATIONS]) {
    const text = fs.readFileSync(path.join(root, it.source_file), "utf8");
    assert.ok(it.quotes.length > 0, `${it.id} quotes nothing`);
    for (const q of it.quotes) assert.ok(text.includes(q), `${it.id}: "${q}" not found in ${it.source_file}`);
  }
  for (const c of BRAIN_COMPONENTS) for (const id of c.long.items) assert.ok(LONG_EVIDENCE[id], `${c.id} cites unknown evidence ${id}`);
});

test("flagRaw: env PRESENT wins (even empty / off); absent -> config default; True -> '1'", () => {
  assert.deepEqual(flagRaw("VOLTRADE_REGIME_ENGINE", {}, BRAIN_CONFIG_MIRROR), { value: "1", source: "config" });
  assert.deepEqual(flagRaw("VOLTRADE_REGIME_ENGINE", { VOLTRADE_REGIME_ENGINE: "" }, BRAIN_CONFIG_MIRROR), { value: "", source: "env" });
  assert.deepEqual(flagRaw("VOLTRADE_SHADOW_ENTRIES", {}, BRAIN_CONFIG_MIRROR), { value: "stocks,options", source: "config" });
  assert.deepEqual(flagRaw("VOLTRADE_REGIME_LABELS", {}, { REGIME_LABELS_DEFAULT: false }), { value: "", source: "config" });
});

test("parse: selection/sizing sets and the options crisis+vrp rule mirror regime_engine", () => {
  assert.deepEqual([...parseNamedSet("all", SELECTION_GUARDS)].sort(), [...SELECTION_GUARDS].sort());
  assert.deepEqual([...parseNamedSet("off", SELECTION_GUARDS)], []);
  assert.deepEqual([...parseNamedSet("knife, bogus,rs", SELECTION_GUARDS)].sort(), ["knife", "rs"]);
  assert.deepEqual([...parseOptionsGates("earnings")].sort(), ["crisis", "earnings", "vrp"]);
  assert.deepEqual([...parseOptionsGates("0")], []);
});

test("status: defaults (both entry sleeves shadowed) — score inputs are SHADOW, floor/exits LIVE, dead code OFF", () => {
  const env = {};
  assert.equal(status("stock_entries", env).status, "shadow");
  assert.equal(status("stock_entries", env).status_source, "config");
  assert.equal(status("options_entries", env).status, "shadow");
  assert.equal(status("news", env).status, "shadow");
  assert.match(status("news", env).detail, /nothing it touches is ordered/);
  assert.equal(status("floor_basket", env).status, "live");
  assert.equal(status("floor_basket", env).status_source, "config");
  assert.equal(status("live_exits", env).status, "live");
  assert.equal(status("live_exits", env).status_source, "code");
  assert.equal(status("strategy_weights", env).status, "off");
  assert.equal(status("google_trends", env).status, "off");
  assert.equal(status("ml_v3", env).status, "shadow");
  assert.equal(status("forecast", env).status, "shadow");
  assert.equal(status("regime_sizing", env).status, "shadow");
  // regime guards are gated but stock entries are shadowed -> they only block would-be entries
  assert.equal(status("sel_knife", env).status, "shadow");
  assert.match(status("sel_knife", env).detail, /would-be entries only/);
  // the stress multiplier is replaced by the regime engine's sizing on the stock sleeve
  assert.equal(status("stress_index", env).status, "off");
  // convexity: CONVEXITY_OVERLAY.enabled is never read; the options shadow decides
  assert.equal(status("convexity", env).status, "shadow");
  assert.match(status("convexity", env).detail, /never read/);
  assert.equal(status("third_leg", env).status, "shadow");
});

test("status: env flips — entries live, regime flags off, an explicit empty env value wins over the config default", () => {
  const live = { VOLTRADE_SHADOW_ENTRIES: "off" };
  assert.equal(status("stock_entries", live).status, "live");
  assert.equal(status("stock_entries", live).status_source, "env");
  assert.equal(status("news", live).status, "live");
  assert.equal(status("sel_knife", live).status, "live");
  assert.equal(status("csp_block_guards", live).status, "live");
  assert.equal(status("convexity", live).status, "live");
  assert.equal(status("third_leg", live).status, "live");
  const selOff = { VOLTRADE_SHADOW_ENTRIES: "off", VOLTRADE_REGIME_SELECTION: "" };
  assert.equal(status("sel_knife", selOff).status, "shadow");
  assert.equal(status("sel_knife", selOff).status_source, "env");
  assert.match(status("sel_knife", selOff).detail, /flag-only/);
  const engOff = { VOLTRADE_REGIME_ENGINE: "0" };
  assert.equal(status("sel_dip", engOff).status, "off");
  assert.equal(status("regime_engine", engOff).status, "off");
  assert.equal(status("regime_engine", engOff).status_source, "env");
  assert.equal(status("regime_engine", {}).status, "live");
  assert.equal(status("regime_sizing", engOff).status, "off");
  assert.equal(status("regime_labels", engOff).status, "off");
  assert.equal(status("stress_index", { ...engOff, VOLTRADE_SHADOW_ENTRIES: "options" }).status, "live");
  assert.equal(status("options_entries", { VOLTRADE_SHADOW_ENTRIES: "stocks" }).status, "live");
});

test("status: config from Python wins over the mirror; mirror used (and said) when Python failed", () => {
  const py = { ...BRAIN_CONFIG_MIRROR, FLOOR_BASKET_ENABLED: false, LEG3_VRP_PCT: 0, TIERS_ENABLED: { "1": true, "4": true } };
  assert.equal(status("floor_basket", {}, py).status, "off");
  assert.equal(status("third_leg", {}, py).status, "off");
  assert.equal(status("tier4", {}, py).status, "live");
  assert.equal(readSwitches({}, py).config_source, "python");
  const sw = readSwitches({}, { error: "ImportError" });
  assert.equal(sw.config_source, "mirror");
  assert.match(deriveStatus(comp("floor_basket").status, sw).detail, /mirror/);
  assert.equal(status("tier4", {}, null).status, "off");
});

test("honesty: with no evidence payload nothing is invented — every component is 'not measured' or doc-only", () => {
  const body = brainBody(null, {}, new Date("2026-10-29T12:00:00Z"), null, "daemon down");
  assert.equal(body.evidence_error, "daemon down");
  for (const c of body.components) {
    assert.equal(c.short_term.measured, false, c.id);
    assert.ok(c.short_term.reason && c.short_term.reason.length > 0, c.id);
    assert.equal(c.short_term.n, undefined, `${c.id} must not carry an n`);
    if (c.long_term.items.length === 0) {
      assert.equal(c.long_term.measured, false);
      assert.match(c.long_term.reason ?? "", /Not yet measured|Not measured|Only the fit-time/);
    }
    assert.notEqual(c.agreement.flag, "agree", `${c.id} cannot agree without short-term evidence`);
  }
  assert.equal(body.combinations.short_term.available, false);
  assert.match(String(body.combinations.short_term.reason), /unavailable/);
  assert.equal(body.rule, BRAIN_RULE);
  assert.ok(body.long_term.gaps.some((g) => /walk-forward/.test(g)));
  const sum = body.status_counts.live + body.status_counts.shadow + body.status_counts.off;
  assert.equal(sum, BRAIN_COMPONENTS.length);
});

test("honesty: a component with no long-horizon test says 'Not yet measured' with a reason, sign insufficient", () => {
  const l = longTermFor(comp("kill_switches"), null);
  assert.equal(l.measured, false);
  assert.equal(l.sign, "insufficient");
  assert.match(l.reason ?? "", /Not yet measured/);
  const news = longTermFor(comp("news"), null);
  assert.equal(news.measured, false);
  assert.match(news.reason ?? "", /walk-forward panel harness is not built/);
});

const EV: EvidencePayload = {
  generated_at: "2026-10-29T11:00:00Z",
  config: { ...BRAIN_CONFIG_MIRROR },
  shadow_log: { records: 9000, first_day: "2026-06-01", last_day: "2026-10-28" },
  spy: { ok: true, first_day: "2026-05-25", last_day: "2026-10-28" },
  buckets: {
    rejected_score: { records: 4000, xa: { n: 400, n_days: 80, mean_pct: -0.6, hit_rate: 0.41, ci95: [-0.9, -0.3], first_day: "2026-06-01", last_day: "2026-10-01", vs_spy: { n: 400, mean_excess_pct: -0.8, ci95: [-1.1, -0.5], beat_rate: 0.38 } }, "+5d": { n: 380, mean_pct: -0.4, hit_rate: 0.4 }, "+20d": { n: 300, mean_pct: -0.2 } },
    "shadow_mode_entry:stocks": { records: 20, xa: { n: 12, n_days: 3, mean_pct: 0.5, hit_rate: 0.6, thin: true } },
    would_reject_regime_knife: { records: 200, xa: { n: 60, n_days: 30, mean_pct: 0.9, hit_rate: 0.6, ci95: [0.2, 1.6], vs_spy: { n: 60, mean_excess_pct: 0.7, ci95: [0.1, 1.3], beat_rate: 0.6 } } },
  },
  source_validation: {
    method: { prereg_version: "source_validation_v2 (2026-10-27)", multiple_testing: "Benjamini-Hochberg across 52 sources at FDR 0.1" },
    population: { n_unique: 900, n_days: 90, first_day: "2026-06-01", last_day: "2026-10-20" },
    summary: { PASS: [], FAIL_INVERTED: ["adj.news"], FAIL: ["adj.alt"], INSUFFICIENT: ["adj.insider_mspr"], CONSTANT: [] },
    sources: {
      "adj.news": { n: 900, n_days: 90, ic: -0.09, ci95: [-0.14, -0.04], q_bh: 0.01, verdict: "FAIL_INVERTED", verdict_reason: "BH-significant NEGATIVE IC" },
      "adj.alt": { n: 900, ic: 0.01, ci95: [-0.04, 0.06], q_bh: 0.8, verdict: "FAIL", verdict_reason: "no BH-significant positive IC" },
      "adj.insider_mspr": { n: 40, verdict: "INSUFFICIENT", verdict_reason: "n=40 < MIN_N=100" },
    },
    reference: { score: { n: 900, ic: -0.06, ci95: [-0.1, -0.02] }, ml_only_score: { n: 900, ic: 0.01, ci95: [-0.03, 0.05] } },
    selection: { features: { status: "NOTHING_SELECTED", reason: "no source added significant out-of-sample IC", n_rows: 800, candidates: 30, steps: [{ candidate: "feat.rsi_14", select_ic: 0.04, gain: 0.04, se: 0.03, p_bonferroni: 1, tried: 30, accepted: false }], selected: [], variants_tried: 30 },
      breakdown: { status: "INSUFFICIENT", reason: "250 rows < MIN_SELECT_ROWS=300", n_rows: 250, candidates: 40, variants_tried: 0 } },
    ablation: { status: "OK", n_rows_with_components: 300, baseline_inverted: true, un_inverts: true, improvement_significant: false, variants: { baseline: { n: 300, ic: -0.05, ci95: [-0.1, 0.0], removed: [] }, drop_unvalidated: { n: 300, ic: 0.02, ci95: [-0.03, 0.07], removed: ["intel", "alt", "social", "news", "fundamental"], ic_change_vs_baseline: 0.07, ic_change_se: 0.05 } } },
  },
  ml_v3: { trained_at: "2026-10-27T23:46:14Z", rows: 66833, date_range: ["2023-12-28", "2026-09-28"], auc_ci95_date_block: [0.479, 0.5155], pooled: { n: 55775, auc: 0.4991, log_loss: 0.6949, base_rate_log_loss: 0.6928 }, top_minus_bottom_pct: -0.125, variants_tried: 1 },
  ml_v3_shadow_ic: { n: 0, n_days: 0 },
  forecast: { generated_at: "2026-10-29T10:00:00Z", accuracy: { n: 7, thin: true, hits: 4, hit_rate: 0.57, brier: 0.24, brier_skill_vs_coinflip: 0.04 }, weeks: 1, holdout: { split_date: "2025-12-04", n_test: 940, n_train: 2160, brier: 0.245, brier_climatology: 0.246, brier_skill: 0.0045, hit_rate: 0.557 }, fit_window: ["2023-10-20", "2026-10-16"] },
};

test("short-term: a blocking rule whose blocked names lost vs SPY 'earns its keep' (positive); window + provenance carried", () => {
  const s = shortTermFor(comp("min_score"), EV);
  assert.equal(s.measured, true);
  assert.equal(s.bucket, "rejected_score");
  assert.equal(s.n, 400);
  assert.deepEqual(s.ci95, [-0.9, -0.3]);
  assert.equal(s.vs_spy?.mean_excess_pct, -0.8);
  assert.equal(s.sign, "positive");
  assert.match(s.sign_note, /earns its keep/);
  assert.deepEqual(s.window, { first: "2026-06-01", last: "2026-10-01" });
  assert.equal(s.as_of, "2026-10-29T11:00:00Z");
  assert.ok(s.horizons?.["+5d"]);
});

test("short-term: a guard whose blocked names beat SPY blocks winners (negative); would_reject bucket used when no gated bucket", () => {
  const s = shortTermFor(comp("sel_knife"), EV);
  assert.equal(s.bucket, "would_reject_regime_knife");
  assert.equal(s.sign, "negative");
  assert.match(s.sign_note, /blocks winners/);
});

test("short-term: thin bucket -> insufficient, never a verdict", () => {
  const s = shortTermFor(comp("stock_entries"), EV);
  assert.equal(s.bucket, "shadow_mode_entry:stocks");
  assert.equal(s.n, 12);
  assert.equal(s.sign, "insufficient");
});

test("short-term: sources carry the BH verdict; FAIL_INVERTED -> negative, FAIL -> null, INSUFFICIENT -> insufficient", () => {
  const n = shortTermFor(comp("news"), EV);
  assert.equal(n.verdict, "FAIL_INVERTED");
  assert.equal(n.q_bh, 0.01);
  assert.equal(n.sign, "negative");
  assert.match(n.source, /BH FDR 0.10/);
  assert.equal(shortTermFor(comp("alt"), EV).sign, "null");
  assert.equal(shortTermFor(comp("insider_mspr"), EV).sign, "insufficient");
  assert.equal(shortTermFor(comp("final_score"), EV).sign, "negative");
  assert.equal(shortTermFor(comp("options_entries"), EV).measured, false);
  assert.match(String(shortTermFor(comp("options_entries"), EV).reason), /not option P&L/);
});

test("long-term: ML v3 live eval is prepended from the volume file; doc evidence keeps its file + date", () => {
  const l = longTermFor(comp("ml_v3"), EV);
  assert.equal(l.items[0].id, "ml_v3_live");
  assert.equal(l.items[0].sign, "null");
  assert.equal(l.items[0].date, "2026-10-27");
  assert.equal(l.items[1].source_file, "research/ml_v3_eval.md");
  assert.equal(longTermFor(comp("forecast"), EV).items[0].id, "forecast_holdout");
});

test("agreement: same sign agrees, opposite disagrees (live beats backtest), any insufficient is insufficient", () => {
  assert.equal(agreementOf("positive", "positive").flag, "agree");
  assert.equal(agreementOf("null", "null").flag, "agree");
  const d = agreementOf("negative", "positive");
  assert.equal(d.flag, "disagree");
  assert.match(d.note, /live beats backtest/);
  assert.equal(agreementOf("null", "positive").flag, "disagree");
  assert.equal(agreementOf("insufficient", "positive").flag, "insufficient");
  assert.equal(agreementOf("positive", "insufficient").flag, "insufficient");
  // end to end: ML v2 short (IC CI spans 0 -> null) vs long (honest AUC ~ chance -> null) = agree
  const body = brainBody(EV, {}, new Date("2026-10-29T12:00:00Z"), "daemon");
  const ml = body.components.find((c) => c.id === "ml_v2");
  assert.equal(ml?.agreement.flag, "agree");
});

test("combinations: forward selection carries variants tried + Bonferroni p; ablation rows; long-term arms carry their discount", () => {
  const body = brainBody(EV, {}, new Date("2026-10-29T12:00:00Z"), "daemon");
  const c = body.combinations.short_term;
  assert.equal(c.available, true);
  assert.equal(c.family, "Benjamini-Hochberg across 52 sources at FDR 0.1");
  const f = c.arms.find((a) => a.arm === "features");
  assert.equal(f?.variants_tried, 30);
  assert.equal(f?.steps[0].p_bonferroni, 1);
  assert.equal(c.ablation?.un_inverts, true);
  assert.equal(c.ablation?.rows.find((r) => r.name === "drop_unvalidated")?.ic_change_vs_baseline, 0.07);
  for (const lc of body.combinations.long_term) { assert.ok(lc.variants_tried > 1); assert.ok(lc.discount.length > 20); }
});

test("wiring: /api/bot/brain is owner-only, cached 10 min, daemon-first with a non-trivial subprocess fallback; daemon routes it", () => {
  const i = bot.indexOf('app.get("/api/bot/brain", requireOwner');
  assert.ok(i > 0, "route missing or not owner-only");
  assert.ok(bot.includes('pythonCall("brain_evidence", {}, BRAIN_EVIDENCE_CMD, { timeout: 85_000'));
  assert.ok(bot.includes("BRAIN_CACHE_MS"));
  assert.ok(daemon.includes('"brain_evidence": ("brain_evidence", "brain_evidence")'));
  const plan = planPythonExec(BRAIN_EVIDENCE_CMD, 85_000);
  assert.equal(plan.lane, "mutex");
  assert.ok(plan.timeoutMs <= 90_000 && plan.timeoutMs >= 60_000);
});
