/**
 * brain.ts — the owner's Brain page (2026-10-29, human: "how would I know
 * what is being used at the moment and what combinations could have produced
 * better results ... the short term is one thing but the long term is
 * different"). GET /api/bot/brain (bot.ts, requireOwner, 10-min cache).
 *
 * Pure shaping (node:test safe). Three inputs:
 *   1. BRAIN_COMPONENTS — a typed registry of every input, rule, model and
 *      sleeve, catalogued from research/brain_inventory.md (each entry names
 *      the exact inventory row text it mirrors; brain.test.ts fails when the
 *      doc row is renamed or removed). Components added after that inventory
 *      (regime engine, entry shadow, ML v3, forecast, edge P1) cite their own
 *      doc instead.
 *   2. STATUS — derived LIVE from the real switches, never from the doc's
 *      status column: the env var when PRESENT (any value wins), else the
 *      system_config BASE_CONFIG default (regime_engine.flag_raw /
 *      entry_shadow.flag_raw rule). The BASE_CONFIG values come from
 *      brain_evidence.config_snapshot() when the Python call succeeded, else
 *      from BRAIN_CONFIG_MIRROR (test_brain_evidence.py pins the mirror).
 *      "live" = reaches a real order today; "shadow" = computed and logged,
 *      nothing ordered; "off" = not computed / not read.
 *   3. EVIDENCE — short-term = live shadow outcomes (brain_evidence.py over
 *      the shadow log: decision buckets with SPY-relative CIs, the
 *      source_validation report); long-term = multi-year tests that EXIST in
 *      the repo (LONG_EVIDENCE below, numbers copied from the cited files —
 *      brain.test.ts checks every quoted figure is still in its file). Where
 *      nothing exists the component says "not yet measured" and why.
 *
 * The agreement flag is mechanical: each side reduces to a sign (helps /
 * hurts / no effect / insufficient); same sign = agree, different = disagree,
 * either insufficient = insufficient. Rule: live beats backtest when they
 * disagree (CLAUDE.md REASONING STANDARD #9).
 */
import { parseShadowSleeves } from "./entryShadow";

export type BrainKind = "input" | "rule" | "model" | "sleeve";
export type BrainStatus = "live" | "shadow" | "off";
export type StatusSource = "config" | "env" | "code";
export type Sign = "positive" | "negative" | "null" | "insufficient";

export const BRAIN_CACHE_MS = 10 * 60_000;
export const BRAIN_RULE = "Live beats backtest when they disagree: a component is promoted only when short-term (live shadow) and long-term (multi-year) evidence agree; when they disagree, the live result wins and the gap is itself the next question.";
export const BRAIN_EXPLAINER = "Short-term = live shadow outcomes (what the bot's own candidates did after it scored them); long-term = multi-year tests. A component is promoted only when both agree.";
export const BRAIN_EVIDENCE_CMD = `python3 -c "import json; from brain_evidence import brain_evidence; print(json.dumps(brain_evidence(), default=str))"`;
export const BUCKET_MIN_N = 30;   // == brain_evidence.BUCKET_MIN_N
export const PRIMARY_BUCKET_OUTCOME = "xa";

/** BASE_CONFIG values the status rules read — a MIRROR used only when the Python call failed (test_brain_evidence.py pins it). */
export const BRAIN_CONFIG_MIRROR: Readonly<Record<string, unknown>> = Object.freeze({
  SHADOW_ENTRY_SLEEVES: "stocks,options",
  REGIME_ENGINE_ENABLED: true,
  REGIME_SELECTION_DEFAULT: "all",
  REGIME_OPTIONS_DEFAULT: "all",
  REGIME_SIZING_DEFAULT: "all",
  REGIME_LABELS_DEFAULT: true,
  FLOOR_BASKET_ENABLED: true,
  DEFENSIVE_FLOOR_ENABLED: true,
  LEG3_VRP_PCT: 0.15,
  DRAWDOWN_HALT_ENABLED: true,
  POSITION_HARD_STOP_ENABLED: true,
  CONVEXITY_OVERLAY_enabled: false,
});

/** env var -> BASE_CONFIG key (regime_engine.FLAG_DEFAULTS + entry_shadow.CONFIG_KEY). */
export const FLAG_CONFIG_KEYS: Readonly<Record<string, string>> = Object.freeze({
  VOLTRADE_REGIME_ENGINE: "REGIME_ENGINE_ENABLED",
  VOLTRADE_REGIME_SELECTION: "REGIME_SELECTION_DEFAULT",
  VOLTRADE_REGIME_OPTIONS: "REGIME_OPTIONS_DEFAULT",
  VOLTRADE_REGIME_SIZING: "REGIME_SIZING_DEFAULT",
  VOLTRADE_REGIME_LABELS: "REGIME_LABELS_DEFAULT",
  VOLTRADE_SHADOW_ENTRIES: "SHADOW_ENTRY_SLEEVES",
});

const OFF_VALUES = new Set(["", "0", "off", "false", "none"]);
const ALL_VALUES = new Set(["1", "all", "true", "on"]);
export const SELECTION_GUARDS = ["knife", "dip", "panic", "rs", "breadth"] as const;
export const OPTIONS_GUARDS = ["crisis", "vrp", "inversion", "elevated", "earnings", "ebp"] as const;
export const SIZING_SLEEVES = ["stock", "options", "tier1"] as const;

export interface FlagValue { value: string; source: "env" | "config" }

/** regime_engine.flag_raw / entry_shadow.flag_raw: env when PRESENT, else the config default as a string (True -> "1"). */
export function flagRaw(envName: string, env: NodeJS.ProcessEnv, config: Readonly<Record<string, unknown>>): FlagValue {
  const v = env[envName];
  if (v !== undefined && v !== null) return { value: String(v), source: "env" };
  const d = config[FLAG_CONFIG_KEYS[envName] ?? ""];
  if (d === true) return { value: "1", source: "config" };
  if (d === undefined || d === null || d === false) return { value: "", source: "config" };
  return { value: String(d), source: "config" };
}

/** regime_engine.selection_gates / sizing_sleeves: off -> none, all -> every name, else the named ones. */
export function parseNamedSet(raw: string, names: readonly string[]): Set<string> {
  const v = String(raw ?? "").trim().toLowerCase();
  if (OFF_VALUES.has(v)) return new Set();
  if (ALL_VALUES.has(v)) return new Set(names);
  return new Set(v.split(",").map((s) => s.trim()).filter((s) => names.includes(s)));
}

/** regime_engine.options_gates: any non-empty value also turns on crisis + vrp. */
export function parseOptionsGates(raw: string): Set<string> {
  const s = parseNamedSet(raw, OPTIONS_GUARDS);
  const v = String(raw ?? "").trim().toLowerCase();
  if (!OFF_VALUES.has(v) && !ALL_VALUES.has(v)) { s.add("crisis"); s.add("vrp"); }
  return s;
}

export interface Switches {
  config: Readonly<Record<string, unknown>>;
  config_source: "python" | "mirror";
  entries: { stocks: boolean; options: boolean; flag: FlagValue };
  engine: { on: boolean; flag: FlagValue };
  selection: { gates: string[]; flag: FlagValue };
  options: { gates: string[]; flag: FlagValue };
  sizing: { sleeves: string[]; flag: FlagValue };
  labels: { on: boolean; flag: FlagValue };
}

export function readSwitches(env: NodeJS.ProcessEnv, pyConfig: Record<string, unknown> | null): Switches {
  const usable = pyConfig && !pyConfig.error && typeof pyConfig === "object" && "SHADOW_ENTRY_SLEEVES" in pyConfig;
  const config = usable ? pyConfig : BRAIN_CONFIG_MIRROR;
  const ent = flagRaw("VOLTRADE_SHADOW_ENTRIES", env, config);
  const sleeves = parseShadowSleeves(ent.value);
  const eng = flagRaw("VOLTRADE_REGIME_ENGINE", env, config);
  const sel = flagRaw("VOLTRADE_REGIME_SELECTION", env, config);
  const opt = flagRaw("VOLTRADE_REGIME_OPTIONS", env, config);
  const siz = flagRaw("VOLTRADE_REGIME_SIZING", env, config);
  const lab = flagRaw("VOLTRADE_REGIME_LABELS", env, config);
  const truthy = (s: string) => ["1", "true", "on", "yes"].includes(s.trim().toLowerCase());
  return {
    config, config_source: usable ? "python" : "mirror",
    entries: { stocks: sleeves.has("stocks"), options: sleeves.has("options"), flag: ent },
    engine: { on: truthy(eng.value), flag: eng },
    selection: { gates: [...parseNamedSet(sel.value, SELECTION_GUARDS)], flag: sel },
    options: { gates: [...parseOptionsGates(opt.value)], flag: opt },
    sizing: { sleeves: [...parseNamedSet(siz.value, SIZING_SLEEVES)], flag: siz },
    labels: { on: truthy(lab.value), flag: lab },
  };
}

// ── status rules ────────────────────────────────────────────────────────────
export type StatusRule =
  | { by: "code"; status: BrainStatus; detail: string }
  | { by: "entries"; sleeve: "stocks" | "options" }
  | { by: "scored" }
  | { by: "regime_engine" }
  | { by: "selection_guard"; guard: string }
  | { by: "options_guard"; guards: string[] }
  | { by: "regime_sizing" }
  | { by: "regime_labels" }
  | { by: "stress_index" }
  | { by: "config_bool"; key: string; detail: string }
  | { by: "third_leg" }
  | { by: "convexity" }
  | { by: "tier"; tier: string; detail: string };

export interface StatusResult { status: BrainStatus; status_source: StatusSource; detail: string }

const NOTHING_ORDERED = "nothing it touches is ordered while both entry sleeves are shadowed";

export function deriveStatus(rule: StatusRule, sw: Switches): StatusResult {
  const entSrc: StatusSource = sw.entries.flag.source;
  const anyEntryLive = !sw.entries.stocks || !sw.entries.options;
  switch (rule.by) {
    case "code":
      return { status: rule.status, status_source: "code", detail: rule.detail };
    case "entries": {
      const shadowed = sw.entries[rule.sleeve];
      return { status: shadowed ? "shadow" : "live", status_source: entSrc,
        detail: `VOLTRADE_SHADOW_ENTRIES=${JSON.stringify(sw.entries.flag.value)} (${entSrc}): ${shadowed ? "would-be entries are logged, no order" : "entries are ordered"}` };
    }
    case "scored":
      return anyEntryLive
        ? { status: "live", status_source: entSrc, detail: `computed every scan; feeds the score that orders ${!sw.entries.stocks ? "stock" : "options"} entries` }
        : { status: "shadow", status_source: entSrc, detail: `computed every scan and logged; ${NOTHING_ORDERED} (VOLTRADE_SHADOW_ENTRIES=${JSON.stringify(sw.entries.flag.value)}, ${entSrc})` };
    case "regime_engine":
      return { status: sw.engine.on ? "live" : "off", status_source: sw.engine.flag.source,
        detail: `VOLTRADE_REGIME_ENGINE=${JSON.stringify(sw.engine.flag.value)} (${sw.engine.flag.source}); its state feeds the selection/CSP guards, sizing and labels below` };
    case "selection_guard": {
      if (!sw.engine.on) return { status: "off", status_source: sw.engine.flag.source, detail: "regime engine off: no state, no guard" };
      const gated = sw.selection.gates.includes(rule.guard);
      if (gated && !sw.entries.stocks) return { status: "live", status_source: sw.selection.flag.source, detail: `gate on (VOLTRADE_REGIME_SELECTION=${JSON.stringify(sw.selection.flag.value)}): blocks real stock entries` };
      return { status: "shadow", status_source: gated ? entSrc : sw.selection.flag.source,
        detail: gated ? "gate on, but stock entries are shadowed: it blocks would-be entries only" : "flag-only: relabels shadow records (would_reject_regime_*), blocks nothing" };
    }
    case "options_guard": {
      if (!sw.engine.on) return { status: "off", status_source: sw.engine.flag.source, detail: "regime engine off: no state, no guard" };
      const gated = rule.guards.some((g) => sw.options.gates.includes(g));
      if (gated && !sw.entries.options) return { status: "live", status_source: sw.options.flag.source, detail: `gate on (VOLTRADE_REGIME_OPTIONS=${JSON.stringify(sw.options.flag.value)}): acts on real CSP entries` };
      return { status: "shadow", status_source: gated ? entSrc : sw.options.flag.source,
        detail: gated ? "gate on, but options entries are shadowed: it acts on would-be entries only" : "flag-only: logged on shadow records, acts on nothing" };
    }
    case "regime_sizing": {
      if (!sw.engine.on || sw.sizing.sleeves.length === 0) return { status: "off", status_source: sw.engine.on ? sw.sizing.flag.source : sw.engine.flag.source, detail: "legacy VIX/stress scalars size every position" };
      const live = (sw.sizing.sleeves.includes("stock") && !sw.entries.stocks) || ((sw.sizing.sleeves.includes("options") || sw.sizing.sleeves.includes("tier1")) && !sw.entries.options);
      return { status: live ? "live" : "shadow", status_source: live ? sw.sizing.flag.source : entSrc,
        detail: `sleeves ${sw.sizing.sleeves.join(", ")} (VOLTRADE_REGIME_SIZING, ${sw.sizing.flag.source})${live ? "" : `; ${NOTHING_ORDERED}`}` };
    }
    case "regime_labels":
      return { status: sw.engine.on && sw.labels.on ? "live" : "off", status_source: sw.labels.flag.source,
        detail: sw.engine.on && sw.labels.on ? "the engine's legacy label replaces the Markov label on records and exit geometry" : "Markov label in use" };
    case "stress_index": {
      const replaced = sw.engine.on && sw.sizing.sleeves.includes("stock");
      if (replaced) return { status: "off", status_source: sw.sizing.flag.source, detail: "replaced by the regime engine's exposure multiplier on the stock sizer (VOLTRADE_REGIME_SIZING)" };
      return { status: sw.entries.stocks ? "shadow" : "live", status_source: entSrc, detail: "sizes stock entries" };
    }
    case "config_bool": {
      const on = sw.config[rule.key] === true;
      return { status: on ? "live" : "off", status_source: "config", detail: `${rule.key}=${String(sw.config[rule.key])}${sw.config_source === "mirror" ? " (mirror: Python config unreadable)" : ""}; ${rule.detail}` };
    }
    case "third_leg": {
      const pct = Number(sw.config.LEG3_VRP_PCT ?? 0);
      if (!(pct > 0)) return { status: "off", status_source: "config", detail: "LEG3_VRP_PCT=0" };
      return { status: sw.entries.stocks ? "shadow" : "live", status_source: sw.entries.stocks ? entSrc : "config",
        detail: `LEG3_VRP_PCT=${pct}; its SVXY buy is a "stocks" entry (entry_shadow.py)${sw.entries.stocks ? " — logged, not ordered; existing SVXY exits stay live" : ""}` };
    }
    case "convexity":
      // bot_engine._run_convexity_overlay has NO reader of CONVEXITY_OVERLAY.enabled (verified 2026-10-29):
      // it runs every scan; only the options entry-shadow gate decides whether the put is ordered.
      return { status: sw.entries.options ? "shadow" : "live", status_source: entSrc,
        detail: `runs every scan (CONVEXITY_OVERLAY.enabled=${String(sw.config.CONVEXITY_OVERLAY_enabled)} is never read); ${sw.entries.options ? "buy-to-open is logged, not ordered (options sleeve shadowed); rolls' closes stay live" : "puts are ordered"}` };
    case "tier": {
      const tiers = sw.config.TIERS_ENABLED as Record<string, boolean> | null | undefined;
      const on = tiers && typeof tiers === "object" ? tiers[rule.tier] === true : null;
      if (on === null) return { status: rule.tier === "4" ? "off" : "live", status_source: "code", detail: `tiered_strategy.TIERS_ENABLED unreadable; ${rule.detail}` };
      return { status: on ? "live" : "off", status_source: "code", detail: `tiered_strategy.TIERS_ENABLED[${rule.tier}]=${on}; ${rule.detail}` };
    }
  }
}

// ── short-term specs ────────────────────────────────────────────────────────
export type ShortSpec =
  | { kind: "source"; name: string }
  | { kind: "reference"; name: string }
  | { kind: "bucket"; buckets: string[]; blocks: boolean; label: string }
  | { kind: "ml_v3_ic" }
  | { kind: "forecast" }
  | { kind: "none"; reason: string };

// ── long-term evidence (copied from the cited files; brain.test.ts pins the quotes) ──
export interface LongItem {
  id: string;
  title: string;
  source_file: string;
  date: string;
  horizon: string;
  finding: string;
  compared_to: string;
  sign: Sign;
  variants_tried?: number;
  quotes: string[];       // exact substrings of source_file
}

export const LONG_EVIDENCE: Readonly<Record<string, LongItem>> = Object.freeze({
  regime_backtest: {
    id: "regime_backtest", title: "Regime-layer backtest (index level, 12 arms)", source_file: "research/regime_backtest_2026-10-28.md", date: "2026-10-28",
    horizon: "2007-08-28 → 2026-10-26, SPY, 2-session lag, 5 bp turnover",
    finding: "engine_quantized: CAGR 8.34%, max DD -20.5%, Sharpe 0.76 vs the old schedule (current_proxy) 4.30% / -20.2% / 0.56; buy-and-hold 11.05% / -55.2% / 0.63. Every gated arm trails buy-and-hold CAGR.",
    compared_to: "the regime schedule it replaced (current_proxy)", sign: "positive", variants_tried: 12,
    quotes: ["| engine_quantized | 8.34% / -20.5% / 0.76", "| current_proxy | 4.30% / -20.2% / 0.56", "| buy_and_hold | 11.05% / -55.2% / 0.63"],
  },
  regime_no_cagr: {
    id: "regime_no_cagr", title: "Regime findings: a regime layer cannot add CAGR here", source_file: "docs/regime/FINDINGS.md", date: "2026-10-28",
    horizon: "19 years of index data (RESEARCH §5)",
    finding: "With leverage capped at 1, every gated arm trails buy-and-hold; the product is a shallower drawdown path, not more growth.",
    compared_to: "buy-and-hold SPY", sign: "negative", quotes: ["Can a regime layer add CAGR here? | No."],
  },
  sleeves_unbacktestable: {
    id: "sleeves_unbacktestable", title: "Stock and options sleeves cannot be backtested honestly yet", source_file: "docs/regime/FINDINGS.md", date: "2026-10-28",
    horizon: "—", finding: "No point-in-time universe, no delisted names, no realistic small-cap fills and ~80 days of options history: the stock and options sleeves are measured forward only. A 2016+ walk-forward panel harness is not built yet.",
    compared_to: "—", sign: "insufficient", quotes: ["Can Section 10's backtest be built? | No."],
  },
  ml_v3_doc: {
    id: "ml_v3_doc", title: "ML v3 candidate, honest out-of-sample CV", source_file: "research/ml_v3_eval.md", date: "2026-10-27",
    horizon: "2023-12-28 → 2026-09-28, 97 stocks, 5 date-block folds, 55,775 OOS rows",
    finding: "AUC 0.499 (95% CI 0.479–0.516); log-loss 0.6949 worse than the base rate's 0.6928; top predicted quintile returned less than the bottom (-0.13 pts). 1 configuration tried.",
    compared_to: "a constant base-rate forecast", sign: "null", variants_tried: 1,
    quotes: ["AUC is 0.499 (95% date-block bootstrap CI 0.479–0.516)"],
  },
  ml_v2_honest: {
    id: "ml_v2_honest", title: "ML v2 procedure with the CV leak removed", source_file: "research/ml_v3_eval.md", date: "2026-10-27",
    horizon: "2023-02-02 → 2026-10-16, 97 tickers, 90,210 rows",
    finding: "The v2 procedure reports AUC 0.658; on identical inputs with date folds and an honest embargo it scores AUC 0.515 and log-loss 0.690 vs 0.684 for the base rate — the reported skill is the leak.",
    compared_to: "a constant base-rate forecast", sign: "null",
    quotes: ["| honest (date folds, 5-date embargo + purge, inner-validation early stop) | **0.515**"],
  },
  live_exit_base_rate: {
    id: "live_exit_base_rate", title: "Random entry under the live exit ladder (base rate)", source_file: "research/ml_v3_eval.md", date: "2026-10-27",
    horizon: "2023-12-28 → 2026-09-28, 97 stocks, net of 0.10% round trip",
    finding: "+0.364% per trade, win rate 51.3% — a base rate for the live exits, not a comparison against another exit model (none tested).",
    compared_to: "nothing (descriptive)", sign: "insufficient", quotes: ["**+0.364% per", "win rate 51.3%"],
  },
  floor_composition: {
    id: "floor_composition", title: "Floor basket vs same-exposure SPY (simulation)", source_file: "research/floor_basket_decision.md", date: "2026-10-27",
    horizon: "2014-11 → 2026-10, 2,990 days, 5 bp/side, live exposure schedule",
    finding: "Basket CAGR 7.82%, max DD -21.6%, Sharpe 0.733 ± 0.291; Sharpe minus same-exposure SPY -0.109 [95% -0.375, +0.156] — indistinguishable from SPY at the same exposure.",
    compared_to: "SPY held at the floor's own exposure", sign: "null", variants_tried: 4,
    quotes: ["| basket (live) | 7.82% | −21.6% | 0.733 ± 0.291 | −0.109 [−0.375, +0.156] |"],
  },
  floor_schedule: {
    id: "floor_schedule", title: "The floor's regime exposure schedule (simulation)", source_file: "research/floor_basket_decision.md", date: "2026-10-27",
    horizon: "2014-11 → 2026-10",
    finding: "SPY at the floor's regime-driven exposure made 7.41%/yr vs 13.58% buy-and-hold at Sharpe 0.843 vs 0.816: the gating roughly halved compounding for no risk-adjusted gain.",
    compared_to: "buy-and-hold SPY", sign: "negative", quotes: ["made **7.41%/yr vs 13.58%**"],
  },
  edge_p1: {
    id: "edge_p1", title: "Edge P1: dealer gamma vs realized/implied vol (gate 2)", source_file: "research/edge_p1_dealer_gamma_result.md", date: "2026-10-28",
    horizon: "2012-05 → 2026-10 (3,640 days); holdout 2019-01 → 2026-10",
    finding: "PASS on the pre-registered rule: b = -0.858 (NW t -9.61) full window, -0.904 (t -7.25) holdout. A signal about SPY 5-day variance, not a tradeable P&L; about a third of the slope is the VIX-vs-5-day horizon gap.",
    compared_to: "no gamma information (b = 0)", sign: "positive", variants_tried: 3,
    quotes: ["Full window b = −0.858, NW t = −9.61. Holdout b = −0.904, NW t = −7.25."],
  },
  convexity_10yr: {
    id: "convexity_10yr", title: "QQQ-put convexity overlay, 10-year backtest", source_file: "system_config.py", date: "2026-04-17",
    horizon: "10 years (the run that disabled it in config)",
    finding: "The overlay bled $307K-$508K in annual premium against 1-2 crash payoffs that did not recoup it. (An older docstring in bot_engine.py claims Sharpe 3.80; the 2026-04-17 run is the later evidence.)",
    compared_to: "no overlay", sign: "negative", quotes: ["10yr backtest showed the QQQ put overlay bled", "$307K-$508K in annual premium vs 1-2 crash payoffs"],
  },
});

export interface LongSpec { items: string[]; live?: "ml_v3" | "forecast"; reason?: string }

export interface BrainComponent {
  id: string;
  name: string;
  kind: BrainKind;
  group: string;
  does: string;
  inventory_ref: string | null;   // exact row text in research/brain_inventory.md
  ref_file?: string;              // the component's own doc when it postdates the inventory
  status: StatusRule;
  short: ShortSpec;
  long: LongSpec;
}

const NO_LONG_SCORE = "Not yet measured: no multi-year test of this input exists yet — the 2016+ point-in-time walk-forward harness, walkforward.py, has not produced a result (research/walkforward.md).";
const src = (name: string): ShortSpec => ({ kind: "source", name });
const scored: StatusRule = { by: "scored" };
const noLong: LongSpec = { items: [], reason: NO_LONG_SCORE };

export const BRAIN_COMPONENTS: readonly BrainComponent[] = Object.freeze([
  // ── inputs ──
  { id: "snapshots", name: "Stock snapshots (15-min delayed)", kind: "input", group: "Market data", does: "Alpaca snapshots of ~12k names feed the quick score, price/volume floors and today's change.", inventory_ref: "Stock snapshots (~12k/scan)", status: scored, short: { kind: "none", reason: "A data feed, not a predictor: its contribution is measured through the score components below." }, long: { items: [], reason: "Not yet measured (a feed, not a signal)." } },
  { id: "news", name: "Headline sentiment", kind: "input", group: "News & social", does: "Alpaca news keyword score: +10/+5/-6/-12 points on the stock score.", inventory_ref: "Headline sentiment", status: scored, short: src("adj.news"), long: noLong },
  { id: "intel", name: "News + insider intel", kind: "input", group: "News & social", does: "News classification, Form 4 counts, MSPR and analyst trend: up to ±15 points.", inventory_ref: "News classification (intel)", status: scored, short: src("adj.intel"), long: noLong },
  { id: "insider_mspr", name: "Insider MSPR (Finnhub)", kind: "input", group: "News & social", does: "Finnhub insider-sentiment ±5 points (inside intel).", inventory_ref: "Insider MSPR", status: scored, short: src("adj.insider_mspr"), long: noLong },
  { id: "social", name: "Social buzz (Reddit, RSS)", kind: "input", group: "News & social", does: "Reddit and news-RSS sentiment: ±15 points (halved at low confidence).", inventory_ref: "Reddit sentiment / buzz", status: scored, short: src("adj.social"), long: noLong },
  { id: "google_trends", name: "Google Trends", kind: "input", group: "News & social", does: "Would weight 35% of the social score; hard-coded to 0 (rate limits).", inventory_ref: "| Google Trends", status: { by: "code", status: "off", detail: "disabled in social_data.py (signal fixed at 0); its 35% weight slot still exists" }, short: { kind: "none", reason: "Off: constant 0, nothing to measure." }, long: { items: [], reason: "Not measured (off)." } },
  { id: "alt", name: "Alt-data composite", kind: "input", group: "Alternative data", does: "Wikipedia attention, short-volume proxy, FTD, GDELT, FRED flags: ±20 points.", inventory_ref: "Alt composite", status: scored, short: src("adj.alt"), long: noLong },
  { id: "fundamentals", name: "Fundamentals (yfinance)", kind: "input", group: "Fundamentals", does: "Rating, target upside, forward P/E, EV/EBITDA: about -18..+16 points.", inventory_ref: "Fundamentals (rating", status: scored, short: src("adj.fundamental"), long: noLong },
  { id: "markov", name: "Markov regime state", kind: "input", group: "Macro & regime", does: "SPY 5-day-return Markov state: an ML feature and a size multiplier.", inventory_ref: "SPY 5-day returns (Markov)", status: scored, short: src("feat.markov_state"), long: noLong },
  // ── rules (score) ──
  { id: "momentum", name: "Momentum sub-score (25%)", kind: "rule", group: "Stock score", does: "12-1 momentum strategy sub-score, weight 0.25 in the rules composite.", inventory_ref: "Rules composite: momentum 0.25", status: scored, short: src("adj.momentum"), long: noLong },
  { id: "mean_reversion", name: "Mean-reversion sub-score (20%)", kind: "rule", group: "Stock score", does: "Mean-reversion sub-score, weight 0.20 (neutralized by the regime dip guard).", inventory_ref: "Rules composite: momentum 0.25", status: scored, short: src("adj.mean_reversion"), long: noLong },
  { id: "vrp", name: "VRP sub-score (25%)", kind: "rule", group: "Stock score", does: "Implied-minus-realized vol sub-score, weight 0.25.", inventory_ref: "Rules composite: momentum 0.25", status: scored, short: src("adj.vrp"), long: noLong },
  { id: "squeeze", name: "Squeeze sub-score (15%)", kind: "rule", group: "Stock score", does: "Short-squeeze sub-score, weight 0.15.", inventory_ref: "Rules composite: momentum 0.25", status: scored, short: src("adj.squeeze"), long: noLong },
  { id: "volume", name: "Volume sub-score (15%)", kind: "rule", group: "Stock score", does: "Volume-surge sub-score, weight 0.15.", inventory_ref: "Rules composite: momentum 0.25", status: scored, short: src("adj.volume"), long: noLong },
  { id: "vol_premium", name: "EWMA vol-premium bonus", kind: "rule", group: "Stock score", does: "+8/+5 points when EWMA implied premium is rich.", inventory_ref: "EWMA-premium bonus", status: scored, short: src("adj.vol_premium"), long: noLong },
  { id: "macro_adj", name: "Macro adjustments", kind: "rule", group: "Stock score", does: "VIX regime ±5/-10, sector ±8, risk on/off -5/+3 points.", inventory_ref: "Macro adjustments: VIX regime", status: scored, short: src("adj.macro"), long: noLong },
  { id: "adx", name: "ADX trend bonus", kind: "rule", group: "Stock score", does: "+6/+3/-4 points by ADX.", inventory_ref: "OBV ±5", status: scored, short: src("adj.adx"), long: noLong },
  { id: "obv", name: "OBV bonus", kind: "rule", group: "Stock score", does: "±5 points by on-balance volume.", inventory_ref: "OBV ±5", status: scored, short: src("adj.obv"), long: noLong },
  { id: "rel_strength", name: "Relative-strength bonus", kind: "rule", group: "Stock score", does: "+6 points for relative strength vs the market.", inventory_ref: "OBV ±5", status: scored, short: src("adj.rel_strength"), long: noLong },
  { id: "recommendation", name: "\"AI recommends\" boost", kind: "rule", group: "Stock score", does: "+8 when analyze.py's action contains BUY (direction-blind).", inventory_ref: "AI recommends", status: scored, short: src("adj.recommendation"), long: noLong },
  { id: "contrarian", name: "Contrarian / squeeze-watch flags", kind: "rule", group: "Stock score", does: "+10/+8/+4 contrarian flags, +10 \"Squeeze Watch\".", inventory_ref: "Contrarian flags", status: scored, short: src("adj.contrarian"), long: noLong },
  // ── rules (gates) ──
  { id: "min_score", name: "Minimum score gate", kind: "rule", group: "Entry gates", does: "Adaptive MIN_SCORE 63/63/67/75 (Node floor 65): candidates below it are rejected.", inventory_ref: "Adaptive MIN_SCORE", status: { by: "entries", sleeve: "stocks" }, short: { kind: "bucket", buckets: ["rejected_score"], blocks: true, label: "candidates it rejected" }, long: noLong },
  { id: "regime_gate", name: "No stock longs in BEAR/PANIC", kind: "rule", group: "Entry gates", does: "The 5-level regime blocks new stock longs in BEAR and PANIC.", inventory_ref: "Regime gate: no stock longs", status: { by: "entries", sleeve: "stocks" }, short: { kind: "bucket", buckets: ["rejected_regime_gate"], blocks: true, label: "candidates it rejected" }, long: noLong },
  { id: "regime_engine", name: "Regime engine (composite state)", kind: "model", group: "Regime engine", does: "One regime state per scan (slow trend + realized-vol dial + fast downgrade votes) that the guards, sizing and labels read.", inventory_ref: null, ref_file: "docs/regime/FINDINGS.md", status: { by: "regime_engine" }, short: { kind: "none", reason: "The state itself places no trade: its consumers (selection/CSP guards, sizing) are measured on their own rows. Daily state is on /api/bot/regime." }, long: { items: ["regime_backtest", "regime_no_cagr"] } },
  ...SELECTION_GUARDS.map((g): BrainComponent => ({
    id: `sel_${g}`, name: `Regime selection guard: ${g}`, kind: "rule", group: "Regime engine",
    does: ({ knife: "Blocks names below 75% of their 52-week high or at a new 20-day low.", dip: "Neutralizes mean-reversion and dip bonuses outside calm, healthy tapes.", panic: "No momentum longs in the panic state.", rs: "Blocks names trailing the index by more than 10 pp over 3 months.", breadth: "No net new stock exposure near index highs with weak breadth." } as Record<string, string>)[g],
    inventory_ref: null, ref_file: "docs/regime/FINDINGS.md", status: { by: "selection_guard", guard: g },
    short: { kind: "bucket", buckets: [`rejected_regime_${g}`, `would_reject_regime_${g}`], blocks: true, label: "candidates it blocked (or would have)" },
    long: { items: ["sleeves_unbacktestable"], reason: "Not yet measured in our data: the guards cite published research (FINDINGS step 4), but the stock sleeve cannot be backtested here; first forward read is due 2026-11-16." },
  })),
  { id: "csp_block_guards", name: "Regime CSP guards: crisis / VRP / earnings", kind: "rule", group: "Regime engine", does: "No new short puts in CRISIS, with negative VRP, or with earnings inside the contract window.", inventory_ref: null, ref_file: "docs/regime/FINDINGS.md", status: { by: "options_guard", guards: ["crisis", "vrp", "earnings"] }, short: { kind: "bucket", buckets: ["rejected_regime_crisis", "rejected_regime_vrp", "rejected_regime_earnings", "would_reject_regime_crisis", "would_reject_regime_vrp", "would_reject_regime_earnings"], blocks: true, label: "CSP names it blocked (underlying long-stock path, not option P&L)" }, long: { items: ["sleeves_unbacktestable"], reason: "Not yet measured: ~80 days of options history; no options simulator (KB #48)." } },
  { id: "csp_size_guards", name: "Regime CSP size guards: inversion / elevated / EBP", kind: "rule", group: "Regime engine", does: "Halves new CSP size on VIX/VIX3M inversion, elevated vol or a positive excess bond premium.", inventory_ref: null, ref_file: "docs/regime/FINDINGS.md", status: { by: "options_guard", guards: ["inversion", "elevated", "ebp"] }, short: { kind: "none", reason: "Not measured: a size change has no blocked-candidate bucket; needs option P&L (no options simulator yet)." }, long: { items: ["sleeves_unbacktestable"], reason: "Not yet measured." } },
  { id: "regime_sizing", name: "Regime exposure multiplier (sizing)", kind: "rule", group: "Regime engine", does: "One exposure multiplier replaces the VIX scalar, the stress multiplier and the Tier-1 VXX ladder.", inventory_ref: null, ref_file: "docs/regime/FINDINGS.md", status: { by: "regime_sizing" }, short: { kind: "none", reason: "Not measured from the shadow log: outcomes are per-candidate returns, not position sizes. Today's applied multipliers are on /api/bot/regime." }, long: { items: ["regime_backtest", "regime_no_cagr"] } },
  { id: "regime_labels", name: "Regime engine labels", kind: "rule", group: "Regime engine", does: "The engine's legacy label replaces the Markov label on records and exit geometry.", inventory_ref: null, ref_file: "docs/regime/FINDINGS.md", status: { by: "regime_labels" }, short: { kind: "none", reason: "A labelling change: no outcome bucket of its own." }, long: { items: [], reason: "Not yet measured." } },
  { id: "stress_index", name: "Stress-index size multiplier", kind: "rule", group: "Sizing", does: "Yield-curve / credit / VIX-term / breadth stress cuts stock size.", inventory_ref: "Stress index (yield curve", status: { by: "stress_index" }, short: { kind: "none", reason: "A sizing scalar: no outcome bucket of its own." }, long: { items: [], reason: "Not yet measured (never validated, inventory §1.5)." } },
  { id: "live_exits", name: "Live stock exit ladder", kind: "rule", group: "Exits", does: "2×ATR stop, scale-outs in thirds, breakeven, trails, 7-day flat time stop — the only stock exit executor.", inventory_ref: "phase-1 stop clamp", status: { by: "code", status: "live", detail: "exits are never gated by entry shadow (bot.ts checkPositionOnTick)" }, short: { kind: "bucket", buckets: ["taken", "shadow_mode_entry:stocks"], blocks: false, label: "would-be and taken entries walked through the live exits (exit-aligned ruler)" }, long: { items: ["live_exit_base_rate"], reason: "Only a base rate exists: no alternative exit model has been tested against it." } },
  { id: "kill_switches", name: "Drawdown halts and kill switches", kind: "rule", group: "Risk", does: "Node -10% drawdown kill, Python 18% halt, daily -3%, circuit breaker, POS-KILL.", inventory_ref: "Node drawdown kill", status: { by: "config_bool", key: "DRAWDOWN_HALT_ENABLED", detail: "risk mechanisms (frozen); not gated by entry shadow" }, short: { kind: "none", reason: "Not measured: kill thresholds have no counterfactual logging yet (inventory §2.11)." }, long: { items: [], reason: "Not yet measured." } },
  // ── models ──
  { id: "ml_v2", name: "ML v2 (LightGBM) blend", kind: "model", group: "Models", does: "Win probability blended into the score at ~0.28–0.40 weight.", inventory_ref: "LightGBM global model", status: scored, short: { kind: "reference", name: "ml_only_score" }, long: { items: ["ml_v2_honest"] } },
  { id: "ml_v3", name: "ML v3 candidate", kind: "model", group: "Models", does: "Honest-CV candidate model; its probability is logged on every shadow record, never blended.", inventory_ref: null, ref_file: "research/ml_v3_eval.md", status: { by: "code", status: "shadow", detail: "ml_model_v3.py: promoted=False hard-coded; logged next to v2, never blended" }, short: { kind: "ml_v3_ic" }, long: { items: ["ml_v3_doc"], live: "ml_v3" } },
  { id: "final_score", name: "Final blended score", kind: "model", group: "Models", does: "Rules composite + adjustments, blended with ML, clamped 0–100: ranks every candidate.", inventory_ref: "rules·rw + ml·mw", status: scored, short: { kind: "reference", name: "score" }, long: noLong },
  { id: "forecast", name: "Weekly forecast screen", kind: "model", group: "Models", does: "Weekly up/down calls with probabilities for SPY, positions and candidates.", inventory_ref: null, ref_file: "forecast.py", status: { by: "code", status: "shadow", detail: "display and tracking only: nothing in the trading loop reads it" }, short: { kind: "forecast" }, long: { items: [], live: "forecast", reason: "Only the fit-time holdout exists (below)." } },
  { id: "strategy_weights", name: "\"LEARN\" strategy-weight loop", kind: "model", group: "Models", does: "Shifts momentum/VRP weights from recent win rate — written and audited, never read.", inventory_ref: "strategyWeights", status: { by: "code", status: "off", detail: "server/bot.ts strategyWeights is write-only: the score uses hard-coded weights (bot_engine.py)" }, short: { kind: "none", reason: "Off (dead code): reads nothing, changes nothing." }, long: { items: [], reason: "Not measured (dead)." } },
  { id: "edge_p1", name: "Dealer-gamma short-premium gate (Edge P1)", kind: "model", group: "Models", does: "Candidate rule: no new short premium when dealer gamma rank < 0.2. Gate 2 passed; not built.", inventory_ref: null, ref_file: "research/edge_p1_dealer_gamma_result.md", status: { by: "code", status: "off", detail: "research only: ladder gate 2 passed, gate 3 simulation not run, nothing trades on it" }, short: { kind: "none", reason: "Not measured live: not built into the bot." }, long: { items: ["edge_p1"] } },
  // ── sleeves ──
  { id: "stock_entries", name: "Stock entries", kind: "sleeve", group: "Sleeves", does: "New stock/ETF positions from the scan (executeTrades, morning queue, Tier-3).", inventory_ref: "Quick-score rank", status: { by: "entries", sleeve: "stocks" }, short: { kind: "bucket", buckets: ["shadow_mode_entry:stocks", "taken"], blocks: false, label: "would-be stock entries (shadow) — or taken, when live" }, long: { items: ["sleeves_unbacktestable"], reason: NO_LONG_SCORE } },
  { id: "options_entries", name: "Options entries (CSPs, scanner, Tier-1)", kind: "sleeve", group: "Sleeves", does: "Every options open: Tier-1 CSPs, scanner trades, spreads, rolls' opening legs.", inventory_ref: "T1 CSP core", status: { by: "entries", sleeve: "options" }, short: { kind: "none", reason: "Not measured: shadow outcomes are long-stock returns of the underlying, not option P&L (no options simulator, KB #48). Realized options P&L is on the Sleeves panel." }, long: { items: ["sleeves_unbacktestable"], reason: "Not yet measured: ~80 days of options history." } },
  { id: "floor_basket", name: "Floor basket (QQQ/SMH/KWEB/VXUS)", kind: "sleeve", group: "Sleeves", does: "Regime-scheduled passive core: 70% BULL / 90% NEUTRAL / 35% CAUTION / 0% BEAR-PANIC.", inventory_ref: "Floor basket (QQQ .50", status: { by: "config_bool", key: "FLOOR_BASKET_ENABLED", detail: "never gated by entry shadow (entry_shadow.py)" }, short: { kind: "none", reason: "Not measured here: the sleeve ledger reports dollars, not a daily return series vs same-exposure SPY (research/floor_basket_decision.md)." }, long: { items: ["floor_composition", "floor_schedule"] } },
  { id: "defensive_gld", name: "Defensive GLD floor", kind: "sleeve", group: "Sleeves", does: "GLD at 8/8/15/30/40% by regime.", inventory_ref: "Defensive GLD floor", status: { by: "config_bool", key: "DEFENSIVE_FLOOR_ENABLED", detail: "never gated by entry shadow" }, short: { kind: "none", reason: "Not measured here (no return series)." }, long: { items: [], reason: "Not yet measured on its own (it is inside the floor simulation's arms)." } },
  { id: "third_leg", name: "Third leg: SVXY VRP harvest", kind: "sleeve", group: "Sleeves", does: "SVXY up to LEG3_VRP_PCT of equity when VXX is 1.05–1.25 and falling, 10% stop.", inventory_ref: "Third leg: SVXY", status: { by: "third_leg" }, short: { kind: "none", reason: "Not measured: no per-leg return series." }, long: { items: [], reason: "Not yet measured." } },
  { id: "convexity", name: "QQQ-put convexity overlay", kind: "sleeve", group: "Sleeves", does: "Far-OTM QQQ puts, 60 DTE, rolled at 21 DTE, 2–4% of equity.", inventory_ref: "QQQ put convexity overlay", status: { by: "convexity" }, short: { kind: "none", reason: "Not measured: option P&L (no options simulator)." }, long: { items: ["convexity_10yr"] } },
  { id: "tier4", name: "Tier 4 SPY tail hedge", kind: "sleeve", group: "Sleeves", does: "SPY tail hedge tier — disabled, replaced by stress-index de-risking.", inventory_ref: "T4 SPY tail hedge", status: { by: "tier", tier: "4", detail: "tail hedge tier" }, short: { kind: "none", reason: "Off." }, long: { items: [], reason: "Not measured (off)." } },
]);

// ── evidence payload (brain_evidence.py) ────────────────────────────────────
export interface MeanStats { n: number; n_days?: number; mean_pct?: number; hit_rate?: number; ci95?: [number, number]; thin?: boolean; first_day?: string; last_day?: string; vs_spy?: { n: number; mean_excess_pct?: number | null; ci95?: [number, number] | null; beat_rate?: number | null } }
export interface IcStats { n?: number; n_days?: number; ic?: number; ci95?: [number, number]; p?: number; q_bh?: number; verdict?: string; verdict_reason?: string; group?: string; kind?: string; market_wide?: boolean; constant?: boolean; first_day?: string; last_day?: string }
export interface SelectionArm { status?: string; reason?: string; n_rows?: number; candidates?: number; steps?: Array<{ candidate: string; select_ic: number; gain: number; se: number | null; p_bonferroni: number; tried: number; accepted: boolean }>; selected?: string[]; weights?: Record<string, number>; holdout_live_score?: IcStats; holdout_selected?: IcStats; holdout_vs_live_score?: { ic_gain: number; se: number | null }; variants_tried?: number; split?: Record<string, { n: number; days: number; first_day: string | null; last_day: string | null }> }
export interface EvidencePayload {
  generated_at?: string;
  error?: string;
  config?: Record<string, unknown>;
  shadow_log?: { records: number; first_day: string | null; last_day: string | null };
  shadow_log_error?: string;
  spy?: { ok: boolean; error?: string | null; first_day?: string | null; last_day?: string | null; method?: string };
  buckets?: Record<string, Record<string, MeanStats | number>> & { error?: string };
  source_validation?: {
    error?: string; method?: Record<string, unknown>; population?: { n_unique?: number; n_days?: number; first_day?: string | null; last_day?: string | null; mean_outcome_pct?: number | null };
    summary?: Record<string, string[]>; sources?: Record<string, IcStats>; reference?: Record<string, IcStats>;
    selection?: { method?: Record<string, unknown>; features?: SelectionArm; breakdown?: SelectionArm };
    ablation?: { status?: string; reason?: string; n_rows_with_components?: number; baseline_inverted?: boolean; un_inverts?: boolean; improvement_significant?: boolean; removed_groups?: string[]; variants?: Record<string, { n?: number; ic?: number; ci95?: [number, number]; status?: string; removed?: string[]; ic_change_vs_baseline?: number; ic_change_se?: number | null; q5_minus_q1_pct?: number }> };
  };
  ml_v3?: { error?: string; trained_at?: string; rows?: number; date_range?: [string, string]; auc_ci95_date_block?: [number, number]; pooled?: { n?: number; auc?: number; log_loss?: number; base_rate_log_loss?: number }; top_minus_bottom_pct?: number; base_rate_mean_net_return_pct?: number; variants_tried?: number; eval_age_days?: number; candidate_exists?: boolean };
  ml_v3_shadow_ic?: IcStats & { error?: string };
  walkforward?: {
    error?: string; schema?: string; generated_at?: string; as_of?: string; code_version?: string | null;
    panel?: { build_id?: string | null }; universe?: { n_symbols?: number; delisted_symbol_share?: number | null; rows?: number };
    items?: Record<string, LongItem & { live?: boolean }>; verdicts?: Record<string, string>; not_measurable?: Record<string, string>;
  };
  forecast?: { error?: string; generated_at?: string; model_version?: string; accuracy?: { n?: number; thin?: boolean; hits?: number; hit_rate?: number | null; brier?: number | null; brier_skill_vs_coinflip?: number | null }; weeks?: number; holdout?: { split_date?: string; n_test?: number; n_train?: number; brier?: number; brier_climatology?: number; brier_skill?: number | null; hit_rate?: number } | null; fit_window?: [string | null, string | null] | null };
}

// ── shaping ─────────────────────────────────────────────────────────────────
export interface ShortTerm {
  measured: boolean;
  reason?: string;
  what: string;
  metric?: "mean_return" | "ic" | "forecast";
  n?: number;
  n_days?: number;
  mean_pct?: number;
  hit_rate?: number;
  ci95?: [number, number] | null;
  vs_spy?: MeanStats["vs_spy"] | null;
  ic?: number;
  q_bh?: number;
  verdict?: string;
  verdict_reason?: string;
  bucket?: string;
  outcome?: string;
  horizons?: Record<string, { n: number; mean_pct?: number; hit_rate?: number; ci95?: [number, number] | null; mean_excess_pct?: number | null }>;
  window?: { first: string | null; last: string | null };
  source: string;
  as_of: string | null;
  sign: Sign;
  sign_note: string;
}

export interface LongTerm { measured: boolean; reason?: string; items: Array<LongItem & { live?: boolean }>; sign: Sign; sign_note: string }

function ciSign(ci: [number, number] | null | undefined, invert: boolean): Sign | null {
  if (!ci) return null;
  if (ci[0] > 0) return invert ? "negative" : "positive";
  if (ci[1] < 0) return invert ? "positive" : "negative";
  return "null";
}

function isStats(v: unknown): v is MeanStats {
  return !!v && typeof v === "object" && typeof (v as MeanStats).n === "number";
}

export function shortTermFor(c: BrainComponent, ev: EvidencePayload | null): ShortTerm {
  const asOf = ev?.generated_at ?? null;
  const base = { source: "brain_evidence.py over the shadow log", as_of: asOf };
  const spec = c.short;
  if (spec.kind === "none") return { ...base, measured: false, reason: spec.reason, what: "—", sign: "insufficient", sign_note: "not measured" };
  if (!ev || ev.error) return { ...base, measured: false, reason: `Evidence unavailable: ${ev?.error ?? "Python call failed"}`, what: "—", sign: "insufficient", sign_note: "evidence call failed" };

  if (spec.kind === "source" || spec.kind === "reference") {
    const sv = ev.source_validation;
    if (!sv || sv.error) return { ...base, measured: false, reason: `source_validation unavailable: ${sv?.error ?? "missing"}`, what: spec.name, sign: "insufficient", sign_note: "harness error" };
    const r = (spec.kind === "source" ? sv.sources : sv.reference)?.[spec.name];
    const pop = sv.population ?? {};
    const window = { first: pop.first_day ?? null, last: pop.last_day ?? null };
    const svSource = `source_validation.py (${String(sv.method?.prereg_version ?? "pre-registered")}), primary outcome +5d, BH FDR 0.10`;
    if (!r) return { ...base, source: svSource, measured: false, reason: `${spec.name} not in the harness output`, what: spec.name, sign: "insufficient", sign_note: "missing", window };
    let sign: Sign;
    let note: string;
    if (spec.kind === "source") {
      const v = r.verdict ?? "INSUFFICIENT";
      sign = v === "PASS" ? "positive" : v === "FAIL_INVERTED" ? "negative" : v === "FAIL" || v === "CONSTANT" ? "null" : "insufficient";
      note = `${v}: ${r.verdict_reason ?? ""}`.trim();
    } else {
      const s = (r.n ?? 0) >= 100 ? ciSign(r.ci95, false) : null;
      sign = s ?? "insufficient";
      note = s ? `IC 95% CI ${s === "null" ? "includes 0" : "excludes 0"} (reference row, outside the BH family)` : `n=${r.n ?? 0} < 100 or no clustered CI`;
    }
    return { ...base, source: svSource, measured: (r.n ?? 0) > 0, what: `rank IC of ${spec.name} vs the +5d outcome`, metric: "ic", n: r.n, n_days: r.n_days, ic: r.ic, ci95: r.ci95 ?? null, q_bh: r.q_bh, verdict: r.verdict, verdict_reason: r.verdict_reason, outcome: "+5d", window, sign, sign_note: note, ...((r.n ?? 0) > 0 ? {} : { reason: "No labeled deduped records yet (+5d labels lag about a week)." }) };
  }

  if (spec.kind === "ml_v3_ic") {
    const r = ev.ml_v3_shadow_ic;
    if (!r || r.error) return { ...base, measured: false, reason: `unavailable: ${r?.error ?? "missing"}`, what: "v3 probability IC", sign: "insufficient", sign_note: "error" };
    const s = (r.n ?? 0) >= 100 ? ciSign(r.ci95, false) : null;
    return { ...base, measured: (r.n ?? 0) > 0, what: "rank IC of the logged v3 probability vs the +5d outcome", metric: "ic", n: r.n, n_days: r.n_days, ic: r.ic, ci95: r.ci95 ?? null, outcome: "+5d", window: { first: r.first_day ?? null, last: r.last_day ?? null }, sign: s ?? "insufficient", sign_note: s ? "IC CI vs 0" : `n=${r.n ?? 0} < 100 or no clustered CI`, ...((r.n ?? 0) > 0 ? {} : { reason: "No labeled shadow record carries a v3 probability yet." }) };
  }

  if (spec.kind === "forecast") {
    const f = ev.forecast;
    if (!f || f.error) return { ...base, source: "forecast.py report()", measured: false, reason: `unavailable: ${f?.error ?? "missing"}`, what: "weekly calls", sign: "insufficient", sign_note: "error" };
    const a = f.accuracy ?? {};
    const n = a.n ?? 0;
    const skill = a.brier_skill_vs_coinflip;
    const sign: Sign = n < BUCKET_MIN_N || skill === null || skill === undefined ? "insufficient" : skill > 0 ? "positive" : "negative";
    return { source: "forecast.py report()", as_of: f.generated_at ?? asOf, measured: n > 0, what: "resolved weekly calls: direction hit rate and Brier skill vs a coin flip", metric: "forecast", n, hit_rate: a.hit_rate ?? undefined, sign, sign_note: n < BUCKET_MIN_N ? `${n} resolved calls < ${BUCKET_MIN_N}` : "Brier skill sign (no CI)", ...(n > 0 ? {} : { reason: "No call has resolved yet." }) };
  }

  // bucket
  const bk = ev.buckets;
  if (!bk || bk.error) return { ...base, measured: false, reason: `buckets unavailable: ${String(bk?.error ?? "missing")}`, what: spec.label, sign: "insufficient", sign_note: "error" };
  const primaryN = (b: string) => { const s = bk[b]?.[PRIMARY_BUCKET_OUTCOME]; return isStats(s) ? s.n : 0; };
  const chosen = spec.buckets.find((b) => primaryN(b) > 0) ?? spec.buckets.find((b) => bk[b]) ?? null;
  if (!chosen || !bk[chosen]) return { ...base, measured: false, reason: `No shadow record in ${spec.buckets.join(" / ")} yet.`, what: spec.label, sign: "insufficient", sign_note: "empty bucket" };
  const cell = bk[chosen];
  const p = cell[PRIMARY_BUCKET_OUTCOME];
  const prim: MeanStats = isStats(p) ? p : { n: 0 };
  const horizons: ShortTerm["horizons"] = {};
  for (const oc of ["xa", "+5d", "+20d"]) {
    const s = cell[oc];
    if (isStats(s)) horizons[oc] = { n: s.n, mean_pct: s.mean_pct, hit_rate: s.hit_rate, ci95: s.ci95 ?? null, mean_excess_pct: s.vs_spy?.mean_excess_pct ?? null };
  }
  const spyOk = !!ev.spy?.ok;
  let sign: Sign = "insufficient";
  let note = `n=${prim.n} (< ${BUCKET_MIN_N} or no clustered CI)`;
  if (prim.n >= BUCKET_MIN_N) {
    const s = spyOk ? ciSign(prim.vs_spy?.ci95 ?? null, spec.blocks) : ciSign(prim.ci95 ?? null, spec.blocks);
    if (s) {
      sign = s;
      const what = spyOk ? "excess vs SPY" : "mean return (SPY unavailable)";
      note = spec.blocks
        ? (s === "positive" ? `blocked names did worse (${what} CI < 0): the rule earns its keep` : s === "negative" ? `blocked names did better (${what} CI > 0): the rule blocks winners` : `${what} CI includes 0: no measurable prevention value`)
        : (s === "positive" ? `${what} CI > 0` : s === "negative" ? `${what} CI < 0` : `${what} CI includes 0`);
    }
  }
  return {
    ...base, measured: prim.n > 0, what: spec.label, metric: "mean_return", bucket: chosen, outcome: "xa (exit-aligned: the live exit ladder, long stock)",
    n: prim.n, n_days: prim.n_days, mean_pct: prim.mean_pct, hit_rate: prim.hit_rate, ci95: prim.ci95 ?? null, vs_spy: spyOk ? prim.vs_spy ?? null : null,
    horizons, window: { first: prim.first_day ?? null, last: prim.last_day ?? null }, sign, sign_note: note,
    ...(prim.n > 0 ? {} : { reason: `No ${chosen} record has an exit-aligned label yet (labels need the exit to happen or 20 bars).` }),
  };
}

/** Brain component id -> walkforward.py input id (research/walkforward.md). final_score maps to the
 *  composite's price-measurable part only; its item title says so. */
export const WALKFORWARD_ITEM_FOR: Readonly<Record<string, string>> = Object.freeze({
  momentum: "momentum", mean_reversion: "mean_reversion", volume: "volume", vol_premium: "vol_premium",
  adx: "adx", obv: "obv", rel_strength: "rel_strength", final_score: "price_composite",
});
const SIGNS: readonly Sign[] = ["positive", "negative", "null", "insufficient"];

export function longTermFor(c: BrainComponent, ev: EvidencePayload | null): LongTerm {
  const items: LongTerm["items"] = c.long.items.map((id) => ({ ...LONG_EVIDENCE[id] }));
  const wf = ev?.walkforward && !ev.walkforward.error && ev.walkforward.schema === "walkforward/v1" ? ev.walkforward : null;
  const wfItem = wf?.items?.[WALKFORWARD_ITEM_FOR[c.id] ?? ""];
  if (wfItem) {
    const sign: Sign = SIGNS.includes(wfItem.sign) ? wfItem.sign : "insufficient";
    items.unshift({ ...wfItem, sign, live: true, quotes: [] });
  }
  const notMeasurable = !wfItem ? wf?.not_measurable?.[c.id] : undefined;
  if (c.long.live === "ml_v3" && ev?.ml_v3 && !ev.ml_v3.error && ev.ml_v3.pooled?.auc !== undefined) {
    const m = ev.ml_v3;
    const ci = m.auc_ci95_date_block;
    const sign: Sign = ci ? (ci[0] > 0.5 ? "positive" : ci[1] < 0.5 ? "negative" : "null") : "insufficient";
    items.unshift({
      id: "ml_v3_live", live: true, title: "ML v3 candidate — latest weekly honest re-evaluation (volume file)", source_file: "DATA_DIR/ml_v3/ml_v3_eval.json", date: String(m.trained_at ?? "").slice(0, 10),
      horizon: m.date_range ? `${m.date_range[0]} → ${m.date_range[1]}, ${m.rows ?? "?"} rows` : "—",
      finding: `OOS AUC ${m.pooled?.auc?.toFixed(3)}${ci ? ` (95% CI ${ci[0].toFixed(3)}–${ci[1].toFixed(3)})` : ""}; log-loss ${m.pooled?.log_loss?.toFixed(4)} vs base rate ${m.pooled?.base_rate_log_loss?.toFixed(4)}; top-minus-bottom quintile ${m.top_minus_bottom_pct ?? "?"} pts. ${m.variants_tried ?? 1} configuration tried.`,
      compared_to: "AUC 0.5 (no skill)", sign, variants_tried: m.variants_tried ?? 1, quotes: [],
    });
  }
  if (c.long.live === "forecast" && ev?.forecast?.holdout) {
    const h = ev.forecast.holdout;
    items.unshift({
      id: "forecast_holdout", live: true, title: "Forecast model — fit-time holdout (latest weekly batch)", source_file: "forecast.py store (latest batch)", date: String(h.split_date ?? ""),
      horizon: ev.forecast.fit_window ? `${ev.forecast.fit_window[0]} → ${ev.forecast.fit_window[1]}; holdout from ${h.split_date}` : "—",
      finding: `Holdout Brier ${h.brier} vs ${h.brier_climatology} always-base-rate (skill ${h.brier_skill ?? "?"}), ${h.n_test ?? "?"} samples, hit rate ${h.hit_rate ?? "?"}.`,
      compared_to: "always predicting the base rate", sign: "insufficient", quotes: [],
    });
  }
  const primary = items[0];
  if (!primary) {
    const reason = notMeasurable ? `Not measured long-term (no point-in-time history): ${notMeasurable}` : c.long.reason ?? "Not yet measured.";
    return { measured: false, reason, items, sign: "insufficient", sign_note: "no long-horizon test exists" };
  }
  const measured = items.some((i) => i.sign !== "insufficient");
  return {
    measured, ...(c.long.reason ? { reason: c.long.reason } : {}), items,
    sign: primary.sign, sign_note: primary.sign === "insufficient" ? "descriptive only / no comparison" : `${primary.title} vs ${primary.compared_to}`,
  };
}

export type AgreementFlag = "agree" | "disagree" | "insufficient";
export function agreementOf(s: Sign, l: Sign): { flag: AgreementFlag; note: string } {
  if (s === "insufficient" || l === "insufficient") {
    const which = s === "insufficient" && l === "insufficient" ? "neither horizon has a usable test" : s === "insufficient" ? "short-term (live shadow) evidence is missing or too thin" : "no long-horizon test exists";
    return { flag: "insufficient", note: which };
  }
  if (s === l) return { flag: "agree", note: `both say ${SIGN_WORD[s]}` };
  return { flag: "disagree", note: `short-term says ${SIGN_WORD[s]}, long-term says ${SIGN_WORD[l]} — live beats backtest` };
}
const SIGN_WORD: Record<Sign, string> = { positive: "it helps", negative: "it hurts", null: "no measurable effect", insufficient: "not enough data" };

export interface BrainBody {
  generated_at: string;
  evidence_generated_at: string | null;
  evidence_via: string | null;
  evidence_error: string | null;
  explainer: string;
  rule: string;
  status_counts: Record<BrainStatus, number>;
  switches: { config_source: "python" | "mirror"; entry_shadow: { stocks: boolean; options: boolean; value: string; source: string }; regime: Record<string, FlagValue> };
  short_term: { window: { first: string | null; last: string | null }; records: number | null; spy: EvidencePayload["spy"] | null; primary_outcome: string; note: string };
  long_term: { sources: Array<{ file: string; date: string }>; gaps: string[] };
  components: Array<Omit<BrainComponent, "status" | "short" | "long"> & StatusResult & { short_term: ShortTerm; long_term: LongTerm; agreement: { flag: AgreementFlag; note: string } }>;
  combinations: ReturnType<typeof combinationsOf>;
}

export function combinationsOf(ev: EvidencePayload | null) {
  const sv = ev?.source_validation;
  const avail = !!sv && !sv.error && !!sv.selection;
  const arms = avail ? (["features", "breakdown"] as const).map((k) => {
    const a: SelectionArm = sv?.selection?.[k] ?? {};
    return {
      arm: k,
      label: k === "features" ? "ML features only (all deduped rows)" : "score adjustments + features (rows with a logged breakdown)",
      status: a.status ?? "MISSING", reason: a.reason ?? null, n_rows: a.n_rows ?? null, candidates: a.candidates ?? null,
      variants_tried: a.variants_tried ?? 0, selected: a.selected ?? [], weights: a.weights ?? null, steps: a.steps ?? [],
      holdout_selected: a.holdout_selected ?? null, holdout_live_score: a.holdout_live_score ?? null, holdout_vs_live_score: a.holdout_vs_live_score ?? null,
      holdout_window: a.split?.holdout ? { first: a.split.holdout.first_day, last: a.split.holdout.last_day } : null,
    };
  }) : [];
  const abl = sv?.ablation;
  const ablation = abl ? {
    status: abl.status ?? null, reason: abl.reason ?? null, n_rows: abl.n_rows_with_components ?? null, baseline_inverted: abl.baseline_inverted ?? null,
    un_inverts: abl.un_inverts ?? null, improvement_significant: abl.improvement_significant ?? null,
    rows: Object.entries(abl.variants ?? {}).map(([name, v]) => ({ name, removed: v.removed ?? [], n: v.n ?? null, ic: v.ic ?? null, ci95: v.ci95 ?? null, ic_change_vs_baseline: v.ic_change_vs_baseline ?? null, ic_change_se: v.ic_change_se ?? null, q5_minus_q1_pct: v.q5_minus_q1_pct ?? null, status: v.status ?? null })),
  } : null;
  const fam = sv?.method?.multiple_testing;
  return {
    short_term: {
      available: avail,
      reason: avail ? null : `source_validation unavailable: ${sv?.error ?? (ev?.error ?? "no evidence payload")}`,
      as_of: ev?.generated_at ?? null,
      source: "source_validation.py forward selection (day-split fit/select/holdout; each step's p-value Bonferroni-multiplied by the candidates tried) + ablation",
      family: typeof fam === "string" ? fam : null,
      summary: sv?.summary ?? null,
      arms, ablation,
    },
    long_term: LONG_COMBINATIONS,
  };
}

/** Multi-variant long-horizon comparisons that exist in the repo, with the discount their source applies. */
export const LONG_COMBINATIONS = Object.freeze([
  {
    id: "regime_arms", title: "Regime-layer arms (SPY, full window)", source_file: "research/regime_backtest_2026-10-28.md", date: "2026-10-28", horizon: "2007-08-28 → 2026-10-26",
    variants_tried: 12, discount: "No deflated Sharpe in the source: 12 arms on one history, so the best arm's edge is optimistic. Pre-registered claims held (gated arms cut drawdown; none beats buy-and-hold CAGR).",
    columns: ["Arm", "CAGR", "Max DD", "Sharpe"],
    rows: [["vol_target_15", "10.13%", "-34.7%", "0.78"], ["vol_target_10", "8.03%", "-25.4%", "0.78"], ["engine_quantized (live)", "8.34%", "-20.5%", "0.76"], ["faber_monthly", "7.93%", "-27.7%", "0.66"], ["buy_and_hold", "11.05%", "-55.2%", "0.63"], ["current_proxy (old)", "4.30%", "-20.2%", "0.56"]],
    quotes: ["| vol_target_15 | 10.13% / -34.7% / 0.78", "| vol_target_10 | 8.03% / -25.4% / 0.78", "| faber_monthly | 7.93% / -27.7% / 0.66"],
  },
  {
    id: "floor_arms", title: "Floor composition arms", source_file: "research/floor_basket_decision.md", date: "2026-10-27", horizon: "2014-11 → 2026-10",
    variants_tried: 4, discount: "Deflated-Sharpe bar for 4 arms is 0.08 — cleared only by qqq_only; every arm's Sharpe is within one standard error of the others.",
    columns: ["Arm", "CAGR", "Max DD", "Sharpe ± SE"],
    rows: [["qqq_only", "9.76%", "-23.6%", "0.838 ± 0.291"], ["basket (live)", "7.82%", "-21.6%", "0.733 ± 0.291"], ["fifty_fifty", "6.26%", "-20.1%", "0.698 ± 0.291"], ["spy_core", "5.46%", "-19.4%", "0.667 ± 0.290"]],
    quotes: ["The DSR bar for 4 arms (0.08) is cleared only by qqq_only.", "| qqq_only | 9.76% | −23.6% | 0.838 ± 0.291"],
  },
]);

export function brainBody(ev: EvidencePayload | null, env: NodeJS.ProcessEnv, now: Date, via: string | null = null, evidenceError: string | null = null): BrainBody {
  const sw = readSwitches(env, ev?.config ?? null);
  const components = BRAIN_COMPONENTS.map((c) => {
    const st = deriveStatus(c.status, sw);
    const s = shortTermFor(c, ev);
    const l = longTermFor(c, ev);
    const rest = { id: c.id, name: c.name, kind: c.kind, group: c.group, does: c.does, inventory_ref: c.inventory_ref, ...(c.ref_file ? { ref_file: c.ref_file } : {}) };
    return { ...rest, ...st, short_term: s, long_term: l, agreement: agreementOf(s.sign, l.sign) };
  });
  const counts: Record<BrainStatus, number> = { live: 0, shadow: 0, off: 0 };
  for (const c of components) counts[c.status] += 1;
  const files = new Map<string, string>();
  for (const it of Object.values(LONG_EVIDENCE)) files.set(it.source_file, it.date);
  const wf = ev?.walkforward && !ev.walkforward.error && ev.walkforward.schema === "walkforward/v1" ? ev.walkforward : null;
  if (wf) files.set("DATA_DIR/walkforward/latest.json", String(wf.generated_at ?? "").slice(0, 10));
  const regime: Record<string, FlagValue> = {};
  for (const n of ["VOLTRADE_REGIME_ENGINE", "VOLTRADE_REGIME_SELECTION", "VOLTRADE_REGIME_OPTIONS", "VOLTRADE_REGIME_SIZING", "VOLTRADE_REGIME_LABELS"]) regime[n] = flagRaw(n, env, sw.config);
  return {
    generated_at: now.toISOString(),
    evidence_generated_at: ev?.generated_at ?? null,
    evidence_via: via,
    evidence_error: evidenceError ?? ev?.error ?? null,
    explainer: BRAIN_EXPLAINER,
    rule: BRAIN_RULE,
    status_counts: counts,
    switches: { config_source: sw.config_source, entry_shadow: { stocks: sw.entries.stocks, options: sw.entries.options, value: sw.entries.flag.value, source: sw.entries.flag.source }, regime },
    short_term: {
      window: { first: ev?.shadow_log?.first_day ?? null, last: ev?.shadow_log?.last_day ?? null },
      records: ev?.shadow_log?.records ?? null,
      spy: ev?.spy ?? null,
      primary_outcome: "xa = exit-aligned ruler (the live exit ladder) for buckets; +5d path ruler for source ICs (source_validation's pre-registered primary)",
      note: "One first-seen record per (ticker, ET entry day). Long-stock returns of the scored name — never option P&L. 95% CIs are day-clustered bootstraps.",
    },
    long_term: {
      sources: [...files.entries()].map(([file, date]) => ({ file, date })),
      gaps: [
        wf
          ? `The 2016+ point-in-time walk-forward (panel ${wf.panel?.build_id ?? "?"}, ${wf.universe?.n_symbols ?? "?"} symbols, ${wf.universe?.delisted_symbol_share ?? "?"} delisted share) covers only the price-derivable score inputs; news, intel, social, alt, fundamentals, VRP and squeeze have no point-in-time history, and the stock sleeve as traded has no multi-year test.`
          : `The 2016+ point-in-time walk-forward (walkforward.py) has not produced a result yet${ev?.walkforward?.error ? ` (${ev.walkforward.error})` : ""}: the stock score, its inputs and the stock sleeve have no multi-year test.`,
        "No options simulator: options entries, CSP guards and the convexity overlay have no option-P&L test (KB #48).",
        "Kill-switch thresholds have no counterfactual logging.",
      ],
    },
    components,
    combinations: combinationsOf(ev),
  };
}
