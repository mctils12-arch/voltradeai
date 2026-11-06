#!/usr/bin/env python3
"""
decision_journal.py — the DECISION JOURNAL (2026-11-06; human refocus spec §7
"log every signal, decision, and outcome", §8 per-pick plain-English cards).

ONE append-only, size-capped JSONL — DATA_DIR/voltrade_decisions.jsonl — with
one row per ENTRY, EXIT or SKIP decision, carrying the stages that produced
it. Python writes from the places its decisions are made (this module's
record_* functions, called by bot_engine scan_market's gates, tiered_strategy's
CSP rejections and master kill, options_manager's exits, the volatility-
managed floor); Node (server/decisionJournal.ts) appends the SAME row shape to
the SAME file for the decisions made in server/bot.ts (executeTrades'
stock/ETF/option entries and its gates, the tier dispatcher, the stock exits).
GET /api/bot/decisions and the "decisions" diag probe serve it.

ROW (schema v1; every reason and the summary are plain English):
    {"v": 1, "id", "ts" (UTC ISO, ms, "Z"), "ticker" (the underlying),
     "action": "buy" (opened a position — whatever the order side, a sold
               cash-secured put is an entry too) | "sell" (closed one) |
               "skip" (a candidate not taken) | "rotate" (a roll: one contract
               closed, its replacement opened) | "scale_in" | "scale_out"
               (the floor or a partial exit),
     "instrument": "stock" | "etf" | "option" | "floor",
     "chosen": the strategy or rule acted on ("buy_stock",
               "sell_cash_secured_put", "profit_target"; "none" on a skip),
     "rejected": [{"instrument", "reason", "would_be"?}]  — the alternatives
               not taken and why (instrument_selector's per-instrument scores
               and ruled-out reasons; on a skip, the trade that was refused and
               the gate). On an entry the selector decided, the instrument it
               passed over across the stock/options line carries `would_be`
               (2026-11-06, instrument_counterfactual.py): its would-be entry
               (the option alternative of a stock / ETF buy, the stock
               alternative of an options entry), and on read the +1/+5/+20
               trading-day outcomes the nightly job settled,
     "stages": {"regime", "calendar", "flows", "sector", "selection",
                "options", "tax"} — each a dict or null. calendar / flows /
               tax are filled by their own modules; a producer passes them in
               a candidate's "decision_stages" dict and they are kept as-is.
     "sizing": {...} | null, "summary": str,
     optional: "source", "signal" (what produced it: "deep_score scan,
               momentum-led", "tier1 CSP harvest", ...), "side" (order side),
               "symbol" (what was ordered: OCC / ETF), "qty", "price",
               "notional", "order" {status, id, ...}, "mode" ("live" |
               "shadow"), "rule" (the gate label), "dedupe", "backfilled"}

SELECTION sub-scores: deep_score's breakdown (shadow_portfolio's 14 + extra
keys) as returned on the candidate ("score_breakdown"). "drivers" are each
term's points in the RULES score relative to a neutral 50 (the five weighted
components: (score - 50) x weight) or its added points (the adjustments) —
before the ML blend. "dominant" is the largest positive driver.

SKIPS are deduplicated: one row per (ticker, gate, ET day) — the shadow
portfolio still records every evaluation; the journal records the decision.
The in-process memory is seeded from the journal's tail once per process and
day, so a subprocess fallback does not repeat what the daemon wrote.

SIZE: above DECISION_JOURNAL.max_bytes the file moves to <file>.1 (one old
generation kept; ~100+ days per generation at a few dozen rows a day). Rows
are cut to ROW_MAX_BYTES. Each batch is ONE os.write on an O_APPEND fd, so a
Node append never interleaves inside a line.

OFF SWITCH: system_config BASE_CONFIG["DECISION_JOURNAL"]["enabled"]; the env
var VOLTRADE_DECISION_JOURNAL wins whenever it is PRESENT ("off"/"0"/"false"
= nothing written) — the same rule entry_shadow uses, mirrored by
server/decisionJournal.ts.

LOGGING ONLY: nothing here changes an order, a size, a gate or a decision;
nothing prints to stdout (bot.ts parses subprocess stdout); every public
recorder swallows and logs its own errors (a journal failure must never
break the loop).
"""
import json
import logging
import math
import os
import threading
import uuid
from datetime import datetime, timezone

try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = "/data/voltrade" if os.path.isdir("/data") else "/tmp"

log = logging.getLogger("voltrade.decision_journal")

JOURNAL_FILENAME = "voltrade_decisions.jsonl"
CONFIG_KEY = "DECISION_JOURNAL"
ENV_VAR = "VOLTRADE_DECISION_JOURNAL"
OFF_VALUES = ("", "0", "off", "false", "no", "none")
DEFAULT_MAX_BYTES = 5_000_000      # mirrors system_config DECISION_JOURNAL.max_bytes
MIN_MAX_BYTES = 100_000
ROW_MAX_BYTES = 8_000
SCHEMA_VERSION = 1
MAX_ROWS_PER_CALL = 60             # one scan writes at most this many rows
SEED_TAIL_BYTES = 262_144          # how much of the tail seeds the skip dedupe

ACTIONS = ("buy", "sell", "skip", "rotate", "scale_in", "scale_out")
STAGE_KEYS = ("regime", "calendar", "flows", "sector", "selection", "options", "tax")
OPTIONAL_KEYS = ("source", "signal", "side", "symbol", "qty", "price", "notional", "order",
                 "mode", "rule", "pnl_pct", "dedupe", "backfilled", "truncated")

_write_lock = threading.Lock()
_seen_lock = threading.Lock()
_seen = {"day": None, "keys": set(), "seeded": False}


# ── switch + paths ────────────────────────────────────────────────────────────
def _config() -> dict:
    try:
        from system_config import BASE_CONFIG
        c = BASE_CONFIG.get(CONFIG_KEY) or {}
        return c if isinstance(c, dict) else {}
    except Exception as e:  # never break a trading path on a config read
        log.warning(f"[DECISION-JOURNAL] system_config unavailable ({e}); defaults used")
        return {}


def _base_config_value(key):
    try:
        from system_config import BASE_CONFIG
        return BASE_CONFIG.get(key)
    except Exception as e:
        log.debug(f"[DECISION-JOURNAL] {key} unavailable: {e}")
        return None


def enabled(env=None, config=None) -> bool:
    """The env var when PRESENT (any value wins), else the config flag (default on)."""
    env = os.environ if env is None else env
    v = env.get(ENV_VAR)
    if v is not None:
        return str(v).strip().lower() not in OFF_VALUES
    cfg = _config() if config is None else config
    return bool(cfg.get("enabled", True))


def max_bytes(config=None) -> int:
    cfg = _config() if config is None else config
    try:
        v = int(cfg.get("max_bytes", DEFAULT_MAX_BYTES))
    except (TypeError, ValueError):
        v = DEFAULT_MAX_BYTES
    return max(MIN_MAX_BYTES, v)


def journal_path() -> str:
    return os.path.join(DATA_DIR, JOURNAL_FILENAME)


# ── small helpers ─────────────────────────────────────────────────────────────
def now_iso(now=None) -> str:
    dt = now or datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def et_day(now=None) -> str:
    """The ET trading date (YYYY-MM-DD) skips are deduplicated on."""
    now = now or datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        return now.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    except Exception as e:
        log.debug(f"[DECISION-JOURNAL] zoneinfo unavailable ({e}); UTC day used")
        return now.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _num(x):
    """A finite float, else None (bools are not numbers here)."""
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _text(x, n=400) -> str:
    s = "" if x is None else str(x)
    return " ".join(s.split())[:n]


def _clean(obj, depth=0):
    """JSON-safe copy: non-finite floats -> None, unknown objects -> str, bounded depth."""
    if depth > 6:
        return None
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float)):
        return obj if math.isfinite(float(obj)) else None
    if isinstance(obj, dict):
        return {str(k)[:60]: _clean(v, depth + 1) for k, v in list(obj.items())[:60]}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(v, depth + 1) for v in list(obj)[:40]]
    return _text(obj, 200)


def _money(x) -> str:
    v = _num(x)
    if v is None:
        return "?"
    return f"{'-' if v < 0 else ''}${abs(v):,.0f}" if abs(v) >= 100 else f"{'-' if v < 0 else ''}${abs(v):,.2f}"


def _price(x) -> str:
    """A per-share / per-contract price: always cents."""
    v = _num(x)
    return "?" if v is None else f"{'-' if v < 0 else ''}${abs(v):.2f}"


def _dumps(row: dict) -> str:
    return json.dumps(row, separators=(",", ":"), default=str)


def _sentence(s) -> str:
    s = _text(s, 400)
    return s if not s or s[-1] in ".!?" else s + "."


def describe_occ(symbol) -> str:
    """'TLT261219P00085000' -> 'TLT $85 put expiring 2026-12-19'; anything else as given."""
    s = str(symbol or "")
    if len(s) > 15 and s[-15:-9].isdigit() and s[-9] in "CP" and s[-8:].isdigit():
        strike = int(s[-8:]) / 1000.0
        exp = f"20{s[-15:-13]}-{s[-13:-11]}-{s[-11:-9]}"
        return f"{s[:-15]} ${strike:g} {'call' if s[-9] == 'C' else 'put'} expiring {exp}"
    return s


# ── plain-English vocabulary (server/decisionJournal.ts mirrors RULE_TEXT) ────
RULE_TEXT = {
    "rejected_held": "it is already held",
    "rejected_stock_slots": "every stock slot was taken",
    "rejected_cooldown": "it was stopped out recently (re-entry cooldown)",
    "rejected_regime_gate": "new stock buys are off in this market regime",
    "rejected_score": "its score was below the bar",
    "rejected_heat": "its sector already holds the maximum number of positions",
    "rejected_spread": "the bid-ask spread was too wide",
    "rejected_other": "a pre-trade check refused it",
    "rejected_dollar_volume": "it trades too little dollar volume",
    "rejected_blocked_ticker": "it is on the blocked list",
    "rejected_extreme_mover": "it already moved more than 50% today",
    "rejected_bearish_stock_signal": "the signal was bearish and stock shorting is off",
    "rejected_sizing": "position sizing gave it no size",
    "rejected_sizing_error": "position sizing failed on it",
    "rejected_zero_shares": "position sizing rounded to zero shares",
    "rejected_volume_share": "the position would be too large a share of its daily volume",
    "rejected_options_slots": "every options slot was taken",
    "rejected_node_score": "its score was below the server's threshold",
    "rejected_daily_block": "it is blocked for the day after repeated stop-losses",
    "rejected_size_cap": "the position was above the per-position size ceiling",
    "rejected_exposure_cap": "total exposure would have gone over the cap",
    "rejected_corr_cap": "its sector would be over-concentrated",
    "rejected_stock_halted": "trading in it was halted",
    "rejected_max_retry": "too many order attempts on it today",
    "rejected_dup_order": "an order for it was already pending",
    "rejected_buying_power": "there was not enough buying power",
    "rejected_killswitch": "the kill switch was on",
    "rejected_options_nocontract": "no suitable option contract was found",
    "rejected_options_error": "the options engine failed on it",
    "rejected_csp_resubmit": "its last put order was just cancelled unfilled or is still working",
    "rejected_masterkill": "the tier engine's master kill switch blocked all new entries",
    "rejected_capital": "there was not enough capital to secure the put",
    "rejected_order": "the broker refused the order",
    "shadow_mode_repeat": "it was already shadow-entered today",
}


def rule_text(decision) -> str:
    d = str(decision or "")
    if d in RULE_TEXT:
        return RULE_TEXT[d]
    if d.startswith("rejected_regime_"):
        return f"the regime guard '{d[len('rejected_regime_'):]}' blocked it"
    return d.replace("rejected_", "").replace("_", " ").strip() or "a gate blocked it"


# options_manager exit rules (the action "type"), in plain English
EXIT_RULE_TEXT = {
    "profit_target": "the profit target was reached",
    "loss_limit": "the loss limit was hit",
    "bought_loss_limit": "a bought option lost half its premium",
    "dte_critical": "expiry was too close",
    "dte_close": "the 21-days-to-expiry exit",
    "dte_close_bought": "the days-to-expiry exit for a bought option",
    "dte_roll": "rolled at the days-to-expiry point",
    "gamma_risk": "gamma risk close to expiry",
    "assignment_close": "assignment risk (deep in the money near expiry)",
    "assignment_roll": "rolled to avoid assignment",
    "earnings_close": "closed before the company reports earnings",
    "leftover_long_close": "a leftover bought leg whose structure was gone",
    "strategy_dte_critical": "expiry was too close for the structure",
    "strategy_earnings_close": "the structure was closed before earnings",
    "strategy_profit_target": "the structure reached its profit target",
    "strategy_ic_max_loss_exit": "the iron condor hit half its maximum loss",
    "strategy_loss_limit": "the structure hit its loss limit",
    "strategy_dte_exit": "the structure's days-to-expiry exit",
    "sabr_edge_exit": "the pricing edge was gone",
}


def exit_rule_text(rule) -> str:
    r = str(rule or "")
    return EXIT_RULE_TEXT.get(r) or (r.replace("_", " ").strip() or "an exit rule fired")


# ── selection: deep_score's sub-scores ────────────────────────────────────────
# bot_engine.deep_score: rules = sum(weight x component) + additive points.
SUBSCORE_WEIGHTS = {"momentum": 0.25, "mean_reversion": 0.20, "vrp": 0.25, "squeeze": 0.15, "volume": 0.15}
SUBSCORE_ADDITIVE = ("vol_premium", "macro", "news", "adx", "obv", "intel", "alt", "social",
                     "fundamental", "recommendation", "contrarian", "rel_strength")
SUBSCORE_LABELS = {
    "momentum": "momentum", "mean_reversion": "mean reversion", "vrp": "volatility risk premium",
    "squeeze": "short squeeze", "volume": "volume", "vol_premium": "volatility premium bonus",
    "macro": "macro backdrop", "news": "news", "adx": "trend strength (ADX)", "obv": "on-balance volume",
    "intel": "insider and analyst intelligence", "alt": "alternative data", "social": "social sentiment",
    "fundamental": "fundamentals", "recommendation": "analyst recommendation", "contrarian": "contrarian signal",
    "rel_strength": "relative strength vs SPY", "insider_mspr": "insider buying (MSPR)",
}
SUBSCORE_KEYS = tuple(SUBSCORE_WEIGHTS) + SUBSCORE_ADDITIVE + ("insider_mspr", "rules_raw")


def score_drivers(breakdown) -> list:
    """[{name, label, points}] by |points|, largest first: each term's points in
    the rules score vs a neutral 50 (weighted components) or as added (the
    adjustments). insider_mspr is inside `intel`, rules_raw is the total — both
    are kept as sub-scores but are not drivers."""
    if not isinstance(breakdown, dict):
        return []
    out = []
    for k, w in SUBSCORE_WEIGHTS.items():
        v = _num(breakdown.get(k))
        if v is not None:
            out.append({"name": k, "label": SUBSCORE_LABELS[k], "points": round((v - 50.0) * w, 1)})
    for k in SUBSCORE_ADDITIVE:
        v = _num(breakdown.get(k))
        if v is not None and v != 0:
            out.append({"name": k, "label": SUBSCORE_LABELS[k], "points": round(v, 1)})
    out.sort(key=lambda d: (-abs(d["points"]), d["name"]))
    return out


def dominant_driver(drivers) -> dict | None:
    """The largest positive driver (what pushed the score up); else the largest by size."""
    pos = [d for d in drivers or [] if d.get("points", 0) > 0]
    if pos:
        return max(pos, key=lambda d: d["points"])
    return (drivers or [None])[0]


def selection_stage(cand: dict, threshold=None) -> dict | None:
    if not isinstance(cand, dict) or not cand:
        return None
    score = _num(cand.get("deep_score", cand.get("score")))
    bd = cand.get("score_breakdown") if isinstance(cand.get("score_breakdown"), dict) else None
    drivers = score_drivers(bd)
    subs = None
    if bd:
        subs = {k: round(v, 1) for k in SUBSCORE_KEYS if (v := _num(bd.get(k))) is not None}
    return {
        "score": round(score, 1) if score is not None else None,
        "threshold": _num(threshold),
        "rules_score": _num(cand.get("rules_only_score", cand.get("rules_score"))),
        "ml_score": _num(cand.get("ml_only_score", cand.get("ml_score_raw"))),
        "subscores": subs,
        "drivers": drivers[:5],
        "dominant": dominant_driver(drivers),
        "reasons": [_text(r, 160) for r in (cand.get("reasons") or [])[:6] if r],
        "recommendation": _text(cand.get("recommendation"), 40) or None,
        "side": _text(cand.get("side"), 12) or None,
    }


def signal_for_candidate(cand: dict) -> str:
    """What produced a scan candidate, in a few words."""
    if not isinstance(cand, dict):
        return "deep_score scan"
    if str(cand.get("regime_at_entry") or "") == "OPTIONS_SCANNER":
        return f"options scanner: {_text(cand.get('setup') or cand.get('options_strategy') or 'setup', 60)}"
    dom = dominant_driver(score_drivers(cand.get("score_breakdown")))
    base = "deep_score scan"
    if str(cand.get("trade_type") or "") == "options":
        base = "deep_score scan (options)"
    return f"{base}, {dom['label']}-led" if dom and dom.get("points", 0) > 0 else base


def budget_view(budget) -> dict | None:
    """The regime risk budget's compact view (regime_engine.risk_budget_view: budget, band,
    summary, drivers, applied, stale) as a stage carries it; None without one or on its error."""
    if not isinstance(budget, dict) or budget.get("error") or _num(budget.get("budget")) is None:
        return None
    return {"budget": _num(budget.get("budget")), "band": _text(budget.get("band"), 40) or None,
            "summary": _text(budget.get("summary"), 400) or None,
            "drivers": [_text(d, 160) for d in (budget.get("drivers") or [])[:6]],
            "applied": bool(budget.get("applied")), "stale": bool(budget.get("stale")),
            "as_of": _text(budget.get("as_of"), 40) or None}


def regime_stage(label=None, gate_regime=None, vxx_ratio=None, reason=None, budget=None) -> dict | None:
    """The regime stage. With the regime risk budget (scan_market's `risk_budget`), its
    ready-made plain-English `summary` is the stage's reason; else the label + VXX read."""
    label = _text(label, 40) or None
    gate = _text(gate_regime, 40) or None
    vxx = _num(vxx_ratio)
    rb = budget_view(budget)
    if not (label or gate or vxx is not None or reason or rb):
        return None
    shown = gate or label
    text = f"{shown} market" if shown else "market regime"
    if vxx is not None:
        text += f" (VXX {vxx:.2f}x its 30-day average)"
    if label and gate and label != gate:
        text += f"; the scorer's regime read was {label}"
    if reason:
        text += f"; {_text(reason, 200)}"
    out = {"label": shown, "scorer_label": label, "gate_regime": gate, "vxx_ratio": vxx, "reason": text}
    if rb:
        out["risk_budget"] = rb
        if rb["summary"]:
            out["reason"] = rb["summary"] + (f" ({text})" if shown or vxx is not None else "")
    return out


def flows_stage(flows) -> dict | None:
    """The institutional-flow stage (flows_model, via the trade's / floor result's `flows`
    block: sector, etf, pressure, multiplier, applied, reason) — kept as-is; None without one."""
    if not isinstance(flows, dict) or not flows:
        return None
    return _clean(flows)


def regime_budget_from_state(regime_state):
    """regime_engine.risk_budget_view of the scan's regime state, when that function exists
    (the regime risk budget, 2026-11-06); None otherwise. Never raises."""
    if not isinstance(regime_state, dict):
        return None
    try:
        import regime_engine
        view = getattr(regime_engine, "risk_budget_view", None)
        return view(regime_state) if callable(view) else None
    except Exception as e:
        log.debug(f"[DECISION-JOURNAL] risk budget view unavailable: {e}")
        return None


def sector_stage(sector, reason=None) -> dict | None:
    s = _text(sector, 60)
    if not s or s.lower() == "unknown":
        return {"name": "Unknown", "reason": _text(reason, 200) or None} if reason else None
    return {"name": s, "reason": _text(reason, 200) or None}


def _extra_stages(cand: dict) -> dict:
    """Stages other modules attached to a candidate (calendar / flows / tax ...)."""
    ds = cand.get("decision_stages") if isinstance(cand, dict) else None
    return {k: v for k, v in (ds or {}).items() if isinstance(v, dict)}


# ── normalize + write ─────────────────────────────────────────────────────────
def normalize(row: dict) -> dict | None:
    """The contract shape, or None for a row that cannot be one (unknown action, no ticker)."""
    if not isinstance(row, dict):
        return None
    action = str(row.get("action") or "").lower()
    ticker = _text(row.get("ticker"), 24).upper()
    if action not in ACTIONS or not ticker:
        return None
    stages_in = row.get("stages") if isinstance(row.get("stages"), dict) else {}
    stages = {k: (_clean(stages_in.get(k)) if isinstance(stages_in.get(k), dict) else None) for k in STAGE_KEYS}
    for k, v in stages_in.items():
        if k not in stages and isinstance(v, dict):
            stages[str(k)[:30]] = _clean(v)
    rejected = []
    for r in row.get("rejected") or []:
        if isinstance(r, dict) and r.get("instrument"):
            alt = {"instrument": _text(r.get("instrument"), 40), "reason": _text(r.get("reason"), 300)}
            if isinstance(r.get("would_be"), dict) and r["would_be"]:
                alt["would_be"] = _clean(r["would_be"])
            rejected.append(alt)
    out = {
        "v": SCHEMA_VERSION,
        "id": _text(row.get("id"), 40) or uuid.uuid4().hex[:16],
        "ts": _text(row.get("ts"), 40) or now_iso(),
        "ticker": ticker,
        "action": action,
        "instrument": _text(row.get("instrument") or "stock", 40),
        "chosen": _text(row.get("chosen"), 80) or ("none" if action == "skip" else ""),
        "rejected": rejected[:6],
        "stages": stages,
        "sizing": _clean(row.get("sizing")) if isinstance(row.get("sizing"), dict) else None,
        "summary": _text(row.get("summary"), 900),
    }
    for k in OPTIONAL_KEYS:
        v = row.get(k)
        if v is None or v == "" or v == [] or v == {}:
            continue
        c = _clean(v)
        if c is not None:
            out[k] = c
    return out


def _fit(row: dict) -> str:
    """One JSON line of at most ROW_MAX_BYTES: long detail is cut, never the contract keys."""
    text = _dumps(row)
    if len(text.encode("utf-8")) <= ROW_MAX_BYTES:
        return text
    r = dict(row)
    r["summary"] = r.get("summary", "")[:400]
    r["rejected"] = [{"instrument": x["instrument"], "reason": x["reason"][:160],
                      **({"would_be": x["would_be"]} if x.get("would_be") else {})} for x in r.get("rejected", [])[:4]]
    st = dict(r.get("stages") or {})
    if isinstance(st.get("selection"), dict):
        st["selection"] = {k: v for k, v in st["selection"].items() if k not in ("reasons", "subscores")}
    r["stages"] = st
    r["truncated"] = True
    text = _dumps(r)
    if len(text.encode("utf-8")) <= ROW_MAX_BYTES:
        return text
    r["stages"] = {k: None for k in STAGE_KEYS}
    r["sizing"] = None
    r.pop("order", None)
    return _dumps(r)


def append_rows(rows) -> int:
    """Normalize and append rows (one write), rotating above max_bytes. Returns rows written; never raises."""
    try:
        if not enabled():
            return 0
        lines = []
        for r in rows or []:
            n = normalize(r)
            if n is not None:
                lines.append(_fit(n) + "\n")
        if not lines:
            return 0
        data = "".join(lines).encode("utf-8")
        p = journal_path()
        with _write_lock:
            if os.path.exists(p) and os.path.getsize(p) > max_bytes():
                os.replace(p, p + ".1")
            fd = os.open(p, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
            finally:
                os.close(fd)
        return len(lines)
    except (OSError, TypeError, ValueError) as e:
        log.warning(f"[DECISION-JOURNAL] rows not written (no decision changed): {e}")
        return 0


def read_rows(include_rotated: bool = True) -> list:
    """Every parseable row, oldest first (both generations). For tests and audits."""
    out = []
    p = journal_path()
    for path in ([p + ".1"] if include_rotated else []) + [p]:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except ValueError as e:
                        log.debug(f"[DECISION-JOURNAL] malformed row skipped: {e}")
        except OSError as e:
            log.warning(f"[DECISION-JOURNAL] {path} unreadable: {e}")
    return out


# ── once-per-day dedupe ───────────────────────────────────────────────────────
def _tail_keys(day: str) -> set:
    keys = set()
    p = journal_path()
    try:
        if not os.path.exists(p):
            return keys
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            f.seek(max(0, size - SEED_TAIL_BYTES))
            chunk = f.read().decode("utf-8", errors="replace")
        for line in chunk.splitlines():
            if '"dedupe"' not in line:
                continue
            try:
                k = json.loads(line).get("dedupe")
            except ValueError as e:
                log.debug(f"[DECISION-JOURNAL] tail line skipped while seeding: {e}")
                continue
            if isinstance(k, str) and k.endswith("|" + day):
                keys.add(k)
    except OSError as e:
        log.debug(f"[DECISION-JOURNAL] dedupe seed unavailable: {e}")
    return keys


def first_today(key: str, now=None) -> str | None:
    """The dedupe key (key + ET day) the first time `key` is offered this ET day; None after."""
    day = et_day(now)
    full = f"{key}|{day}"
    with _seen_lock:
        if _seen["day"] != day:
            _seen.update(day=day, keys=set(), seeded=False)
        if not _seen["seeded"]:
            _seen["keys"] |= _tail_keys(day)
            _seen["seeded"] = True
        if full in _seen["keys"]:
            return None
        _seen["keys"].add(full)
        return full


def _reset_dedupe() -> None:
    """Tests: forget what was seen."""
    with _seen_lock:
        _seen.update(day=None, keys=set(), seeded=False)


# ── producers ─────────────────────────────────────────────────────────────────
def _instrument_of(cand: dict) -> str:
    tt = str((cand or {}).get("trade_type") or "").lower()
    return "option" if tt in ("options", "csp", "csp_options") else "stock"


def skip_summary(ticker, decision, reason="", score=None, threshold=None) -> str:
    s = f"Skipped {ticker}: {rule_text(decision)}"
    reason = _text(reason, 220)
    if reason:
        s += f" ({reason})"
    sc, th = _num(score), _num(threshold)
    if sc is not None and "score" not in reason.lower():
        s += f". Its score was {sc:.1f}" + (f" against {th:g} needed" if th is not None else "")
    return s + "."


def scan_skip_row(ticker, decision, reason, cand=None, scan_regime=None, threshold=None,
                  vxx_ratio=None, source="bot_engine.scan_market", dedupe=None, risk_budget=None) -> dict:
    cand = cand if isinstance(cand, dict) else {}
    inst = _instrument_of(cand)
    proposed = "a stock buy" if inst == "stock" else f"an options trade ({_text(cand.get('action_label') or 'options', 40)})"
    stages = {
        "regime": regime_stage(cand.get("regime_label"), scan_regime, vxx_ratio, budget=risk_budget),
        "flows": flows_stage(cand.get("flows")),
        "sector": sector_stage(cand.get("sector")),
        "selection": selection_stage(cand, threshold),
    }
    stages.update(_extra_stages(cand))
    return {
        "ts": now_iso(), "ticker": ticker, "action": "skip", "instrument": inst, "chosen": "none",
        "rejected": [{"instrument": inst, "reason": f"{proposed} was refused because {rule_text(decision)}"
                      + (f": {_text(reason, 200)}" if reason else "")}],
        "stages": stages, "sizing": None,
        "summary": skip_summary(ticker, decision, reason, cand.get("deep_score", cand.get("score")), threshold),
        "source": source, "signal": signal_for_candidate(cand), "rule": str(decision), "dedupe": dedupe,
    }


def record_scan_skips(skips, candidates=None, scan_regime=None, threshold=None, vxx_ratio=None,
                      source="bot_engine.scan_market", now=None, regime_state=None) -> int:
    """scan_market's per-candidate gates: skips = [{"ticker", "decision", "reason"}] (the
    shadow relabels + the sector / spread / sizing blocks); candidates = the deep-scored
    dicts (score, sub-scores, sector, regime); regime_state = the scan's regime state (its
    risk budget's summary becomes the regime stage's reason). "already held" is not journaled."""
    try:
        if not skips or not enabled():
            return 0
        risk_budget = regime_budget_from_state(regime_state)
        by_t = {}
        for c in candidates or []:
            if isinstance(c, dict) and c.get("ticker"):
                by_t[str(c["ticker"]).upper()] = c
        rows = []
        for s in skips:
            if not isinstance(s, dict):
                continue
            t = _text(s.get("ticker"), 24).upper()
            dec = _text(s.get("decision"), 60)
            if not t or not dec or dec == "rejected_held":
                continue
            key = first_today(f"skip|{t}|{dec}", now)
            if key is None:
                continue
            rows.append(scan_skip_row(t, dec, s.get("reason"), by_t.get(t), scan_regime, threshold,
                                      vxx_ratio, source, key, risk_budget))
            if len(rows) >= MAX_ROWS_PER_CALL:
                break
        return append_rows(rows)
    except Exception as e:
        log.warning(f"[DECISION-JOURNAL] scan skips not recorded (scan unaffected): {e}")
        return 0


def _csp_options_stage(meta: dict, strategy="sell_cash_secured_put") -> dict:
    meta = meta if isinstance(meta, dict) else {}
    out = {"strategy": strategy}
    for k in ("target_delta", "target_dte_min", "target_dte_max", "take_profit_pct", "min_iv_rank"):
        v = _num(meta.get(k))
        if v is not None:
            out[k] = v
    return out


def record_tier_rejections(rejections, source="tiered_strategy.log_csp_gate_shadow", now=None) -> int:
    """SELL_CSP actions a gate refused (tier1's live regime guards, the dispatcher's slot /
    resubmit gates). Shadow-only verdicts ("would_reject_*") are NOT decisions: the name
    was still proposed, so they are not journaled."""
    try:
        if not rejections or not enabled():
            return 0
        rows = []
        for r in rejections:
            if not isinstance(r, dict):
                continue
            t = _text(r.get("ticker"), 24).upper()
            dec = _text(r.get("decision"), 60)
            if not t or not dec.startswith("rejected_"):
                continue
            key = first_today(f"skip|{t}|{dec}", now)
            if key is None:
                continue
            meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
            reason = _text(r.get("reason"), 220)
            rows.append({
                "ts": now_iso(), "ticker": t, "action": "skip", "instrument": "option", "chosen": "none",
                "rejected": [{"instrument": "option", "reason": f"a cash-secured put was refused because {rule_text(dec)}"
                              + (f": {reason}" if reason else "")}],
                "stages": {"regime": regime_stage(meta.get("regime_label"), None, meta.get("vxx_ratio")),
                           "options": _csp_options_stage(meta)},
                "summary": skip_summary(t, dec, reason),
                "source": source, "signal": "tier1 CSP harvest", "rule": dec, "dedupe": key,
            })
            if len(rows) >= MAX_ROWS_PER_CALL:
                break
        return append_rows(rows)
    except Exception as e:
        log.warning(f"[DECISION-JOURNAL] tier rejections not recorded (tiers unaffected): {e}")
        return 0


def record_tier_kill(actions, kill_reason, regime=None, vxx_ratio=None,
                     source="tiered_strategy.master_kill_switch", now=None) -> int:
    """The CSPs tier1 would have proposed while the master kill switch blocked every entry."""
    try:
        if not actions or not enabled():
            return 0
        rows = []
        for a in actions:
            t = _text(getattr(a, "ticker", None) if not isinstance(a, dict) else a.get("ticker"), 24).upper()
            if not t:
                continue
            key = first_today(f"skip|{t}|rejected_masterkill", now)
            if key is None:
                continue
            meta = (getattr(a, "metadata", None) if not isinstance(a, dict) else a.get("metadata")) or {}
            rows.append({
                "ts": now_iso(), "ticker": t, "action": "skip", "instrument": "option", "chosen": "none",
                "rejected": [{"instrument": "option", "reason": f"a cash-secured put was refused because "
                              f"{rule_text('rejected_masterkill')}: {_text(kill_reason, 200)}"}],
                "stages": {"regime": regime_stage(meta.get("regime_label") or regime, None,
                                                  meta.get("vxx_ratio", vxx_ratio)),
                           "options": _csp_options_stage(meta)},
                "summary": skip_summary(t, "rejected_masterkill", kill_reason),
                "source": source, "signal": "tier1 CSP harvest", "rule": "rejected_masterkill", "dedupe": key,
            })
            if len(rows) >= MAX_ROWS_PER_CALL:
                break
        return append_rows(rows)
    except Exception as e:
        log.warning(f"[DECISION-JOURNAL] master-kill skips not recorded (kill unaffected): {e}")
        return 0


_EXIT_KINDS = ("CLOSE", "CLOSE_STRATEGY", "ROLL")


def option_exit_row(a: dict, source="options_manager.manage_options_positions", dedupe=None) -> dict | None:
    kind = str(a.get("action") or "")
    if kind not in _EXIT_KINDS:
        return None
    order = a.get("order") if isinstance(a.get("order"), dict) else {}
    status = _text(order.get("status"), 30) or "unknown"
    legs = [str(x) for x in (a.get("legs") or []) if x] or ([str(a["occ_symbol"])] if a.get("occ_symbol") else [])
    ticker = _text(a.get("ticker"), 24).upper() or (legs[0][:-15] if legs and len(legs[0]) > 15 else "")
    if not ticker:
        return None
    rule = _text(a.get("type"), 60) or "exit"
    strategy = _text(a.get("strategy"), 40)
    what = f"{strategy} ({len(legs)} legs)" if strategy and strategy != "unknown" and len(legs) > 1 \
        else (strategy if strategy and strategy != "unknown" else "option")
    verb = "Rolled" if kind == "ROLL" else "Closed"
    pnl = _num(a.get("combined_pnl"))
    contract = f" ({describe_occ(legs[0])})" if len(legs) == 1 else ""
    summary = (f"{verb} the {ticker} {what}{contract}: {exit_rule_text(rule)}. "
               f"{_sentence(a.get('reason'))}")
    if pnl is not None:
        summary += f" Unrealized P&L at the decision: {_money(pnl)}."
    summary += f" Order {status}" + (f": {_text(order.get('detail'), 140)}" if status in ("error", "failed") else "") + "."
    return {
        "ts": now_iso(), "ticker": ticker, "action": "rotate" if kind == "ROLL" else "sell", "instrument": "option",
        "chosen": rule, "rejected": [],
        "stages": {"options": {"strategy": strategy or None, "legs": legs[:4], "rule": rule,
                               "reason": _text(a.get("reason"), 300), "unrealized_at_decision": pnl,
                               "dte": _num(a.get("dte"))}},
        "summary": summary, "source": source, "signal": f"exit: {exit_rule_text(rule)}",
        "symbol": legs[0] if len(legs) == 1 else None,
        "order": {"status": status, "id": _text(order.get("order_id"), 60) or None,
                  "method": _text(order.get("method"), 30) or None,
                  "detail": _text(order.get("detail"), 200) or None},
        "rule": rule, "dedupe": dedupe,
    }


def record_option_exits(actions, source="options_manager.manage_options_positions", now=None) -> int:
    """options_manager's CLOSE / CLOSE_STRATEGY / ROLL actions that reached the broker (a
    deferred close is re-judged on regular-hours marks, so it is not a decision yet). One
    row per (contract or legs, rule, order status) per ET day."""
    try:
        if not actions or not enabled():
            return 0
        rows = []
        for a in actions:
            if not isinstance(a, dict) or a.get("action") not in _EXIT_KINDS:
                continue
            order = a.get("order") if isinstance(a.get("order"), dict) else {}
            status = str(order.get("status") or "")
            if status == "deferred":
                continue
            legs = sorted(str(x) for x in (a.get("legs") or []) if x) or [str(a.get("occ_symbol") or a.get("ticker") or "")]
            key = first_today(f"exit|{','.join(legs)}|{a.get('type')}|{status}", now)
            if key is None:
                continue
            row = option_exit_row(a, source, key)
            if row:
                rows.append(row)
            if len(rows) >= MAX_ROWS_PER_CALL:
                break
        return append_rows(rows)
    except Exception as e:
        log.warning(f"[DECISION-JOURNAL] option exits not recorded (exits unaffected): {e}")
        return 0


_FLOOR_ORDER_TYPES = ("floor_vm_buy", "floor_vm_sell", "floor_vm_legacy_sell")


def record_floor(result, source="bot_engine._manage_vol_managed_floor") -> int:
    """The volatility-managed floor's orders: a buy toward the target is a scale_in, a sell
    (toward the target, or of a legacy floor ticker) a scale_out — each with the plan's
    numbers (target and current exposure, 20-day volatility, capital)."""
    try:
        if not isinstance(result, dict) or not enabled():
            return 0
        rows = []
        sigma = _num(result.get("sigma"))
        e_star = _num(result.get("target_exposure"))
        e_now = _num(result.get("current_exposure"))
        reduce_only = bool(result.get("reduce_only"))
        target_vol = _num(_base_config_value("FLOOR_VM_TARGET_VOL")) or 0.16
        why = (f"20-day volatility {sigma * 100:.1f}% against a {target_vol * 100:.0f}% target puts the target "
               f"exposure at {e_star:.2f}x (now {e_now:.2f}x)") \
            if sigma is not None and e_star is not None and e_now is not None \
            else "the volatility-managed floor rebalanced"
        _fl = result.get("flows") if isinstance(result.get("flows"), dict) else {}
        if _fl.get("applied") and _fl.get("reason"):
            why += f"; institutional flows: {_text(_fl.get('reason'), 200)}"
        for a in result.get("actions") or []:
            if not isinstance(a, dict) or a.get("type") not in _FLOOR_ORDER_TYPES:
                continue
            sym = _text(a.get("ticker"), 12).upper()
            qty, px = _num(a.get("qty")), _num(a.get("limit"))
            if not sym or not qty:
                continue
            buy = a["type"] == "floor_vm_buy"
            legacy = a["type"] == "floor_vm_legacy_sell"
            ok = bool(a.get("ok"))
            what = (f"sold all {qty:g} {sym} (a floor ticker the volatility-managed floor no longer holds)" if legacy
                    else f"{'bought' if buy else 'sold'} {qty:g} {sym} at {_price(px)}")
            summary = f"Floor: {what}. Why: {why}" + (" (drawdown halt: reduce only)" if reduce_only else "") + "."
            if not ok:
                summary += f" The order was refused: {_text(a.get('error'), 160)}."
            rows.append({
                "ts": now_iso(), "ticker": sym, "action": "scale_in" if buy else "scale_out", "instrument": "floor",
                "chosen": "sell_legacy_floor" if legacy else "vol_managed_spy", "rejected": [],
                "stages": {"regime": {"label": "volatility-managed floor", "sigma20": sigma, "reason": why},
                           "flows": flows_stage(result.get("flows"))},
                "sizing": {"target_exposure": e_star, "current_exposure": e_now, "capital": _num(result.get("capital")),
                           "target_base_dollars": _num(result.get("target_base_dollars")),
                           "target_levered_dollars": _num(result.get("target_levered_dollars")),
                           "reduce_only": reduce_only},
                "summary": summary, "source": source, "signal": "floor: volatility-managed SPY",
                "side": "buy" if buy else "sell", "symbol": sym, "qty": qty, "price": px,
                "notional": round(qty * px, 2) if px else None,
                "order": {"status": "submitted" if ok else "error", "id": a.get("order_id"),
                          "detail": _text(a.get("error"), 200) or None},
            })
        return append_rows(rows)
    except Exception as e:
        log.warning(f"[DECISION-JOURNAL] floor decisions not recorded (floor unaffected): {e}")
        return 0
