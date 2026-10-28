#!/usr/bin/env python3
"""
regime_engine.py — the single owner of market-regime state.

PROGRAM: docs/regime/FINDINGS.md (AlphaDesk Market Regime & Volatility Engine,
human directive 2026-10-27), Stage B step 2. Evidence: docs/regime/RESEARCH.md
(§1.6, §2.1, §3.1, §4.3, §5, §6 ranks 1–12); what it replaces: docs/regime/
AUDIT.md §2 (five regime vocabularies, none of them owned). Every number
below is a documented starting value from the research, not a tuned result;
scripts/regime_backtest.py evaluates this module's exposure path on the
repo's own index bars (same code, one implementation).

FLAG: VOLTRADE_REGIME_ENGINE (env; "1"/"true"/"on"). OFF by default — the
module is then never imported on the scan path. ON = SHADOW: the state is
computed once per scan, persisted and served; no consumer reads it until its
own flag (VOLTRADE_REGIME_SELECTION / _OPTIONS / _SIZING / _LABELS) is set.

WHAT THE STATE IS
  trend_state   UP / NEUTRAL / DOWN from the SLOW ensemble only: the share
                of 120/160/200/240-day price-vs-SMA votes that are positive
                (each vote has a ±1% band so it cannot flip on noise); UP at
                >= 3 of 4, DOWN at <= 1 of 4, passed through the same 6-day
                majority filter as the composite. This is the primary regime
                and the flip budget (research: 0.5–2 changes a year; measured
                2.7/yr on SPY 2007–2026 after the filter, 4.7 raw; alarm > 6).
  fast_state    UP / MIXED / DOWN from the 22- and 65-day return signs (one
                vote per horizon — never TSMOM and MA-cross on the same
                horizon). Fast legs only DOWNGRADE exposure; they never set
                the regime label (RESEARCH §4.3a).
  vol_state     LOW / NORMAL / ELEVATED / CRISIS from the EWMA realized vol
                of the gating index (20-day half-life) ranked against its own
                history (LOW < 20th pct, ELEVATED ≥ 80th, CRISIS = ELEVATED
                and (VIX/VIX3M > 1 or ≥ 95th pct)), each state left only 10
                percentile points below its entry level (hysteresis).
  vol_dial      min(1, σ_target / σ̂), σ_target = the index's own long-run
                realized vol fixed ex ante (mean of σ̂ over the prior 2–10
                years) — de-risk only, never levers up.
  breadth_state HEALTHY / NARROW / WEAK = share of a fixed basket (11 sector
                SPDRs + IWM + MDY) above its 200-day SMA; a caution flag and a
                selection guard, never a sell trigger.
  panic_state   gating index 24-month return < 0 AND 126-day variance in its
                top quintile (Daniel–Moskowitz) — conditions stock selection.
  composite     CALM_BULL / VOLATILE_BULL / CHOPPY / STRESSED / BEAR / CRISIS
                by the decode table; passed through a 6-day majority filter.
  exposure      exposure_multiplier ∈ {0, .25, .5, .75, 1} for NEW
                discretionary risk = quarter-quantized
                (0.5·f_slow + 0.25·f_fast + 0.25·breadth_bin) × vol_dial;
                downgrades apply as soon as the filter clears, upgrades glide
                +0.25 per 5 sessions (tranched, no calendar-day luck);
                core_multiplier = max(0.5, exposure_continuous) for the index
                floor sleeve — never fully out (RESEARCH §5.5).
  plus          legacy_label (BULL/NEUTRAL/CAUTION/BEAR/PANIC for consumers
                that only know the old vocabulary), allowed_strategies,
                confidence, days_in_regime, pending_transition, flips_12m and
                an alarm above 6 flips a year, stress_flags (VIX/VIX3M
                inversion, VRP sign, 1-year VIX percentile, VVIX), the Fed
                excess bond premium, and every input with its date and source.

DATA. Daily closes via etf_data_sources.fetch_bars_alpaca (one batched call)
with the Yahoo chart endpoint as the fallback and for the index family
(^VIX, ^VIX3M, ^VVIX); the Fed EBP CSV monthly. Everything is cached for the
day under storage_config.CACHE_DIR; the state and its history live in
DATA_DIR (storage_config.REGIME_STATE_PATH / REGIME_HISTORY_PATH).

NEVER: trades, places orders, edits a risk limit, or raises into the scan —
regime_state_rpc / update_state return an error dict instead.
"""
from __future__ import annotations

import collections
import json
import logging
import math
import os
import statistics as st
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

log = logging.getLogger("voltrade.regime")

PARAMS_VERSION = "2026-10-28.1"

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLC", "XLRE"]
BREADTH_BASKET = SECTOR_ETFS + ["IWM", "MDY"]
CREDIT_PAIR = ["HYG", "IEF"]
INDEX_FAMILY = ["^VIX", "^VIX3M", "^VVIX"]

DEFAULT_PARAMS = {
    # trend legs (RESEARCH §2.1, §4.3, §4.5: one vote per horizon, ensembled lookbacks)
    "slow_horizons": (120, 160, 200, 240),
    "fast_horizons": (22, 65),
    "band": 0.01,                 # ±1% hysteresis band on every trend vote (STW/BLL; churn control only)
    "trend_up": 0.75,             # trend_score >= -> UP
    "trend_down": 0.25,           # trend_score <= -> DOWN
    # volatility (RESEARCH §3.1: EWMA ~20-day half-life, act in the extreme quintile, cap 1)
    "vol_halflife": 20,
    "vol_warmup": 270,            # sessions before the dial may act (Harvey et al.)
    "vol_pct_min_obs": 504,       # history needed before a percentile means anything
    "vol_pct_window": 2520,       # ≤ 10 years of own history for the percentile
    "vol_low_pct": 0.20,
    "vol_elevated_pct": 0.80,
    "vol_crisis_pct": 0.95,
    "vol_exit_margin": 0.10,      # hysteresis: leave a vol state 10 percentile points below its entry level
    "vol_target_min_obs": 504,    # σ_target = mean σ̂ over the prior [min_obs, window] sessions
    "vol_target_window": 2520,
    # breadth (RESEARCH §2.3: flag + guard, never a throttle)
    "breadth_healthy": 0.60,
    "breadth_weak": 0.40,
    "breadth_min_members": 6,
    # transitions (RESEARCH §4.3: 6-day majority filter; fast out, tranched in)
    "filter_days": 6,
    "upgrade_step": 0.25,
    "upgrade_every": 5,
    # panic state (RESEARCH §2.2, Daniel–Moskowitz)
    "panic_lookback": 504,
    "panic_var_window": 126,
    "panic_var_pct": 0.80,
    "panic_min_obs": 504,
    # outputs
    "core_floor": 0.50,           # never fully out on the index sleeve (RESEARCH §5.5)
    "flip_alarm_12m": 6,          # ≈ the daily-200-day artefact rate (RESEARCH §4.3e)
    "stale_market_days": 2,
    "rv_window": 21,              # realized vol window for the VRP gauge
}

COMPOSITES = ("CALM_BULL", "VOLATILE_BULL", "CHOPPY", "STRESSED", "BEAR", "CRISIS")
LEGACY_MAP = {"CALM_BULL": "BULL", "CHOPPY": "NEUTRAL", "VOLATILE_BULL": "CAUTION",
              "STRESSED": "CAUTION", "BEAR": "BEAR", "CRISIS": "PANIC"}
TRUTHY = ("1", "true", "on", "yes")


# ── flag ─────────────────────────────────────────────────────────────────────
def engine_allowed(env=None, config=None) -> bool:
    """Pure gate: VOLTRADE_REGIME_ENGINE env flag OR system_config
    BASE_CONFIG["REGIME_ENGINE_ENABLED"] (default OFF — AUDIT.md's five
    vocabularies keep running untouched until one of them is set)."""
    env = os.environ if env is None else env
    if str(env.get("VOLTRADE_REGIME_ENGINE", "0")).strip().lower() in TRUTHY:
        return True
    if config is None:
        try:
            from system_config import BASE_CONFIG as config
        except Exception:
            config = {}
    return bool(config.get("REGIME_ENGINE_ENABLED", False))


def params_with(overrides=None) -> dict:
    p = dict(DEFAULT_PARAMS)
    if overrides:
        p.update(overrides)
    return p


# ── pure arithmetic ──────────────────────────────────────────────────────────
def sma(values, i, n):
    """Average of values[i-n+1..i]; None when there is not enough (non-None) history."""
    if i < n - 1:
        return None
    window = values[i - n + 1:i + 1]
    if any(v is None for v in window):
        return None
    return sum(window) / n


def ewma_vol_series(closes, halflife):
    """Annualized EWMA volatility of log returns, one value per close (None at index 0 and while closes are missing)."""
    lam = 0.5 ** (1.0 / float(halflife))
    var = None
    out = [None] * len(closes)
    for i in range(1, len(closes)):
        a, b = closes[i], closes[i - 1]
        if a is None or b is None or a <= 0 or b <= 0:
            out[i] = math.sqrt(var * 252) if var else None
            continue
        r = math.log(a / b)
        var = r * r if var is None else lam * var + (1 - lam) * r * r
        out[i] = math.sqrt(var * 252)
    return out


def percentile_rank(history, x):
    """Share of `history` values <= x (0..1); None when history is empty."""
    vals = [h for h in history if h is not None]
    if not vals or x is None:
        return None
    return sum(1 for h in vals if h <= x) / len(vals)


def quantize_quarter(x):
    return max(0.0, min(1.0, round(float(x) * 4) / 4))


def majority(window, current):
    """The label that > half of `window` agrees on, else `current` (hold)."""
    if not window:
        return current
    counts = collections.Counter(window)
    label, n = counts.most_common(1)[0]
    return label if n * 2 > len(window) else current


# ── signals ──────────────────────────────────────────────────────────────────
def signal_series(dates, gating, basket, vix=None, vix3m=None, params=None):
    """Per-day signal dicts for a gating index.

    dates    : ascending trading dates
    gating   : closes of the gating index aligned to `dates` (None allowed before inception)
    basket   : {symbol: closes aligned to dates} for the breadth basket (None before inception)
    vix/vix3m: closes aligned to dates (or None lists) — the index family may be absent
    """
    p = params_with(params)
    n = len(dates)
    if len(gating) != n:
        raise ValueError("gating closes must align to dates")
    vix = vix or [None] * n
    vix3m = vix3m or [None] * n
    sig = ewma_vol_series(gating, p["vol_halflife"])
    slow_state = {h: 1 for h in p["slow_horizons"]}
    fast_state = {h: 1 for h in p["fast_horizons"]}
    var_hist = []
    out = []
    for i in range(n):
        c = gating[i]
        # ── trend votes with a hysteresis band ──
        votes = []
        for h in p["slow_horizons"]:
            m = sma(gating, i, h)
            if m is None or c is None:
                votes.append(slow_state[h])
                continue
            x = c / m - 1.0
            s = slow_state[h]
            if s == 1 and x < -p["band"]:
                s = 0
            elif s == 0 and x > p["band"]:
                s = 1
            slow_state[h] = s
            votes.append(s)
        f_slow = sum(votes) / len(votes)
        m200 = sma(gating, i, 200)
        above_200 = (c > m200) if (m200 is not None and c is not None) else None
        hi_win = [x for x in gating[max(0, i - 251):i + 1] if x is not None]
        dd52 = (c / max(hi_win) - 1.0) if (c is not None and hi_win) else None
        fv = []
        for h in p["fast_horizons"]:
            if i < h or c is None or gating[i - h] is None:
                fv.append(fast_state[h])
                continue
            x = c / gating[i - h] - 1.0
            s = fast_state[h]
            if s == 1 and x < -p["band"]:
                s = 0
            elif s == 0 and x > p["band"]:
                s = 1
            fast_state[h] = s
            fv.append(s)
        f_fast = sum(fv) / len(fv)
        trend_score = 0.5 * f_slow + 0.25 * fv[0] + 0.25 * fv[1] if len(fv) == 2 else 0.5 * f_slow + 0.5 * f_fast
        # ── breadth ──
        up = tot = 0
        for sym, closes in basket.items():
            m = sma(closes, i, 200)
            if m is None or closes[i] is None:
                continue
            tot += 1
            up += 1 if closes[i] > m else 0
        breadth = (up / tot) if tot >= p["breadth_min_members"] else None
        breadth_bin = quantize_quarter(breadth) if breadth is not None else 1.0
        # ── volatility state and dial ──
        s_now = sig[i]
        lo = max(0, i - p["vol_pct_window"])
        hist = [v for v in sig[lo:i] if v is not None]
        vol_pct = percentile_rank(hist, s_now) if (s_now is not None and len(hist) >= p["vol_pct_min_obs"]) else None
        t_hist = [v for v in sig[max(0, i - p["vol_target_window"]):i] if v is not None]
        sigma_target = (sum(t_hist) / len(t_hist)) if len(t_hist) >= p["vol_target_min_obs"] else None
        if s_now and sigma_target and i >= p["vol_warmup"]:
            m_vol = min(1.0, sigma_target / s_now)
        else:
            m_vol = 1.0
        # ── VIX family ──
        v, v3 = vix[i], vix3m[i]
        ts = (v / v3) if (v and v3) else None
        rv = None
        w = p["rv_window"]
        if i >= w and all(gating[t] is not None for t in range(i - w, i + 1)):
            rets = [math.log(gating[t] / gating[t - 1]) for t in range(i - w + 1, i + 1)]
            rv = st.pstdev(rets) * math.sqrt(252) * 100.0
        vrp = (v - rv) if (v is not None and rv is not None) else None
        vix_hist = [x for x in vix[max(0, i - 251):i + 1] if x is not None]
        vix_pct_1y = percentile_rank(vix_hist, v) if (v is not None and len(vix_hist) >= 126) else None
        # ── panic state ──
        lb, vw = p["panic_lookback"], p["panic_var_window"]
        mom = (c / gating[i - lb] - 1.0) if (i >= lb and c is not None and gating[i - lb]) else None
        var126 = None
        if i >= vw and all(gating[t] is not None for t in range(i - vw, i + 1)):
            rets = [math.log(gating[t] / gating[t - 1]) for t in range(i - vw + 1, i + 1)]
            var126 = st.pvariance(rets)
        var_pct = percentile_rank(var_hist, var126) if (var126 is not None and len(var_hist) >= p["panic_min_obs"]) else None
        if var126 is not None:
            var_hist.append(var126)
        panic = bool(mom is not None and mom < 0 and var_pct is not None and var_pct >= p["panic_var_pct"])
        out.append({
            "f_slow": f_slow, "f_fast": f_fast, "trend_score": trend_score,
            "breadth": breadth, "breadth_bin": breadth_bin,
            "sigma": s_now, "sigma_target": sigma_target, "vol_pct": vol_pct, "m_vol": m_vol,
            "ts": ts, "rv": rv, "vrp": vrp, "vix": v, "vix_pct_1y": vix_pct_1y,
            "mom24m": mom, "var126_pct": var_pct, "panic": panic, "above_200d": above_200, "dd52": dd52,
        })
    return out


def trend_state_of(f_slow, params=None):
    """PRIMARY trend from the slow ensemble only (share of 120–240-day votes positive):
    UP at >= 3 of 4, DOWN at <= 1 of 4. The fast legs never set the regime label —
    they downgrade EXPOSURE (RESEARCH §4.3a: fast out as signal speed, slow leg re-enters)."""
    p = params_with(params)
    if f_slow >= p["trend_up"]:
        return "UP"
    if f_slow <= p["trend_down"]:
        return "DOWN"
    return "NEUTRAL"


def fast_state_of(f_fast):
    """The 22/65-day legs: DOWN when both are negative, UP when both positive, else MIXED."""
    if f_fast <= 0.0:
        return "DOWN"
    if f_fast >= 1.0:
        return "UP"
    return "MIXED"


def vol_state_of(vol_pct, ts, prev=None, params=None):
    """Volatility state with hysteresis: a state is entered at its threshold and left
    only `vol_exit_margin` below it (LOW is the mirror), so a percentile hovering at a
    boundary does not flip the label every few days."""
    p = params_with(params)
    if vol_pct is None:
        return "NORMAL"
    m = p["vol_exit_margin"]
    inverted = bool(ts is not None and ts > 1.0)
    crisis_in = vol_pct >= p["vol_elevated_pct"] and (inverted or vol_pct >= p["vol_crisis_pct"])
    crisis_hold = prev == "CRISIS" and vol_pct >= p["vol_elevated_pct"] - m and (inverted or vol_pct >= p["vol_crisis_pct"] - m)
    if crisis_in or crisis_hold:
        return "CRISIS"
    elevated_in = vol_pct >= p["vol_elevated_pct"]
    elevated_hold = prev in ("ELEVATED", "CRISIS") and vol_pct >= p["vol_elevated_pct"] - m
    if elevated_in or elevated_hold:
        return "ELEVATED"
    low_in = vol_pct < p["vol_low_pct"]
    low_hold = prev == "LOW" and vol_pct < p["vol_low_pct"] + m
    if low_in or low_hold:
        return "LOW"
    return "NORMAL"


def breadth_state_of(breadth, params=None):
    p = params_with(params)
    if breadth is None:
        return None
    if breadth >= p["breadth_healthy"]:
        return "HEALTHY"
    if breadth < p["breadth_weak"]:
        return "WEAK"
    return "NARROW"


def decode(trend_state, vol_state):
    """The documented decode table (docs/regime/FINDINGS.md step 2)."""
    if vol_state == "CRISIS":
        return "CRISIS"
    if trend_state == "UP":
        return "VOLATILE_BULL" if vol_state == "ELEVATED" else "CALM_BULL"
    if trend_state == "NEUTRAL":
        return "STRESSED" if vol_state == "ELEVATED" else "CHOPPY"
    return "BEAR"


def legacy_label(composite):
    return LEGACY_MAP.get(composite, "NEUTRAL")


def allowed_strategies(composite, breadth_state, panic):
    """What the state permits to OPEN (nothing here closes anything)."""
    if composite == "CALM_BULL":
        out = ["momentum_long", "csp"]
        if breadth_state == "HEALTHY":
            out.append("dip_buy")
    elif composite == "VOLATILE_BULL":
        out = ["momentum_long", "csp_reduced"]
    elif composite == "CHOPPY":
        out = ["momentum_long", "csp"]
        if breadth_state == "HEALTHY":
            out.append("dip_buy")
    elif composite == "STRESSED":
        out = ["csp_reduced"]
    elif composite == "BEAR":
        out = ["csp_reduced"]
    else:
        out = []
    if panic and "momentum_long" in out:
        out.remove("momentum_long")
    return out


# ── transitions ──────────────────────────────────────────────────────────────
def run_states(signals, params=None):
    """Daily state machine over a signal series: majority-filtered composite and
    exposure level, downgrade-fast / upgrade-glide exposure, flips and dwell."""
    p = params_with(params)
    k = max(1, int(p["filter_days"]))
    comp_win = collections.deque(maxlen=k)
    lvl_win = collections.deque(maxlen=k)
    trend_win = collections.deque(maxlen=k)
    trend_f = None         # the majority-filtered PRIMARY trend label (the flip budget counts this)
    composite = None
    level = 1.0
    target = 1.0
    last_upgrade_i = -10 ** 9
    days_in = 0
    changes = []           # indices where the (filtered) composite changed
    trend_changes = []     # indices where the PRIMARY trend state changed (the flip budget)
    prev_trend = None
    prev_vs = None
    out = []
    for i, s in enumerate(signals):
        ts_raw = trend_state_of(s["f_slow"], p)
        trend_win.append(ts_raw)
        ts_ = ts_raw if trend_f is None else majority(list(trend_win), trend_f)
        trend_f = ts_
        vs = vol_state_of(s["vol_pct"], s["ts"], prev_vs, p)
        prev_vs = vs
        raw = decode(ts_raw, vs)
        comp_win.append(raw)
        new_comp = raw if composite is None else majority(list(comp_win), composite)
        if composite is None:
            composite, days_in = new_comp, 1
        elif new_comp != composite:
            composite, days_in = new_comp, 1
            changes.append(i)
        else:
            days_in += 1
        if prev_trend is not None and ts_ != prev_trend:
            trend_changes.append(i)
        prev_trend = ts_
        raw_level = quantize_quarter(0.5 * s["f_slow"] + 0.25 * s["f_fast"] + 0.25 * s["breadth_bin"])
        lvl_win.append(raw_level)
        agreed = majority(list(lvl_win), target) if i >= k - 1 else raw_level
        if agreed < level:                      # fast out: once the filter clears
            level, target = agreed, agreed
        elif agreed > level:                    # slow in: glide a quarter every `upgrade_every` sessions
            target = agreed
            if i - last_upgrade_i >= p["upgrade_every"]:
                level = min(target, level + p["upgrade_step"])
                last_upgrade_i = i
        else:
            target = agreed
        exposure_c = level * s["m_vol"]
        exposure_q = quantize_quarter(exposure_c)
        core = max(p["core_floor"], exposure_c)
        flips_12m = sum(1 for c in trend_changes if c > i - 252)
        composite_changes_12m = sum(1 for c in changes if c > i - 252)
        votes_agree = sum(1 for c in comp_win if c == composite) / len(comp_win)
        pending = None
        if raw != composite:
            pending = {"candidate": raw, "votes": sum(1 for c in comp_win if c == raw), "needed": k // 2 + 1, "window": len(comp_win)}
        out.append({
            "trend_state": ts_, "trend_state_raw": ts_raw, "fast_state": fast_state_of(s["f_fast"]), "vol_state": vs,
            "raw_composite": raw, "composite": composite,
            "level": level, "level_target": target, "exposure_continuous": exposure_c,
            "exposure_multiplier": exposure_q, "core_multiplier": core, "days_in_regime": days_in,
            "flips_12m": flips_12m, "composite_changes_12m": composite_changes_12m,
            "votes_agree": votes_agree, "pending_transition": pending,
        })
    return out


# ── state object ─────────────────────────────────────────────────────────────
@dataclass
class RegimeState:
    as_of: str
    gating_index: str
    trend_state: str
    fast_state: str
    vol_state: str
    breadth_state: str | None
    breadth_universe: float | None
    composite_regime: str
    legacy_label: str
    exposure_multiplier: float
    exposure_continuous: float
    core_multiplier: float
    vol_dial: float
    allowed_strategies: list
    confidence: float
    days_in_regime: int
    pending_transition: dict | None
    flips_12m: int
    composite_changes_12m: int
    alarm: bool
    panic_state: bool
    stress_flags: dict
    ebp: dict | None
    inputs: dict
    data_quality: dict
    params_version: str = PARAMS_VERSION
    computed_at: str = ""
    stale: bool = False
    engine: str = "regime_engine"


def state_from_series(dates, signals, states, gating="SPY", sources=None, ebp=None,
                      breadth_universe=None, params=None, quality_factor=1.0):
    """The RegimeState for the last day of a computed series."""
    p = params_with(params)
    i = len(dates) - 1
    s, r = signals[i], states[i]
    breadth_state = breadth_state_of(s["breadth"], p)
    comp = r["composite"]
    stress = {
        "vix": s["vix"], "vix_vix3m": s["ts"], "vix_vix3m_inverted": bool(s["ts"] is not None and s["ts"] > 1.0),
        "vrp_points": s["vrp"], "vrp_negative": bool(s["vrp"] is not None and s["vrp"] < 0),
        "vix_pct_1y": s["vix_pct_1y"], "vvix": (sources or {}).get("vvix_latest"),
        "realized_vol_21d_pct": s["rv"], "ewma_vol_pct": (s["sigma"] * 100.0) if s["sigma"] else None,
        "vol_pct_own_history": s["vol_pct"], "sigma_target_pct": (s["sigma_target"] * 100.0) if s["sigma_target"] else None,
    }
    conf = round(min(1.0, max(0.0, r["votes_agree"] * quality_factor)), 3)
    return RegimeState(
        as_of=dates[i], gating_index=gating,
        trend_state=r["trend_state"], fast_state=r["fast_state"], vol_state=r["vol_state"], breadth_state=breadth_state,
        breadth_universe=breadth_universe, composite_regime=comp, legacy_label=legacy_label(comp),
        exposure_multiplier=r["exposure_multiplier"], exposure_continuous=round(r["exposure_continuous"], 4),
        core_multiplier=round(r["core_multiplier"], 4), vol_dial=round(s["m_vol"], 4),
        allowed_strategies=allowed_strategies(comp, breadth_state, s["panic"]),
        confidence=conf, days_in_regime=r["days_in_regime"], pending_transition=r["pending_transition"],
        flips_12m=r["flips_12m"], composite_changes_12m=r["composite_changes_12m"],
        alarm=bool(r["flips_12m"] > p["flip_alarm_12m"]), panic_state=bool(s["panic"]),
        stress_flags=stress, ebp=ebp,
        inputs={"f_slow": s["f_slow"], "f_fast": s["f_fast"], "trend_score": round(s["trend_score"], 4),
                "above_200d": s["above_200d"], "gating_dd52": (round(s["dd52"], 4) if s["dd52"] is not None else None),
                "breadth": s["breadth"], "breadth_bin": s["breadth_bin"], "mom24m": s["mom24m"],
                "var126_pct": s["var126_pct"], "level": r["level"], "level_target": r["level_target"],
                "raw_composite": r["raw_composite"], "sessions": len(dates), "first_date": dates[0],
                "sources": sources or {}},
        data_quality={"quality_factor": quality_factor, "votes_agree": round(r["votes_agree"], 3)},
        computed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def compute_state(dates, gating_closes, basket, vix=None, vix3m=None, gating="SPY", sources=None,
                  ebp=None, breadth_universe=None, params=None, quality_factor=1.0):
    """Pure: inputs in, RegimeState out (no I/O)."""
    signals = signal_series(dates, gating_closes, basket, vix, vix3m, params)
    states = run_states(signals, params)
    return state_from_series(dates, signals, states, gating, sources, ebp, breadth_universe, params, quality_factor)


def exposure_paths(dates, gating_closes, basket, vix=None, vix3m=None, params=None):
    """For scripts/regime_backtest.py: (continuous, quantized, core, composites) aligned to dates."""
    signals = signal_series(dates, gating_closes, basket, vix, vix3m, params)
    states = run_states(signals, params)
    return ([r["exposure_continuous"] for r in states], [r["exposure_multiplier"] for r in states],
            [r["core_multiplier"] for r in states], [r["composite"] for r in states])


# ── inputs (I/O) ─────────────────────────────────────────────────────────────
def _cache_dir():
    try:
        from storage_config import CACHE_DIR
        os.makedirs(CACHE_DIR, exist_ok=True)
        return CACHE_DIR
    except Exception as e:
        log.debug(f"storage_config CACHE_DIR unavailable ({e}); using /tmp")
        return "/tmp"


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        log.debug(f"read {path}: {e}")
        return None


def _write_json(path, payload):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, default=str)
    os.replace(tmp, path)


def _yahoo_closes(symbol, days, timeout=20):
    """{date: close} from the Yahoo chart endpoint (the same path backtest_v2 falls back to)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.request.quote(symbol)}"
           f"?period1={int(start.timestamp())}&period2={int(end.timestamp())}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    res = raw["chart"]["result"][0]
    stamps = res["timestamp"]
    closes = res["indicators"]["quote"][0]["close"]
    out = {}
    for t, c in zip(stamps, closes):
        if c is None:
            continue
        out[datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")] = float(c)
    return out


def fetch_closes(symbols, days=3800, fetch_alpaca=None, fetch_yahoo=None):
    """{symbol: {date: close}} plus {symbol: source}. Alpaca batch first (equities),
    Yahoo for anything missing and for the index family. Pure w.r.t. the injected fetchers."""
    sources = {}
    out = {}
    equities = [s for s in symbols if not s.startswith("^")]
    if fetch_alpaca is None:
        def fetch_alpaca(syms, d):
            from etf_data_sources import fetch_bars_alpaca
            return fetch_bars_alpaca(syms, d)
    if fetch_yahoo is None:
        fetch_yahoo = _yahoo_closes
    got = {}
    if equities:
        try:
            got = fetch_alpaca(equities, days) or {}
        except Exception as e:
            log.warning(f"[REGIME] alpaca bars failed: {e}")
            got = {}
    for sym in equities:
        rows = got.get(sym)
        if rows:
            out[sym] = {d: float(c) for d, c in rows}
            sources[sym] = "alpaca"
    for sym in symbols:
        if sym in out:
            continue
        try:
            out[sym] = fetch_yahoo(sym, days)
            sources[sym] = "yahoo"
        except Exception as e:
            log.warning(f"[REGIME] yahoo closes failed for {sym}: {e}")
            sources[sym] = "missing"
    return out, sources


def fetch_ebp(timeout=10, fetch=None):
    """Latest Fed excess-bond-premium row ({date, ebp, est_prob}) or None. Cached 7 days."""
    cache = os.path.join(_cache_dir(), "regime_ebp.json")
    cached = _read_json(cache)
    if cached and time.time() - float(cached.get("fetched_at", 0)) < 7 * 86400:
        return cached.get("row")
    url = "https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv"
    try:
        if fetch is None:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            text = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")
        else:
            text = fetch(url)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        header = [h.strip().lower() for h in lines[0].split(",")]
        last = [x.strip() for x in lines[-1].split(",")]
        row = dict(zip(header, last))
        out = {"date": row.get("date"), "ebp": float(row["ebp"]) if row.get("ebp") else None,
               "est_prob": float(row["est_prob"]) if row.get("est_prob") else None, "source": "federalreserve.gov"}
        _write_json(cache, {"fetched_at": time.time(), "row": out})
        return out
    except Exception as e:
        log.warning(f"[REGIME] EBP fetch failed: {e}")
        return cached.get("row") if cached else None


def align(series_by_symbol, anchor):
    """Dates of the anchor series; every series forward-filled onto them (None before inception)."""
    dates = sorted(series_by_symbol[anchor])
    out = {}
    for sym, s in series_by_symbol.items():
        vals = []
        last = None
        for d in dates:
            v = s.get(d)
            if v is not None:
                last = v
            vals.append(last)
        out[sym] = vals
    return dates, out


def build_inputs(gating="SPY", days=3800, fetch_alpaca=None, fetch_yahoo=None, use_cache=True, ebp=True):
    """Everything compute_state needs, fetched once per day and cached.
    Returns dict(dates, gating, basket, vix, vix3m, sources, ebp, vvix_latest, cache_hit)."""
    symbols = [gating] + [s for s in BREADTH_BASKET if s != gating] + CREDIT_PAIR + INDEX_FAMILY
    cache = os.path.join(_cache_dir(), f"regime_bars_{gating}.json")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cached = _read_json(cache) if use_cache else None
    if cached and cached.get("day") == today and cached.get("closes", {}).get(gating):
        closes, sources, hit = cached["closes"], cached.get("sources", {}), True
    else:
        closes, sources = fetch_closes(symbols, days, fetch_alpaca, fetch_yahoo)
        hit = False
        if use_cache and closes.get(gating):
            try:
                _write_json(cache, {"day": today, "closes": closes, "sources": sources})
            except OSError as e:
                log.debug(f"[REGIME] bars cache write failed: {e}")
    if not closes.get(gating):
        raise RuntimeError(f"no closes for the gating index {gating}")
    dates, aligned = align(closes, gating)
    basket = {s: aligned[s] for s in BREADTH_BASKET if s in aligned and s != gating}
    vvix = closes.get("^VVIX") or {}
    return {
        "dates": dates, "gating": aligned[gating], "basket": basket,
        "vix": aligned.get("^VIX"), "vix3m": aligned.get("^VIX3M"),
        "sources": dict(sources, vvix_latest=(vvix[max(vvix)] if vvix else None)),
        "ebp": fetch_ebp() if ebp else None, "cache_hit": hit,
    }


def quality_factor_of(sources, dates, today=None):
    """1.0 when everything is fresh; lower when the index family is missing or the bars are stale."""
    q = 1.0
    if sources.get("^VIX") == "missing" or sources.get("^VIX3M") == "missing":
        q *= 0.7
    if dates:
        today = today or datetime.now(timezone.utc).date()
        last = datetime.strptime(dates[-1], "%Y-%m-%d").date()
        if (today - last).days > 5:
            q *= 0.5
    return q


# ── persistence ──────────────────────────────────────────────────────────────
def _paths():
    try:
        from storage_config import REGIME_STATE_PATH, REGIME_HISTORY_PATH
        return REGIME_STATE_PATH, REGIME_HISTORY_PATH
    except Exception:
        from storage_config import DATA_DIR
        return (os.path.join(DATA_DIR, "voltrade_regime_state.json"),
                os.path.join(DATA_DIR, "voltrade_regime_history.jsonl"))


def load_state(path=None):
    path = path or _paths()[0]
    return _read_json(path)


def save_state(state_dict, path=None, history_path=None):
    """Persist the latest state; append one history line per NEW as_of date. Returns {'saved', 'appended'}."""
    sp, hp = _paths()
    path = path or sp
    history_path = history_path or hp
    prev = _read_json(path)
    _write_json(path, state_dict)
    appended = False
    if not prev or prev.get("as_of") != state_dict.get("as_of") or prev.get("composite_regime") != state_dict.get("composite_regime"):
        with open(history_path, "a") as f:
            f.write(json.dumps(state_dict, default=str) + "\n")
        appended = True
    return {"saved": True, "appended": appended, "previous": prev}


def read_history(history_path=None, limit=400):
    hp = history_path or _paths()[1]
    try:
        with open(hp) as f:
            lines = f.readlines()
    except OSError:
        return []
    rows = []
    for ln in lines[-limit:]:
        try:
            rows.append(json.loads(ln))
        except ValueError:
            log.debug("[REGIME] skipping a malformed history line")
    return rows


def state_age_market_days(state, today=None):
    """Approximate market days between the state's as_of and today (weekdays only)."""
    if not state or not state.get("as_of"):
        return None
    today = today or datetime.now(timezone.utc).date()
    d = datetime.strptime(state["as_of"][:10], "%Y-%m-%d").date()
    n = 0
    while d < today:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


# ── consumer: stock selection (docs/regime/FINDINGS.md step 4) ───────────────
SELECTION_GUARDS = ("knife", "dip", "panic", "rs", "breadth")
SELECTION_PARAMS = {
    "knife_high_frac": 0.75,        # price below 75% of its 252-day high = momentum-crash fuel, not value (George & Hwang)
    "knife_low_window": 20,         # a close at a new 20-day low is a falling knife (practitioner guard, counterfactual-logged)
    "rs_block_pct": -10.0,          # 3-month return vs the gating index below -10 pp (placeholder, counterfactual-logged)
    "breadth_high_proximity": 0.03, # index within 3% of its 52-week high AND breadth WEAK -> no net new stock exposure
    "mr_weight": 0.20,              # deep_score's mean-reversion weight (bot_engine combined score)
}
_STATE_CACHE = {"at": 0.0, "state": None}


def selection_gates(env=None) -> set:
    """VOLTRADE_REGIME_SELECTION: comma list of guards promoted from SHADOW (flags on the
    shadow record) to GATE (the candidate is blocked / the score changes); 'all' = every
    guard. Default: none — every guard only labels (RULE REVIEW: no rule ships without
    counterfactual data)."""
    env = os.environ if env is None else env
    raw = str(env.get("VOLTRADE_REGIME_SELECTION", "")).strip().lower()
    if not raw or raw in ("0", "off", "false", "none"):
        return set()
    if raw in ("1", "all", "true", "on"):
        return set(SELECTION_GUARDS)
    return {g.strip() for g in raw.split(",") if g.strip() in SELECTION_GUARDS}


def cached_state(max_age_s=60.0, loader=None, now=None):
    """The persisted state, re-read at most every `max_age_s` (deep_score runs per candidate)."""
    t = time.time() if now is None else now
    if _STATE_CACHE["state"] is not None and t - _STATE_CACHE["at"] < max_age_s:
        return _STATE_CACHE["state"]
    st = (loader or load_state)()
    _STATE_CACHE["at"], _STATE_CACHE["state"] = t, st
    return st


def selection_verdict(state, candidate, params=None) -> dict:
    """Pure: what the regime guards say about ONE stock candidate.

    candidate: price, high_52w, closes (recent daily closes, oldest first),
    relative_strength_pct (vs the gating index), mean_reversion_score (deep_score's
    0-100 sub-score), dip_bonus (points the contrarian 'Buy the Dip' bonus adds).
    Returns flags (0/1 per guard, persisted on the shadow record), reasons, the
    guards that WOULD block the entry, and the score points the dip
    neutralization would remove (mr_delta) — so the gate's prevention-P&L can be
    read before it is ever turned on."""
    p = dict(SELECTION_PARAMS)
    p.update(params or {})
    st = state if isinstance(state, dict) else {}
    inputs = st.get("inputs") if isinstance(st.get("inputs"), dict) else {}
    flags = {g: 0 for g in SELECTION_GUARDS}
    reasons = {}
    price = float(candidate.get("price") or 0.0)
    hi = candidate.get("high_52w")
    closes = [float(x) for x in (candidate.get("closes") or []) if x is not None]
    if price > 0 and hi and float(hi) > 0 and price < p["knife_high_frac"] * float(hi):
        flags["knife"] = 1
        reasons["knife"] = f"price {price:.2f} below {p['knife_high_frac']:.0%} of its 52-week high {float(hi):.2f}"
    if price > 0 and len(closes) >= p["knife_low_window"]:
        lo = min(closes[-p["knife_low_window"]:])
        if price <= lo:
            flags["knife"] = 1
            reasons["knife"] = (reasons.get("knife", "") + "; " if reasons.get("knife") else "") + f"new {p['knife_low_window']}-day low ({lo:.2f})"
    allowed = set(st.get("allowed_strategies") or [])
    above = inputs.get("above_200d")
    dip_ok = ("dip_buy" in allowed) and (above is not False)
    mr = float(candidate.get("mean_reversion_score") if candidate.get("mean_reversion_score") is not None else 50.0)
    bonus = float(candidate.get("dip_bonus") or 0.0)
    is_dip = mr > 50.0 or bonus > 0.0
    if st and is_dip and not dip_ok:
        flags["dip"] = 1
        reasons["dip"] = (f"dip-buying not allowed: regime {st.get('composite_regime')}, breadth {st.get('breadth_state')}, "
                          f"index above 200-day {above}")
    if st.get("panic_state"):
        flags["panic"] = 1
        reasons["panic"] = "panic state (24-month return < 0 and top-quintile variance): no momentum longs"
    rs = candidate.get("relative_strength_pct")
    if rs is not None and float(rs) < p["rs_block_pct"]:
        flags["rs"] = 1
        reasons["rs"] = f"relative strength {float(rs):.1f}% vs the index, below {p['rs_block_pct']:.0f}%"
    dd52 = inputs.get("gating_dd52")
    if st.get("breadth_state") == "WEAK" and dd52 is not None and float(dd52) >= -p["breadth_high_proximity"]:
        flags["breadth"] = 1
        reasons["breadth"] = f"breadth WEAK with the index {float(dd52) * 100:.1f}% from its 52-week high: no net new stock exposure"
    mr_delta = (max(0.0, mr - 50.0) * p["mr_weight"] + bonus) if flags["dip"] else 0.0
    return {
        "flags": flags, "reasons": reasons, "neutralize_mr": bool(flags["dip"]), "mr_delta": round(mr_delta, 2),
        "would_block": [g for g in ("knife", "panic", "rs", "breadth") if flags[g]],
        "composite": st.get("composite_regime"), "as_of": st.get("as_of"),
    }


# ── entry points ─────────────────────────────────────────────────────────────
def update_state(gating="SPY", force=False, params=None, build=None):
    """Compute (if the bar date advanced or `force`), persist, and return the state dict.
    Never raises: returns {'error': ...} with the last persisted state when it fails."""
    prev = load_state()
    try:
        inputs = (build or build_inputs)(gating=gating)
        last_date = inputs["dates"][-1]
        if (prev and not force and prev.get("as_of") == last_date and prev.get("gating_index") == gating
                and prev.get("params_version") == PARAMS_VERSION):
            prev["cache_hit"] = True
            return prev
        qf = quality_factor_of(inputs["sources"], inputs["dates"])
        state = compute_state(inputs["dates"], inputs["gating"], inputs["basket"], inputs["vix"], inputs["vix3m"],
                              gating=gating, sources=inputs["sources"], ebp=inputs.get("ebp"), params=params,
                              quality_factor=qf)
        d = asdict(state)
        d["transition"] = None
        if prev and prev.get("composite_regime") and prev.get("composite_regime") != d["composite_regime"]:
            d["transition"] = {"from": prev.get("composite_regime"), "to": d["composite_regime"],
                               "from_as_of": prev.get("as_of"), "to_as_of": d["as_of"]}
        saved = save_state(d)
        d["persisted"] = {"appended": saved["appended"]}
        return d
    except Exception as e:
        log.warning(f"[REGIME] update_state failed: {e}")
        out = dict(prev or {})
        out["error"] = str(e)[:300]
        out["stale"] = True
        return out


def regime_state_rpc(gating="SPY", force=False, compute=True):
    """Daemon method `regime_state` (voltrade_daemon.py). Never raises."""
    try:
        enabled = engine_allowed()
        if not enabled:
            return {"enabled": False, "state": load_state(), "note": "VOLTRADE_REGIME_ENGINE is off; serving the last persisted state if any"}
        if not compute:
            return {"enabled": True, "state": load_state()}
        return {"enabled": True, "state": update_state(gating=gating, force=bool(force))}
    except Exception as e:
        return {"enabled": engine_allowed(), "state": None, "error": str(e)[:300]}


if __name__ == "__main__":
    import sys
    g = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    print(json.dumps(regime_state_rpc(gating=g, force=True), indent=1, default=str))
