"""live_exit_policy.py — a PURE mirror of the live stock exit engine.

[RULE-REVIEW] measurement code (CLAUDE.md MEASUREMENT INTEGRITY), 2026-10-16.

The only code that places live stock exits is server/bot.ts
checkPositionOnTick (levels fed by syncMonitoredPositions from
bot_engine.manage_positions' stop state). research/open_questions.md KNOWN
BROKEN #48: the configured exits (system_config ATR_STOP_MULTIPLIER/
TIME_STOP_DAYS block — read by nothing), the backtest's exits (backtest_v2
1.5xATR clamp 2-6%, 2R TP, regime time stop) and these live exits were three
different models. This module is the live one, written so any simulator can
drive it on daily bars.

PROVENANCE. The rule set is copied from the reviewed, fuzz-verified mirror
merged in PR #1355 (shadow_portfolio._XaPosition / _label_exit_aligned).
test_backtest_v2_exit_model.py asserts the two produce identical exits on
fuzzed paths. FOLLOW-UP: make shadow_portfolio import LivePosition and
walk_bar from this module and delete its duplicate (until then the fuzz test is the only
drift guard between the two copies).

NOT MODELLED (bias direction stated, MEASUREMENT INTEGRITY):
  * Sampled peaks: each sync overwrites highestPnl/highestPrice/phase with
    bot_engine's 45s-sampled values (bot.ts ~6631-6686), so live trails run
    from LOWER peaks than the daily high used here -> OPTIMISTIC on trails.
  * Phantom first scale-out (KNOWN BROKEN #47): Python can mark scale-out 1
    done without an order, with third = max(1, orig//3) (bot_engine.py
    ~2083) -> live may keep more size than modelled; direction mixed.
  * Daily-bar touches: stops/targets fill exactly at the level on a
    continuous O->L->H->C walk; live checks 1-minute closes past the level
    -> slightly OPTIMISTIC.
  * Exit orders are limit orders (getOrderParams); unfilled exits are not
    modelled -> OPTIMISTIC.
  * Intrabar exit cost uses bar k's liquidity_cost_pct (full-day volume);
    measured effect <= 0.001 Sharpe.

Two extensions over #1355's copy, both opt-in, both mirroring live behaviour
that a labeler could not see but a backtest can:
  * original_qty: integer share counts. bot.ts scales out
    Math.floor(originalQty / 3) shares and only while remainingQty > 1 and
    thirdQty >= 1 — so a 1- or 2-share position never scales out (and never
    arms breakeven). original_qty=None keeps #1355's fractional thirds.
  * refresh(atr_pct, regime_class): bot_engine.manage_positions rewrites
    current_stop_pct (= the CURRENT ATR%) and regime every cycle, and
    syncMonitoredPositions recomputes the phase-1 stop, the TP and the trail
    widths from them; initial_risk_pct (the R unit) is written once and kept.
    The labeler fixes both at entry; a backtest that knows the prior bar's
    ATR and regime can follow live more closely. Never called = #1355's
    behaviour exactly.

Every constant names the bot.ts / bot_engine.py expression it copies;
test_backtest_v2_exit_model.py's drift pin asserts those literals still exist,
so a live exit change fails CI instead of silently staling this mirror.
No imports from trading code: pure stdlib, no I/O.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

RISK_ATR_MULT = 2.0             # bot.ts `ps.initial_risk_pct || atrPct * 2.0`; bot_engine: atr_pct * 2.0 (UNclamped)
P1_STOP_ATR_MULT = 2.0          # bot.ts `Math.max(1.5, Math.min(atrPct * 2.0, 8.0))`
P1_STOP_MIN_PCT = 1.5           #   ... lower clamp
P1_STOP_MAX_PCT = 8.0           #   ... upper clamp
SCALE1_R = 1.0                  # bot.ts `const scaleOut1R = 1.0`
SCALE2_R = {"bullish": 3.0, "bearish": 1.5, "neutral": 2.0}          # bot.ts scaleOut2R
TRAIL4_ATR_MULT = {"bullish": 3.0, "bearish": 1.5, "neutral": 2.0}   # bot.ts trailAtrMult, from the highest PRICE
P2_TRAIL_MIN_PCT, P2_TRAIL_ATR_MULT = 1.5, 1.5   # bot.ts `Math.max(1.5, atr * 1.5)`, from peak P&L
P3_TRAIL_MIN_PCT, P3_TRAIL_ATR_MULT = 1.0, 1.0   # bot.ts `Math.max(1.0, atr * 1.0)`, from peak P&L
TP_MIN_PCT, TP_ATR_MULT, TP_MAX_PCT = 4.0, 3.0, 15.0  # bot.ts `Math.max(4.0, Math.min(atrPct * 3.0, 15.0))`
TIME_STOP_DAYS = 7              # bot.ts `daysHeld >= 7 && Math.abs(pnlPct) < 2.0` (CALENDAR days)
TIME_STOP_BAND_PCT = 2.0        #   ... the |pnl| band
TRANCHE = 1.0 / 3.0             # bot.ts `Math.floor(pos.originalQty / 3)` per scale-out
ATR_PERIOD = 14                 # bot_engine._get_atr: Wilder ATR(14), seeded with the mean of the first 14 TRs
ATR_LOOKBACK_DAYS = 60          # bot_engine._get_atr: lookback_days = max(60, period * 4)
ATR_LIMIT_BARS = ATR_PERIOD * 3  # bot_engine._get_atr: limit={period * 3}
ATR_DEFAULT_PCT = 2.0           # bot_engine.manage_positions: `... else 2.0` when ATR is unavailable

_TOL = 1e-9     # pct-point slack so a level's own float round-trip still counts as touched
_NUDGE = 1e-9   # relative step past a level crossed INSIDE a continuous leg

# Exit reasons in bot.ts vocabulary (stopType / exitType), for trade records.
REASON_NAMES = {
    "stop": "stop_loss",            # phase-1 stop
    "trail": "trailing_stop",       # phase 2-4 trail
    "breakeven": "breakeven_stop",  # bot.ts stopType 'trailing_stop', reason "WS BREAKEVEN STOP"
    "take_profit": "take_profit",
    "time_stop": "time_stop",
}


def regime_class(regime: Optional[str]) -> str:
    """bot.ts isBullish / isBearish on the 5-level regime label."""
    r = str(regime or "NEUTRAL")
    if r in ("BULL", "NEUTRAL_BULL"):
        return "bullish"
    if r in ("BEAR", "PANIC", "CAUTION"):
        return "bearish"
    return "neutral"


def wilder_atr(highs: List[float], lows: List[float], closes: List[float],
               period: int = ATR_PERIOD) -> Optional[float]:
    """bot_engine._get_atr's method on the given bars (oldest first): true
    ranges from bar 1 on, seeded with the mean of the first `period`, then
    Wilder-smoothed. None with fewer than period+1 bars or a non-positive ATR."""
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        h, lo, pc = float(highs[i]), float(lows[i]), float(closes[i - 1])
        trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr if atr > 0 else None


def time_eligible(entry_date: str, bar_date: str) -> bool:
    """bot.ts daysHeld = floor((now - entryDate) / 1 day) >= 7, entryDate the
    entry's calendar date: on a daily bar, calendar days between the dates."""
    d0 = datetime.strptime(entry_date[:10], "%Y-%m-%d").date()
    d1 = datetime.strptime(bar_date[:10], "%Y-%m-%d").date()
    return (d1 - d0).days >= TIME_STOP_DAYS


class LivePosition:
    """One long stock position under the live exit policy. tick() is a
    line-by-line mirror of bot.ts checkPositionOnTick (P&L in percent of the
    entry FILL price, 'R' = initial risk %)."""

    def __init__(self, entry: float, atr_pct: float, regime_class: str,
                 original_qty: Optional[int] = None):
        self.entry = entry
        self.risk = atr_pct * RISK_ATR_MULT          # written once (bot_engine initial_risk_pct)
        self.original_qty = original_qty
        self.remaining_qty = original_qty
        self.third_qty = (original_qty // 3) if original_qty is not None else None
        self._set_vol(atr_pct, regime_class)
        self.phase = 1
        self.peak_pnl = 0.0                         # bot.ts addPositionToMonitor: highestPnl: 0
        self.peak_price = entry
        self.scales = 0
        self.breakeven = False
        # (size, price): size is a fraction of the position when original_qty
        # is None (#1355's labeler mode), else a share count
        self.fills: List[Tuple[float, float]] = []
        self.exit_reason: Optional[str] = None
        self.max_px = entry
        self.min_px = entry

    def _set_vol(self, atr_pct: float, regime_cls: str) -> None:
        self.atr = atr_pct
        self.p1_stop = max(P1_STOP_MIN_PCT, min(atr_pct * P1_STOP_ATR_MULT, P1_STOP_MAX_PCT))
        self.tp = max(TP_MIN_PCT, min(atr_pct * TP_ATR_MULT, TP_MAX_PCT))
        self.regime_class = regime_cls
        self.scale2_r = SCALE2_R[regime_cls]
        self.trail4 = TRAIL4_ATR_MULT[regime_cls]

    def refresh(self, atr_pct: float, regime_cls: str) -> None:
        """A new manage cycle's ATR% and regime (see the module docstring).
        The R unit (self.risk) is NOT changed — live keeps initial_risk_pct."""
        self._set_vol(atr_pct, regime_cls)

    def _pnl(self, px: float) -> float:
        return (px - self.entry) / self.entry * 100.0

    def _can_scale(self) -> bool:
        if self.original_qty is None:
            return True
        # bot.ts: `pos.remainingQty > 1` and `thirdQty >= 1`
        return self.remaining_qty > 1 and self.third_qty >= 1

    def _scale(self, px: float) -> None:
        if self.original_qty is None:
            self.fills.append((TRANCHE, px))
        else:
            self.fills.append((self.third_qty, px))
            self.remaining_qty -= self.third_qty
        self.scales += 1

    def _close(self, px: float, reason: str) -> None:
        if self.original_qty is None:
            self.fills.append((1.0 - TRANCHE * self.scales, px))
        else:
            self.fills.append((self.remaining_qty, px))
            self.remaining_qty = 0
        self.exit_reason = reason

    def tick(self, px: float, time_ok: bool) -> Optional[str]:
        """'scale' (a tranche sold — the caller re-ticks at the same price,
        standing in for the next tick), an exit reason, or None."""
        self.max_px = max(self.max_px, px)
        self.min_px = min(self.min_px, px)
        pnl = self._pnl(px)
        self.peak_pnl = max(self.peak_pnl, pnl)
        self.peak_price = max(self.peak_price, px)
        # phases only move up, from the PEAK R multiple
        if self.peak_pnl >= 3.0 * self.risk - _TOL and self.phase < 4:
            self.phase = 4
        elif self.peak_pnl >= 2.0 * self.risk - _TOL and self.phase < 3:
            self.phase = 3
        elif self.peak_pnl >= 1.0 * self.risk - _TOL and self.phase < 2:
            self.phase = 2
        # scale-outs (bot.ts returns right after one: no full exit on that tick)
        if self.scales < 2 and self._can_scale():
            if self.scales == 0 and pnl >= SCALE1_R * self.risk - _TOL:
                self._scale(px)
                self.breakeven = True
                return "scale"
            if self.scales == 1 and pnl >= self.scale2_r * self.risk - _TOL:
                self._scale(px)
                return "scale"
        reason: Optional[str] = None
        if self.phase >= 2:
            if self.phase >= 4:
                dd_from_high = (self.peak_price - px) / self.peak_price * 100.0
                if dd_from_high >= self.atr * self.trail4 - _TOL:
                    reason = "trail"
            else:
                trail = (max(P3_TRAIL_MIN_PCT, self.atr * P3_TRAIL_ATR_MULT) if self.phase >= 3
                         else max(P2_TRAIL_MIN_PCT, self.atr * P2_TRAIL_ATR_MULT))
                if self.peak_pnl - pnl >= trail - _TOL:
                    reason = "trail"
            if reason is None and self.breakeven and pnl < -_TOL:
                reason = "breakeven"
        elif pnl <= -self.p1_stop + _TOL:
            reason = "stop"
        if reason is None and pnl >= self.tp - _TOL and self.phase < 3 and self.scales >= 2:
            reason = "take_profit"
        if reason is None and time_ok and abs(pnl) < TIME_STOP_BAND_PCT - _TOL:
            reason = "time_stop"
        if reason is not None:
            self._close(px, reason)
        return reason

    def settle(self, px: float, time_ok: bool) -> bool:
        """tick() at px, re-ticking after each scale-out. True once closed."""
        for _ in range(3):
            if self.tick(px, time_ok) != "scale":
                break
        return self.exit_reason is not None

    def _levels(self, time_ok: bool) -> List[float]:
        """Every price at which some tick() condition can flip, given the
        current state (on a falling leg the peak is fixed; on a rising leg the
        trail levels sit below the price and cannot fire)."""
        e = self.entry
        lv = [e, e * (1 + self.tp / 100.0), e * (1 - self.p1_stop / 100.0)]
        lv += [e * (1 + k * self.risk / 100.0) for k in (1.0, 2.0, 3.0, self.scale2_r)]
        if time_ok:
            lv += [e * (1 - TIME_STOP_BAND_PCT / 100.0), e * (1 + TIME_STOP_BAND_PCT / 100.0)]
        lv.append(e * (1 + (self.peak_pnl - max(P2_TRAIL_MIN_PCT, self.atr * P2_TRAIL_ATR_MULT)) / 100.0))
        lv.append(e * (1 + (self.peak_pnl - max(P3_TRAIL_MIN_PCT, self.atr * P3_TRAIL_ATR_MULT)) / 100.0))
        lv.append(self.peak_price * (1 - self.atr * self.trail4 / 100.0))
        return [x for x in lv if x > 0]

    def leg(self, a: float, b: float, time_ok: bool) -> bool:
        """Continuous move a -> b: tick just past every level crossed, in
        order, then at b. A level crossed inside the leg fills at the level
        (+/- a 1e-9 relative nudge). True once closed."""
        cur = a
        for _ in range(64):
            if b == cur:
                return False
            up = b > cur
            inside = [x for x in self._levels(time_ok) if (cur <= x < b if up else b < x <= cur)]
            if inside:
                nxt = min(inside) if up else max(inside)
                px = nxt * (1 + _NUDGE) if up else nxt * (1 - _NUDGE)
                if (up and px > b) or (not up and px < b):
                    px = b
            else:
                px = b
            if self.settle(px, time_ok):
                return True
            cur = px
        return self.settle(b, time_ok)


def walk_bar(pos: LivePosition, o: float, h: float, lo: float, c: float,
             time_ok: bool) -> bool:
    """One daily bar, walked OPEN -> LOW -> HIGH -> CLOSE (the adverse leg
    first for a long, so a bar touching both a stop and a profit level
    resolves STOP FIRST). The jump to the open is a gap: anything it skips
    past fills AT THE OPEN. True once the position is closed."""
    hi, low = max(o, h, lo, c), min(o, h, lo, c)
    return (pos.settle(o, time_ok) or pos.leg(o, low, time_ok)
            or pos.leg(low, hi, time_ok) or pos.leg(hi, c, time_ok))
