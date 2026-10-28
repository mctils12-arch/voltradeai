"""Parity tests — audit §6.8 N6 ETF max-hold exit, [REPAIR] 2026-10-27.

instrument_selector.ETF_MAX_HOLD_DAYS ("never hold leveraged ETF longer than
5 days") gated only the ENTRY choice; no exit enforced it. The live exit
engine (server/bot.ts checkPositionOnTick) now does, using a TS mirror of the
limit and of the substitute-ETF universe in server/etfSubstitution.ts. These
pin the mirror to its Python sources so the two cannot drift.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TS = open(os.path.join(HERE, "server", "etfSubstitution.ts")).read()


def test_max_hold_days_mirrors_instrument_selector():
    import instrument_selector
    m = re.search(r"export const ETF_MAX_HOLD_DAYS\s*=\s*(\d+)\s*;", TS)
    assert m, "ETF_MAX_HOLD_DAYS not exported from server/etfSubstitution.ts"
    assert int(m.group(1)) == instrument_selector.ETF_MAX_HOLD_DAYS


def test_substitute_etf_set_mirrors_analyze_leveraged_etfs():
    from analyze import LEVERAGED_ETFS
    m = re.search(r"export const SUBSTITUTE_LEVERAGED_ETFS[^=]*=\s*new Set\(\[([^\]]*)\]\)", TS)
    assert m, "SUBSTITUTE_LEVERAGED_ETFS not found in server/etfSubstitution.ts"
    ts_set = set(re.findall(r'"([A-Z]+)"', m.group(1)))
    py_set = {v for mp in LEVERAGED_ETFS.values() for k, v in mp.items()
              if k in ("bull", "bear") and v}
    assert ts_set == py_set


def test_exit_is_wired_into_the_live_exit_engine():
    bot = open(os.path.join(HERE, "server", "bot.ts")).read()
    start = bot.find("async function checkPositionOnTick(")
    end = bot.find("// ── FIRE EXIT ORDER (remaining position)", start)
    assert start > 0 and end > start
    section = bot[start:end]
    assert "etfMaxHoldExceeded(ticker, pos.entryDate" in section
    assert "if (!shouldStop && !shouldTP && !shouldTimeStop && !shouldEtfMaxHold) return;" in section
