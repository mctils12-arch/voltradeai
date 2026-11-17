"""
gate_counterfactuals_probe.py — READ-ONLY report: what the tier engine's gates
blocked or cut, and what those dollars would have made.

WHY ([RULE-REVIEW] 2026-11-12). CLAUDE.md RULE REVIEW: a rule's threshold may
move only on its prevention P&L (what it stopped, and what that would have
returned). The tier engine's gates are the exposure cap (the master kill at
invested >= MAX_TOTAL_EXPOSURE and the allocator's scale-down under it), the
daily loss limit, the portfolio drawdown kill and the cash reserve.
tiered_strategy logs every tier action one of them drops or scales as a shadow
record (decision "rejected_masterkill" or "rejected_cash_reserve") with a
`gate` block: the tier, dropped or scaled, the dollars cut, and the share of
that cut no other gate would have stopped. The nightly backfill labels those
records at +5 / +10 / +20 trading days. This module reads them back per gate
and per tier.

WHAT IT REPORTS, per gate and per (tier, dropped | scaled):
  rows        raw records (a Tier-1 kill record is written every scan)
  name_days   one per (ticker, ET day)
  per horizon the outcome distribution over the first labeled record of each
              name-day, the SAME-PERIOD base rates (the scanner's `taken`
              candidates and every other logged candidate, first seen per
              name-day, same ruler), and a DOLLAR estimate over
              non-overlapping positions (one position per name at a time: a
              name blocked again before its earlier label's exit date is the
              same would-be trade, not a second one)
  by_regime   the same counts and dollars split by the system_config regime
              whose cap and reserve applied (each regime has its own)

EVERY DOLLAR FIGURE IS AN ESTIMATE. The outcome is the shadow labeler's long
stock price path (+2% target / -4% stop / N-day timeout). For a cash-secured
put it is a directional proxy, never option P&L, and the dollars are the
collateral budget the tier asked for, an upper bound on what one contract
count would have used. prevention_pnl_dollars = -(what the cut dollars would
have made): positive when the gate stopped a loss.

It writes nothing and makes no network call. It imports shadow_portfolio only
(no bot_engine: that import is the ~300 MB cold start; no tiered_strategy).
Served by voltrade_daemon.py as the `gate_counterfactuals` RPC method, with
the python3 -c fallback in server/bot.ts's `gate_counterfactuals` diag probe.
"""
from __future__ import annotations

import json
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import shadow_portfolio as sp

MIN_N = 5   # the shadow summaries' floor (shadow_portfolio._win_rate_by_masterkill_reason): below it only counts
GATE_ORDER = ("exposure_ceiling", "daily_loss", "drawdown", "cash_reserve", "unknown")
GATE_TEXT = {
    "exposure_ceiling": "the regime's MAX_TOTAL_EXPOSURE (system_config, less the regime risk budget): the master "
                        "kill at invested >= cap, and the allocator's proportional scale-down under the same cap",
    "daily_loss": "the tier engine's daily loss limit (tiered_strategy.DAILY_LOSS_LIMIT_PCT): no new tier entries that day",
    "drawdown": "the tier engine's portfolio drawdown kill (tiered_strategy.PORTFOLIO_DRAWDOWN_KILL)",
    "cash_reserve": "the regime's cash reserve (position_mgmt), an account-wide gate. Tier 1/2/3: the allocator's "
                    "drop or scale-down when the reserve's room is the smaller one. Tier `unrecorded`: a tier CSP "
                    "refused before the 2026-11-12 change or by bot.ts at dispatch (no tier or size). Tier `scanner`: "
                    "the scanner's own stock entries the reserve blocked; they are not tier actions",
    "unknown": "rejected_masterkill records whose reason text matches no master_kill_switch message",
}
DOLLAR_BASIS = ("ESTIMATE. cut dollars x the stock price-path label's return (long the underlying, +2% target / "
                "-4% stop / N-day timeout). Not option P&L: for SELL_CSP rows the dollars are the collateral "
                "budget and the stock return is a directional proxy.")
LIMITS = [
    "Outcomes are shadow_portfolio's path labels of the UNDERLYING stock, long. A blocked cash-secured put is "
    "bullish, so the sign is right; the size of its P&L is not (premium, delta and assignment are not modelled).",
    "Records from before the 2026-11-12 change carry no tier and no size. Master-kill ones are Tier-1 CSPs a kill "
    "dropped (the only writer then), so they count under tier 1 / dropped; cash-reserve ones count under tier "
    "unrecorded (a tier CSP) or scanner (the scanner's own stock entry). All are left out of the dollar sums; "
    "position_mean_return_pct still covers them.",
    "Under an exposure_ceiling kill Tier 2 and Tier 3 check the same cap themselves, so they would have proposed "
    "nothing and no Tier-2/3 row can exist for that gate. tier2_status / tier3_status show what each would have done.",
    "Tier 3 proposes nothing while the floor is the volatility-managed SPY/SSO floor (FLOOR_MODE vol_managed_spy). "
    "Tier 4 (the hedge, disabled) is not shadowed: a long-stock label has the wrong sign for a bought put.",
    "A master kill's shadow sees only the first three names Tier 1 proposes, on every scan. Nothing is sold, so it "
    "keeps proposing the same names: name_days counts a persistent block once per day, and the dollar figures "
    "count one position per name at a time.",
    "net_* figures keep only the share of a cut that no other gate would have stopped at that moment: the other "
    "master conditions, the exposure cap's room and the cash reserve's room. Working orders are not known to the "
    "tier engine; bot.ts applies the cash reserve again at dispatch.",
    "A row needs an entry price from the layer-2 cache or a held position. A name without one is not recorded.",
    "The engine-wide kill switch (risk_kill_switch) runs before the tier engine and skips it: while it is latched "
    "no tier action is computed and nothing is recorded here. It has its own drawdown kill, checked first; while "
    "the two thresholds are equal the tier engine's own drawdown gate never gets to fire, so that gate is normally empty.",
    "The base rate `taken` is the scanner's candidates that cleared the score threshold, first seen per name-day, "
    "under the same labels. A tier CSP that was actually sold leaves no shadow record, so this is the nearest "
    "same-ruler population, not the tier engine's own fills.",
]


def _labeled(rec: dict, key: str) -> Optional[dict]:
    """rec's outcome at horizon `key` when it carries a real win/loss label and a return."""
    out = (rec.get("outcomes") or {}).get(key)
    if isinstance(out, dict) and out.get("label") in (0, 1) and out.get("return_pct") is not None:
        return out
    return None


def _unlabelable(rec: dict, key: str) -> bool:
    out = (rec.get("outcomes") or {}).get(key)
    return isinstance(out, dict) and out.get("label") == -1


def tier_and_action(rec: dict) -> Tuple[str, str]:
    """(tier, action) of a gate record. One without a `gate` block:
      - master kill: a Tier-1 CSP a kill dropped (before 2026-11-12
        log_masterkill_csp_shadow was the only writer): tier "1";
      - cash reserve with model features: the SCANNER's stock candidate, whose
        own record bot_engine / bot.ts relabelled when the reserve blocked the
        entry. Not a tier action: tier "scanner";
      - cash reserve without features: a tier CSP the allocator dropped before
        2026-11-12 or bot.ts's dispatcher refused: tier "unrecorded"."""
    g = rec.get("gate")
    if isinstance(g, dict) and g.get("tier") is not None and g.get("action"):
        return str(g["tier"]), str(g["action"])
    if rec.get("decision") == "rejected_masterkill":
        return "1", "dropped"
    return ("scanner" if rec.get("features") else "unrecorded"), "dropped"


_REASON_REGIME = re.compile(r"\(regime=([A-Za-z_]+)\)")


def regime_of(rec: dict) -> str:
    """The system_config regime whose thresholds applied to a gate record: the
    block's `regime`; else the "(regime=X)" a master-kill reason carries (the
    cap's own regime: each regime has its own cap); else, for a cash-reserve
    record, its regime_label (the tier action's caps regime). "unrecorded"
    otherwise: a pre-change daily-loss record's regime_label is the tier
    context's label, another taxonomy."""
    g = rec.get("gate")
    if isinstance(g, dict) and g.get("regime"):
        return str(g["regime"])
    m = _REASON_REGIME.search(str(rec.get("decision_reason") or ""))
    if m:
        return m.group(1)
    label = str(rec.get("regime_label") or "")
    if rec.get("decision") == "rejected_cash_reserve" and label and label != "unknown":
        return label
    return "unrecorded"


def _first_labeled_per_name_day(rows: List[Tuple[dict, Optional[str]]], key: str) -> Dict[Tuple[str, str], dict]:
    """{(ticker, ET day): the earliest record labeled at `key`}: the house
    first-seen rule (shadow_portfolio._first_seen_per_ticker_day) over rows
    whose ET day is already known."""
    first: Dict[Tuple[str, str], dict] = {}
    for rec, day in rows:
        if day is None or _labeled(rec, key) is None:
            continue
        k = (str(rec.get("ticker") or ""), day)
        cur = first.get(k)
        if cur is None or str(rec.get("timestamp", "")) < str(cur.get("timestamp", "")):
            first[k] = rec
    return first


def _dist(recs: List[dict], key: str, min_n: int) -> dict:
    """The outcome distribution of labeled records at one horizon."""
    outs = [_labeled(r, key) for r in recs]
    n = len(outs)
    if n < min_n:
        return {"labeled": n}
    rets = sorted(float(o["return_pct"]) for o in outs)
    wins = sum(1 for o in outs if o["label"] == 1)

    def pct(p: float) -> float:
        return rets[min(n - 1, int(round(p * (n - 1))))]

    return {
        "labeled": n, "wins": wins, "losses": n - wins, "win_rate_pct": round(100.0 * wins / n, 1),
        "return_pct": {"mean": round(sum(rets) / n, 3), "median": round(statistics.median(rets), 3),
                       "p10": round(pct(0.10), 3), "p90": round(pct(0.90), 3),
                       "min": round(rets[0], 3), "max": round(rets[-1], 3)},
        "exits": dict(Counter(str(o.get("exit_reason") or "unrecorded") for o in outs)),
    }


def _regime_cell(recs: List[dict], positions: List[dict], key: str, min_n: int) -> dict:
    """One regime's slice of a group at one horizon: the labeled name-days'
    win rate and mean, and the dollar figures of the positions it holds."""
    outs = [_labeled(r, key) for r in recs]
    cell: dict = {"labeled": len(outs)}
    if len(outs) >= min_n:
        cell["win_rate_pct"] = round(100.0 * sum(1 for o in outs if o["label"] == 1) / len(outs), 1)
        cell["mean_return_pct"] = round(sum(float(o["return_pct"]) for o in outs) / len(outs), 3)
    d = _dollars(positions, key)
    for k in ("positions", "position_mean_return_pct", "sized_positions", "cut_dollars", "prevention_pnl_dollars",
              "net_prevention_pnl_dollars"):
        cell[k] = d[k]
    return cell


def _base(pairs: List[Tuple[float, int]], min_n: int) -> dict:
    """A base-rate cell from (return_pct, label) pairs."""
    n = len(pairs)
    if n < min_n:
        return {"n": n}
    rets = [p[0] for p in pairs]
    return {"n": n, "win_rate_pct": round(100.0 * sum(p[1] for p in pairs) / n, 1),
            "mean_return_pct": round(sum(rets) / n, 3), "median_return_pct": round(statistics.median(rets), 3)}


def _positions(first: Dict[Tuple[str, str], dict], key: str, horizon: int) -> List[dict]:
    """Non-overlapping would-be positions out of labeled name-days: per name,
    in day order, a record is a new position only when its ET day is after the
    previous position's exit date (the label's exit_date, else the end of the
    window the backfill read: shadow_portfolio._shadow_label_end_date)."""
    by_name: Dict[str, List[Tuple[str, dict]]] = defaultdict(list)
    for (ticker, day), rec in first.items():
        by_name[ticker].append((day, rec))
    taken: List[dict] = []
    for ticker in sorted(by_name):
        busy_until = None
        for day, rec in sorted(by_name[ticker], key=lambda x: x[0]):
            if busy_until is not None and day <= busy_until:
                continue
            taken.append(rec)
            busy_until = sp._shadow_label_end_date(rec.get("timestamp", ""), _labeled(rec, key), horizon) or day
    return taken


def _no_dollars() -> dict:
    return {"estimate": True, "positions": 0, "sized_positions": 0, "unsized_positions": 0,
            "cut_dollars": 0.0, "would_have_made_dollars": 0.0, "prevention_pnl_dollars": 0.0,
            "net_cut_dollars": 0.0, "net_would_have_made_dollars": 0.0, "net_prevention_pnl_dollars": 0.0}


def _dollars(positions: List[dict], key: str) -> dict:
    """The dollar ESTIMATE over would-be positions that recorded a size, and
    the mean return of ALL the positions (the size-free reading: x10 is the
    dollars per $1,000 cut, the only one a record without a size can give)."""
    out = _no_dollars()
    out["positions"] = len(positions)
    rets = [float(_labeled(rec, key)["return_pct"]) for rec in positions]
    out["position_mean_return_pct"] = round(sum(rets) / len(rets), 3) if rets else None
    for rec in positions:
        g = rec.get("gate")
        cut = g.get("cut_dollars") if isinstance(g, dict) else None
        if not isinstance(cut, (int, float)) or isinstance(cut, bool):
            continue
        ret = float(_labeled(rec, key)["return_pct"]) / 100.0
        net = g.get("net_dollars")
        net = float(net) if isinstance(net, (int, float)) and not isinstance(net, bool) else 0.0
        out["sized_positions"] += 1
        out["cut_dollars"] += float(cut)
        out["would_have_made_dollars"] += float(cut) * ret
        out["net_cut_dollars"] += net
        out["net_would_have_made_dollars"] += net * ret
    out["unsized_positions"] = out["positions"] - out["sized_positions"]
    out["prevention_pnl_dollars"] = 0.0 - out["would_have_made_dollars"]          # 0.0 - x: never a "-0.0"
    out["net_prevention_pnl_dollars"] = 0.0 - out["net_would_have_made_dollars"]
    return {k: (round(v, 2) if isinstance(v, float) and k != "position_mean_return_pct" else v) for k, v in out.items()}


def _add_dollars(total: dict, part: dict) -> None:
    for k in total:
        if k != "estimate":
            total[k] = round(total[k] + part[k], 2) if isinstance(part[k], float) else total[k] + part[k]


def _base_index(rows: List[Tuple[dict, Optional[str]]], key: str) -> Dict[str, List[Tuple[float, int]]]:
    """{ET day: [(return_pct, label), ...]} over the first labeled record of each name-day."""
    out: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
    for (_ticker, day), rec in _first_labeled_per_name_day(rows, key).items():
        o = _labeled(rec, key)
        out[day].append((float(o["return_pct"]), int(o["label"])))
    return out


def _on_days(index: Dict[str, List[Tuple[float, int]]], days) -> List[Tuple[float, int]]:
    return [pair for d in days for pair in index.get(d, [])]


def build_report(records: List[dict], min_n: int = MIN_N) -> dict:
    """The report over shadow records. Pure: records are not modified."""
    horizons = [(f"+{h}d", h) for h in sp.FORWARD_HORIZONS_DAYS]
    rows = [(r, sp._xa_entry_date_et(r.get("timestamp", ""))) for r in records if isinstance(r, dict)]
    gate_rows: Dict[str, Dict[Tuple[str, str], List[Tuple[dict, Optional[str]]]]] = defaultdict(lambda: defaultdict(list))
    other_rows: List[Tuple[dict, Optional[str]]] = []
    for rec, day in rows:
        gate = sp.gate_reason(rec)
        if gate is None:
            other_rows.append((rec, day))
        else:
            gate_rows[gate][tier_and_action(rec)].append((rec, day))
    taken_rows = [(r, d) for r, d in other_rows if r.get("decision") == "taken"]
    base_taken = {key: _base_index(taken_rows, key) for key, _ in horizons}
    base_scanned = {key: _base_index(other_rows, key) for key, _ in horizons}

    gates: Dict[str, dict] = {}
    for gate in GATE_ORDER:
        if gate == "unknown" and gate not in gate_rows:
            continue
        groups = []
        names_days, all_rows = set(), 0
        totals = {key: _no_dollars() for key, _ in horizons}
        t2, t3 = Counter(), Counter()
        for (tier, action), grp in sorted(gate_rows.get(gate, {}).items()):
            name_days = {(str(r.get("ticker") or ""), d) for r, d in grp if d is not None}
            names_days |= name_days
            all_rows += len(grp)
            days = sorted(d for _, d in name_days)
            first_any: Dict[Tuple[str, str], dict] = {}
            for r, d in grp:   # the first record of each name-day, labeled or not: its block is the day's reading
                k = (str(r.get("ticker") or ""), d)
                if d is not None and (k not in first_any or str(r.get("timestamp", "")) < str(first_any[k].get("timestamp", ""))):
                    first_any[k] = r
            blocks = [r["gate"] for r in first_any.values() if isinstance(r.get("gate"), dict)]
            co = Counter()
            for g in blocks:
                also = g.get("also") or []
                co.update(also if also else ["no_other_gate"])
                if tier == "1" and action == "dropped":
                    if g.get("t2"):
                        t2[g["t2"]] += 1
                    if g.get("t3"):
                        t3[g["t3"]] += 1
            regimes = Counter(regime_of(r) for r in first_any.values())
            by_regime: Dict[str, dict] = {reg: {"name_days": n} for reg, n in sorted(regimes.items())}
            group = {
                "tier": tier, "action": action, "rows": len(grp), "name_days": len(name_days),
                "names": len({t for t, _ in name_days}), "sized_name_days": len(blocks),
                "first_day": days[0] if days else None, "last_day": days[-1] if days else None,
                "other_gates": dict(co), "by_regime": by_regime,
            }
            for key, h in horizons:
                first = _first_labeled_per_name_day(grp, key)
                cell = _dist(list(first.values()), key, min_n)
                dead = {(str(r.get("ticker") or ""), d) for r, d in grp if _unlabelable(r, key)} - set(first)
                cell["unlabelable"] = len(dead)
                cell["pending"] = len(name_days) - len(first) - len(dead)
                ldays = sorted({d for _, d in first})
                span = [d for d in base_scanned[key] if ldays and ldays[0] <= d <= ldays[-1]]
                taken = _base(_on_days(base_taken[key], ldays), min_n)
                cell["base_taken_same_days"] = taken
                cell["base_taken_same_span"] = _base(_on_days(base_taken[key], span), min_n)
                cell["base_scanned_same_days"] = _base(_on_days(base_scanned[key], ldays), min_n)
                if "return_pct" in cell and "mean_return_pct" in taken:
                    cell["mean_minus_taken_same_days_pct"] = round(cell["return_pct"]["mean"] - taken["mean_return_pct"], 3)
                positions = _positions(first, key, h)
                cell["dollars"] = _dollars(positions, key)
                _add_dollars(totals[key], cell["dollars"])
                group[key] = cell      # flat on purpose: server/diag.ts sanitizeDiag stops at depth 8
                for reg in by_regime:  # REASONING STANDARD #2: never the pooled result alone
                    by_regime[reg][key] = _regime_cell([r for r in first.values() if regime_of(r) == reg],
                                                       [r for r in positions if regime_of(r) == reg], key, min_n)
            groups.append(group)
        gates[gate] = {
            "what": GATE_TEXT[gate], "rows": all_rows, "name_days": len(names_days),
            "names": len({t for t, _ in names_days}),
            "tier2_status": dict(t2), "tier3_status": dict(t3),
            "dollars_all_tiers": totals, "groups": groups,
        }

    all_days = sorted(d for _, d in rows if d is not None)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "log": {"records": len(rows), "first_day": all_days[0] if all_days else None,
                "last_day": all_days[-1] if all_days else None,
                "gate_rows": sum(g["rows"] for g in gates.values()),
                "taken_rows": len(taken_rows)},
        "labeling": {"method": "path_dependent_v2", "win_pct": sp.WIN_THRESHOLD_PCT, "stop_pct": sp.LOSS_THRESHOLD_PCT,
                     "horizons_trading_days": list(sp.FORWARD_HORIZONS_DAYS),
                     "note": "return_pct is entry -> exit under the labeler's target / stop / timeout rules, "
                             "long the underlying stock"},
        "min_n": min_n,
        "dollar_basis": DOLLAR_BASIS,
        "reading": {
            "name_days": "one per (ticker, ET day); a horizon's statistics use the first record of that day "
                         "that carries its label",
            "positions": "non-overlapping name-days: a name counts again only after its earlier label's exit date",
            "cut_dollars": "summed over those positions: a name re-entered after an exit counts again, so this is "
                           "dollars put to work over the period, not capital held at one time",
            "position_mean_return_pct": "the mean label return of those positions, sized or not; x10 = dollars per $1,000 cut",
            "prevention_pnl_dollars": "-(what the cut dollars would have made): positive = the gate stopped a loss",
            "net": "only the share of the cut no other gate would have stopped at that moment",
            "dollars_all_tiers": "the sum of the gate's groups; a name scaled in one scan and dropped in a later "
                                 "one the same day is in both",
            "tier2_status": "what Tier 2 (and tier3_status: Tier 3) would have done at each Tier-1 name-day of a "
                            "master kill: levered / proposed, or the first of its own gates that stops it",
            "other_gates": "per name-day, the other gates that would also have cut it (no_other_gate: none)",
            "by_regime": "the group split by the system_config regime whose thresholds applied (each regime has its "
                         "own exposure cap and cash reserve); its positions and dollars add up to the group's",
            "base": "same labels over the scanner's `taken` candidates (and every other logged candidate), "
                    "first seen per name-day, on the same ET days / over the same span as this cell's labeled name-days",
        },
        "gates": gates,
        "limits": LIMITS,
    }


def gate_counterfactuals_report() -> dict:
    """Production entry point (daemon method "gate_counterfactuals"). Reads the
    shadow log read-only; never raises: a failure is reported as {"error"}."""
    t0 = time.time()
    try:
        out = build_report(sp._load_shadow_log(readonly=True))
    except Exception as e:  # noqa: BLE001 — a probe that crashes hides the state it was built to show
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


if __name__ == "__main__":  # operator one-liner: python3 gate_counterfactuals_probe.py
    print(json.dumps(gate_counterfactuals_report(), indent=2, default=str))
