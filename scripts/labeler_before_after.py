#!/usr/bin/env python3
"""labeler_before_after.py — the two shadow rulers on IDENTICAL inputs.

MEASUREMENT INTEGRITY (CLAUDE.md) requires every ruler change to state what
the metric reported before vs. after on identical historical inputs. This
replays, case by case and over the same bars:

  OLD  shadow_portfolio._label_from_path   +2% target / -4% stop / N-bar
       timeout (N = 5, 10, 20), exactly as backfill_outcomes() calls it
  NEW  shadow_portfolio._exit_aligned_outcome   the live checkPositionOnTick
       policy (2xATR R-ladder, scale-outs in thirds, breakeven, phase trails,
       conditional TP, 7-calendar-day flat time stop), 20-bar censoring cap

and prints win rate / mean / median return per ruler, split by entry-day
change band, regime class and ATR bucket, plus the paired old(+10d) x new
2x2 matrix.

  python3 scripts/labeler_before_after.py --fixtures      # hand-built bar fixtures (offline)
  python3 scripts/labeler_before_after.py --yf            # real sample via yfinance daily bars
  python3 scripts/labeler_before_after.py --yf --feature-atr   # same sample, features.atr_pct branch
  python3 scripts/labeler_before_after.py --yf --json out.json

Real-sample design (--yf): ~40 tickers (20 liquid large caps + 20 volatile
small/mid caps) x ~15 entry dates spread over 2026-06-01..2026-09-11, entry
at that day's CLOSE, so every path has >= 20 forward sessions by 2026-10-14.
By default these synthetic entries carry no features.atr_pct, so the new
ruler takes its FALLBACK branch (Wilder ATR14 strictly before the entry
session) — NOT the branch production records mostly take. --feature-atr
sets features.atr_pct the way deep_score computes it at scan time (Wilder
ATR14 over the daily bars up to AND including the entry session, % of
price, 2 decimals; here on yfinance bars, so an approximation of the
Alpaca-bar value), exercising the primary branch. regime_class comes from
SPY/50dMA + VXX/30d-mean on the entry date (the 200-day inputs live also
passes are not modelled — see shadow_band_report's not_modelled). Both
rulers are GROSS.

PAIRED PATHS: here both rulers walk the same forward bars (entries are
stamped 20:00 UTC, same UTC and ET date). In PRODUCTION paired_vs_old,
scans made 20:00-24:00 ET carry the next UTC date, so the old ruler's path
starts one session later than this ruler's — those pairs compare
DIFFERENT forward paths.
"""
import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import shadow_portfolio as sp  # noqa: E402

HORIZONS = (5, 10, 20)
LIQUID = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "UNH", "HD",
          "KO", "PEP", "WMT", "COST", "AMD", "NFLX", "CRM", "CAT", "BA", "TSLA"]
SMALL = ["PLUG", "SOFI", "RIOT", "MARA", "UPST", "AFRM", "RIVN", "LCID", "OPEN", "BBAI",
         "SOUN", "IONQ", "RKLB", "ACHR", "JOBY", "HIMS", "CELH", "FUBO", "CLSK", "RUN"]
BIAS_NOTE = (
    "Direction the NEW labels could bias results: (1) daily OHLC overstates level touches vs live's "
    "1-minute IEX bar CLOSES — scale-outs at 1R and trail high-water marks fire MORE often and from "
    "HIGHER peaks than live, which flatters the right tail (optimistic); (2) the adverse-first "
    "O->L->H->C walk and fill-at-open on gaps are pessimistic for bars that really went up first; "
    "(3) runners still open at bar 20 are marked to market (censored) — understated if they would "
    "have kept trailing; (4) gross of costs (net = gross - 0.10pp); (5) live's phantom first "
    "scale-out race (Python's 45s manage cycle marks the 1R third sold without an order; the WS "
    "monitor adopts it) is not modelled — this ruler always banks 1/3 at +1R: optimistic on "
    "1R-touch-then-breakeven paths, pessimistic on big runners; (6) live trails measure from "
    "45s-sampled peaks, lower than any daily high (optimistic on trail exits); (7) the 200-day SPY "
    "regime inputs live passes are not recorded, so BULL/NEUTRAL may be read where live sees "
    "NEUTRAL/BEAR (mixed). Net direction is NOT known a priori: a 'looks better' result is suspect "
    "until checked against real closed trades (voltrade_trade_feedback.json)."
)


# ── case construction ────────────────────────────────────────────────────────
def _bar(day, o, h, lo, c):
    return {"t": f"{day}T04:00:00Z", "o": float(o), "h": float(h), "l": float(lo), "c": float(c)}


def _weekdays(start, n, before=False):
    d = datetime.strptime(start, "%Y-%m-%d").date()
    out = []
    while len(out) < n:
        d += timedelta(days=-1 if before else 1)
        if d.weekday() < 5:
            out.append(d.isoformat())
    return sorted(out)


def make_case(name, entry_day, entry_price, bars, vxx=1.0, spy_vs_ma50=1.0, change_pct=0.0,
              atr_feature=None, group=""):
    features = {"change_pct_today": change_pct, "spy_vs_ma50": spy_vs_ma50}
    if atr_feature is not None:
        features["atr_pct"] = atr_feature
    rec = {"ticker": name, "timestamp": f"{entry_day}T20:00:00+00:00", "entry_price": entry_price,
           "vxx_ratio": vxx, "features": features}
    return {"name": name, "group": group, "entry_day": entry_day, "entry": entry_price,
            "bars": sorted(bars, key=lambda b: b["t"]), "rec": rec, "change_pct": change_pct}


def fixture_cases():
    """Hand-built paths (entry 100 on Mon 2026-06-01, flat prior bars => ATR 2%)."""
    e = "2026-06-01"
    prior = [_bar(d, 100, 101, 99, 100) for d in _weekdays(e, 41, before=True)] + [_bar(e, 100, 101, 99, 100)]
    fwd = _weekdays(e, 25)

    def drift(level, days):
        return [_bar(d, level, level + 0.3, level - 0.3, level) for d in days]

    bull, bear, neut = (0.90, 1.02), (1.20, 1.00), (1.00, 1.00)
    specs = [
        ("stop_1st_bar", neut, None, [_bar(fwd[0], 100, 100.5, 95, 96)] + drift(97, fwd[1:])),
        ("gap_through_stop", neut, None, [_bar(fwd[0], 90, 91, 89, 90.5)] + drift(91, fwd[1:])),
        ("clamp_low_atr0.5", neut, 0.5, [_bar(fwd[0], 100, 100.2, 98, 98.5)] + drift(99, fwd[1:])),
        ("clamp_high_atr6", neut, 6.0, [_bar(fwd[0], 100, 100.2, 91, 91.5)] + drift(92, fwd[1:])),
        ("scale_then_breakeven", neut, 0.6, [_bar(fwd[0], 100.2, 101.3, 100.1, 101), _bar(fwd[1], 100.8, 100.9, 99, 99.5)] + drift(99.5, fwd[2:])),
        ("bull_runner_trail", bull, None, [_bar(fwd[0], 100.5, 104.5, 100, 104), _bar(fwd[1], 104, 113, 103.5, 112.5), _bar(fwd[2], 113, 130, 112, 129), _bar(fwd[3], 128, 128.5, 120, 121)] + drift(121, fwd[4:])),
        ("bear_take_profit", bear, None, [_bar(fwd[0], 100.5, 104.5, 100, 104), _bar(fwd[1], 104, 107, 103.5, 106.5)] + drift(106, fwd[2:])),
        ("neutral_phase3_trail", neut, None, [_bar(fwd[0], 100.5, 104.5, 100, 104), _bar(fwd[1], 104, 109, 103.5, 108.5), _bar(fwd[2], 108.5, 109, 105, 105.5)] + drift(105.5, fwd[3:])),
        ("time_stop_day7", neut, None, drift(101, fwd[:4]) + drift(101.4, fwd[4:])),
        ("censored_at_cap", neut, None, drift(103, fwd)),
        ("same_bar_ambiguous", neut, None, [_bar(fwd[0], 100, 105, 95, 104)] + drift(104, fwd[1:])),
        ("slow_grind_up", neut, None, [_bar(d, 100 + i * 0.5, 100.4 + i * 0.5, 99.8 + i * 0.5, 100.3 + i * 0.5) for i, d in enumerate(fwd)]),
    ]
    return [make_case(name, e, 100.0, prior + path, vxx=v, spy_vs_ma50=s, atr_feature=atr, group="fixture")
            for name, (v, s), atr, path in specs]


def _deep_score_atr_pct(bars_upto_entry, price):
    """bot_engine.deep_score's features.atr_pct: Wilder ATR14 (seeded with the
    mean of the first 14 TRs) over the bars up to and including the entry
    session, as % of price, rounded to 2 decimals; None if < 15 bars."""
    if len(bars_upto_entry) < 15 or price <= 0:
        return None
    trs = [max(b["h"] - b["l"], abs(b["h"] - p["c"]), abs(b["l"] - p["c"]))
           for p, b in zip(bars_upto_entry, bars_upto_entry[1:])]
    atr = sum(trs[:14]) / 14
    for tr in trs[14:]:
        atr = (atr * 13 + tr) / 14
    return round(atr / price * 100, 2)


def yf_cases(tickers, start, end, n_dates, feature_atr=False):
    import yfinance as yf
    import pandas as pd
    first = datetime.strptime(start, "%Y-%m-%d").date()
    dl_start = (first - timedelta(days=110)).isoformat()
    dl_end = (date.today() + timedelta(days=1)).isoformat()
    df = yf.download(tickers + ["SPY", "VXX"], start=dl_start, end=dl_end, auto_adjust=False,
                     group_by="ticker", progress=False, threads=True)
    spy = df["SPY"]["Close"].dropna()
    vxx = df["VXX"]["Close"].dropna()
    session_days = [d.date().isoformat() for d in spy.index if start <= d.date().isoformat() <= end]
    if len(session_days) <= n_dates:
        entry_days = session_days
    else:   # evenly spaced across the WHOLE range, first and last session included
        entry_days = sorted({session_days[round(i * (len(session_days) - 1) / (n_dates - 1))]
                             for i in range(n_dates)})
    cases, skipped = [], defaultdict(int)
    for t in tickers:
        if t not in df.columns.get_level_values(0):
            skipped["no_download"] += 1
            continue
        sub = df[t][["Open", "High", "Low", "Close"]].dropna()
        if sub.empty:
            skipped["no_download"] += 1
            continue
        bars = [_bar(ix.date().isoformat(), r.Open, r.High, r.Low, r.Close) for ix, r in sub.iterrows()]
        by_day = {b["t"][:10]: i for i, b in enumerate(bars)}
        for d in entry_days:
            i = by_day.get(d)
            if i is None or i < 1:
                skipped["no_entry_bar"] += 1
                continue
            lo = (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=sp.XA_ATR_LOOKBACK_DAYS)).date().isoformat()
            hi = (datetime.strptime(d, "%Y-%m-%d") + timedelta(days=sp.XA_WINDOW_END_DAYS)).date().isoformat()
            window = [b for b in bars if lo <= b["t"][:10] <= hi]
            entry = bars[i]["c"]
            cp = (entry / bars[i - 1]["c"] - 1) * 100
            ts = pd.Timestamp(d)
            spy_hist = spy[spy.index <= ts]
            vxx_hist = vxx[vxx.index <= ts]
            s50 = float(spy_hist.iloc[-1] / spy_hist.iloc[-50:].mean())
            v30 = float(vxx_hist.iloc[-1] / vxx_hist.iloc[-30:].mean())
            feat = _deep_score_atr_pct(bars[: i + 1], entry) if feature_atr else None
            cases.append(make_case(t, d, entry, window, vxx=v30, spy_vs_ma50=s50, change_pct=cp,
                                   atr_feature=feat, group="liquid" if t in LIQUID else "small"))
    return cases, entry_days, dict(skipped)


# ── replay ───────────────────────────────────────────────────────────────────
def replay(case):
    fwd = [b for b in case["bars"] if b["t"][:10] > case["entry_day"]]
    row = {"name": case["name"], "group": case["group"], "band": sp._band_of(case["change_pct"])}
    for h in HORIZONS:
        lbl, px, reason, _ = sp._label_from_path(case["entry"], fwd, max_bars=h) if fwd else (-1, None, "no_data", None)
        row[f"old+{h}d"] = None if lbl == -1 else {
            "label": lbl, "reason": reason,
            "ret": round((px - case["entry"]) / case["entry"] * 100, 3) if px else 0.0}
    xa = sp._exit_aligned_outcome(case["rec"], case["bars"])
    row["xa"] = xa if (xa and xa.get("label") in (0, 1)) else None
    row["xa_unlabelable"] = None if row["xa"] else (xa or {}).get("exit_reason", "no_price_data")
    return row


def _stats(rets, wins):
    n = len(rets)
    if not n:
        return {"n": 0}
    return {"n": n, "win_rate": round(100 * wins / n, 1), "mean": round(sum(rets) / n, 3),
            "median": round(statistics.median(rets), 3)}


def summarize(rows):
    out = {}

    def cut(key_fn):
        cells = defaultdict(lambda: defaultdict(lambda: ([], 0)))
        for r in rows:
            k = key_fn(r)
            if k is None:
                continue
            for ruler in [f"old+{h}d" for h in HORIZONS] + ["xa"]:
                o = r.get(ruler)
                if not o:
                    continue
                ret = o["ret"] if ruler != "xa" else o["return_pct"]
                rets, wins = cells[k][ruler]
                rets.append(ret)
                cells[k][ruler] = (rets, wins + (1 if o["label"] == 1 else 0))
        return {k: {ru: _stats(*v) for ru, v in d.items()} for k, d in sorted(cells.items())}

    out["overall"] = cut(lambda r: "all")
    out["by_group"] = cut(lambda r: r["group"])
    out["by_change_band"] = cut(lambda r: r["band"])
    out["by_regime_class"] = cut(lambda r: r["xa"]["regime_class"] if r["xa"] else None)
    out["by_atr_bucket"] = cut(lambda r: None if not r["xa"] else
                               ("atr<2" if r["xa"]["atr_pct"] < 2 else "atr2-4" if r["xa"]["atr_pct"] < 4
                                else "atr4-6" if r["xa"]["atr_pct"] < 6 else "atr>=6"))
    reasons = defaultdict(list)
    for r in rows:
        if r["xa"]:
            reasons[r["xa"]["exit_reason"]].append(r["xa"]["return_pct"])
    out["xa_exit_reasons"] = {k: {"n": len(v), "mean": round(sum(v) / len(v), 3)} for k, v in sorted(reasons.items())}
    xa_rows = [r["xa"] for r in rows if r["xa"]]
    out["xa_censored_pct"] = round(100 * sum(1 for x in xa_rows if x["censored"]) / len(xa_rows), 1) if xa_rows else None
    out["xa_mean_net"] = round(sum(x["return_pct"] for x in xa_rows) / len(xa_rows) - sp.XA_ROUND_TRIP_COST_PCT, 3) if xa_rows else None
    unl = defaultdict(int)
    for r in rows:
        if r["xa_unlabelable"]:
            unl[r["xa_unlabelable"]] += 1
    out["xa_unlabelable"] = dict(unl)
    src = defaultdict(int)
    for x in xa_rows:
        src[x.get("atr_src", "?")] += 1
    out["xa_atr_src"] = dict(src)
    m = {"old1_xa1": 0, "old1_xa0": 0, "old0_xa1": 0, "old0_xa0": 0}
    for r in rows:
        if r["xa"] and r["old+10d"]:
            m[f"old{r['old+10d']['label']}_xa{r['xa']['label']}"] += 1
    out["paired_old10d_vs_xa"] = m
    return out


def _fmt_table(title, block):
    lines = [f"\n## {title}", f"{'cell':<18}{'ruler':<9}{'n':>5}{'win%':>8}{'mean%':>9}{'median%':>9}"]
    for k, rulers in block.items():
        for ru in [f"old+{h}d" for h in HORIZONS] + ["xa"]:
            s = rulers.get(ru)
            if not s or not s.get("n"):
                continue
            lines.append(f"{str(k):<18}{ru:<9}{s['n']:>5}{s['win_rate']:>8.1f}{s['mean']:>9.3f}{s['median']:>9.3f}")
    return "\n".join(lines)


def report(rows, header):
    s = summarize(rows)
    print(header)
    for key, title in (("overall", "Overall"), ("by_group", "By universe group"),
                       ("by_change_band", "By entry-day change band (%)"),
                       ("by_regime_class", "By regime class at entry"), ("by_atr_bucket", "By ATR% at entry")):
        print(_fmt_table(title, s[key]))
    print("\n## New ruler: exit reasons (n, mean return %)")
    for k, v in s["xa_exit_reasons"].items():
        print(f"  {k:<14} n={v['n']:<5} mean={v['mean']:+.3f}")
    print(f"\n  censored at the 20-bar cap: {s['xa_censored_pct']}%   mean NET (gross - 0.10pp): {s['xa_mean_net']}")
    print(f"  unlabelable by the new ruler: {s['xa_unlabelable']}")
    print(f"  ATR branch taken by the new ruler (atr_src): {s['xa_atr_src']}")
    print(f"\n## Paired old(+10d) x new label matrix: {s['paired_old10d_vs_xa']}")
    print("\n" + BIAS_NOTE)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--yf", action="store_true")
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end", default="2026-09-11")
    ap.add_argument("--dates", type=int, default=15)
    ap.add_argument("--feature-atr", action="store_true",
                    help="--yf: set features.atr_pct as deep_score does (exercises the primary ATR branch)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if not (a.fixtures or a.yf):
        a.fixtures = True
    dump = {}
    if a.fixtures:
        rows = [replay(c) for c in fixture_cases()]
        print("# (a) FIXTURES — per case")
        print(f"{'case':<24}{'old+5d':>14}{'old+10d':>14}{'old+20d':>14}   {'new (exit_aligned_v1)':<40}")
        for r in rows:
            cells = []
            for h in HORIZONS:
                o = r[f"old+{h}d"]
                cells.append(f"{o['label']}/{o['ret']:+.2f}" if o else "-1")
            x = r["xa"]
            xs = (f"{x['label']}/{x['return_pct']:+.3f} {x['exit_reason']} b{x['bars_held']} s{x['scale_outs']}"
                  if x else f"-1 {r['xa_unlabelable']}")
            print(f"{r['name']:<24}{cells[0]:>14}{cells[1]:>14}{cells[2]:>14}   {xs:<40}")
        dump["fixtures"] = report(rows, "\n# (a) FIXTURES — summary")
    if a.yf:
        tickers = LIQUID + SMALL
        cases, entry_days, skipped = yf_cases(tickers, a.start, a.end, a.dates, feature_atr=a.feature_atr)
        rows = [replay(c) for c in cases]
        hdr = (f"\n# (b) REAL SAMPLE — yfinance daily bars, {len(tickers)} tickers x {len(entry_days)} entry dates "
               f"({entry_days[0]}..{entry_days[-1]}), entry at the day's close; {len(cases)} cases; skipped {skipped}\n"
               f"# ATR branch: {'features.atr_pct (deep_score method on yfinance bars)' if a.feature_atr else 'prior-bars FALLBACK only (no features.atr_pct; production records mostly take the feature branch)'}\n"
               f"# paired matrix: same forward path for both rulers here; in production, scans 20:00-24:00 ET pair different paths")
        dump["yf"] = report(rows, hdr)
        dump["yf"]["entry_days"] = entry_days
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dump, f, indent=2)


if __name__ == "__main__":
    main()
