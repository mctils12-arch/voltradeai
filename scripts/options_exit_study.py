#!/usr/bin/env python3
"""Options exit study (2026-11-02, human: "figure out what's best for taking
profit"). Replays candidate exit rules over the account's OWN closed option
lots, using the diag probes shipped the same day:
  /api/diag/lots         — the sleeve ledger's closed-lot walk (entry/exit per share)
  /api/diag/option_bars  — Alpaca daily option bars for each contract's life
Every rule is evaluated on DAILY CLOSES between the lot's open and its actual
close (conservative: a rule fires on the first close that satisfies it and
exits at that close; a lot closed the same day it opened has no path and
keeps its actual result). Output: study/results.json + study/results.md.
Pure read; prints a summary. Run with DIAG_TOKEN in the environment.
"""
import json, os, re, sys, time, urllib.request, urllib.parse, datetime as dt, statistics, collections

BASE = "https://voltradeai.com/api/diag"
TOKEN = os.environ.get("DIAG_TOKEN", "")
HERE = os.environ.get("OPTIONS_EXIT_STUDY_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research", "options_exit_study_runs")
os.makedirs(HERE, exist_ok=True)
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)
OCC = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")

def get(probe, **q):
    url = f"{BASE}/{probe}" + ("?" + urllib.parse.urlencode(q) if q else "")
    req = urllib.request.Request(url, headers={"x-diag-token": TOKEN})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

def ymd(iso): return iso[:10]
def d(iso): return dt.date.fromisoformat(iso[:10])

# ── 1. closed option lots ───────────────────────────────────────────────
lots_path = os.path.join(HERE, "lots.json")
if os.path.exists(lots_path) and "--refresh" not in sys.argv:
    lots = json.load(open(lots_path))
else:
    lots = get("lots", limit=5000)
    json.dump(lots, open(lots_path, "w"))
rows = [r for r in lots["rows"] if r.get("option")]
print(f"lots: {lots['count']} events total, {len(rows)} option events, since {lots.get('since')}, truncated={lots.get('truncated')}")

# merge partial closes of one lot: key = symbol + openedAt
by_lot = collections.OrderedDict()
for r in rows:
    k = (r["symbol"], r.get("openedAt"))
    L = by_lot.setdefault(k, {"symbol": r["symbol"], "openedAt": r.get("openedAt"), "side": r["side"], "qty": 0, "basis": 0.0, "pnl": 0.0, "closes": []})
    L["qty"] += r["qty"]; L["basis"] += r["basis"] or 0; L["pnl"] += r["pnl"] or 0
    L["closes"].append({"t": r["closedAt"], "qty": r["qty"], "exit": r["exitPrice"], "entry": r["entryPrice"]})
lots_list = list(by_lot.values())
for L in lots_list:
    L["closedAt"] = max(c["t"] for c in L["closes"])
    L["entry"] = L["closes"][0]["entry"]
    mult = 100
    L["exit"] = (L["entry"] + (L["pnl"] / (L["qty"] * mult)) * (1 if L["side"] == "long" else -1)) if L["entry"] is not None and L["qty"] else None
    L["ret"] = L["pnl"] / L["basis"] if L["basis"] else None
    L["days"] = (d(L["closedAt"]) - d(L["openedAt"])).days if L.get("openedAt") else None
    m = OCC.match(L["symbol"]); L["root"] = m.group(1) if m else L["symbol"]; L["cp"] = m.group(3) if m else "?"
    L["expiry"] = ("20" + m.group(2)[:2] + "-" + m.group(2)[2:4] + "-" + m.group(2)[4:]) if m else None
print(f"option lots (merged): {len(lots_list)}; with known open time: {sum(1 for L in lots_list if L['openedAt'])}; multi-day: {sum(1 for L in lots_list if (L['days'] or 0) >= 1)}")

# ── 2. strategy groups: same root + expiry, opened within 3 minutes ─────
groups = collections.defaultdict(list)
for L in lots_list:
    if not L["openedAt"]: continue
    key = (L["root"], L["expiry"], L["openedAt"][:16])  # minute bucket
    groups[key].append(L)
for L in lots_list:
    L["legs_opened_together"] = len(groups.get((L["root"], L["expiry"], (L["openedAt"] or "")[:16]), [L]))

# ── 3. bars for every multi-day lot ─────────────────────────────────────
need = [L for L in lots_list if L["openedAt"] and (L["days"] or 0) >= 1 and L["entry"]]
syms = sorted({L["symbol"] for L in need})
bars = {}
for s in syms:
    p = os.path.join(CACHE, s + ".json")
    if os.path.exists(p): bars[s] = json.load(open(p))
todo = [s for s in syms if s not in bars]
print(f"contracts needing bars: {len(syms)}; cached {len(syms)-len(todo)}; fetching {len(todo)}")
# window per symbol: earliest open − 1 d .. latest close + 1 d
win = {}
for L in need:
    a, b = d(L["openedAt"]) - dt.timedelta(days=1), d(L["closedAt"]) + dt.timedelta(days=1)
    w = win.get(L["symbol"]); win[L["symbol"]] = (min(a, w[0]) if w else a, max(b, w[1]) if w else b)
errors = []
for i in range(0, len(todo), 10):
    batch = todo[i:i+10]
    start = min(win[s][0] for s in batch).isoformat(); end = max(win[s][1] for s in batch).isoformat()
    try:
        res = get("option_bars", symbols=",".join(batch), start=start, end=end)
        for s in batch:
            bars[s] = res.get("bars", {}).get(s, [])
            json.dump(bars[s], open(os.path.join(CACHE, s + ".json"), "w"))
        if res.get("truncated"): errors.append(f"truncated page set for batch starting {batch[0]}")
    except Exception as e:
        errors.append(f"{batch[0]}..: {e}"); print("  bars error:", e)
    time.sleep(0.6)
print(f"bars fetched; contracts with ≥1 bar: {sum(1 for s in syms if bars.get(s))}; errors: {len(errors)}")

# ── 4. replay rules on daily closes ─────────────────────────────────────
def path(L):
    """closes strictly inside (open day, close day] — the days a daily rule could have acted before the actual exit"""
    o, c = d(L["openedAt"]), d(L["closedAt"])
    out = []
    for b in bars.get(L["symbol"], []):
        day = d(b["t"])
        if o < day < c and b.get("c"): out.append((day, float(b["c"]), float(b["h"] or b["c"]), float(b["l"] or b["c"])))
    return out

def pnl_at(L, px):
    return (px - L["entry"]) * L["qty"] * 100 * (1 if L["side"] == "long" else -1)

RULES_LONG = {
    "actual": lambda L, P: None,
    "tp+50%": lambda L, P: next((px for day, px, h, l in P if px >= L["entry"] * 1.5), None),
    "tp+100%": lambda L, P: next((px for day, px, h, l in P if px >= L["entry"] * 2.0), None),
    "tp+200%": lambda L, P: next((px for day, px, h, l in P if px >= L["entry"] * 3.0), None),
}
def trail_long(arm, giveback):
    def f(L, P):
        peak = L["entry"]; armed = False
        for day, px, h, l in P:
            peak = max(peak, px)
            if peak >= L["entry"] * (1 + arm): armed = True
            if armed and px <= peak * (1 - giveback): return px
        return None
    return f
RULES_LONG["trail35%after+50%"] = trail_long(0.5, 0.35)
RULES_LONG["trail40%after+50%"] = trail_long(0.5, 0.40)
RULES_LONG["trail25%after+100%"] = trail_long(1.0, 0.25)

RULES_SHORT = {
    "actual": lambda L, P: None,
    "keep25%": lambda L, P: next((px for day, px, h, l in P if px <= L["entry"] * 0.75), None),
    "keep50%": lambda L, P: next((px for day, px, h, l in P if px <= L["entry"] * 0.50), None),
    "keep75%": lambda L, P: next((px for day, px, h, l in P if px <= L["entry"] * 0.25), None),
}

def evaluate(sub, rules):
    out = {}
    for name, rule in rules.items():
        tot = 0.0; fired = 0; rets = []; pnls = []
        for L in sub:
            P = path(L)
            px = rule(L, P) if P else None
            v = pnl_at(L, px) if px is not None else L["pnl"]
            if px is not None: fired += 1
            tot += v; pnls.append(v); rets.append(v / L["basis"] if L["basis"] else 0)
        n = len(sub)
        out[name] = {"n": n, "fired": fired, "total": round(tot, 0), "mean": round(tot / n, 1) if n else None,
                     "median": round(statistics.median(pnls), 1) if pnls else None,
                     "winners": sum(1 for v in pnls if v > 0), "worst": round(min(pnls), 0) if pnls else None, "best": round(max(pnls), 0) if pnls else None,
                     "mean_ret_pct": round(100 * statistics.mean(rets), 1) if rets else None}
    return out

long_lots = [L for L in lots_list if L["side"] == "long" and L["entry"]]
short_lots = [L for L in lots_list if L["side"] == "short" and L["entry"]]
long_multi = [L for L in long_lots if (L["days"] or 0) >= 1 and bars.get(L["symbol"])]
short_multi = [L for L in short_lots if (L["days"] or 0) >= 1 and bars.get(L["symbol"])]
results = {
    "lots": {"option_lots": len(lots_list), "long": len(long_lots), "short": len(short_lots),
             "long_multi_day_with_bars": len(long_multi), "short_multi_day_with_bars": len(short_multi),
             "bars_errors": errors},
    "hold": collections.Counter(("<1d" if (L["days"] or 0) < 1 else "1-7d" if L["days"] <= 7 else "8-30d" if L["days"] <= 30 else ">30d") for L in lots_list),
    "long_rules_all": evaluate(long_lots, RULES_LONG),
    "long_rules_multi_day": evaluate(long_multi, RULES_LONG),
    "short_rules_all": evaluate(short_lots, RULES_SHORT),
    "short_rules_multi_day": evaluate(short_multi, RULES_SHORT),
    "long_lots_multi_day": [{k: L[k] for k in ("symbol", "side", "qty", "entry", "exit", "pnl", "ret", "days", "openedAt", "closedAt", "legs_opened_together")} for L in long_multi],
}
json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=str)
def table(title, res):
    lines = [f"### {title}", "", "| rule | n | fired | total $ | mean $ | median $ | winners | worst | best | mean ret % |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in res.items():
        lines.append(f"| {k} | {v['n']} | {v['fired']} | {v['total']} | {v['mean']} | {v['median']} | {v['winners']} | {v['worst']} | {v['best']} | {v['mean_ret_pct']} |")
    return "\n".join(lines) + "\n"
md = ["# Options exit study — account's own closed lots (run " + dt.datetime.now(dt.timezone.utc).isoformat()[:16] + "Z)", "",
      f"option lots {len(lots_list)} (long {len(long_lots)}, short {len(short_lots)}); multi-day with bars: long {len(long_multi)}, short {len(short_multi)}; hold buckets {dict(results['hold'])}", ""]
md += [table("Bought lots, all (same-day lots keep their actual result)", results["long_rules_all"]),
       table("Bought lots, multi-day with a price path", results["long_rules_multi_day"]),
       table("Sold lots, all", results["short_rules_all"]),
       table("Sold lots, multi-day with a price path", results["short_rules_multi_day"])]
open(os.path.join(HERE, "results.md"), "w").write("\n".join(md))
print("\n".join(md))
