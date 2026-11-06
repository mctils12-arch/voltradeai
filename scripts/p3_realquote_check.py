#!/usr/bin/env python3
"""Edge P3 gate-1 real-quote clause (research/edge_preregistration.md, P3).

Recomputes the pre-registered Cremers-Weinbaum VS from Cboe's free delayed
quotes (a real exchange-quote reference, NOT Alpaca's indicative feed) and
rank-correlates it with the archived Alpaca-indicative VS for the same
underlyings and trading day. Kill clause: Spearman < 0.3 kills the DATASET.

Definition mirrors server/chainVs.ts exactly: expiry 14..60 DTE from the day,
call delta in [0.25, 0.75], same strike+expiry call/put pair, both iv > 0,
>= 3 pairs, VS = mean(iv_call - iv_put).

Usage:
  python3 scripts/p3_realquote_check.py --day 2026-11-02 --vs-json chain_vs.json \
      [--n 40] [--min-pairs-ours 10] [--out result.json]
chain_vs.json = body of GET /api/diag/chain_vs?day=<day> (needs DIAG_TOKEN).
Run it right after the close: Cboe serves the LAST quote, so the Cboe day must
equal --day (the script refuses a snapshot whose ET date differs).
"""
import argparse, json, os, re, sys, time, urllib.request
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

VS_MIN_DTE, VS_MAX_DTE = 14, 60
CALL_DELTA_BAND = (0.25, 0.75)
VS_MIN_PAIRS = 3
OCC = re.compile(r"^([A-Z.]{1,6})(\d{6})([CP])(\d{8})$")


def vs_from_options(options, day):
    """options: Cboe rows {option, iv, delta}. Returns (vs|None, n_pairs)."""
    d0 = date.fromisoformat(day)
    calls, puts = {}, {}
    for o in options:
        m = OCC.match(o.get("option", ""))
        if not m:
            continue
        exp = date(2000 + int(m[2][:2]), int(m[2][2:4]), int(m[2][4:]))
        dte = (exp - d0).days
        if dte < VS_MIN_DTE or dte > VS_MAX_DTE:
            continue
        iv = o.get("iv")
        if not iv or iv <= 0:
            continue
        key = (m[2], m[4])
        if m[3] == "C":
            dl = o.get("delta")
            if dl is None or not (CALL_DELTA_BAND[0] <= dl <= CALL_DELTA_BAND[1]):
                continue
            calls[key] = iv
        else:
            puts[key] = iv
    diffs = [calls[k] - puts[k] for k in calls if k in puts]
    if len(diffs) < VS_MIN_PAIRS:
        return None, len(diffs)
    return sum(diffs) / len(diffs), len(diffs)


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def et_day(ts):
    """Cboe's snapshot timestamp is UTC wall time; the trading day is its ET date."""
    t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return t.astimezone(ZoneInfo("America/New_York")).date().isoformat()


def fetch_cboe(sym):
    req = urllib.request.Request(
        f"https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json",
        headers={"User-Agent": "Mozilla/5.0 voltradeai-research"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)


def pool_results(results, boot=2000, seed=7):
    """Apply the pre-registered pooling rule (experiments.md 2026-11-02): pool
    every (day, name) pair from the per-day result files; pooled Spearman < 0.3
    kills the Alpaca-indicative dataset at gate 1. The bar is not lowered.
    Also reports the per-day values and a bootstrap 95% CI (disclosure only,
    the verdict uses the point estimate exactly as the first day's did)."""
    import random
    ours, theirs, per_day = [], [], {}
    for r in results:
        pr = r.get("pairs") or []
        per_day[r["day"]] = {"n": len(pr), "spearman": r.get("spearman")}
        ours += [p[1] for p in pr]
        theirs += [p[2] for p in pr]
    if len(ours) < 30:
        return {"days": per_day, "n": len(ours), "verdict": "insufficient_n"}
    rho = spearman(ours, theirs)
    rng, n, bs = random.Random(seed), len(ours), []
    for _ in range(boot):
        ix = [rng.randrange(n) for _ in range(n)]
        bs.append(spearman([ours[i] for i in ix], [theirs[i] for i in ix]))
    bs = sorted(x for x in bs if x == x)
    return {"days": per_day, "n": n, "pooled_spearman": round(rho, 4),
            "ci95": [round(bs[int(.025 * len(bs))], 4), round(bs[int(.975 * len(bs)) - 1], 4)],
            "datasetKill": rho < 0.3,
            "verdict": "KILLED_AT_GATE_1" if rho < 0.3 else "PASSES_GATE_1_CLAUSE"}


DIAG_BASE = "https://voltradeai.com/api/diag"
ARCHIVE_READY_ET = (16, 30)  # options chain archive runs 16:30-18:30 ET; Cboe last quote is final after the close


def too_early(day, now=None):
    """True if `day`'s close-time data cannot exist yet (before 16:30 ET on that ET date)."""
    now = now or datetime.now(ZoneInfo("America/New_York"))
    ready = datetime.combine(date.fromisoformat(day), datetime.min.time(),
                             tzinfo=ZoneInfo("America/New_York")).replace(hour=ARCHIVE_READY_ET[0], minute=ARCHIVE_READY_ET[1])
    return now < ready


def fetch_chain_vs(day, token, opener=urllib.request.urlopen):
    """GET /api/diag/chain_vs?day=<day> (x-diag-token). Compiles the manual fetch step."""
    if not token:
        raise SystemExit("DIAG_TOKEN not set; cannot --fetch")
    req = urllib.request.Request(f"{DIAG_BASE}/chain_vs?day={day}", headers={"x-diag-token": token})
    with opener(req, timeout=120) as r:
        body = json.load(r)
    if not any(x.get("day") == day for x in body.get("rows", [])):
        raise SystemExit(f"chain_vs has no rows for {day} (archive not run yet?)")
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", nargs="+", metavar="RESULT_JSON",
                    help="pool per-day result files and print the pre-registered verdict")
    ap.add_argument("--day")
    ap.add_argument("--vs-json")
    ap.add_argument("--fetch", action="store_true",
                    help="fetch chain_vs for --day using DIAG_TOKEN instead of --vs-json; refuses before 16:30 ET on --day")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min-pairs-ours", type=int, default=10)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.pool:
        print(json.dumps(pool_results([json.load(open(f)) for f in a.pool]), indent=1))
        return 0
    if a.fetch and a.day and not a.vs_json:
        if too_early(a.day):
            raise SystemExit(f"too early: {a.day} close-time data is not final before 16:30 ET (Cboe serves the last quote)")
        vs_body = fetch_chain_vs(a.day, os.environ.get("DIAG_TOKEN", ""))
    elif a.day and a.vs_json:
        vs_body = json.load(open(a.vs_json))
    else:
        ap.error("--day with --vs-json (or --fetch) is required unless --pool is given")
    rows = [r for r in vs_body["rows"]
            if r["day"] == a.day and r["vs"] is not None and r["nPairs"] >= a.min_pairs_ours]
    rows.sort(key=lambda r: r["u"])
    # deterministic, selection-free sample: evenly spaced through the alphabetical list
    step = max(1, len(rows) // a.n)
    sample = rows[::step][: a.n]
    ours, theirs, used, skipped = [], [], [], []
    for r in sample:
        try:
            j = fetch_cboe(r["u"])
        except Exception as e:  # noqa: BLE001
            skipped.append((r["u"], f"fetch:{type(e).__name__}"))
            continue
        if et_day(j.get("timestamp", "1970-01-01 00:00:00")) != a.day:
            skipped.append((r["u"], f"stale_ts:{j.get('timestamp')}"))
            continue
        v, n = vs_from_options(j["data"]["options"], a.day)
        if v is None:
            skipped.append((r["u"], f"cboe_pairs:{n}"))
            continue
        ours.append(r["vs"]); theirs.append(v); used.append((r["u"], r["vs"], v, r["nPairs"], n))
        time.sleep(0.4)
    out = {"day": a.day, "sampled": len(sample), "used": len(used), "skipped": skipped}
    if len(used) >= 10:
        out["spearman"] = round(spearman(ours, theirs), 4)
        out["median_abs_diff_volpts"] = round(sorted(abs(x - y) for x, y in zip(ours, theirs))[len(used) // 2] * 100, 2)
        out["mean_ours"] = round(sum(ours) / len(ours), 4)
        out["mean_cboe"] = round(sum(theirs) / len(theirs), 4)
        out["datasetKill"] = out["spearman"] < 0.3
    out["pairs"] = used
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
