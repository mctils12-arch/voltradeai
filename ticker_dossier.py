#!/usr/bin/env python3
"""
ticker_dossier.py — Ticker Dossier v1 ([PRODUCT] 2026-10-29, human-asked).

"For RKLB: the week and the day outlook, the market, the sector (space), the
stock, the options side, and events the price hasn't factored in yet — like
weather: accurate because it looks at a vast area, then the region, then the
spot." Plus (same day, scope addition) a TRADE PLAN: levels from the range
model, a documented stop and target, the probability of touching the target
before the stop, and the bot's own metrics for this name.

Layers, top-down, each {status: live|partial|unavailable, reason, as_of, source}:
  market     regime_engine's persisted state (load_state; read-only)
  sector     a documented static ticker -> industry ETF map (thematic names),
             else bot_engine's sector resolution -> its SPDR ETF; trend and
             relative strength vs SPY (20/60 sessions) and the stock vs it
  stock      price, trend, realized vol, RANGE FORECASTS (dossier_ranges:
             HAR/EWMA -> 68%/90% lognormal bands, 1 and 5 sessions) with their
             walk-forward coverage and the live calibration ledger
  direction  ONLY where a measured track record exists: the weekly forecast
             screen (forecast.py fc-logit-v1) when this ticker is in the current
             week's set, with the model's measured hit rate / calibration bin;
             otherwise "no validated direction signal — near coin flip"
  bot        the bot's shadow-candidate outcomes by deep_score decile (all
             decisions, +5d/+20d) — the client matches the bot's live score
             (/api/bot/lookup) to its decile; regime_engine.selection_verdict
             for this ticker ("would the regime guards block it today?")
  options    Alpaca indicative chain (options_scanner._fetch_options_chain):
             ATM IV term, implied 1d/5d move, OTM put-call IV skew, next
             earnings when a Finnhub key is configured; implied vs our model
  events     SEC EDGAR submissions (8-K items decoded, Form 4 counts, offerings,
             periodic reports) + Alpaca news, each with a SOURCE CREDIBILITY
             TIER (static, documented — a tier, not a score) and an UNVALIDATED
             "possibly not priced in" reaction check
  plan       dossier_ranges.trade_plan from the live price
  ledger     the dossier's own running record (dossier_ranges.ledger_report)

Read-only with respect to trading: no order, gate, size or threshold reads
anything here. Every network step is guarded and reported under `errors`.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import dossier_ranges as DR

log = logging.getLogger("voltrade.ticker_dossier")

SYMBOL_RE = re.compile(r"^[A-Z]{1,6}$")
HISTORY_CAL_DAYS = 1150           # ~790 sessions: HAR fit window + backtest + 200d SMA
EVENT_WINDOW_DAYS = 60
NEWS_LIMIT = 15
SEC_TICKERS_TTL_S = 24 * 3600
SEC_SUBMISSIONS_TTL_S = 15 * 60
SHADOW_TTL_S = 30 * 60
REACTION_SMALL_Z = 0.5            # |move since| / expected move under this = "small reaction"
REACTION_MAX_SESSIONS = 10

# ── sector map (documented, static) ─────────────────────────────────────────
# Thematic industries a broad SPDR sector misdescribes. ETFs are tried in order;
# the first with enough history is used. Membership is a judgment call recorded
# here, not a classification service — the UI labels the source "static map".
INDUSTRY_MAP: Dict[str, Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = {
    "space": ("Space / aerospace", ("UFO", "ARKX", "ITA"),
              ("RKLB", "ASTS", "LUNR", "PL", "RDW", "BKSY", "SPCE", "IRDM", "SATS", "GSAT", "VSAT", "MNTS", "SIDU", "KRMN")),
    "semis": ("Semiconductors", ("SMH", "SOXX"),
              ("NVDA", "AMD", "AVGO", "MU", "INTC", "TSM", "QCOM", "AMAT", "LRCX", "KLAC", "MRVL", "ON", "TXN", "ARM", "SMCI", "ADI", "NXPI", "MCHP", "ASML")),
    "software": ("Software", ("IGV",), ("CRM", "NOW", "ADBE", "SNOW", "PLTR", "ORCL", "MSFT", "INTU", "WDAY", "DDOG", "CRWD", "PANW", "ZS", "NET", "MDB", "TEAM", "SHOP")),
    "biotech": ("Biotech", ("XBI", "IBB"), ("MRNA", "BNTX", "VRTX", "REGN", "BIIB", "ALNY", "INCY", "EXEL", "SRPT", "NVAX", "CRSP", "BEAM")),
    "gold_miners": ("Gold miners", ("GDX",), ("NEM", "GOLD", "AEM", "KGC", "AU", "WPM", "FNV")),
    "homebuilders": ("Homebuilders", ("XHB", "ITB"), ("DHI", "LEN", "PHM", "TOL", "NVR", "KBH")),
    "airlines": ("Airlines", ("JETS",), ("DAL", "UAL", "AAL", "LUV", "ALK", "JBLU")),
    "uranium": ("Uranium / nuclear fuel", ("URA", "URNM"), ("CCJ", "UEC", "NXE", "DNN", "UUUU", "LEU", "SMR", "OKLO")),
    "solar": ("Solar", ("TAN",), ("FSLR", "ENPH", "SEDG", "RUN", "ARRY", "NXT")),
    "oil_ep": ("Oil & gas E&P", ("XOP",), ("OXY", "DVN", "EOG", "FANG", "APA", "COP", "MRO", "CTRA")),
    "crypto": ("Crypto-linked equities", ("BITQ",), ("COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT")),
}
# bot_engine/macro_data sector names -> SPDR ETF (macro_data.SECTOR_ETFS inverted)
SECTOR_SPDR = {"Technology": "XLK", "Financials": "XLF", "Energy": "XLE", "Healthcare": "XLV", "Industrials": "XLI",
               "Communications": "XLC", "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Utilities": "XLU",
               "Real Estate": "XLRE", "Materials": "XLB"}
INDEX_ETFS = {"SPY", "QQQ", "IWM", "DIA", "VOO", "VTI"}

# ── source credibility tiers (static, documented; a TIER, not a score) ──────
CREDIBILITY_TIERS = {
    1: "primary — the issuer's own legal filing (SEC EDGAR)",
    2: "wire / major outlet — newsroom with editorial standards, or an issuer press release carried by a wire",
    3: "aggregator / financial blog — rewrites, summaries, opinion",
    4: "social / unclassified — unverified or unknown source",
}
SOURCE_TIER = {
    "sec": 1, "edgar": 1,
    "reuters": 2, "ap": 2, "associated press": 2, "bloomberg": 2, "wsj": 2, "wall street journal": 2, "dow jones": 2,
    "marketwatch": 2, "cnbc": 2, "financial times": 2, "ft": 2, "barron's": 2, "barrons": 2, "new york times": 2,
    "business wire": 2, "businesswire": 2, "pr newswire": 2, "prnewswire": 2, "globenewswire": 2, "globe newswire": 2, "accesswire": 2,
    "benzinga": 3, "seeking alpha": 3, "seekingalpha": 3, "yahoo": 3, "yahoo finance": 3, "motley fool": 3, "zacks": 3,
    "investorplace": 3, "tipranks": 3, "thefly": 3, "the fly": 3, "investing.com": 3,
    "reddit": 4, "stocktwits": 4, "twitter": 4, "x": 4,
}

# SEC Form 8-K items (Form 8-K General Instructions) — label + whether material
ITEM_8K = {
    "1.01": ("Entry into a material definitive agreement", True), "1.02": ("Termination of a material agreement", True),
    "1.03": ("Bankruptcy or receivership", True), "1.04": ("Mine safety", False), "1.05": ("Material cybersecurity incident", True),
    "2.01": ("Acquisition or disposition of assets", True), "2.02": ("Results of operations (earnings)", True),
    "2.03": ("New direct financial obligation", True), "2.04": ("Triggering event accelerating an obligation", True),
    "2.05": ("Exit or disposal costs (restructuring)", True), "2.06": ("Material impairment", True),
    "3.01": ("Delisting notice / listing-standard failure", True), "3.02": ("Unregistered sale of equity", True),
    "3.03": ("Material modification of holders' rights", True), "4.01": ("Change of auditor", True),
    "4.02": ("Non-reliance on prior financials (restatement)", True), "5.01": ("Change in control", True),
    "5.02": ("Director / officer departure or appointment", True), "5.03": ("Charter or bylaw amendment", False),
    "5.07": ("Shareholder vote results", False), "7.01": ("Regulation FD disclosure", False),
    "8.01": ("Other events (often litigation, settlements, product news)", True), "9.01": ("Financial statements and exhibits", False),
}
FORM_LABEL = {
    "4": ("Insider transaction (Form 4)", False), "3": ("Initial insider ownership (Form 3)", False),
    "10-K": ("Annual report — includes Item 3 Legal Proceedings", True), "10-Q": ("Quarterly report — Part II Item 1 Legal Proceedings", True),
    "S-1": ("Registration statement (possible offering / dilution)", True), "S-3": ("Shelf registration (possible offering / dilution)", True),
    "424B5": ("Prospectus supplement (offering)", True), "424B3": ("Prospectus (offering / resale)", True), "424B4": ("Prospectus (offering)", True),
    "SC 13D": ("Activist / 5%+ holder stake (13D)", True), "SC 13G": ("Passive 5%+ holder stake (13G)", False),
    "SCHEDULE 13D": ("Activist / 5%+ holder stake (13D)", True), "SCHEDULE 13G": ("Passive 5%+ holder stake (13G)", False),
    "DEF 14A": ("Proxy statement", False), "6-K": ("Foreign issuer current report", True), "20-F": ("Foreign issuer annual report", True),
    "144": ("Proposed insider sale (Form 144)", False), "8-K": ("Current report (8-K)", True),
}


def _now_iso(now: datetime) -> str:
    return now.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e)[:160]}"


def credibility_tier(source: Optional[str]) -> int:
    s = str(source or "").strip().lower()
    if s in SOURCE_TIER:
        return SOURCE_TIER[s]
    for k, v in SOURCE_TIER.items():
        if len(k) > 3 and k in s:
            return v
    return 4


def sector_for(symbol: str, resolved_sector: Optional[str]) -> dict:
    """{key, label, etfs (try in order), source}."""
    if symbol in INDEX_ETFS:
        return {"key": "index", "label": "Broad index ETF", "etfs": [], "source": "static map (index ETF)"}
    for key, (label, etfs, members) in INDUSTRY_MAP.items():
        if symbol in members:
            return {"key": key, "label": label, "etfs": list(etfs), "source": "static industry map (ticker_dossier.INDUSTRY_MAP)"}
    if resolved_sector and resolved_sector in SECTOR_SPDR:
        return {"key": resolved_sector.lower().replace(" ", "_"), "label": resolved_sector, "etfs": [SECTOR_SPDR[resolved_sector]],
                "source": "bot sector map / sector cache (bot_engine._get_sector) -> SPDR sector ETF"}
    return {"key": "unknown", "label": "Unknown sector", "etfs": [], "source": "no mapping for this ticker"}


# ── bar math (pure) ─────────────────────────────────────────────────────────
def _closes(bars: Sequence[dict]) -> List[float]:
    return [float(b["c"]) for b in bars]


def ret_pct(closes: Sequence[float], n: int) -> Optional[float]:
    if len(closes) <= n or closes[-1 - n] <= 0:
        return None
    return round((closes[-1] / closes[-1 - n] - 1) * 100, 3)


def sma(closes: Sequence[float], n: int) -> Optional[float]:
    return sum(closes[-n:]) / n if len(closes) >= n else None


def trend_of(closes: Sequence[float]) -> dict:
    px = closes[-1] if closes else None
    s20, s50, s200 = sma(closes, 20), sma(closes, 50), sma(closes, 200)
    above = [x for x in (s20, s50, s200) if x is not None and px is not None and px > x]
    known = [x for x in (s20, s50, s200) if x is not None]
    label = "unknown"
    if px is not None and known:
        label = "up" if len(above) == len(known) else ("down" if not above else "mixed")
    return {"label": label, "sma20": _r(s20), "sma50": _r(s50), "sma200": _r(s200),
            "vs_sma50_pct": round((px / s50 - 1) * 100, 2) if px and s50 else None,
            "vs_sma200_pct": round((px / s200 - 1) * 100, 2) if px and s200 else None,
            "ret_5d_pct": ret_pct(closes, 5), "ret_20d_pct": ret_pct(closes, 20), "ret_60d_pct": ret_pct(closes, 60),
            "ret_250d_pct": ret_pct(closes, 250)}


def _r(x: Optional[float], d: int = 4) -> Optional[float]:
    return round(x, d) if isinstance(x, (int, float)) and math.isfinite(x) else None


def rel_strength(a: Sequence[float], b: Sequence[float], n: int) -> Optional[float]:
    ra, rb = ret_pct(a, n), ret_pct(b, n)
    return round(ra - rb, 3) if ra is not None and rb is not None else None


def realized_vol(closes: Sequence[float], n: int = 20) -> Optional[float]:
    r = DR.log_returns(closes[-(n + 1):])
    if len(r) < n:
        return None
    m = sum(r) / len(r)
    sd = math.sqrt(sum((x - m) ** 2 for x in r) / (len(r) - 1))
    return round(sd * math.sqrt(252) * 100, 2)


def reaction_check(event_day: str, after_close: bool, bars: Sequence[dict], live_price: float, sigma_1d: Optional[float]) -> Optional[dict]:
    """UNVALIDATED: the stock's move since the last close BEFORE the event vs its
    typical move over the same number of sessions (sigma_1d * sqrt(n))."""
    if not sigma_1d or live_price <= 0:
        return None
    days = [str(b.get("t", ""))[:10] for b in bars]
    # reaction starts on the event day, or the next session when filed after the close
    prior = [i for i, d in enumerate(days) if (d <= event_day if after_close else d < event_day)]
    if not prior:
        return None
    i0 = prior[-1]
    base = float(bars[i0]["c"])
    n = max(1, len(bars) - 1 - i0)
    move = math.log(live_price / base)
    exp_move = sigma_1d * math.sqrt(n)
    z = move / exp_move if exp_move > 0 else None
    return {"base_date": days[i0], "sessions_since": n, "move_since_pct": round((math.exp(move) - 1) * 100, 2),
            "typical_move_pct": round((math.exp(exp_move) - 1) * 100, 2), "z": round(z, 2) if z is not None else None,
            "small_reaction": bool(z is not None and abs(z) < REACTION_SMALL_Z and n <= REACTION_MAX_SESSIONS),
            "label": "UNVALIDATED — gate-2 pending (event-reaction study)"}


def parse_submissions(sub: dict, since: str) -> List[dict]:
    """EDGAR submissions JSON -> filings on/after `since`, newest first."""
    rec = ((sub or {}).get("filings") or {}).get("recent") or {}
    cols = {k: rec.get(k) or [] for k in ("form", "filingDate", "acceptanceDateTime", "items", "accessionNumber", "primaryDocument", "primaryDocDescription")}
    cik = str(sub.get("cik") or "").lstrip("0")
    out = []
    for i, form in enumerate(cols["form"]):
        fd = cols["filingDate"][i] if i < len(cols["filingDate"]) else ""
        if not fd or fd < since:
            continue
        acc = cols["accessionNumber"][i] if i < len(cols["accessionNumber"]) else ""
        doc = cols["primaryDocument"][i] if i < len(cols["primaryDocument"]) else ""
        items = [x.strip() for x in str(cols["items"][i] if i < len(cols["items"]) else "").split(",") if x.strip()]
        out.append({"form": form, "filing_date": fd, "accepted": cols["acceptanceDateTime"][i] if i < len(cols["acceptanceDateTime"]) else None,
                    "items": items, "accession": acc,
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}" if cik and acc and doc else None,
                    "description": cols["primaryDocDescription"][i] if i < len(cols["primaryDocDescription"]) else ""})
    out.sort(key=lambda f: (f["filing_date"], f.get("accepted") or ""), reverse=True)
    return out


def filing_event(f: dict) -> dict:
    base = str(f["form"]).upper().replace("/A", "")
    label, material = FORM_LABEL.get(base, (f"{f['form']} filing", False))
    items = [{"item": it, "label": ITEM_8K.get(it, (f"Item {it}", False))[0]} for it in f.get("items") or []]
    if base in ("8-K", "6-K") and items:
        material = any(ITEM_8K.get(it["item"], ("", False))[1] for it in items)
        label = "; ".join(it["label"] for it in items if it["item"] != "9.01") or label
    accepted = f.get("accepted") or ""
    after_close = False
    try:
        # EDGAR acceptance times are US Eastern wall clock; after 16:00 ET the market reacts next session
        hh = int(accepted[11:13]) if len(accepted) >= 13 else 0
        after_close = hh >= 16
    except ValueError:
        after_close = False
    return {"kind": "filing", "form": f["form"], "title": label, "items": items, "material": material,
            "time": accepted or f["filing_date"], "date": f["filing_date"], "after_close": after_close,
            "source": "SEC EDGAR", "tier": 1, "url": f.get("url")}


def news_event(a: dict) -> dict:
    src = a.get("source") or "unknown"
    ts = str(a.get("created_at") or a.get("updated_at") or "")
    return {"kind": "news", "title": str(a.get("headline") or "")[:220], "time": ts, "date": ts[:10],
            "after_close": len(ts) >= 13 and ts[11:13].isdigit() and int(ts[11:13]) >= 20,  # UTC 20:00 ~= 16:00 ET (EDT); approximate
            "source": src, "tier": credibility_tier(src), "url": a.get("url"), "author": a.get("author"), "material": True}


# ── shadow outcomes by score decile (pure) ──────────────────────────────────
def score_deciles(records: Sequence[dict]) -> Optional[dict]:
    """Shadow candidates with a settled +5d or +20d outcome, bucketed by score
    decile (edges from the same records). Each bucket: n, positive-return rate,
    mean return, the bot's +2%/-4% rule win rate, decisions; plus the all-records
    base rate so a bucket is read against it."""
    rows = []
    for r in records or []:
        s = r.get("score")
        oc = r.get("outcomes") or {}
        if not isinstance(s, (int, float)):
            continue
        o5, o20 = oc.get("+5d"), oc.get("+20d")
        if not (isinstance(o5, dict) or isinstance(o20, dict)):
            continue
        rows.append((float(s), o5 if isinstance(o5, dict) else None, o20 if isinstance(o20, dict) else None, str(r.get("decision") or "")))
    if len(rows) < 20:
        return {"n": len(rows), "buckets": [], "base": None, "thin": True}
    scores = sorted(x[0] for x in rows)
    edges = [scores[0]] + [scores[min(len(scores) - 1, int(len(scores) * k / 10))] for k in range(1, 10)] + [scores[-1]]

    def stats(sel):
        vals = [o for o in sel if o is not None and isinstance(o.get("return_pct"), (int, float))]
        n = len(vals)
        if not n:
            return {"n": 0, "pos_rate": None, "mean_pct": None, "rule_win_rate": None}
        pos = sum(1 for o in vals if o["return_pct"] > 0)
        lab = [o for o in vals if o.get("label") in (0, 1)]
        return {"n": n, "pos_rate": round(pos / n, 4), "mean_pct": round(sum(o["return_pct"] for o in vals) / n, 3),
                "rule_win_rate": round(sum(1 for o in lab if o["label"] == 1) / len(lab), 4) if lab else None}

    buckets = []
    for k in range(10):
        lo, hi = edges[k], edges[k + 1]
        sel = [x for x in rows if (lo <= x[0] < hi) or (k == 9 and x[0] == hi)]
        dec: Dict[str, int] = {}
        for x in sel:
            dec[x[3]] = dec.get(x[3], 0) + 1
        buckets.append({"decile": k + 1, "lo": round(lo, 2), "hi": round(hi, 2), "count": len(sel),
                        "d5": stats([x[1] for x in sel]), "d20": stats([x[2] for x in sel]),
                        "decisions": dict(sorted(dec.items(), key=lambda kv: -kv[1])[:5])})
    base = {"d5": stats([x[1] for x in rows]), "d20": stats([x[2] for x in rows])}
    return {"n": len(rows), "buckets": buckets, "base": base, "thin": len(rows) < 300}


# ── direction (pure over the forecast store) ────────────────────────────────
def direction_from_store(symbol: str, batches: List[dict], forecasts: List[dict], scores: Dict[str, dict], today: date) -> dict:
    import forecast as FC
    resolved = [(f, scores[f["id"]]) for f in forecasts if scores.get(f["id"], {}).get("status") == "resolved"]
    acc = FC.accuracy(resolved)
    batches = sorted(batches, key=lambda b: (b.get("week_key") or "", b.get("made_at") or ""))
    latest = batches[-1] if batches else None
    call = None
    if latest:
        for f in forecasts:
            if f.get("week_key") == latest.get("week_key") and f.get("symbol") == symbol:
                call = f
                break
    if call and str(call.get("horizon_end") or "") < today.isoformat():
        call = None  # the week's horizon has passed — no current call
    base = {"model_version": FC.MODEL_VERSION, "model_accuracy": {k: acc.get(k) for k in ("n", "thin", "hits", "hit_rate", "brier", "brier_skill_vs_coinflip", "in_range_rate")},
            "source": "weekly forecast screen (forecast.py, SHADOW, experimental)"}
    if not call:
        return {**base, "status": "none", "call": None,
                "statement": "No validated direction signal for this ticker — over a day or a week, direction is close to a coin flip."}
    p = float(call.get("p_up") or 0.5)
    bins = FC.calibration_bins([(float(f["p_up"]), int(s["outcome_up"])) for f, s in resolved])
    mine = next((b for b in bins if b.get("lo", 0) <= p < b.get("hi", 1) or (p == 1.0 and b.get("hi") == 1.0)), None)
    own = [(f, s) for f, s in resolved if f.get("symbol") == symbol]
    return {**base, "status": "call",
            "call": {"direction": call.get("direction"), "p_up": round(p, 4), "p_call": call.get("p_call"), "base_date": call.get("base_date"),
                     "horizon_end": call.get("horizon_end"), "week_key": call.get("week_key"), "made_at": call.get("made_at"),
                     "range_lo": call.get("range_lo"), "range_hi": call.get("range_hi")},
            "calibration_bin": mine, "ticker_resolved": len(own), "ticker_hits": sum(int(s["hit"]) for _, s in own),
            "statement": "Experimental weekly model — read its measured hit rate and calibration before trusting the call."}


# ── options (pure over contracts) ───────────────────────────────────────────
def options_summary(contracts: List[dict], price: float, sigma5: Optional[float], sigma1: Optional[float]) -> dict:
    by_exp: Dict[str, List[dict]] = {}
    for c in contracts or []:
        if isinstance(c.get("iv"), (int, float)) and c["iv"] > 0 and c.get("exp_date"):
            by_exp.setdefault(str(c["exp_date"]), []).append(c)
    term = []
    for exp, cs in sorted(by_exp.items()):
        k_atm = min((float(c.get("strike") or 0) for c in cs), key=lambda k: abs(k - price))
        near = [c for c in cs if float(c.get("strike") or 0) == k_atm]   # the call and put at the nearest strike
        ivs = [float(c["iv"]) for c in near]
        if ivs:
            term.append({"exp_date": exp, "days_out": int(near[0].get("days_out") or 0), "atm_iv": round(sum(ivs) / len(ivs), 4), "n": len(cs)})
    if not term:
        return {"contracts": len(contracts or []), "term": [], "atm_iv": None}
    front = min((t for t in term if t["days_out"] >= 2), key=lambda t: abs(t["days_out"] - 7), default=term[0])
    month = min(term, key=lambda t: abs(t["days_out"] - 30))
    iv = front["atm_iv"]
    m1 = iv * math.sqrt(1 / 252)
    m5 = iv * math.sqrt(5 / 252)
    skew = None
    cs = by_exp.get(month["exp_date"], [])
    puts = [float(c["iv"]) for c in cs if c.get("opt_type") == "put" and 0.88 <= float(c.get("moneyness") or 1) <= 0.95]
    calls = [float(c["iv"]) for c in cs if c.get("opt_type") == "call" and 1.05 <= float(c.get("moneyness") or 1) <= 1.12]
    if puts and calls:
        skew = round((sum(puts) / len(puts) - sum(calls) / len(calls)) * 100, 2)
    out = {"contracts": len(contracts), "term": term[:8], "front": front, "atm_iv": round(iv * 100, 2), "atm_iv_30d": round(month["atm_iv"] * 100, 2),
           "implied_move_1d_pct": round((math.exp(m1) - 1) * 100, 2), "implied_move_5d_pct": round((math.exp(m5) - 1) * 100, 2),
           "skew_put_minus_call_pts": skew, "skew_strikes": "OTM puts at 88–95% of spot vs OTM calls at 105–112%, expiry nearest 30 days"}
    if sigma5:
        out["model_move_5d_pct"] = round((math.exp(sigma5) - 1) * 100, 2)
        out["implied_vs_model_5d"] = round(m5 / sigma5, 3)
    if sigma1:
        out["model_move_1d_pct"] = round((math.exp(sigma1) - 1) * 100, 2)
    return out


# ── I/O (defaults; injectable for tests) ────────────────────────────────────
def _io_fetch_bars(symbols: List[str], start: str, end: str) -> Tuple[Dict[str, List[dict]], set]:
    return DR._default_fetch(symbols, start, end, time.monotonic() + 40)


def _io_regime() -> Optional[dict]:
    import regime_engine
    return regime_engine.load_state()


def _io_sector(symbol: str) -> Optional[str]:
    import bot_engine
    return bot_engine._get_sector(symbol)


def _cache_path(name: str) -> str:
    try:
        from storage_config import CACHE_DIR
    except ImportError:
        CACHE_DIR = "/tmp"
    return os.path.join(CACHE_DIR, name)


def _cached_json(name: str, ttl: float, loader: Callable[[], Any]) -> Any:
    p = _cache_path(name)
    try:
        if time.time() - os.path.getmtime(p) < ttl:
            with open(p) as f:
                return json.load(f)
    except (OSError, ValueError) as e:
        log.debug("dossier cache miss %s (%s)", name, e)
    v = loader()
    try:
        with open(p, "w") as f:
            json.dump(v, f)
    except OSError as e:
        log.debug("dossier cache write failed %s (%s)", name, e)
    return v


def _io_edgar(symbol: str) -> Optional[dict]:
    import sec_filings_text as sft
    client = sft.EdgarClient(max_retries=1, timeout=12)
    tickers = _cached_json("dossier_sec_tickers.json", SEC_TICKERS_TTL_S, lambda: client.get_json(sft.COMPANY_TICKERS_URL))
    cik = None
    for row in (tickers or {}).values():
        if str(row.get("ticker", "")).upper() == symbol:
            cik = sft.cik10(row["cik_str"])
            break
    if not cik:
        return None
    return _cached_json(f"dossier_sec_sub_{cik}.json", SEC_SUBMISSIONS_TTL_S,
                        lambda: client.get_json(f"https://data.sec.gov/submissions/CIK{cik}.json"))


def _io_news(symbol: str) -> List[dict]:
    import requests
    h = {"APCA-API-KEY-ID": os.environ.get("ALPACA_KEY", ""), "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET", "")}
    r = requests.get(f"https://data.alpaca.markets/v1beta1/news?limit={NEWS_LIMIT}&sort=desc&symbols={symbol}", headers=h, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Alpaca news HTTP {r.status_code}")
    return list(r.json().get("news") or [])


def _io_chain(symbol: str, price: float) -> List[dict]:
    from options_scanner import _fetch_options_chain
    return _fetch_options_chain(symbol, price, 1, 45) or []


def _io_earnings(symbol: str) -> Optional[dict]:
    import finnhub_data
    if not finnhub_data.FINNHUB_KEY:
        return {"status": "unavailable", "reason": "no Finnhub key configured"}
    cal = finnhub_data.get_earnings_calendar(symbol) or []
    rows = [c for c in cal if isinstance(c, dict) and str(c.get("symbol", "")).upper() == symbol]
    if not rows:
        return {"status": "none", "reason": "no earnings date in the next 30 days (Finnhub calendar)"}
    nxt = sorted(rows, key=lambda c: str(c.get("date")))[0]
    return {"status": "live", "date": nxt.get("date"), "hour": nxt.get("hour"), "source": "Finnhub earnings calendar"}


def _io_forecast_store():
    import forecast as FC
    return FC.load_store()


_SHADOW_CACHE: Dict[str, Any] = {"at": 0.0, "mtime": None, "value": None}


def _io_shadow_deciles() -> Optional[dict]:
    import shadow_portfolio as SP
    try:
        mt = os.path.getmtime(SP.SHADOW_LOG_PATH)
    except OSError:
        return None
    if _SHADOW_CACHE["value"] is not None and _SHADOW_CACHE["mtime"] == mt and time.time() - _SHADOW_CACHE["at"] < SHADOW_TTL_S:
        return _SHADOW_CACHE["value"]
    v = score_deciles(SP._load_shadow_log())
    _SHADOW_CACHE.update(at=time.time(), mtime=mt, value=v)
    return v


DEFAULT_IO: Dict[str, Callable[..., Any]] = {
    "bars": _io_fetch_bars, "regime": _io_regime, "sector": _io_sector, "edgar": _io_edgar, "news": _io_news,
    "chain": _io_chain, "earnings": _io_earnings, "forecast_store": _io_forecast_store, "shadow": _io_shadow_deciles,
    "record": DR.record, "ledger_report": DR.ledger_report,
}


def _layer(status: str, source: str, as_of: Optional[str] = None, reason: Optional[str] = None, **kw) -> dict:
    return {"status": status, "source": source, "as_of": as_of, "reason": reason, **kw}


# ── the orchestrator ────────────────────────────────────────────────────────
def build(symbol: str, now: Optional[datetime] = None, io: Optional[Dict[str, Callable[..., Any]]] = None) -> dict:
    """The dossier for `symbol`. JSON-safe; never raises."""
    now = now or datetime.now(timezone.utc)
    io = {**DEFAULT_IO, **(io or {})}
    sym = str(symbol or "").strip().upper()
    t0 = time.time()
    out: Dict[str, Any] = {"symbol": sym, "generated_at": _now_iso(now), "errors": {}, "timings": {},
                           "credibility_tiers": CREDIBILITY_TIERS, "research_not_advice": True}
    if not SYMBOL_RE.match(sym):
        out["errors"]["symbol"] = "a stock symbol is 1–6 letters"
        return out

    # bars: the stock, SPY, the sector candidates
    try:
        resolved = io["sector"](sym)
    except Exception as e:  # sector cache / bot_engine import is best-effort
        resolved = None
        out["errors"]["sector_resolve"] = _err(e)
    sec = sector_for(sym, resolved)
    want = [sym, "SPY"] + [e for e in sec["etfs"] if e not in (sym, "SPY")]
    start = (now.date() - timedelta(days=HISTORY_CAL_DAYS)).isoformat()
    try:
        raw, unfetched = io["bars"](want, start, now.date().isoformat())
    except Exception as e:
        raw, unfetched = {}, set(want)
        out["errors"]["bars"] = _err(e)
    from forecast import completed_bars
    all_bars = {k: sorted(v or [], key=lambda b: str(b.get("t"))) for k, v in (raw or {}).items()}
    done = {k: completed_bars(v, now) for k, v in all_bars.items()}
    out["timings"]["bars_s"] = round(time.time() - t0, 2)

    # MARKET
    try:
        st = io["regime"]()
    except Exception as e:
        st = None
        out["errors"]["market"] = _err(e)
    if st:
        out["market"] = _layer("live", "regime engine (regime_engine.py persisted state, gating index SPY)", st.get("as_of"),
                               composite=st.get("composite_regime"), legacy_label=st.get("legacy_label"),
                               exposure=st.get("exposure_multiplier"), exposure_continuous=st.get("exposure_continuous"),
                               trend=st.get("trend_state"), fast=st.get("fast_state"), vol=st.get("vol_state"),
                               breadth=st.get("breadth_state"), confidence=st.get("confidence"), days_in_regime=st.get("days_in_regime"),
                               panic=st.get("panic_state"), stress=st.get("stress_flags") or {}, allowed=st.get("allowed_strategies") or [],
                               computed_at=st.get("computed_at"))
        spy = _closes(done.get("SPY") or [])
        if spy:
            out["market"]["spy"] = {"close": spy[-1], "trend": trend_of(spy)}
    else:
        out["market"] = _layer("unavailable", "regime engine", reason="no persisted regime state yet")

    sb = done.get(sym) or []
    if len(sb) < 30:
        out["errors"]["stock"] = "no price history for this symbol" if sym in unfetched or not sb else "under 30 sessions of history"
        out["stock"] = _layer("unavailable", "Alpaca daily bars", reason=out["errors"]["stock"])
        out["timings"]["total_s"] = round(time.time() - t0, 2)
        return out
    closes = _closes(sb)
    spy_c = _closes(done.get("SPY") or [])
    last_bar = (all_bars.get(sym) or sb)[-1]
    live_price = float(last_bar["c"])
    live_day = str(last_bar.get("t", ""))[:10]
    base_date = str(sb[-1].get("t"))[:10]
    spot = closes[-1]

    # SECTOR
    etf, etf_c = None, []
    for e in sec["etfs"]:
        c = _closes(done.get(e) or [])
        if len(c) >= 61:
            etf, etf_c = e, c
            break
    if etf:
        out["sector"] = _layer("live", sec["source"], str((done.get(etf) or [{}])[-1].get("t", ""))[:10],
                               key=sec["key"], label=sec["label"], etf=etf, tried=sec["etfs"], trend=trend_of(etf_c),
                               rs_vs_spy_20d=rel_strength(etf_c, spy_c, 20), rs_vs_spy_60d=rel_strength(etf_c, spy_c, 60),
                               stock_vs_sector_20d=rel_strength(closes, etf_c, 20), stock_vs_sector_60d=rel_strength(closes, etf_c, 60))
    else:
        out["sector"] = _layer("unavailable" if sec["key"] != "index" else "partial", sec["source"], key=sec["key"], label=sec["label"], tried=sec["etfs"],
                               reason=("no sector mapping for this ticker — read the market and the stock layers" if not sec["etfs"]
                                       else f"no usable bars for {', '.join(sec['etfs'])}") if sec["key"] != "index" else "this is a broad index ETF; the market layer is its sector",
                               stock_vs_spy_20d=rel_strength(closes, spy_c, 20), stock_vs_spy_60d=rel_strength(closes, spy_c, 60))

    # STOCK + RANGES
    t1 = time.time()
    sig = DR.sigma_forecast(closes)
    h1, h5 = sig["horizons"]["1"], sig["horizons"]["5"]
    ranges = {}
    for h, hz in (("1", h1), ("5", h5)):
        b = DR.bands_for(spot, hz.get("sigma"))
        ranges[h] = {"h": int(h), "model": hz.get("model"), "fallback_reason": hz.get("fallback_reason"),
                     "sigma_pct": round(hz["sigma"] * 100, 3) if hz.get("sigma") else None, "bands": b,
                     "backtest": DR.walk_forward_coverage(closes, int(h))}
    hi52 = max(closes[-252:])
    out["stock"] = _layer("live", "Alpaca daily bars (split-adjusted) · dossier_ranges " + DR.MODEL_VERSION, base_date,
                          price=round(live_price, 4), price_day=live_day, price_is_partial_session=live_day != base_date,
                          base_close=round(spot, 4), base_date=base_date, trend=trend_of(closes),
                          realized_vol_20d_pct=realized_vol(closes, 20), realized_vol_60d_pct=realized_vol(closes, 60),
                          high_52w=round(hi52, 4), low_52w=round(min(closes[-252:]), 4), vs_spy_20d=rel_strength(closes, spy_c, 20),
                          vs_spy_60d=rel_strength(closes, spy_c, 60), sessions=len(closes),
                          ranges=ranges, expected_move=DR.expected_move_context(closes, h1.get("sigma")),
                          range_method=("68% and 90% bands of a zero-drift lognormal around the last close, sigma from a HAR-RV model "
                                        "on daily squared returns (EWMA lambda 0.94 when HAR is unavailable or implausible). Daily returns "
                                        "are fat-tailed: the backtest and the live ledger show how often the bands actually held."))
    out["timings"]["ranges_s"] = round(time.time() - t1, 2)

    # REGIME GUARDS for this ticker (would the bot's regime layer block it?)
    if st:
        try:
            import regime_engine
            out["bot_guards"] = regime_engine.selection_verdict(st, {"price": live_price, "high_52w": hi52, "closes": closes[-60:],
                                                                     "relative_strength_pct": rel_strength(closes, spy_c, 63)})
            out["bot_guards"]["source"] = "regime_engine.selection_verdict (read-only; mean-reversion/dip inputs not computed here)"
        except Exception as e:
            out["errors"]["bot_guards"] = _err(e)

    # DIRECTION
    try:
        batches, forecasts, scores = io["forecast_store"]()
        out["direction"] = direction_from_store(sym, batches, forecasts, scores, now.date())
    except Exception as e:
        out["errors"]["direction"] = _err(e)
        out["direction"] = {"status": "none", "call": None, "statement": "No validated direction signal for this ticker — near a coin flip."}

    # BOT TRACK RECORD by score decile
    try:
        out["bot_track"] = io["shadow"]()
        if out["bot_track"] is not None:
            out["bot_track"]["source"] = "shadow portfolio (every scanned candidate, all decisions; +5d/+20d outcomes, nightly backfill)"
    except Exception as e:
        out["errors"]["bot_track"] = _err(e)

    # OPTIONS
    t2 = time.time()
    try:
        contracts = io["chain"](sym, live_price)
        if contracts:
            o = options_summary(contracts, live_price, h5.get("sigma"), h1.get("sigma"))
            out["options"] = _layer("live" if o.get("atm_iv") else "partial", "Alpaca options chain (indicative feed; options_scanner)", _now_iso(now),
                                    reason=None if o.get("atm_iv") else "contracts without usable IV", **o)
        else:
            out["options"] = _layer("unavailable", "Alpaca options chain", reason="no listed options in the 1–45 day window (or the chain fetch returned nothing)")
    except Exception as e:
        out["errors"]["options"] = _err(e)
        out["options"] = _layer("unavailable", "Alpaca options chain", reason=_err(e))
    try:
        out["options"]["earnings"] = io["earnings"](sym)
    except Exception as e:
        out["options"]["earnings"] = {"status": "unavailable", "reason": _err(e)}
    out["timings"]["options_s"] = round(time.time() - t2, 2)

    # EVENTS
    t3 = time.time()
    events: List[dict] = []
    ev_status = []
    since = (now.date() - timedelta(days=EVENT_WINDOW_DAYS)).isoformat()
    form4 = 0
    try:
        sub = io["edgar"](sym)
        if sub:
            for f in parse_submissions(sub, since):
                if str(f["form"]) == "4":
                    form4 += 1
                    if form4 > 5:
                        continue
                events.append(filing_event(f))
            ev_status.append("edgar")
        else:
            out["errors"]["edgar"] = "ticker not in SEC company_tickers.json (no EDGAR filer match)"
    except Exception as e:
        out["errors"]["edgar"] = _err(e)
    try:
        for a in io["news"](sym):
            ev = news_event(a)
            if ev["date"] >= since:
                events.append(ev)
        ev_status.append("news")
    except Exception as e:
        out["errors"]["news"] = _err(e)
    for ev in events:
        if ev.get("material"):
            ev["reaction"] = reaction_check(ev["date"], bool(ev.get("after_close")), sb, live_price, h1.get("sigma"))
    events.sort(key=lambda e: str(e.get("time") or ""), reverse=True)
    out["events"] = _layer("live" if len(ev_status) == 2 else ("partial" if ev_status else "unavailable"),
                           "SEC EDGAR submissions + Alpaca news", _now_iso(now),
                           reason=None if len(ev_status) == 2 else "; ".join(f"{k}: {out['errors'][k]}" for k in ("edgar", "news") if k in out["errors"]),
                           items=events[:40], form4_count=form4, window_days=EVENT_WINDOW_DAYS,
                           reaction_note=("Reaction check: the move since the last close before the event vs the stock's typical move over "
                                          "the same sessions. UNVALIDATED — a small reaction is NOT evidence the news is unpriced until the "
                                          "event-reaction study passes gate 2. Form 4 clusters and contract awards failed gate 2 here: raw events, not signals."))
    out["timings"]["events_s"] = round(time.time() - t3, 2)

    # TRADE PLAN
    plan = DR.trade_plan(live_price, h5.get("sigma_daily"))
    out["plan"] = plan
    # LEDGER: record what was served, then report the running record
    served = _now_iso(now)
    rows = DR.range_rows(sym, base_date, spot, sig, served)
    if plan:
        rows.append(DR.plan_row(sym, now.astimezone(timezone.utc).date().isoformat(), base_date, spot, plan, served))
    try:
        out["ledgered"] = io["record"](rows)
        out["ledger"] = io["ledger_report"](sym)
    except Exception as e:
        out["errors"]["ledger"] = _err(e)
    out["timings"]["total_s"] = round(time.time() - t0, 2)
    return out


def score_ledger() -> dict:
    """Nightly entry point (bot.ts, 22:00 UTC slot)."""
    return DR.score_ledger()


if __name__ == "__main__":  # python3 ticker_dossier.py RKLB
    import sys
    print(json.dumps(build(sys.argv[1] if len(sys.argv) > 1 else "SPY"), indent=1, default=str))
