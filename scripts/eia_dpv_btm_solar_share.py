#!/usr/bin/env python3
"""eia_dpv_btm_solar_share.py — FUSION HYPOTHESIS (b)'s gate-1 (DATA)
follow-up to SWPP solar's residual overshoot, resolving the 2026-09-14
(third session) entry's own filed NEXT(1): "EIA's separately published
small-scale (behind-the-meter) solar capacity estimates for SWPP's/ERCO's
footprint states, to test candidate mechanism (ii) directly."

BACKGROUND (research/open_questions.md, FUSION HYPOTHESES (b)): after
registry-completeness (ruled out) and registry-currency/EIA-860M (closed
ERCO, narrowed SWPP to 1.275x-1.286x across sessions) and the AC/DC
nameplate ambiguity (ruled out against EIA-860's own form instructions,
same UTC day, earlier session), the one remaining candidate mechanism is
(ii): EIA-930's SWPP respondent folds some behind-the-meter (BTM) /
distributed solar into its reported `SUN` hourly total, while our registry
(built from EIA-860, which covers utility-scale generators only) has no
row for BTM capacity at all — so a real physical quantity is on the
numerator (EIA-930 generation) but structurally missing from the
denominator (registry nameplate).

WHAT THIS SCRIPT DOES: pulls EIA's own "estimated small scale solar
photovoltaic" (DPV) generation series — a REAL, separately published,
free EIA statistic (Form EIA-923, `electric-power-operational-data`
route, distinct from the EIA-930 respondent-level series
`grid_generation_gate1_ba.py` reads) — alongside utility-scale `SUN`
generation for the same states, and reports what SHARE of total
state-level solar generation DPV represents. This is a MAGNITUDE
plausibility check, not a mechanism confirmation: it cannot itself prove
SWPP's specific EIA-930 respondent total includes BTM (that would need
SWPP's own internal accounting documentation, which the immediately
preceding session's WebSearch pass could not find — general EIA guidance
says BTM is "often" excluded from BA-level hourly reporting, not that it
always is). What it CAN do is answer a narrower, still-useful question:
is DPV's share of state solar generation even in the right ballpark to
explain a ~27-29% overshoot, or is it two orders of magnitude too small
to be a plausible mechanism at all? A "too small" result would rule out
(ii) the same clean way the AC/DC check ruled out (i); a "right ballpark"
result narrows the search without closing it.

SCOPE CAVEAT, stated up front (REASONING STANDARD #7/#10): this is
STATE-level EIA-923 data, not SWPP-BA-level. SWPP's real RTO footprint
spans all or part of 14 states and does not follow state lines (the same
ERCO/SWPP Texas-Panhandle seam `scripts/grid_ba_capacity.py`'s own
docstring already names, and the same class of approximation
`research/open_questions.md`'s 2026-09-11 entry rejected for a
CAPACITY-ceiling check specifically). Using it here for a SHARE
(a ratio, not an absolute MW figure attributed to SWPP) is a much weaker
claim than attributing state totals to SWPP's own capacity — the same
reasoning `eia860_regional_capacity_check.py` used the EIA-860 BA-code
column to avoid, which does not exist in this monthly generation route.
Reported honestly as a magnitude sanity check over SWPP's core,
least-ambiguous states (OK, KS, NE — all three sit almost entirely
inside SWPP territory, unlike TX/MO/ND/SD/MN/IA/AR/LA/NM/MT/WY, which
straddle multiple BAs) rather than a full 14-state footprint sum that
would reintroduce exactly the cross-BA attribution ambiguity this
thread already spent two sessions resolving for the CAPACITY side.

Usage:
    EIA_API_KEY=... python3 scripts/eia_dpv_btm_solar_share.py \
        --states OK,KS,NE --overshoot-ratio 1.275
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

EIA_OPERATIONAL_DATA_URL = "https://api.eia.gov/v2/electricity/electric-power-operational-data/data/"
DEFAULT_STATES = ("OK", "KS", "NE")  # SWPP's least cross-BA-ambiguous core states — see module docstring


def parse_generation_rows(api_rows):
    """api_rows: EIA v2 response.data entries {location, fueltypeid, period,
    generation, ...}. Returns {(location, fueltypeid, period): mwh}, and a
    count of rows with a non-numeric generation value (kept out of the dict,
    never silently dropped — surfaced via the count, same convention as
    eia930_solar_exceedance_pattern.parse_rows)."""
    out = {}
    unparseable = 0
    for row in api_rows:
        loc = row.get("location")
        fuel = row.get("fueltypeid")
        period = row.get("period")
        raw = row.get("generation")
        if raw in (None, ""):
            continue
        try:
            out[(loc, fuel, period)] = float(raw)
        except (TypeError, ValueError):
            unparseable += 1
    return out, unparseable


def latest_common_period(gen_by_key, states, fuels=("DPV", "SUN")):
    """Returns the most recent period for which EVERY (state, fuel) pair in
    `states` x `fuels` has a value — comparing a period where one state's
    figure hasn't landed yet against another's would silently bias the
    share computed below. None if no such period exists."""
    periods = {period for (_loc, _fuel, period) in gen_by_key}
    for period in sorted(periods, reverse=True):
        if all((state, fuel, period) in gen_by_key for state in states for fuel in fuels):
            return period
    return None


def dpv_btm_share(gen_by_key, states, period):
    """For the given period, sums DPV and SUN generation across `states` and
    returns the DPV share of (DPV + SUN) per state and in aggregate. A share
    near the observed gate-1 overshoot's implied excess fraction is
    SUGGESTIVE of mechanism (ii); a share two orders of magnitude smaller
    RULES IT OUT the same way the AC/DC check ruled out mechanism (i) — see
    module docstring for why this cannot be read as confirmation either
    way."""
    per_state = {}
    total_dpv = 0.0
    total_sun = 0.0
    for state in states:
        dpv = gen_by_key.get((state, "DPV", period))
        sun = gen_by_key.get((state, "SUN", period))
        if dpv is None or sun is None:
            per_state[state] = {"dpv_mwh": dpv, "sun_mwh": sun, "dpv_share": None}
            continue
        total_dpv += dpv
        total_sun += sun
        denom = dpv + sun
        per_state[state] = {
            "dpv_mwh": round(dpv, 1),
            "sun_mwh": round(sun, 1),
            "dpv_share": round(dpv / denom, 4) if denom else None,
        }
    agg_denom = total_dpv + total_sun
    return {
        "period": period,
        "per_state": per_state,
        "aggregate": {
            "total_dpv_mwh": round(total_dpv, 1),
            "total_sun_mwh": round(total_sun, 1),
            "dpv_share": round(total_dpv / agg_denom, 4) if agg_denom else None,
        },
    }


def implied_overshoot_from_btm_share(dpv_share):
    """If a BA's reported SUN total were (utility-scale + BTM) while the
    registry denominator is utility-scale-only, the overshoot ratio implied
    by folding in BTM at share s is 1 / (1 - s) — e.g. s=0.20 implies a
    1.25x ratio. Returns None if dpv_share is None or >= 1 (undefined).
    Purely arithmetic — does NOT assert this is what SWPP's respondent
    actually does; it converts the measured DPV share into the same units
    as the gate-1 overshoot ratio so the two can be compared directly."""
    if dpv_share is None or dpv_share >= 1:
        return None
    return round(1 / (1 - dpv_share), 4)


# ── Network (excluded from unit tests, exercised only by running this file directly) ──

def fetch_state_generation(states, fuels, api_key, months_back=3, timeout=30):
    """Monthly DPV/SUN generation for the given states. The route has no
    per-state date range this repo's other scripts rely on carrying
    forward, so `start` is derived from `months_back` (this series runs
    back to 2014, and an unbounded query returns hundreds of rows even for
    a handful of states/fuels — confirmed live this session, not assumed)
    to keep a single page correct."""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    start = (now.replace(day=1) - timedelta(days=31 * months_back)).strftime("%Y-%m")
    params = [
        ("api_key", api_key),
        ("frequency", "monthly"),
        ("data[0]", "generation"),
        ("start", start),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "desc"),
        ("length", "5000"),
    ]
    for f in fuels:
        params.append(("facets[fueltypeid][]", f))
    for s in states:
        params.append(("facets[location][]", s))
    params.append(("facets[sectorid][]", "99"))  # "All Sectors" — matches gate-1's own all-sector framing
    url = f"{EIA_OPERATIONAL_DATA_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    resp = body.get("response") or {}
    data = resp.get("data") or []
    total = resp.get("total")
    if total is not None and int(total) > len(data):
        raise RuntimeError(f"fetch_state_generation: response truncated ({len(data)} of {total} rows)")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=",".join(DEFAULT_STATES))
    ap.add_argument("--overshoot-ratio", type=float, default=None,
                     help="the gate-1 overshoot ratio to compare against (e.g. SWPP solar's 1.275-1.286x across sessions)")
    args = ap.parse_args()

    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        print("EIA_API_KEY not set — cannot run live fetch", file=sys.stderr)
        sys.exit(1)

    states = [s.strip() for s in args.states.split(",") if s.strip()]
    api_rows = fetch_state_generation(states, ("DPV", "SUN"), api_key)
    gen_by_key, unparseable = parse_generation_rows(api_rows)
    if unparseable:
        print(f"[eia_dpv_btm_solar_share] {unparseable} row(s) had a non-numeric generation value, excluded", file=sys.stderr)

    period = latest_common_period(gen_by_key, states)
    if period is None:
        print(json.dumps({"check": "eia_dpv_btm_solar_share", "error": "no period has data for every requested state"}, indent=2))
        sys.exit(1)

    share = dpv_btm_share(gen_by_key, states, period)
    implied = implied_overshoot_from_btm_share(share["aggregate"]["dpv_share"])

    report = {
        "check": "eia_dpv_btm_solar_share",
        "note": ("magnitude plausibility check for FUSION HYPOTHESIS (b) mechanism (ii) — "
                 "NOT a confirmation; state-level EIA-923 DPV/SUN share used as a proxy "
                 "for how large a BTM contribution COULD plausibly be, compared against "
                 "the gate-1 overshoot ratio via 1/(1-share). See module docstring for the "
                 "state-vs-BA scope caveat."),
        "states": states,
        "eia_rows_fetched": len(api_rows),
        **share,
        "implied_overshoot_ratio_if_sun_includes_btm": implied,
        "observed_gate1_overshoot_ratio": args.overshoot_ratio,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
