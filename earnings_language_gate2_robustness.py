#!/usr/bin/env python3
"""earnings_language_gate2_robustness.py — post-hoc ROBUSTNESS DIAGNOSTICS for
the SEC 8-K earnings-language gate 2 (scripts/earnings_language_gate2.py).

Reads that script's `--out` JSON (per-horizon `samples`). It NEVER changes the
pre-registered PASS/FAIL verdict — it only reports whether a mechanical PASS
survives the checks CLAUDE.md REASONING STANDARD #2/#4/#8 demand:
  - Spearman rank correlation + one-sided permutation p-value (outlier-proof)
  - date-demeaned alpha (strips same-day market/cluster effects)
  - split by filing calendar MONTH (replication across sub-periods/regimes)
Verdict: REPLICATES only if the pooled permutation p < 0.05 AND every month
with n >= MIN_MONTH_N has a same-signed Spearman >= 0.05; else FRAGILE.

Usage:
  python3 scripts/earnings_language_gate2.py \
      --api-url "https://voltradeai.com/api/data/earnings-language/history?days=90" \
      --out g2.json   # NB: --days is ignored on the API path; put ?days=90 in the URL
  python3 scripts/earnings_language_gate2_robustness.py g2.json
"""
from __future__ import annotations

import json
import random
import statistics as st
import sys
from collections import defaultdict

MIN_MONTH_N = 30
PERM = 5000
MIN_MONTH_RHO = 0.05


def _rank(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for k, i in enumerate(order):
        r[i] = float(k)
    return r


def _pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    vx = sum((a - mx) ** 2 for a in x) ** 0.5
    vy = sum((b - my) ** 2 for b in y) ** 0.5
    if vx == 0 or vy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (vx * vy)


def spearman(x: list[float], y: list[float]) -> float | None:
    return _pearson(_rank(x), _rank(y))


def perm_p(x: list[float], y: list[float], n_perm: int = PERM, seed: int = 1) -> float | None:
    obs = spearman(x, y)
    if obs is None:
        return None
    rng = random.Random(seed)
    rx, ry = _rank(x), _rank(y)
    hits = 0
    for _ in range(n_perm):
        p = ry[:]
        rng.shuffle(p)
        r = _pearson(rx, p)
        if r is not None and r >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def date_demeaned_alpha(samples: list[dict]) -> list[float]:
    by_date: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        by_date[s["filedAt"]].append(s["alpha"])
    return [s["alpha"] - st.mean(by_date[s["filedAt"]]) for s in samples]


def horizon_report(samples: list[dict]) -> dict:
    tone = [s["tone_per_1000w"] for s in samples]
    alpha = [s["alpha"] for s in samples]
    rho = spearman(tone, alpha)
    p = perm_p(tone, alpha)
    rho_dm = spearman(tone, date_demeaned_alpha(samples)) if samples else None
    months: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        months[s["filedAt"][:7]].append(s)
    by_month = {}
    for m, ss in sorted(months.items()):
        if len(ss) >= MIN_MONTH_N:
            by_month[m] = {"n": len(ss),
                           "spearman": round(spearman([s["tone_per_1000w"] for s in ss],
                                                      [s["alpha"] for s in ss]) or 0.0, 3)}
    month_ok = bool(by_month) and all(v["spearman"] >= MIN_MONTH_RHO for v in by_month.values())
    replicates = p is not None and p < 0.05 and month_ok
    return {
        "n": len(samples),
        "spearman": round(rho, 3) if rho is not None else None,
        "perm_p": round(p, 4) if p is not None else None,
        "spearman_date_demeaned": round(rho_dm, 3) if rho_dm is not None else None,
        "by_month": by_month,
        "verdict": "REPLICATES" if replicates else "FRAGILE",
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    with open(argv[1]) as f:
        data = json.load(f)
    out = {h: horizon_report(ss) for h, ss in sorted(data.get("samples", {}).items())}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
