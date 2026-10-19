"""
Serve-time regime inputs for the ML feature vector, computed with the SAME
formulas the training pipeline uses (ml_model_v2._compute_features), from
the macro snapshot (macro_data.get_macro_snapshot()).

REPAIR 2026-10-16 (bot cockpit audit, S3 rows 13/15/16): deep_score read
`intel.get("vxx_ratio", 1.0)` — intelligence.py never emits that key, so
vxx_ratio was pinned at 1.0 on every live candidate (20,000 / 20,000 shadow
records), the bull regime model was unreachable, and markov_state /
regime_score came from the Markov module fed an EMPTY return list, while
training computes them from five SPY daily returns and a closed-form score.
macro_data already carries the real readings (vxx_ratio = VXX close / 30-d
average, exactly training's definition; spy_closes_60d; spy_vs_ma50).

Pure function, no I/O. Tested in test_regime_inputs.py.
"""
from __future__ import annotations

# Training thresholds (ml_model_v2.py, _compute_features): markov_state is
# 2 if the mean of the last five daily SPY returns > +0.2 %, 0 if < −0.2 %,
# else 1. regime_score is the closed form below, clipped to [0, 100].
MARKOV_RETURN_THRESHOLD = 0.002
MARKOV_WINDOW = 5


def _f(v, default: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if x != x:  # NaN
        return default
    return x


def regime_inputs(macro) -> dict:
    """Return the regime block of the feature vector plus provenance.

    Keys: vxx_ratio, spy_vs_ma50, markov_state, regime_score (the four
    feature values, training definitions), spy_returns (all daily returns
    derivable from the snapshot, oldest first, as fractions), spy_returns_5
    (the training window), vxx_source ("macro" when a real VXX series was
    behind the ratio, "default" when macro_data substituted 1.0),
    spy_closes_n (how many closes were available).
    """
    m = macro if isinstance(macro, dict) else {}
    vxx = _f(m.get("vxx_ratio"), 1.0)
    if vxx <= 0:
        vxx = 1.0
    vxx_source = "macro" if (m.get("vxx_closes") and vxx != 1.0) or (m.get("vxx_closes") and m.get("vxx_latest")) else "default"
    spy_ma = _f(m.get("spy_vs_ma50"), 1.0)
    if spy_ma <= 0:
        spy_ma = 1.0

    closes = []
    for c in (m.get("spy_closes_60d") or []):
        x = _f(c, 0.0)
        if x > 0:
            closes.append(x)
    rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
    rets5 = rets[-MARKOV_WINDOW:]
    if len(rets5) >= 3:
        mean = sum(rets5) / len(rets5)
        markov_state = 2.0 if mean > MARKOV_RETURN_THRESHOLD else (0.0 if mean < -MARKOV_RETURN_THRESHOLD else 1.0)
    else:
        markov_state = 1.0

    regime_score = (spy_ma - 0.94) / 0.12 * 50 + (1.3 - vxx) / 0.6 * 30 + 20
    regime_score = max(0.0, min(100.0, regime_score))

    return {
        "vxx_ratio": round(vxx, 4),
        "vxx_source": vxx_source,
        "spy_vs_ma50": round(spy_ma, 4),
        "spy_returns": rets,
        "spy_returns_5": rets5,
        "markov_state": markov_state,
        "regime_score": round(regime_score, 2),
        "spy_closes_n": len(closes),
    }
