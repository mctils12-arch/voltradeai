"""
AlphaDesk — standalone tax estimator (capital gains + income aware).

A real, account-aware estimate of the US tax on a year's realized trades, given
the user's filing status, state, and ordinary income (W-2 / 1099). It replaces
the old inline "after-tax horizon" toy with a proper engine:

  - Ordinary income (W-2 + 1099 + other) sets the federal bracket, after the
    standard deduction.
  - Short-term gains (held <= 365 days) stack on top of ordinary income and are
    taxed at ordinary rates; long-term gains (> 365 days) get the preferential
    0 / 15 / 20% rate, stacked above ordinary income.
  - NIIT (3.8%) applies to net investment income above the MAGI threshold.
  - A flat state rate is applied to taxable-account gains (states generally
    don't distinguish ST vs LT).
  - Account type decides whether a gain is taxed at all: taxable accounts are;
    Traditional IRA/401k are tax-DEFERRED (no tax on realization); Roth and HSA
    are tax-FREE. So a winner inside a Roth costs $0 today.
  - Wash sales: a realized loss in a taxable account with a repurchase of the
    same symbol within +/-30 days is flagged and (optionally) disallowed.

IMPORTANT — accuracy + honesty:
Bracket/threshold tables below are TAX YEAR 2025 values (the most recent
confirmed figures). Verify against IRS guidance for your filing year. This is an
ESTIMATE for planning/education, not tax advice or a filed return. Self-employment
tax on 1099 income is estimated separately and flagged as approximate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Tuple

TAX_YEAR = 2025

# ---- 2025 federal ordinary-income brackets (taxable income, marginal) ------ #
# (lower_bound, rate); each rate applies to income above its bound up to the next.
_ORDINARY = {
    "single": [(0, .10), (11925, .12), (48475, .22), (103350, .24),
               (197300, .32), (250525, .35), (626350, .37)],
    "mfj":    [(0, .10), (23850, .12), (96950, .22), (206700, .24),
               (394600, .32), (501050, .35), (751600, .37)],
    "hoh":    [(0, .10), (17000, .12), (64850, .22), (103350, .24),
               (197300, .32), (250500, .35), (626350, .37)],
    "mfs":    [(0, .10), (11925, .12), (48475, .22), (103350, .24),
               (197300, .32), (250525, .35), (375800, .37)],
}

# ---- 2025 long-term cap-gains thresholds (0% up to a, 15% up to b, else 20%) #
_LTCG = {
    "single": (48350, 533400),
    "mfj":    (96700, 600050),
    "hoh":    (64750, 566700),
    "mfs":    (48350, 300000),
}

# ---- 2025 standard deduction ---------------------------------------------- #
_STD_DEDUCTION = {"single": 15000, "mfj": 30000, "hoh": 22500, "mfs": 15000}

# ---- NIIT (3.8% surtax) MAGI thresholds ----------------------------------- #
_NIIT_THRESHOLD = {"single": 200000, "hoh": 200000, "mfj": 250000, "mfs": 125000}
_NIIT_RATE = 0.038

# ---- Self-employment tax (rough) ------------------------------------------ #
_SS_WAGE_BASE = 176100      # 2025 Social Security wage base
_SE_RATE_SS = 0.124         # 12.4% OASDI
_SE_RATE_MED = 0.029        # 2.9% Medicare
_SE_NET_FACTOR = 0.9235     # net earnings adjustment

# Account types that shelter gains from current tax.
_DEFERRED = {"traditional_ira", "traditional_401k", "sep_ira", "simple_ira"}
_TAXFREE = {"roth_ira", "roth_401k", "hsa"}
TAXABLE = "taxable"


def normalize_status(s: str) -> str:
    s = (s or "").strip().lower().replace(" ", "_")
    aliases = {
        "married": "mfj", "married_joint": "mfj", "married_filing_jointly": "mfj",
        "joint": "mfj", "married_separate": "mfs", "married_filing_separately": "mfs",
        "head_of_household": "hoh", "head": "hoh",
    }
    s = aliases.get(s, s)
    return s if s in _ORDINARY else "single"


@dataclass
class Trade:
    symbol: str
    proceeds: float                 # total sale proceeds
    cost_basis: float               # total cost basis
    buy_date: str = ""              # ISO yyyy-mm-dd
    sell_date: str = ""             # ISO yyyy-mm-dd; blank = still open (ignored)
    account: str = TAXABLE          # taxable / roth_ira / traditional_401k / hsa ...

    @property
    def gain(self) -> float:
        return round(self.proceeds - self.cost_basis, 2)

    @property
    def holding_days(self) -> Optional[int]:
        if not self.buy_date or not self.sell_date:
            return None
        try:
            b = date.fromisoformat(self.buy_date)
            s = date.fromisoformat(self.sell_date)
            return (s - b).days
        except ValueError:
            return None

    @property
    def is_long_term(self) -> bool:
        d = self.holding_days
        return d is not None and d > 365


@dataclass
class TaxProfile:
    filing_status: str = "single"
    w2_income: float = 0.0
    income_1099: float = 0.0        # self-employment / contractor income
    other_income: float = 0.0       # interest, non-qualified dividends, etc.
    state_rate: float = 0.0         # flat effective state rate on gains, 0..1
    state: str = ""                 # label only


@dataclass
class TaxResult:
    tax_year: int
    filing_status: str
    # income
    ordinary_income: float
    taxable_ordinary_income: float
    marginal_rate: float
    # realized gains (taxable accounts only)
    short_term_gain: float
    long_term_gain: float
    sheltered_gain: float           # gains inside IRA/Roth/HSA (untaxed now)
    wash_sale_disallowed: float
    # tax line items
    federal_st_tax: float           # incremental fed tax on ST gains
    federal_lt_tax: float           # fed tax on LT gains (0/15/20 stacked)
    niit: float
    state_tax: float
    se_tax: float
    total_tax: float                # tax attributable to the trades (ST+LT+NIIT+state)
    effective_rate_on_gains: float  # total_tax / net taxable gain
    after_tax_gain: float
    notes: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)


def _progressive_tax(income: float, status: str) -> float:
    """Federal ordinary income tax on `income` using the bracket table."""
    if income <= 0:
        return 0.0
    brackets = _ORDINARY[status]
    tax = 0.0
    for i, (lo, rate) in enumerate(brackets):
        hi = brackets[i + 1][0] if i + 1 < len(brackets) else float("inf")
        if income > lo:
            tax += (min(income, hi) - lo) * rate
        else:
            break
    return round(tax, 2)


def marginal_rate(income: float, status: str) -> float:
    if income <= 0:
        return _ORDINARY[status][0][1]
    brackets = _ORDINARY[status]
    rate = brackets[0][1]
    for lo, r in brackets:
        if income > lo:
            rate = r
    return rate


def _ltcg_tax(ordinary_taxable: float, lt_gain: float, status: str) -> float:
    """LT gains stacked on top of ordinary taxable income, taxed 0/15/20%."""
    if lt_gain <= 0:
        return 0.0
    a, b = _LTCG[status]
    lo = max(ordinary_taxable, 0.0)
    hi = lo + lt_gain
    tax = 0.0
    # 0% band
    band0 = max(0.0, min(hi, a) - lo)
    # 15% band
    band15 = max(0.0, min(hi, b) - max(lo, a))
    # 20% band
    band20 = max(0.0, hi - max(lo, b))
    tax = band15 * 0.15 + band20 * 0.20
    return round(tax, 2)


def detect_wash_sales(trades: List[Trade]) -> List[int]:
    """Indices of taxable-account losing trades with a same-symbol repurchase
    within +/-30 days of the sale (simplified wash-sale flag)."""
    flagged = []
    for i, t in enumerate(trades):
        if t.account != TAXABLE or t.gain >= 0 or not t.sell_date:
            continue
        try:
            sell = date.fromisoformat(t.sell_date)
        except ValueError:
            continue
        for j, u in enumerate(trades):
            if j == i or u.symbol.upper() != t.symbol.upper() or not u.buy_date:
                continue
            try:
                buy = date.fromisoformat(u.buy_date)
            except ValueError:
                continue
            if abs((buy - sell).days) <= 30:
                flagged.append(i)
                break
    return flagged


def estimate(profile: TaxProfile, trades: List[Trade],
             disallow_wash_sales: bool = True) -> TaxResult:
    status = normalize_status(profile.filing_status)

    # 1) Ordinary income -> taxable ordinary income (after standard deduction).
    ordinary_income = round(profile.w2_income + profile.income_1099 + profile.other_income, 2)
    taxable_ordinary = max(0.0, ordinary_income - _STD_DEDUCTION[status])

    # 2) Realized gains by treatment.
    st = lt = sheltered = 0.0
    closed = [t for t in trades if t.sell_date]
    wash_idx = set(detect_wash_sales(trades))
    wash_disallowed = 0.0
    for i, t in enumerate(closed):
        g = t.gain
        if t.account in _DEFERRED or t.account in _TAXFREE:
            sheltered += g
            continue
        if t.account != TAXABLE:
            sheltered += g
            continue
        if disallow_wash_sales and i in wash_idx and g < 0:
            wash_disallowed += g
            continue  # loss disallowed this year
        if t.is_long_term:
            lt += g
        else:
            st += g
    st, lt, sheltered = round(st, 2), round(lt, 2), round(sheltered, 2)

    # 3) Federal tax on ST gains = incremental ordinary tax of stacking ST on top.
    #    (ST gains taxed as ordinary income.)
    fed_base = _progressive_tax(taxable_ordinary, status)
    fed_with_st = _progressive_tax(taxable_ordinary + max(st, 0.0), status)
    federal_st_tax = round(max(0.0, fed_with_st - fed_base), 2)

    # 4) Federal tax on LT gains (0/15/20 stacked above ordinary + ST).
    federal_lt_tax = _ltcg_tax(taxable_ordinary + max(st, 0.0), max(lt, 0.0), status)

    # 5) NIIT — 3.8% on lesser of net investment income and MAGI over threshold.
    net_inv_income = max(0.0, st) + max(0.0, lt)
    magi = ordinary_income + max(0.0, st) + max(0.0, lt)
    over = max(0.0, magi - _NIIT_THRESHOLD[status])
    niit = round(_NIIT_RATE * min(net_inv_income, over), 2)

    # 6) State — flat rate on net taxable gains (ST + LT), not below zero.
    net_taxable_gain = st + lt
    state_tax = round(profile.state_rate * max(0.0, net_taxable_gain), 2)

    # 7) Self-employment tax on 1099 income (rough, flagged).
    se_tax = 0.0
    if profile.income_1099 > 0:
        net_se = profile.income_1099 * _SE_NET_FACTOR
        ss = min(net_se, _SS_WAGE_BASE) * _SE_RATE_SS
        med = net_se * _SE_RATE_MED
        se_tax = round(ss + med, 2)

    total_tax = round(federal_st_tax + federal_lt_tax + niit + state_tax, 2)
    eff = round(total_tax / net_taxable_gain, 4) if net_taxable_gain > 0 else 0.0
    after_tax_gain = round(net_taxable_gain - total_tax, 2)

    notes = [
        f"Tax-year {TAX_YEAR} brackets — verify for your filing year.",
        "Estimate for planning/education, not tax advice or a filed return.",
    ]
    if sheltered:
        notes.append(f"${sheltered:,.0f} of gains were inside tax-advantaged accounts (no tax now).")
    flags = []
    if wash_idx:
        flags.append(f"{len(wash_idx)} possible wash sale(s) flagged"
                     + (f"; ${abs(wash_disallowed):,.0f} of losses disallowed this year." if disallow_wash_sales else "."))
    if se_tax:
        flags.append(f"~${se_tax:,.0f} self-employment tax on 1099 income (rough estimate, not in the gains total).")

    return TaxResult(
        tax_year=TAX_YEAR, filing_status=status,
        ordinary_income=ordinary_income, taxable_ordinary_income=round(taxable_ordinary, 2),
        marginal_rate=marginal_rate(taxable_ordinary + max(st, 0.0), status),
        short_term_gain=st, long_term_gain=lt, sheltered_gain=sheltered,
        wash_sale_disallowed=round(wash_disallowed, 2),
        federal_st_tax=federal_st_tax, federal_lt_tax=federal_lt_tax,
        niit=niit, state_tax=state_tax, se_tax=se_tax,
        total_tax=total_tax, effective_rate_on_gains=eff, after_tax_gain=after_tax_gain,
        notes=notes, flags=flags,
    )


# ---- (de)serialization helpers for the CLI/API bridge --------------------- #
def result_to_dict(r: TaxResult) -> dict:
    return {
        "tax_year": r.tax_year, "filing_status": r.filing_status,
        "ordinary_income": r.ordinary_income,
        "taxable_ordinary_income": r.taxable_ordinary_income,
        "marginal_rate": r.marginal_rate,
        "short_term_gain": r.short_term_gain, "long_term_gain": r.long_term_gain,
        "sheltered_gain": r.sheltered_gain, "wash_sale_disallowed": r.wash_sale_disallowed,
        "federal_st_tax": r.federal_st_tax, "federal_lt_tax": r.federal_lt_tax,
        "niit": r.niit, "state_tax": r.state_tax, "se_tax": r.se_tax,
        "total_tax": r.total_tax, "effective_rate_on_gains": r.effective_rate_on_gains,
        "after_tax_gain": r.after_tax_gain, "notes": r.notes, "flags": r.flags,
    }


def estimate_from_payload(payload: dict) -> dict:
    """payload = {profile: {...}, trades: [{...}], disallow_wash_sales?: bool}"""
    p = payload.get("profile", {}) or {}
    profile = TaxProfile(
        filing_status=p.get("filing_status", "single"),
        w2_income=float(p.get("w2_income", 0) or 0),
        income_1099=float(p.get("income_1099", 0) or 0),
        other_income=float(p.get("other_income", 0) or 0),
        state_rate=float(p.get("state_rate", 0) or 0),
        state=p.get("state", ""),
    )
    trades = []
    for t in payload.get("trades", []) or []:
        try:
            trades.append(Trade(
                symbol=str(t.get("symbol", "")).upper(),
                proceeds=float(t.get("proceeds", 0) or 0),
                cost_basis=float(t.get("cost_basis", 0) or 0),
                buy_date=str(t.get("buy_date", "") or ""),
                sell_date=str(t.get("sell_date", "") or ""),
                account=str(t.get("account", TAXABLE) or TAXABLE),
            ))
        except (TypeError, ValueError):
            continue
    disallow = bool(payload.get("disallow_wash_sales", True))
    return result_to_dict(estimate(profile, trades, disallow_wash_sales=disallow))
