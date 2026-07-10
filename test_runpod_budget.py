"""
test_runpod_budget.py — cost-cap gate + ledger math (RunPod spend control).

Exercises the PURE functions of scripts/runpod_budget.py: the balance model,
the worst-case reservation, and the three refusal reasons of authorize_job —
above all the mandate's hard rule that an UNBOUNDED job (no finite time cap)
can never be authorized.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
import runpod_budget as rb  # noqa: E402


def _open_row(job, hourly, max_hours, reserved=None):
    return {
        "job_id": job, "workload": "grid-detector", "gpu": "RTX4090",
        "hourly": hourly, "max_hours": max_hours,
        "reserved_cost": reserved if reserved is not None else round(hourly * max_hours, 6),
        "actual_cost": None, "opened_utc": "2026-07-07T00:00:00Z",
        "closed_utc": None, "note": "",
    }


def _closed_row(job, actual):
    r = _open_row(job, 0.34, 6)
    r["actual_cost"] = actual
    r["closed_utc"] = "2026-07-07T06:00:00Z"
    return r


# --- balance model -----------------------------------------------------------

def test_empty_ledger_is_full_balance():
    assert rb.remaining([]) == rb.START_BALANCE
    assert rb.spent([]) == 0.0


def test_open_row_reserves_worst_case():
    rows = [_open_row("j1", 0.34, 6)]  # worst case 2.04
    assert rb.spent(rows) == 2.04
    assert rb.remaining(rows) == round(rb.START_BALANCE - 2.04, 6)


def test_closed_row_counts_actual_not_reserved():
    # Reserved worst case was 2.04, but the job actually cost 1.10.
    rows = [_closed_row("j1", 1.10)]
    assert rb.spent(rows) == 1.10
    assert rb.remaining(rows) == round(rb.START_BALANCE - 1.10, 6)


def test_mixed_open_and_closed():
    rows = [_closed_row("j1", 1.10), _open_row("j2", 1.64, 3)]  # 1.10 + 4.92
    assert rb.spent(rows) == round(1.10 + 4.92, 6)


# --- append-only reconciliation: open row THEN a separate close row ----------
# The CLI `close` and the launcher both APPEND a new close row rather than
# rewriting the open row, so a finished job appears as TWO rows with the same
# job_id. spend must count the ACTUAL once (not reserved + actual), and the job
# must no longer be considered open. (Regression: the reader used to sum every
# row, double-counting the reservation and leaving a phantom open job forever.)

def test_open_then_close_same_job_counts_actual_once():
    open_r = _open_row("j1", 0.34, 6)          # reserves 2.04
    close_r = dict(open_r)
    close_r["actual_cost"] = 1.10              # closed later, real cost 1.10
    close_r["closed_utc"] = "2026-07-07T06:00:00Z"
    rows = [open_r, close_r]                    # append-only: both rows present
    assert rb.spent(rows) == 1.10              # NOT 2.04 + 1.10 = 3.14
    assert rb.remaining(rows) == round(rb.START_BALANCE - 1.10, 6)


def test_reconcile_drops_finished_job_from_open_list():
    open_r = _open_row("j1", 0.34, 6)
    close_r = dict(open_r)
    close_r["actual_cost"] = 0.017639
    close_r["closed_utc"] = "2026-07-07T06:00:00Z"
    reconciled = rb.reconcile([open_r, close_r])
    assert len(reconciled) == 1                              # one effective row
    assert reconciled[0]["actual_cost"] == 0.017639          # the close wins
    open_jobs = [r for r in reconciled if r.get("actual_cost") is None]
    assert open_jobs == []                                   # no phantom open


def test_reconcile_keeps_still_open_job_open():
    # An opened-but-not-yet-closed job stays open and reserves its worst case.
    rows = [_open_row("j1", 0.34, 6)]
    reconciled = rb.reconcile(rows)
    assert len(reconciled) == 1
    assert reconciled[0]["actual_cost"] is None
    assert rb.spent(rows) == 2.04                            # reservation held


# --- authorize_job: the unbounded guard (the mandate's hard rule) -----------

def test_refuses_unbounded_none():
    res = rb.authorize_job([], "RTX4090", 0.34, None)
    assert res["authorized"] is False
    assert res["reason"] == "unbounded"
    assert res["max_runtime_seconds"] is None


def test_refuses_unbounded_zero_and_negative():
    for bad in (0, -1, -0.5):
        res = rb.authorize_job([], "RTX4090", 0.34, bad)
        assert res["authorized"] is False
        assert res["reason"] == "unbounded"


def test_refuses_unbounded_infinity_and_nan():
    for bad in (math.inf, math.nan):
        res = rb.authorize_job([], "RTX4090", 0.34, bad)
        assert res["authorized"] is False
        assert res["reason"] == "unbounded"


# --- authorize_job: rate sanity guard ---------------------------------------

def test_refuses_bad_rate():
    for bad in (0, -0.1, rb.MAX_SANE_HOURLY + 1, math.inf, math.nan, None):
        res = rb.authorize_job([], "RTX4090", bad, 6)
        assert res["authorized"] is False
        assert res["reason"] == "bad_rate"


# --- authorize_job: floor-buffer guard --------------------------------------

def test_refuses_when_worst_case_breaches_floor():
    # remaining 50; a job whose worst case leaves < FLOOR_BUFFER must be refused.
    # 47 hours * $1/hr = $47 worst -> would leave $3 < $5 floor.
    res = rb.authorize_job([], "A100", 1.0, 47)
    assert res["authorized"] is False
    assert res["reason"] == "insufficient"
    assert res["worst_case"] == 47.0


def test_refuses_when_prior_spend_leaves_too_little():
    rows = [_open_row("j1", 1.0, 43)]  # reserves 43, remaining 7
    # A $3 worst-case job would leave $4 < $5 floor -> refuse.
    res = rb.authorize_job(rows, "RTX4090", 1.0, 3)
    assert res["authorized"] is False
    assert res["reason"] == "insufficient"


# --- authorize_job: the happy path ------------------------------------------

def test_authorizes_small_job_and_returns_hard_cap():
    res = rb.authorize_job([], "RTX4090", 0.34, 6)
    assert res["authorized"] is True
    assert res["reason"] == "ok"
    assert res["worst_case"] == 2.04
    assert res["remaining_after_if_run"] == round(rb.START_BALANCE - 2.04, 6)
    # The hard auto-terminate cap the launcher MUST hand to RunPod.
    assert res["max_runtime_seconds"] == 6 * 3600


def test_authorizes_exactly_at_floor():
    # Worst case that leaves EXACTLY the floor buffer is allowed (>=).
    res = rb.authorize_job([], "A100", 1.0, rb.START_BALANCE - rb.FLOOR_BUFFER)
    assert res["authorized"] is True
    assert res["remaining_after_if_run"] == rb.FLOOR_BUFFER


# --- low_balance / top-up PO -------------------------------------------------

def test_low_balance_flags_below_threshold_with_po():
    rows = [_closed_row("j1", rb.START_BALANCE - 4.0)]  # remaining 4 < 10
    lb = rb.low_balance(rows)
    assert lb["low"] is True
    assert lb["remaining"] == 4.0
    assert "PURCHASE ORDER" in lb["po_line"]
    assert "top-up" in lb["po_line"]


def test_low_balance_quiet_above_threshold():
    rows = [_closed_row("j1", 5.0)]  # remaining 45 > 10
    lb = rb.low_balance(rows)
    assert lb["low"] is False
    assert lb["po_line"] is None


# --- load_ledger robustness --------------------------------------------------

def test_load_ledger_missing_file_is_empty(tmp_path):
    assert rb.load_ledger(str(tmp_path / "nope.jsonl")) == []


def test_load_ledger_skips_blank_and_garbage(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text('{"job_id":"a","reserved_cost":2.0,"actual_cost":null}\n'
                 '\n'
                 'not json at all\n'
                 '{"job_id":"b","actual_cost":1.5}\n')
    rows = rb.load_ledger(str(p))
    assert len(rows) == 2
    assert rb.spent(rows) == 3.5


def test_monkeypatched_module_ledger_redirects_default_path_calls(tmp_path, monkeypatch):
    # BUG FIX 2026-07-10 (found via gv-div4-ks): load_ledger/_append_row used to
    # default `path=LEDGER`, a value bound ONCE at module-def time. A test that
    # did `monkeypatch.setattr(rb, "LEDGER", tmp)` and then called
    # `load_ledger()`/`_append_row(row)` with NO explicit path silently kept
    # writing to the REAL production `datacore/runpod/ledger.jsonl` — caught
    # when a test for run_launch() did exactly that and polluted it with fake
    # rows. Both now resolve the CURRENT module-level LEDGER at call time.
    fake = tmp_path / "fake_ledger.jsonl"
    monkeypatch.setattr(rb, "LEDGER", str(fake))
    assert rb.load_ledger() == []  # no explicit path -- must read the FAKE path
    rb._append_row({"job_id": "z", "actual_cost": 1.0})  # no explicit path either
    assert fake.exists()  # proves it wrote to the fake path, not the real ledger
    assert rb.load_ledger() == [{"job_id": "z", "actual_cost": 1.0}]
