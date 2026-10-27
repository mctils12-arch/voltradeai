"""Ratchet: scan_market must assign current_positions before the tie-break
block reads it (2026-10-27: the read preceded the assignment, so the tie-break
raised UnboundLocalError on every scan and was silently skipped)."""
import ast
import pathlib

SRC = pathlib.Path(__file__).parent / "bot_engine.py"


def _scan_market():
    tree = ast.parse(SRC.read_text())
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_scan_market_inner")


def test_current_positions_assigned_before_first_read():
    fn = _scan_market()
    stores, loads = [], []
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and n.id == "current_positions":
            (stores if isinstance(n.ctx, ast.Store) else loads).append(n.lineno)
    assert stores and loads
    assert min(stores) < min(loads), (
        f"current_positions read at line {min(loads)} before first assignment at {min(stores)}")
