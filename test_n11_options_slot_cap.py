"""N11: Python's options-slot cap must be the system_config value Node uses."""
import inspect
import bot_engine
from system_config import BASE_CONFIG


def test_options_cap_matches_config():
    assert bot_engine.MAX_OPTIONS_POSITIONS == BASE_CONFIG["MAX_OPTIONS_POSITIONS"]


def test_options_cap_is_not_a_hardcoded_literal():
    src = inspect.getsource(bot_engine)
    line = [l for l in src.splitlines() if l.startswith("MAX_OPTIONS_POSITIONS =")][0]
    assert "BASE_CONFIG" in line
