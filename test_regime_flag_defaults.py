"""
Regime flags default ON through system_config (human directive 2026-10-28,
paper account: engine live, all flags, RUNBOOK §6 order, shadow wait skipped).

Contract pinned here: every flag reader (engine_allowed, selection_gates,
options_gates, sizing_sleeves, labels_enabled) uses the env var when it is
PRESENT — any value, "off" / "0" / "" included, wins — and the BASE_CONFIG
default only when the env var is ABSENT; so a Railway env var of `off` rolls a
flag back without a deploy. The Node mirror (server/regimeReport.ts
REGIME_FLAG_DEFAULTS) must agree with BASE_CONFIG.
"""
import os
import re
import unittest

import regime_engine as re_
from system_config import BASE_CONFIG

ROOT = os.path.dirname(os.path.abspath(__file__))

OFF_VALUES = ("off", "0", "", "false", "none", " OFF ")


class TestDefaults(unittest.TestCase):
    def test_base_config_defaults_are_on(self):
        self.assertIs(BASE_CONFIG["REGIME_ENGINE_ENABLED"], True)
        self.assertEqual(BASE_CONFIG["REGIME_SELECTION_DEFAULT"], "all")
        self.assertEqual(BASE_CONFIG["REGIME_OPTIONS_DEFAULT"], "all")
        self.assertEqual(BASE_CONFIG["REGIME_SIZING_DEFAULT"], "all")
        self.assertIs(BASE_CONFIG["REGIME_LABELS_DEFAULT"], True)

    def test_absent_env_reads_the_config_default(self):
        self.assertTrue(re_.engine_allowed({}))
        self.assertEqual(re_.selection_gates({}), set(re_.SELECTION_GUARDS))
        self.assertEqual(re_.options_gates({}), set(re_.OPTIONS_GUARDS))
        self.assertEqual(re_.sizing_sleeves({}), set(re_.SIZING_SLEEVES))
        self.assertTrue(re_.labels_enabled({}))

    def test_absent_env_and_no_config_is_off(self):
        self.assertFalse(re_.engine_allowed({}, {}))
        self.assertEqual(re_.selection_gates({}, {}), set())
        self.assertEqual(re_.options_gates({}, {}), set())
        self.assertEqual(re_.sizing_sleeves({}, {}), set())
        self.assertFalse(re_.labels_enabled({}, {}))


class TestExplicitEnvWins(unittest.TestCase):
    def test_explicit_off_disables_each_flag_even_with_the_default_on(self):
        for v in OFF_VALUES:
            with self.subTest(value=v):
                self.assertFalse(re_.engine_allowed({"VOLTRADE_REGIME_ENGINE": v}))
                self.assertFalse(re_.engine_allowed({"VOLTRADE_REGIME_ENGINE": v}, {"REGIME_ENGINE_ENABLED": True}))
                self.assertEqual(re_.selection_gates({"VOLTRADE_REGIME_SELECTION": v}), set())
                self.assertEqual(re_.options_gates({"VOLTRADE_REGIME_OPTIONS": v}), set())
                self.assertEqual(re_.sizing_sleeves({"VOLTRADE_REGIME_SIZING": v}), set())
                self.assertFalse(re_.labels_enabled({"VOLTRADE_REGIME_LABELS": v}))

    def test_explicit_partial_values_still_narrow(self):
        self.assertEqual(re_.selection_gates({"VOLTRADE_REGIME_SELECTION": "knife"}), {"knife"})
        self.assertEqual(re_.sizing_sleeves({"VOLTRADE_REGIME_SIZING": "stock"}), {"stock"})
        self.assertEqual(re_.options_gates({"VOLTRADE_REGIME_OPTIONS": "earnings"}), {"earnings", "crisis", "vrp"})

    def test_consumers_honour_an_explicit_off(self):
        st = {"as_of": "2026-10-27", "composite_regime": "STRESSED", "legacy_label": "CAUTION",
              "exposure_multiplier": 0.5, "allowed_strategies": ["csp_reduced"]}
        re_._STATE_CACHE.update(at=0.0, state=None)
        try:
            self.assertIsNone(re_.sizing_for("stock", env={"VOLTRADE_REGIME_SIZING": "off"}, loader=lambda: st, today="2026-10-28"))
            self.assertIsNone(re_.sizing_for("stock", env={"VOLTRADE_REGIME_ENGINE": "off"}, loader=lambda: st, today="2026-10-28"))
            self.assertIsNone(re_.label_for({"VOLTRADE_REGIME_LABELS": "off"}, loader=lambda: st, today="2026-10-28"))
            on = re_.sizing_for("stock", env={}, loader=lambda: st, today="2026-10-28")
            self.assertEqual((on["multiplier"], on["source"]), (0.5, "engine"))
            self.assertEqual(re_.label_for({}, loader=lambda: st, today="2026-10-28")["label"], "CAUTION")
        finally:
            re_._STATE_CACHE.update(at=0.0, state=None)

    def test_effective_flags_report_value_and_source(self):
        f = re_.effective_flags({"VOLTRADE_REGIME_SIZING": "off"})
        self.assertEqual(set(f), set(re_.FLAG_DEFAULTS))
        self.assertEqual(f["VOLTRADE_REGIME_SIZING"], {"value": "off", "source": "env"})
        self.assertEqual(f["VOLTRADE_REGIME_ENGINE"], {"value": "1", "source": "config"})
        self.assertEqual(f["VOLTRADE_REGIME_SELECTION"], {"value": "all", "source": "config"})


class TestNodeMirror(unittest.TestCase):
    def test_regime_report_defaults_match_base_config(self):
        src = open(os.path.join(ROOT, "server", "regimeReport.ts")).read()
        block = re.search(r"REGIME_FLAG_DEFAULTS[^=]*=\s*Object\.freeze\(\{(.*?)\}\)", src, re.S)
        self.assertIsNotNone(block, "REGIME_FLAG_DEFAULTS table not found in server/regimeReport.ts")
        ts = dict(re.findall(r"(VOLTRADE_REGIME_[A-Z]+):\s*\"([^\"]*)\"", block.group(1)))
        self.assertEqual(set(ts), set(re_.FLAG_DEFAULTS))
        for name in re_.FLAG_DEFAULTS:
            self.assertEqual(ts[name], re_.flag_raw(name, {})[0], name)

    def test_node_reader_treats_a_present_env_var_as_authoritative(self):
        src = open(os.path.join(ROOT, "server", "regimeReport.ts")).read()
        self.assertIn('if (v !== undefined && v !== null) return { value: String(v), source: "env" };', src)
        self.assertIn('regimeFlagRaw("VOLTRADE_REGIME_ENGINE", env)', src)


if __name__ == "__main__":
    unittest.main()


def test_missing_state_keeps_legacy_sizing():
    """No usable regime state -> sizing_for returns None (the caller keeps its legacy scalars)."""
    import regime_engine as re_
    env = {"VOLTRADE_REGIME_ENGINE": "1", "VOLTRADE_REGIME_SIZING": "all"}
    for st in ({}, {"as_of": "2026-10-28"}):
        assert re_.sizing_for("stock", env=env, loader=lambda st=st: st) is None
