"""
[REPAIR] 2026-11-06 — ml_model_v2.py named its logger "ml_model_v3" (the
shadow v3 candidate's name), so the live model's warnings were attributed to
the shadow model. Each ML module now logs under its own name. Fails on the
previous commit (ml_model_v2.logger.name == "ml_model_v3").
"""
import logging

import ml_model_v2
import ml_model_v3


def test_each_ml_module_logs_under_its_own_name():
    assert ml_model_v2.logger.name == "ml_model_v2"
    assert ml_model_v3.logger.name == "ml_model_v3"
    assert ml_model_v2.logger is not ml_model_v3.logger


def test_a_v2_warning_is_attributed_to_v2(caplog):
    with caplog.at_level(logging.WARNING):
        ml_model_v2.logger.warning("v2 training probe")
    names = {r.name for r in caplog.records if r.getMessage() == "v2 training probe"}
    assert names == {"ml_model_v2"}
