# Pins the 2026-07-28 contract-selection instrumentation: select_contract's
# catch-all previously reduced any failure to a bare message ("'>' not
# supported between instances of 'NoneType' and 'int'"), which since the
# v1.0.498 indicative-feed fallback has been firing on every Tier2 candidate
# with no way to locate the comparison. The error string must now carry the
# exception's last frame (file:line:function) so the live audit trail is
# diagnosable without a reproducible local setup. Offline-safe.
from options_execution import _error_with_frame


def _raise_the_production_shape():
    threshold = 5
    bad_field = None    # what the indicative feed hands back
    return bad_field > threshold


def test_frame_appended_with_file_line_function():
    try:
        _raise_the_production_shape()
    except TypeError as e:
        msg = _error_with_frame(e)
    assert "'>' not supported" in msg
    assert "@test_options_error_frames.py:" in msg
    assert msg.rstrip().endswith("_raise_the_production_shape")


def test_deep_frame_is_the_last_one():
    def outer():
        def inner():
            return None > 1
        return inner()
    try:
        outer()
    except TypeError as e:
        msg = _error_with_frame(e)
    assert msg.rstrip().endswith("inner")


def test_exception_without_traceback_still_returns_message():
    # A hand-constructed exception has no __traceback__ — the instrumentation
    # must degrade to the plain message, never raise.
    assert _error_with_frame(ValueError("plain")) == "plain"
