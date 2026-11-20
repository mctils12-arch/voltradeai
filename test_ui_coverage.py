"""UI coverage register (owner directive 2026-11-19; CLAUDE.md STANDING
BEHAVIORS, UI INTEGRATION AND ONE ORDER PATH).

The rule says every PR that adds or changes a backend feature updates its row
in UI_COVERAGE.md. These tests are what makes that true: a new server route or
diagnostic probe that no page calls and nobody classified fails the build, and
so does a register that was not regenerated.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import ui_coverage as uc  # noqa: E402


def test_the_committed_register_is_current_and_complete():
    b = uc.build()
    assert uc.problems(b) == [], (
        "Backend items without a UI classification, or notes that no longer apply. "
        "Fix ci/ui_coverage_notes.json, then run: python3 scripts/ui_coverage.py --write"
    )
    with open(uc.OUT) as f:
        assert f.read() == uc.render(b), (
            "UI_COVERAGE.md is stale. Run: python3 scripts/ui_coverage.py --write  and commit the result."
        )


def test_every_note_states_a_status_and_says_something():
    notes = json.load(open(uc.NOTES))
    for section in ("routes", "probes", "strategies"):
        for key, val in notes[section].items():
            assert isinstance(val, list) and len(val) == 2, key
            assert val[0] in uc.STATUSES, f"{key}: {val[0]}"
            assert len(val[1].strip()) >= 8, f"{key}: the note must say what the item is"
            if val[0] in ("missing", "partial"):
                assert "lacement" in val[1] or "as above" in val[1] or len(val[1]) > 60, (
                    f"{key}: a missing/partial item says where it belongs in the existing UI"
                )


def _tree(tmp_path, routes_ts, client_ts, notes, bot_ts=""):
    (tmp_path / "server").mkdir()
    (tmp_path / "client" / "src").mkdir(parents=True)
    (tmp_path / "ci").mkdir()
    (tmp_path / "datacore").mkdir()
    (tmp_path / "server" / "routes.ts").write_text(routes_ts)
    (tmp_path / "server" / "bot.ts").write_text(bot_ts)
    (tmp_path / "server" / "routes.test.ts").write_text('app.get("/api/only/in-a-test", h);\n')
    (tmp_path / "client" / "src" / "page.tsx").write_text(client_ts)
    (tmp_path / "client" / "src" / "page.test.ts").write_text('fetch("/api/data/secret")\n')
    (tmp_path / "ci" / "ui_coverage_notes.json").write_text(json.dumps(notes))
    (tmp_path / "datacore" / "layers.json").write_text(json.dumps({"layers": [{"id": "a", "status": "live"}]}))
    return str(tmp_path)


def test_a_new_route_no_page_calls_is_unclassified_until_someone_decides(tmp_path):
    root = _tree(
        tmp_path,
        'app.get("/api/data/shown", h);\napp.get("/api/data/secret", h);\napp.get("/api/data/plane/:hex", h);\n',
        'fetch("/api/data/shown"); fetch(`/api/data/plane/${hex}`);',
        {"routes": {}, "probes": {}, "strategies": {}},
    )
    b = uc.build(root)
    by = {r[1]: r[3] for r in b["rows"]}
    assert by["GET /api/data/shown"] == "surfaced"
    assert by["GET /api/data/plane/:hex"] == "surfaced", "a parameterised route matches on its fixed part"
    assert by["GET /api/data/secret"] == "UNCLASSIFIED", "a test file's fetch is not a page"
    assert "GET /api/only/in-a-test" not in by, "routes registered in test files are not backend features"
    assert any("UNCLASSIFIED" in p for p in uc.problems(b))


def test_a_note_classifies_it_and_a_note_that_no_longer_applies_is_reported(tmp_path):
    root = _tree(
        tmp_path,
        'app.get("/api/data/shown", h);\napp.get("/api/data/secret", h);\n',
        'fetch("/api/data/shown");',
        {"routes": {"GET /api/data/secret": ["missing", "A thing. Placement: the existing panel."],
                    "GET /api/data/shown": ["missing", "was missing once; a page calls it now"],
                    "GET /api/data/deleted": ["plumbing", "the route is gone"]},
         "probes": {}, "strategies": {}},
    )
    b = uc.build(root)
    assert {r[1]: r[3] for r in b["rows"]}["GET /api/data/secret"] == "missing"
    assert sorted(b["stale_notes"]) == ["GET /api/data/deleted", "GET /api/data/shown"]
    assert b["unclassified"] == []


def test_diagnostic_probes_are_read_from_the_switch_and_need_a_note(tmp_path):
    bot = (
        '  app.get("/api/diag/:probe", async (req, res) => {\n'
        '      switch (req.params.probe) {\n'
        '        case "audit": {\n'
        '          switch (x) { case "inner": break; }\n'
        '        case "new_probe": {\n'
        '  });\n'
        '  app.post("/api/bot/trade", h);\n'
        '        case "not_a_probe": {\n'
    )
    root = _tree(tmp_path, "", "", {"routes": {"GET /api/diag/:probe": ["plumbing", "token-gated diagnostics"],
                                               "POST /api/bot/trade": ["plumbing", "owner action, no page"]},
                                    "probes": {"audit": ["surfaced", "activity log"]}, "strategies": {}}, bot_ts=bot)
    assert uc.diag_probes(root) == ["audit", "new_probe"]
    assert uc.build(root)["unclassified"] == ["probe new_probe"]


def test_the_keyed_api_counts_as_surfaced_only_when_the_catalog_lists_it(tmp_path):
    root = _tree(
        tmp_path,
        'app.get("/api/v1/data/x", h);\napp.get("/api/v1/data/y", h);\nconst CAT = [{ path: "/api/v1/data/x", desc: "" }];\n',
        "",
        {"routes": {}, "probes": {}, "strategies": {}},
    )
    by = {r[1]: r[3] for r in uc.build(root)["rows"]}
    assert by["GET /api/v1/data/x"] == "surfaced" and by["GET /api/v1/data/y"] == "UNCLASSIFIED"


def test_a_route_is_not_called_by_a_page_that_reads_a_longer_path(tmp_path):
    """2026-11-20: "a page calls it" was a substring test, so a route whose path
    starts another route's path was marked surfaced by the longer one
    (/api/news by /api/newsletter/issues; /api/data/aircraft/track/:hex by
    /api/data/aircraft/tracked). The match must end where the route's path ends."""
    root = _tree(
        tmp_path,
        'app.get("/api/news", h);\n'
        'app.get("/api/newsletter/issues", h);\n'
        'app.get("/api/data/aircraft/track/:hex", h);\n'
        'app.get("/api/data/aircraft/tracked", h);\n'
        'app.get("/api/data/insider", h);\n'
        'app.get("/api/data/insider/history", h);\n'
        'app.get("/api/ticker-dossier/:symbol", h);\n'
        'app.get("/api/bot/decisions", h);\n'
        'app.get("/api/data/tiles/*", h);\n',
        'fetch("/api/newsletter/issues"); fetch("/api/data/aircraft/tracked");\n'
        'fetch("/api/data/insider/history"); fetch(`/api/ticker-dossier/${sym}`);\n'
        'fetch("/api/bot/decisions?limit=200"); const t = "/api/data/tiles/" + z;\n',
        {"routes": {}, "probes": {}, "strategies": {}},
    )
    b = uc.build(root)
    status = {r[1]: r[3] for r in b["rows"] if r[0] == "route"}
    # read by a page: an exact path, a fixed path with a query, a parameter's value after "/"
    for key in ("GET /api/newsletter/issues", "GET /api/data/aircraft/tracked", "GET /api/data/insider/history",
                "GET /api/ticker-dossier/:symbol", "GET /api/bot/decisions", "GET /api/data/tiles/*"):
        assert status[key] == "surfaced", key
    # NOT read by a page: only a longer path that starts the same way is
    for key in ("GET /api/news", "GET /api/data/aircraft/track/:hex", "GET /api/data/insider"):
        assert status[key] == "UNCLASSIFIED", key
    assert sorted(b["unclassified"]) == ["GET /api/data/aircraft/track/:hex", "GET /api/data/insider", "GET /api/news"]


def test_page_calls_ends_the_match_where_the_route_ends():
    assert uc.page_calls("/api/bot/close", 'apiRequest("POST", "/api/bot/close")')
    assert uc.page_calls("/api/bot/close", "fetch(`/api/bot/close`)")
    assert not uc.page_calls("/api/bot/close", 'getJson("/api/bot/closed-lots?limit=200")')
    assert not uc.page_calls("/api/bot/close", 'getJson("/api/bot/close/all")')
    assert uc.page_calls("/api/x/:id", 'fetch("/api/x/" + id)')
    assert uc.page_calls("/api/x/:id", 'queryKey: ["/api/x", id]')
    assert not uc.page_calls("/api/x/:id", 'fetch("/api/xy/" + id)')
    assert not uc.page_calls("/:anything", "whatever")

