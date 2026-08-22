"""test_law_iv_context_modules.py — MASTER PROGRAM Q7 (T2.1/T2.2), finding
F-B in research/PROGRAM_STATE.md.

test_audit_critical.py's `TestRenderingMotionLaw._layer_modules()` selects
Law IV's audited files by `basename.endswith("Layer.ts")` — a deliberate
naming rule (its own docstring says so), not a hardcoded list. That rule
still misses the two heaviest render surfaces in the repo: celestialSky.ts
(a second WebGL2 context) and spaceFrame.ts (4,300+ LOC of CPU 2D
rasterisation), because neither is named `*Layer.ts` — both instead export
a top-level `mount<Name>()` function that returns a Handle with its own
`dispose()`, the same lifecycle contract a MapLibre CustomLayerInterface
class gives the five existing `*Layer.ts` modules under a different shape.

WIDENED PREDICATE (below): `*Layer.ts` UNION any file under the same layer
directories exporting a top-level `mount<Name>()` function. Still a naming/
structural rule, not a hardcoded file list — a future module following
either pattern is bound the moment it is written. Three files in the layer
dirs call `getContext()` for a one-shot OFFSCREEN canvas (baking a sprite
or tile texture, then discarding it — no owned render surface, no `mount*`
export): lunarSymbols.ts, spaceAssets.ts, moonTiles.ts. The predicate must
NOT catch these — `test_offscreen_texture_bakers_are_correctly_excluded`
guards that.

WHY THIS IS ITS OWN FILE AND NOT A WIDER PREDICATE IN test_audit_critical.py
DIRECTLY: that file's Law IV test is REQUIRED (gates every merge). Widening
it in place would have turned CI red for a gap the originating session was
documenting, not fully closing — celestialSky.ts was fixed in that same PR
(`maxFeatures`/`vramBudget`, derived from its real buffer sizes, not
invented), but spaceFrame.ts owned half a dozen CPU-rasterised canvases at
different, data-dependent sizes; a good-faith budget number for it needed
its own read of the whole module. This file was kept separate from
test_audit_critical.py's REQUIRED gate for exactly that reason — never
actually listed in ci/quarantine.txt (it was already green on its other two
tests), just structurally isolated so the one known-failing assertion below
couldn't turn the required suite red.

UPDATE (MASTER PROGRAM Q7, budget-derivation session): spaceFrame.ts now
carries real `maxFeatures`/`vramBudget` too (derived from its own owned
buffers — the texture manager and moon/NAC tile managers it creates and
disposes, the shared sprite cache, the moon-patch buffers, the sky/star
canvases — see the export's own comment for the full byte-math). The gap
this file existed to isolate is closed: `expected` below is narrowed to
`[]` per this file's own standing instruction ("if this SHRANK, celebrate
and narrow `expected`"). Kept as its own file rather than folded into
test_audit_critical.py's `_layer_modules()` — that predicate change plus
retiring this file is a second, separable structural edit (one logical
change per PR, CLAUDE.md PROMOTION RULES #5); this file still runs
every session and will fail loudly the moment either module regresses.
"""

import glob
import os
import re
import unittest

LAYER_DIRS = [
    "client/src/lib/orbital",
    "client/src/lib/air",
    "client/src/lib/celestial",
]

MOUNT_RE = re.compile(r"^export function mount[A-Z]\w*\(", re.M)


def _repo(*parts):
    return os.path.join(os.path.dirname(__file__), *parts)


def _layer_sources():
    out = []
    for d in LAYER_DIRS:
        for path in sorted(glob.glob(_repo(d, "*.ts"))):
            if path.endswith(".test.ts"):
                continue
            out.append(path)
    return out


def _context_acquiring_modules():
    out = []
    for path in _layer_sources():
        if os.path.basename(path).endswith("Layer.ts"):
            out.append(path)
            continue
        with open(path, encoding="utf-8") as f:
            if MOUNT_RE.search(f.read()):
                out.append(path)
    return out


class TestLawIVContextAcquiringModules(unittest.TestCase):
    def test_predicate_widens_to_seven(self):
        names = sorted(os.path.basename(p) for p in _context_acquiring_modules())
        self.assertEqual(
            names,
            [
                "airLayer.ts",
                "arcLayer.ts",
                "celestialSky.ts",
                "flightTrackLayer.ts",
                "modelLayer.ts",
                "satLayer.ts",
                "spaceFrame.ts",
            ],
            "Law IV context-acquiring module set changed (a new mount*() module "
            "or *Layer.ts file appeared or disappeared). Update this list "
            "deliberately — it is the audit trail for F-B, not a tautology.",
        )

    def test_offscreen_texture_bakers_are_correctly_excluded(self):
        """Guard the guard: these three call getContext() for a throwaway
        offscreen canvas and export no mount*() — confirms the predicate
        distinguishes 'calls getContext' from 'owns a render surface'
        rather than over-widening to every getContext call in the layer
        dirs (the shell counter in scripts/program_status.sh does exactly
        that broader, informational scan — this test is deliberately
        stricter)."""
        names = {os.path.basename(p) for p in _context_acquiring_modules()}
        for excluded in ("lunarSymbols.ts", "spaceAssets.ts", "moonTiles.ts"):
            self.assertNotIn(excluded, names, f"{excluded} should not be a context-acquiring module")

    def test_law_IV_budgets_and_teardown(self):
        """Mirrors test_audit_critical.py's
        `test_law_IV_every_layer_declares_budgets_and_teardown`, run against
        the WIDENED set. Pinned to `[]` (MASTER PROGRAM Q7 CLOSED, budget-
        derivation session) — both celestialSky.ts and spaceFrame.ts now
        declare real `maxFeatures`/`vramBudget`. ANY future failure here means
        a module regressed (lost its budget/dispose) or a new context-
        acquiring module was added without them — fix it, never widen this
        assertion back open to match a real regression."""
        missing = []
        for path in _context_acquiring_modules():
            rel = os.path.relpath(path, os.path.dirname(__file__))
            with open(path, encoding="utf-8") as f:
                src = f.read()
            if not re.search(r"^export const maxFeatures\b", src, re.M):
                missing.append(f"{rel}: no `export const maxFeatures`")
            if not re.search(r"^export const vramBudget\b", src, re.M):
                missing.append(f"{rel}: no `export const vramBudget`")
            if not re.search(r"^\s*dispose\s*\(\s*\)\s*:", src, re.M):
                missing.append(f"{rel}: no `dispose()` teardown method")
        self.assertEqual(
            sorted(missing),
            [],
            "Law IV context-acquiring gap changed. A module lost its budget/"
            "dispose, or a new context-acquiring module was added without "
            "them — fix it, don't widen this assertion to match.",
        )


if __name__ == "__main__":
    unittest.main()
