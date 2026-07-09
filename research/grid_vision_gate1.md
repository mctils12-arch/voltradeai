# GRID VISION — Gate-1 closure + national-rollout program (PRE-STATED criteria)

> Filed 2026-07-09 BEFORE any of the runs below, per the honesty mandate: a
> metric only counts if its bar was written before the run. Program charter:
> research/grid_vision.md; v0/v1 history: research/experiments.md 2026-07-09;
> hypothesis record: research/open_questions.md. Budget: RunPod ledger
> (datacore/runpod/ledger.jsonl), $49.42 remaining at filing, $5 gate floor ⇒
> ~$44 usable. First rollout state (human directive 2026-07-09): **NEVADA**.

## Where v1 left us (the baseline every bar is measured against)

`gv-detector-v1-2` (yolov8n, tiling 640/512, image-split, both AZ+KS regions in
train): **tile-level val mAP50 0.566, recall 0.499, precision 0.732.** That is
in-domain, out-of-sample by IMAGE only. The four gate-1 items below test what
that number does NOT: cross-region transfer, scene-level (not tile) accuracy,
NAIP domain, and headroom.

## PHASE 1 — the four gate-1 items, each with its bar STATED NOW

### (a) Held-out-REGION generalization
- **Design:** two folds. Fold-AZ = train on KS only, eval on AZ (unseen).
  Fold-KS = train on AZ only, eval on KS (unseen). yolov8n, tiling 640/512,
  60 epochs (v1 config), `--holdout-region`.
- **PRIOR (moderate):** cross-region transfer from a SINGLE training region is
  hard (AZ desert vs KS irrigated plains are different biomes); expect AP50
  0.25–0.45, below the 0.566 in-domain number.
- **PASS bar:** AP50 ≥ 0.30 AND recall ≥ 0.30 on the unseen region, for BOTH
  folds. ONE fold only ⇒ report PARTIAL (directional transfer, region-sensitive).
  Both < 0.30 ⇒ FAIL (no cross-region signal; need more training regions).
- **Why it matters for Nevada:** Fold-AZ (desert Southwest) is the closest proxy
  we have for Nevada; its number is the honest prior for Nevada tower recall.

### (b) Scene-level AP (stitching + cross-seam NMS), not tile-level
- **Design:** run the v1 model over every val scene's tiles, map detections back
  to scene-pixel coords, global NMS (IoU 0.5) across seams, match to scene GT at
  IoU 0.5, sweep confidence for a PR curve ⇒ scene AP50 + P/R at best-F1.
- **PRIOR:** scene-level < tile-level 0.566 (edge duplicates, seam misses);
  expect scene AP50 0.40–0.55.
- **BAR (measurement, not pass/fail):** publish scene AP50 + P/R honestly. Flag
  if scene-level < 0.40 (production geometry materially worse than tile metric).
  The scene number — not the tile number — is what per-state accuracy uses.

### (c) Real-NAIP domain test
- **Design:** run the ortho-trained v1 model on REAL NAIP tiles over an OSM-tower
  corridor (NAIP streaming = gridvision_naip_stac; OSM tower seeds via Overpass,
  since build_power_tiles omits power=tower). Measure recall vs OSM tower nodes.
- **HONESTY:** OSM under-maps towers ⇒ this is a **lower bound on recall**;
  precision needs a human-sampled pass = **BLOCKED-FOR-MIKE** (deferred, not faked).
- **PRIOR:** ortho→NAIP is a radiometry shift; expect NAIP recall 0.20–0.45.
- **PASS bar:** NAIP recall (vs OSM) ≥ 0.25 ⇒ the detector transfers to the
  imagery we'll actually run on. < 0.25 ⇒ NAIP training data required before
  rollout inference is trustworthy (feeds Phase 2's self-bootstrap).

### (d) Recall headroom — pull the untouched levers
- **Design:** yolov8s (one tier up), tiling 640/512, image-split (SAME val as v1
  for a fair delta), 80 epochs, mild extra augmentation. (yolov8m only if s clears
  the bar and budget allows — avoid a variant fish.)
- **PRIOR:** yolov8s + more epochs lifts AP50 to 0.60–0.68, recall to 0.55–0.62.
- **PASS bar (improvement counts only beyond noise):** AP50 ≥ 0.59 (v1 0.566 +
  ~0.025) AND recall ≥ 0.55 on the identical image-split val. Below ⇒ nano was
  not the bottleneck; report and keep v1 as champion.

## PHASE 2 — accuracy ratchet (permanent)

- **Leaderboard:** `datacore/gridvision/leaderboard.json` — one row per model
  version: {version, config, per-region + scene metrics, timestamp, honest
  notes}. Session-owned (serial-merge authority); pods push results to their own
  branches, the session collates. `scripts/gridvision_leaderboard.py` appends and
  **fails (exit non-zero) on any per-region regression** vs the current champion —
  a regression fails the build.
- **Champion rule:** a model is promoted to champion only if it does not regress
  ANY region's scene AP50 beyond a −0.02 noise band. Monotonic-or-better.
- **Self-bootstrap (already filed):** high-confidence, OSM-corroborated detections
  become new labels; model's own uncertain guesses NEVER do (that teaches its own
  errors). NAIP-domain + substation labels are the priority gaps.

## PHASE 3 — national rollout, NEVADA FIRST

Per state: detection pass over NAIP → detections tagged provenance
(osm-verified / ml-extended / ml-discovered) + confidence; accuracy measured vs
OSM; honest coverage manifest (mapped / absent / imagery vintage); PR merged
before the next state. Poor-imagery state ⇒ achievable partial + dated revisit
trigger, never silent-skip, never faked-complete. Per-state accuracy published;
NO single national number hiding weak states. Accuracy-gated promotion:
detections rise to higher-trust presentation only as measured accuracy clears
its pre-stated bar. **Nevada is state #1.**

## PHASE 4 — overlay (ships per the viz spec, not a demo)

Zoom-adaptive geometry, voltage/element color scheme, distinct provenance
rendering, clickable entity-graph-joined detail, per-state + per-provenance
toggles, freshness chips, perf-harness-gated vector tiles, mobile-flawless 390px.

## Budget plan + burn (honest)

- Gate-1 (a–d): ~4 GPU runs ≈ **$0.4–1.5** total (secure ~$0.11/run; spot cheaper
  where partial output is recoverable).
- Rollout: **$4–15/state** (to be refined by the Nevada pilot's measured burn).
- **$44 usable ⇒ gate-1 + roughly 3–8 states, NOT 50.** National completion needs
  a top-up; the exact PO is filed in wishlist.md once the Nevada pilot fixes the
  real $/state. **This is stated plainly: tonight advances the program materially
  and honestly; it does not finish it.**

## RESUME / STOP protocol

Stop cleanly when remaining hits the $5 gate floor. Leave: this file's status
updated, leaderboard.json current, a resume-state block in experiments.md naming
the exact next run, and a top-up PO in wishlist.md stating what the next $N buys.

---

## STATUS 2026-07-09 (results in, bars unchanged above)

- (a) held-out-region: **FAIL** — fold-AZ 0.056/0.143, fold-KS 0.059/0.092, both
  < the 0.30/0.30 bar. Single-region training does not transfer. → data-diversity
  is the rollout bottleneck.
- (d) recall headroom: **PASS** — yolov8s/80ep 0.642/0.588 ≥ 0.59/0.55. v2 champion.
- (b) scene-level: PENDING — pure core built+tested; needs on-pod inference driver.
- (c) NAIP domain: PENDING — needs the same driver + OSM/Overpass tower fetch.
- Revised spend order (logged in experiments.md): + training regions (Duke-US) →
  retrain multi-region → NAIP self-bootstrap → Nevada seed (honest low-accuracy).
