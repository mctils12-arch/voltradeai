# GRID VISION — Phase A Research

Research artifacts for the GRID VISION program (charter:
research/grid_vision.md). Append-only; each item is a dated section
assembled from the four parallel Phase A subagents (GV-A1..A4,
2026-07-07). Ladder context: these are DATA-layer inputs (gate 1) for a
power-infrastructure detector; Phase B begins with VERIFY over known OSM
corridors, so evaluation-set fitness is weighted above training fitness.
VERIFIED = the agent fetched and read the source; REPORTED = secondary.
NOTHING BUILDS until this doc + the A2 products plan
(research/grid_vision_products.md) are filed.

### Cross-cutting findings (assembler's summary — updated as items land)

- The April-2025 T&F paper (Item 1) is a validated RECIPE, not a
  deployable model: no released weights, no training annotations in
  the repo (despite its Data Availability statement), imagery
  ESA-locked. Adapting = re-executing the method on our imagery.
  Tower AP50 ~73% @ 0.3 m is the honest bar.
- LICENSE WALL (Items 1 and 2 converge independently): Esri World
  Imagery bulk tile use for ML training is outside its basemap terms —
  NAIP (public domain, 0.3–1 m CONUS) is the clean substrate for
  anything feeding a sellable model. At NAIP resolution TOWERS and
  SUBSTATIONS are realistic targets; distribution POLES are not
  (≤10 px). Honest Phase B scope: transmission towers + substations.
- Weights hygiene: Detectron2 COCO starting weights are CC-BY-SA 3.0 —
  fine-tuned weights stay INTERNAL; selling detections/outputs is
  fine. OSM labels are ODbL — a published corrected-OSM database
  stays ODbL; design the product boundary around that (Item 2.8).
- Evaluation design (Item 2.10): two-layer benchmark — Duke US zips
  (real ground truth, stale/narrow) + OSM-corridor recall on current
  NAIP with human-sampled precision. OSM supports RECALL only; never
  report "accuracy vs OSM" as if OSM were exhaustive.
- Esri wall now TRIPLE-CONFIRMED with quoted contract clauses (Item
  4.1: E204 §3.2(c) no scraping, §3.3(h) no AI/ML training, §3.3(b)
  no sold compilations; E300 fn.96 no deep learning on image
  services). Display basemap + identify capture-date metadata (our
  shipped use) remains the permitted category. NAIP streams free via
  Planetary Computer STAC (2010–2023) with no account.
- Detection consensus across four independent groups (Items 1, 3.1):
  tower mAP 0.6–0.85 at 0.3 m; ≥0.3 m needed to see half the towers;
  the universal weak link is line/GRAPH inference (F1 ~0.63) — which
  our OSM-as-base framing sidesteps (detector confirms/denies along
  known bearings instead of solving blind topology).
- The proven operating pattern is the World Bank/DevSeed HYBRID (Item
  3.2): high-recall model → candidate overlay → human adjudication UI
  (33× km²/h speedup, threshold 0.97, published false-positive
  taxonomy). Their mosaic-quality-boundary artifact becomes our
  ratchet: evaluate detectors PER-MOSAIC-SOURCE, never pooled —
  regime conditioning applied to imagery.
- Compute is NOT the obstacle (Item 4.6–4.7): TX corridor-verify is
  CPU-feasible (~7 h @0.6 m, streamable free from MPC); full-state
  GPU sweeps are $5–25/state and national re-scans $100–400 total;
  training $50–100. Data movement is the real cost — stream COG
  windows or compute next to the data. RunPod purchase order filed in
  wishlist (training needs it regardless: no released weights exist).
- SAR at 10 m: verification-only ladder experiment (gate-1 ROC vs
  offset controls), never blind discovery (Item 3.3).
  Shadow+Sundial: attribute estimator for tower height→voltage class,
  labeled estimate (Item 3.4) — time-of-day recoverable from shadow
  azimuth (2.1±3.4 min), so Esri's date-only metadata does not kill
  it.

---

## Item 1 — The April-2025 T&F power-infrastructure detection paper (GV-A1)

Date: 2026-07-07. Session type: [RESEARCH]. Repo cloned and inspected;
paper full text retrieved (VERIFIED throughout unless noted).

### 1.1 Citation block

> **Ye, Mengqi; Ward, Philip J.; De Plaen, Joël J.-F. G.; Koks,
> Elco E.** (2025). "A deep learning pipeline to power infrastructure
> detection in high-resolution satellite images." *Big Earth Data*,
> 9(3), 525–546. Taylor & Francis. DOI:
> 10.1080/20964471.2025.2490408. Received 2024-11-29, accepted
> 2025-03-20, published online **2025-04-16**, print issue
> 2025-07-03. License: **CC-BY 4.0** (gold OA). Funding: China
> Scholarship Council; EU Horizon CoCliCo (101003598), MIRACA
> (101093854), MYRIAD-EU (101003276). Affiliation: IVM, VU Amsterdam
> (Koks' Infrastructure Risk & Resilience group).

**Stated scope (exact):** detects **power towers and poles only** —
two classes (`tower`, `pole`; code optionally supports a third class
`streetlight`). It does **NOT** detect substations, lines, or
corridors (line/graph inference is cited as related work —
GridTracer — not implemented). Imagery: **pan-sharpened WorldView-3,
0.30–0.40 m, resampled to 0.30 m**, 4 bands requested (RGB+NIR) but
the pipeline converts to **RGB PNG** for training/inference. ~730 km²
over 9 AOIs in Vietnam, acquired 2019–2023 under ESA license.
Architecture: **Detectron2 (PyTorch) Faster R-CNN**, baseline
backbone **ResNet-101+FPN 3×** (also swept R50-FPN/C4/DC5,
R101-C4/DC5, X101-FPN); LR 0.001, batch 4, 1024×1024 tiles, early
stopping (patience 4, eval every 50 iters).

### 1.2 Code location — VERIFIED facts

- **Repo: https://github.com/Mengqi-Ye/PI-Detection** — named in the
  paper's Data Availability statement; public; verified via
  `git ls-remote` (HEAD `f8de4e7`, branches `main`, `tmp`,
  `tmp240606`, `tmp240616/06`) and full clone (1.3 GB, 839 files).
- **License:** MIT (© 2023 MengqiYe). Paper text CC-BY 4.0.
- **Language/framework:** ~99.7% Jupyter notebooks + a few Python
  scripts. Detectron2 on **PyTorch 1.10.2+cu113, Python 3.9, CUDA
  11.3** (pinned in `setup_detectron2.md`, WSL2-oriented).
  `environment.yml` covers only the geo stack
  (gdal/geopandas/rasterio) — torch/detectron2 install is manual.
- **Model weights: NOT released.** The only `.pth` in the repo is
  `2_model_training/eval/instances_predictions.pth` (79 KB — an eval
  predictions dump, not weights). No releases, no Zenodo/figshare
  deposit found.
- **Annotation dataset: NOT actually in the repo**, despite the Data
  Availability statement claiming ground-truth data is publicly
  available there. Checked `main` and `tmp` branches: zero
  `.geojson`/`.shp`/`via_region_data*` files. Only
  `eval/val_coco_format.json` (239 KB, val split in COCO pixel
  coordinates, no images). Raw WV3 imagery is ESA-restricted and
  cannot be redistributed regardless.
- **README quality: minimal to poor.** README is one line;
  `workflow.md` is developer notes (mixed English/Chinese, open TODO
  checkboxes). `src/03_config_train_evaluate_python.py` contains a
  **syntax error as committed** (`num_patience = ` — empty
  assignment), so the headline training script doesn't run without
  editing. Committed `wandb/` logs (~1.2 GB), `trash/`, and
  `.ipynb_checkpoints/` dirs. Reusable core:
  `2_model_training/train.py` (`setup_cfg`, `MyTrainer` with early
  stopping) and the tiling/COCO-JSON prep notebooks. Training
  hardware per wandb metadata: **single NVIDIA GTX 1080 (8 GB)**.

### 1.3 Reported metrics (quoted from the paper)

Test set: 15% of 9,104 tiles (70/15/15 split), 1024×1024 px @ 0.3 m,
WV3 Vietnam; dataset = 1,920 tower + 2,380 pole polygon annotations
(pole annotations include shadows; tower annotations exclude them).
F1/accuracy are tile-level classification metrics; AP is COCO-style
bbox.

| Metric | Baseline (Exp 1: R101-FPN-3×, LR 0.001, BS 4, all empty tiles) | Best variant (Exp 10: no empty tiles) |
|---|---|---|
| AP (IoU 0.50:0.95) | 30.3% | 69.0%* |
| **AP50 (all classes)** | **60.6%** | — |
| AP50 — tower | **72.9%** | 77.2% |
| AP50 — pole | **49.8%** | 61.2% |
| F1-score | **74.7%** | 96.2%* |
| Accuracy | **90.9%** | 92.7% |

*Caveat: Section 5 of the paper has internally inconsistent metric
labels between paragraphs (one paragraph calls 60.6 "overall AP" and
74.7 "AP50"); the abstract and conclusion consistently state AP50
60.6 / F1 74.7 / accuracy 90.9 / tower 72.9 / pole 49.8 — treat those
as authoritative. Ranges across all 26 experiments: AP50 40.4–69.0%,
F1 49.4–96.2%, accuracy 66.7–92.7%. Paper's own comparison: their
tower AP50 0.73 vs GridTracer's 0.52 (Huang et al. 2022) and Qiao et
al. 2020 Faster R-CNN AP 0.654 (hard subset)/0.871 (standard). The
committed eval snapshot in the repo (`results.json`: AP50 38.8,
AP-tower 28.1, AP-pole 11.0; tile precision 0.386/recall 0.771/F1
0.515) is an intermediate run, far below the paper baseline — the
final model is NOT reproducible from repo artifacts alone.

### 1.4 Adaptability assessment

**Verdict: ADAPTABLE-WITH-WORK (heavy) — the recipe is usable, the
artifacts are not.** What transfers is the method (tiling, OSM-seeded
annotation, COCO-JSON prep, Detectron2 config, hyperparameters,
baseline expectations). What's missing is everything that would save
us time: no trained weights, no training annotations, imagery
license-locked.

- **Input match:** RGB 1024×1024 @ 0.3 m/px. **Esri World Imagery
  z19 ≈ 0.22–0.30 m/px** at US latitudes — excellent resolution
  match; a 1024² patch = 4×4 stitched XYZ tiles (~307 m square). Glue
  code is modest (~150–250 lines: tile fetch/stitch, WebMercator
  pixel→lon/lat back-projection, cross-tile NMS) — and we already run
  tile plumbing in datacore (owmTiles). **NAIP 0.6–1 m:** 2–3×
  coarser than training res; poles (median 39 m² incl. shadow ≈
  20×20 px @0.3 m) shrink to ≤10 px and would largely be lost; towers
  (median 194 m²) plausibly survive but expect material AP
  degradation without retraining at NAIP resolution. **Sentinel-2
  10 m: NOT-USABLE** (a tower is 1–2 px).
- **Compute:** trained on one GTX 1080 (8 GB) — retraining is
  hobby-GPU scale. Detectron2 model zoo lists R101-FPN-3× at
  0.051 s/im on V100; **CPU inference works** in Detectron2 and runs
  roughly 50–100× slower — ballpark 2–8 s per 1024² tile per core.
  Feasible for targeted corridor/substation-area scans (thousands of
  tiles overnight), not for state-wide sweeps. (GV-A4 quantifies.)
- **Retraining: mandatory, not optional.** Since neither weights nor
  training annotations ship, "adapting" = re-executing their recipe:
  seed candidate locations from OSM `power=tower/pole` (their exact
  method), annotate ~2–4k instances on the target imagery, fine-tune
  COCO-pretrained Faster R-CNN (or, honestly, a modern YOLO — the
  paper itself concedes Faster R-CNN was chosen for comparability,
  not superiority). The best open US training substrate is the Duke
  dataset (Item 2.1).
- **License constraints on commercial use:** paper CC-BY 4.0
  (attribution) — fine. Repo MIT — fine. Detectron2 code Apache-2.0 —
  fine. **Detectron2 model-zoo COCO starting weights are CC-BY-SA
  3.0** ("All models available for download through this document are
  licensed under the Creative Commons Attribution-ShareAlike 3.0
  license") — selling *detections/outputs* is fine, but
  redistributing fine-tuned *weights* would inherit ShareAlike; keep
  weights internal. **Esri World Imagery**: bulk tile scraping for ML
  outside ArcGIS products violates Esri's terms — for a monetized
  product the clean path is **NAIP (public domain)**, which is
  exactly the resolution where poles get hard: expect towers-only
  capability, or budget the resolution-vs-rights tradeoff explicitly.

### 1.5 Nearest open alternatives (since weights are absent)

No off-the-shelf satellite-view tower-detection weights exist in the
open ecosystem — every candidate requires training:

- LisavilaLee/SCAResNet_mmdet (GitHub) — IEEE GRSL, ResNet variant
  for tiny transmission/distribution towers, mmdetection; code only,
  no released checkpoints.
- Duke GridTracer dataset (Huang et al. 2022) — best US-relevant
  annotations (Item 2.1); the published bar is tower AP50 0.52.
- gubbriaco/ttpla-detector + TTPLA dataset — towers/lines but **UAV
  imagery**, wrong domain for overhead tiles (Item 2.3).
- Substation segmentation (different target, relevant to scope):
  thisishardik/electrical_substation_detection (UNet, ICETCI 2021
  challenge, TF/Keras),
  arnabk001/Electrical-Substation-Detection-from-Satellite-Images.

**Bottom line:** this paper is a validated recipe with honest
baseline numbers (tower AP50 ~73% @ 0.3 m is the bar), not a
deployable model. Glue code is cheap; the annotation+retraining cycle
(~2–4k instances, one consumer GPU or rented spot instance) is the
real cost. Poles are out of reach on public-domain US imagery (NAIP);
towers are the realistic first target.

Sources (all VERIFIED by the agent): T&F article full text
(tandfonline.com/doi/full/10.1080/20964471.2025.2490408), Crossref
metadata, PI-Detection repo (cloned & inspected), VU research portal,
MIRACA project page, Detectron2 MODEL_ZOO, Big Earth Data
announcement.

---

## Item 2 — Labeled datasets for power-infrastructure detection from overhead imagery (GV-A2)

Date: 2026-07-07. Session type: [RESEARCH]. All URLs probed live this date
unless noted. "Probe" = HTTP request from this session (through the agent
proxy); figshare/HF/GCS probes are authoritative, GitHub release-asset
probes are NOT (our proxy blocks unauthenticated GitHub asset downloads —
flagged where it matters).

### 2.1 Duke Electric Transmission and Distribution Infrastructure Imagery Dataset — THE CANONICAL ONE

- **URL / DOI**: https://figshare.com/articles/dataset/Electric_Transmission_and_Distribution_Infrastructure_Imagery_Dataset/6931088 — DOI `10.6084/m9.figshare.6931088.v1` (published 2018-08-03; Bradbury, Han, Nair, Pathirathna, You — Duke Data+ "Energy Infrastructure Map of the World").
- **Downloadable today**: YES, verified. figshare API (`api.figshare.com/v2/articles/6931088`) returns 16 files with live `ndownloader.figshare.com` URLs; we followed one 302 to signed S3 and actually downloaded `Documentation.pdf` (5.5 MB, 13 pages) this session.
- **Size**: 28.1 GB total, 16 zips (per-city) + docs + 256 MB `Sample Data.zip`.
- **Contents**: 511 images, ~321 km², 14 cities, 6 countries. US portion (5 zips, our target): Hartford CT (25 img, 0.15 m, UConn/CT ECO), Clyde NC (8), Wilmington NC (12), Colwich+Maize KS (15) all 0.15 m USGS orthoimagery 10,000×10,000 px; Tucson AZ (12, 0.30 m). Non-US: Matamoros MX (0.15 m), 5 NZ cities (0.125 m LINZ), Khartoum/Shanghai/Rio (0.30 m WorldView-3 via SpaceNet, 1300×1300 chips).
- **Label classes (7)**: Distribution tower (DT), Distribution line (DL), Transmission tower (TT), Transmission line (TL), Other tower (OT), Other line (OL), **Substation (SS)**. Towers are points, lines are polylines, substations are polygons.
- **Annotation format**: per-image GeoJSON (WGS84 geo-coords + pixel coords + metadata), CSV (per-vertex), and rendered multiclass masks (.npz + .tif, mask values 1–7). Mask-generation script included.
- **License**: figshare license field = **"CC BY 4.0"** (`https://creativecommons.org/licenses/by/4.0/`). Commercial use and model training permitted with attribution. CAVEAT: the three SpaceNet-derived cities (Sudan/China/Brazil) rest on SpaceNet imagery whose upstream license is CC BY-SA 4.0 — ShareAlike contamination risk for derivatives of those images. The US zips (USGS / Connecticut state orthoimagery) carry no such problem. **For a US-only eval set, license is clean.**
- **Known reuse as detection benchmark**: GridTracer (arXiv 2101.06390) and SCAResNet (arXiv 2404.04179) both train/evaluate tower+line+substation detection on it — it is the community's de-facto benchmark for nadir T&D detection.
- **Domain gaps for us**: imagery vintage ~2016–2018 (grid has changed; fine for detector eval, weak for current-state verification); 0.125–0.3 m is finer than our 0.3–1 m target — downsample to simulate NAIP GSD; distribution poles labeled at 0.15 m will be invisible at 1 m (transmission towers and substations survive downsampling).

### 2.2 SRSPTD — Duke re-cropped into ready-made YOLO tower boxes

- **URL**: https://github.com/ZX815/LSKF-YOLO (dataset committed directly in-repo: folders `SRSPTD Dataset` (YOLO), `SRSPTD(VOC)` 7:2:1 splits, `SRSPTD（5-k cross_val)`). From LSKF-YOLO, IEEE TGRS 2024, doi 10.1109/TGRS.2024.3389056 (North China Electric Power University).
- **Downloadable today**: YES — repo public, README and folder listing fetched this session (raw.githubusercontent.com verified; dataset files live in the git tree, no external host).
- **Contents**: 512×512 crops of the Duke dataset's six best regions — Tucson AZ, Hartford CT, Colwich&Maize KS, Wilmington NC + Tauranga & Dunedin NZ — re-annotated with labelImg into **bounding boxes**, 2 classes: transmission tower, distribution tower (distribution towers labeled together with their shadows, deliberately). Image count not stated in README; it is the tiling of ~72 large source images.
- **Annotation format**: YOLO txt (primary) + Pascal VOC XML.
- **License**: NO license file in repo. Derivative of CC BY 4.0 Duke data, so redistribution should carry CC BY, but the repo itself is legally ambiguous. Usable for internal evaluation; do not redistribute without pinning this down (authors publish contact emails in README).
- **Why it matters**: someone already did the polygon/point→detection-box conversion labor on exactly the US regions we care about. Fastest path to a tower-detection eval set.

### 2.3 TTPLA — aerial (drone) transmission towers + power lines

- **URL**: https://github.com/R3ab/ttpla_dataset (paper: arXiv 2010.10032, ACCV 2020).
- **Downloadable today**: YES, verified — Google Drive file `data_original_size_v1.zip` **(4.2 GB)** confirmed live this session (id `1Yz59yXCiPKS0_X4K3x9mW22NLnxjvrr0`; large-file confirm page returns filename+size).
- **Contents**: 1,100 images (3840×2160) extracted from 80 videos shot by a Parrot-ANAFI UAV "in two different states in USA". 8,987 instances: 8,083 cable, 330 tower_lattice, 283 tower_wooden, 168 tower_tucohy (concrete/steel/hybrid), 173 void. Camera angles: front, top, AND side views — mostly oblique, not nadir.
- **Annotation format**: LabelMe polygons → COCO JSON (instance segmentation; supports detection + semantic seg).
- **License**: **Apache 2.0** — LICENSE file verified in repo root. Commercial use, modification, redistribution all fine with notice preservation.
- **Domain gap**: SEVERE for our task. Drone-altitude oblique frames of tower sides bear little resemblance to 0.3–1 m nadir satellite pixels where a tower is a ~10–40 px lattice footprint + shadow. Useful for cable/line segmentation ideas and tower-type taxonomy, not for evaluating a nadir detector.

### 2.4 PLAD / STN PLAD — drone close-up power line assets

- **URL**: https://github.com/andreluizbvs/PLAD (paper arXiv 2108.07944, SIBGRAPI 2021). Download: GitHub release `1.0` asset `plad.zip` + `labels.zip`, Google Drive mirror.
- **Downloadable today**: PROBABLY — release page verified to exist with 3 assets ("STN PLAD", tag 1.0); direct asset probe returned 403 **through our sandbox proxy** (GitHub asset downloads are blocked for this session — this is our network, not a takedown). Re-probe from an unproxied machine before relying on it.
- **Contents**: only **133 images** (5472×3078/3648, UAV close-range inspection flights along Brazilian transmission lines — STN = Sistema de Transmissão Nordeste), 2,409 instances, 5 classes: transmission tower, insulator, spacer, tower plate, Stockbridge damper.
- **License**: repo LICENSE file = **GPL-3.0** (verified). The arXiv listing shows CC BY-NC-ND on the paper. GPL on a dataset is legally murky and copyleft; treat as unusable for a commercial model.
- **Transfer to satellite view: NO.** These are near-field oblique photos of tower hardware (insulator strings, dampers). Zero of the five classes except "transmission tower" even exists at satellite GSD, and the tower views are side-on close-ups. Successor **InsPLAD** (github.com/andreluizbvs/InsPLAD, 10k+ images) has the same character. Not a candidate; recorded to close the question.

### 2.5 SubstationDataset (TransitionZero / Lindsay-Lab) — OSM substations over Sentinel-2

- **URL**: https://huggingface.co/datasets/neurograce/SubstationDataset (paper arXiv 2409.17363; training code github.com/Lindsay-Lab/substation-seg). Also a split in GEO-Bench-2.
- **Downloadable today**: YES, verified via HF API — public, not gated: `images.zip`+`.z01`+`.z02` (~76 GB), `annotations.json` (30 MB), `mask.tar.gz`, split CSVs. (The older GCS tarballs referenced in the repo README are 404 — HF is the live host.)
- **Contents**: 26,522 image-mask pairs, global, locations from OSM `power=substation`; 228×228 px, 13-band Sentinel-2 at **10 m GSD**, 4–5 revisits per location.
- **License**: **cc-by-4.0** (HF card metadata, verified via API). Underlying labels are OSM (ODbL — attribution + share-alike on the *database*), imagery is free-use Copernicus.
- **Fit**: wrong resolution regime for detection at 0.3–1 m — a substation is ~5–30 Sentinel pixels. Its value to us is (a) a cleaned global substation *location list* (OSM-derived, deduplicated, >10 km² filter), (b) a working precedent for exactly the OSM-weak-label pipeline we plan, including their label-noise handling.

### 2.6 GridNet-HD — high-res oblique imagery + LiDAR of power corridors

- **URL**: https://huggingface.co/datasets/heig-vd-geo/GridNet-HD (arXiv 2601.13052; HEIG-VD, Switzerland; 10.7k HF downloads).
- **Downloadable today**: YES — public HF dataset, verified via API.
- **Contents**: 7,694 images + 2.5 B LiDAR points, co-referenced, 11 semantic classes of power-line assets; European corridors; oblique/helicopter-style capture.
- **License**: **cc-by-4.0** (verified via HF API).
- **Fit**: wrong modality (oblique + LiDAR, non-US). Relevant only if GRID VISION later grows a 3D/corridor-inspection leg.

### 2.7 ICETCI 2021 substation challenge — right GSD, effectively unobtainable

- ~1 m resolution satellite chips over India, 100 training chips each containing a substation, polygons provided; the one dataset at exactly our GSD with substation labels. Hosted behind CodaLab competition 32132 registration; competition ended 2021, data not republished (checked entrant repos — code only). **Treat as unavailable**; not worth the acquisition effort given Duke SS polygons + OSM cover the need.

### 2.8 OSM-as-labels over NAIP/Esri imagery — the weak-label approach

**Geometry source**: Overpass API queries for `power=tower` (nodes), `power=pole`, `power=line` (ways), `power=substation` (polygons/nodes). This is exactly how SubstationDataset (2.5) and several papers built their label sets.

**Tooling, verified state:**
- **Raster Vision** (Element 84): ALIVE — PyPI `rastervision` 0.31.1, released 2024-08-30. Native GeoJSON-vector-labels → chip pipelines for detection/segmentation. The strongest off-the-shelf fit.
- **torchgeo** (Microsoft): actively maintained; `RasterDataset`/`VectorDataset` intersection gives NAIP-tiles × OSM-geometry sampling in a few dozen lines.
- **label-maker** (Development Seed): DEAD for practical purposes — last PyPI release 0.9.1, 2020-11-19, and it depends on OSM QA Tiles, which are no longer updated. Do not build on it.
- **DeepOSM** (trailbehind): proof-of-concept ancestor (NAIP + OSM PBF), archived-stale. Precedent only.

**Imagery licensing — decisive**: NAIP is USDA public domain at 0.3–1 m over CONUS — free for commercial ML, no restrictions. **Esri World Imagery is NOT safe**: its basemap terms prohibit bulk extraction/ML training without an appropriate license. Papers that trained on Esri tiles do not transfer us their rights. Use NAIP (or state orthoimagery programs, mostly public records) for anything that feeds a sellable model.

**Label-noise caveats (why OSM is weak, not ground truth):**
- **Completeness is voltage-stratified**: global transmission coverage ~75% (MapYourGrid, 2026; targeting 98% by 2028). US ≥345 kV is well mapped; 69–161 kV subtransmission uneven; distribution largely absent. Individual `power=tower` nodes are mapped far less completely than the lines they carry — many corridors have the line way but no tower nodes.
- **Positional offset**: towers were digitized against imagery of varying vintage/orthorectification; expect meters of offset — enough to matter for a 10–40 px object. Use buffered matching (e.g., 15–25 m) for scoring, not IoU on points.
- **Temporal mismatch**: OSM edit date vs NAIP acquisition date can differ by years; new lines and demolished corridors both occur.
- **Asymmetric evidence**: OSM presence ≈ reliable positive (validation studies ~75%+ accuracy across countries); OSM ABSENCE proves nothing. Consequence for evaluation: OSM corridors support **recall** measurement; **precision** (false-positive rate over OSM-empty land) requires human review of a detector-output sample. Never report "accuracy vs OSM" as if OSM were exhaustive.
- OSM data is ODbL: attribution required; share-alike applies to derivative *databases* (a published tower database seeded from OSM must be ODbL) — model weights and internal signals are generally regarded as produced works, but a sold "US tower database" built by correcting OSM would carry ODbL. Design the product boundary with this in mind.

### 2.9 Ranking for OUR task (transmission towers + substations, 0.3–1 m US nadir imagery)

1. **Duke figshare dataset (US zips)** — only nadir, georeferenced dataset with BOTH tower and substation labels on US soil at relevant-adjacent GSD; clean CC BY 4.0 for the US portion; verified downloadable. Use downsampled to 0.3/0.6/1.0 m to measure GSD sensitivity. Gaps: 2016–18 vintage, 5 regions only (geography bias: no Midwest flatland-with-pivot-irrigation confusers, no Northeast dense canopy beyond CT), point labels for towers (need box synthesis or use SRSPTD).
2. **SRSPTD** — the same data already converted to YOLO/VOC tower boxes with a transmission/distribution split; verified in-repo. License ambiguity (no LICENSE file) confines it to internal eval until clarified.
3. **OSM-over-NAIP weak labels** — not a "dataset" but the only source that covers ALL of CONUS on current imagery with unrestricted licensing; recall-only evaluation without human review; this IS the Phase B VERIFY substrate.
4. **SubstationDataset (HF)** — wrong GSD (10 m) but a vetted global substation location list + the closest methodological precedent; CC BY 4.0; verified.
5. **TTPLA** — clean Apache 2.0, US, but oblique drone domain; mine it for tower-type taxonomy and possibly synthetic-augmentation, not evaluation.
6. **GridNet-HD** — high quality, CC BY 4.0, but oblique+LiDAR European corridors; park it.
7. **PLAD/InsPLAD** — close-up asset inspection, GPL/unclear license; no transfer to satellite. Closed.
8. **ICETCI 2021** — right GSD, unobtainable. Closed.

### 2.10 Verdict — do existing labels suffice to EVALUATE (not train) a detector on US imagery?

**YES, with a two-layer design.** (a) *Static benchmark*: Duke US zips (+SRSPTD boxes) give ~70 large georeferenced US images across 5 regions with TT/DT/SS labels — enough for per-class PR curves at multiple simulated GSDs, which is what Phase A needs before any model is trusted. (b) *Live-domain benchmark*: because Duke's imagery is ~8–10 years old and 5-region-biased, final evaluation must ALSO run over current NAIP with OSM corridors as recall targets (buffered matching) plus a human-reviewed sample of detections over OSM-empty tiles for precision. Neither layer alone is honest: Duke has real ground truth but stale/narrow imagery; OSM has current, CONUS-wide imagery but one-sided labels. Together they bound the detector's true performance. For TRAINING a production model, existing sets are thin (Duke 5 regions + weak labels) — expect to need pseudo-labeling + a self-annotated hard-negative set in Phase C, but that does not block Phase B VERIFY.

**License bottom line for a sellable model**: build eval and training exclusively on Duke-US (CC BY 4.0, attribute), TTPLA (Apache 2.0), NAIP (public domain), OSM (ODbL — attribute; keep any published corrected-OSM database ODbL or independently re-derived). Avoid: PLAD (GPL), Esri imagery tiles (terms), Duke's SpaceNet cities (CC BY-SA upstream).

---

## Item 3 — Detection methods survey: routing, hybrid, SAR, shadow-height (GV-A3)

Date: 2026-07-07. Session type: [RESEARCH]. VERIFIED = fetched and
read; REPORTED = secondary/search-snippet. License flags are for a
commercial data-products platform.

### 3.1 Family 1 — Optical tower detection + inter-tower routing / corridor extraction

Primary works (all resolve):

1. **GridTracer** — Huang, Yang, Streltsov, Bradbury, Collins, Malof,
   *IEEE JSTARS* 15 (2022); arXiv:2101.06390. VERIFIED (abstract +
   full text via ar5iv). The most complete precedent for our exact
   pipeline: (a) tower detection = Faster R-CNN/Inception-V2 with
   shrunken anchors {10²…200²}px; (b) line segmentation = StackNetMTL
   (road-segmentation architecture); (c) **graph inference** =
   candidate tower pairs within 600 m scored by integrating
   line-segmentation probability along the straight path between
   them, connect if mean score ≥ γ=0.2. Metrics at **0.3 m**:
   distance-based tower mAP AZ 0.73 / KS 0.54 / NZ 0.55 (avg 0.61);
   graph F1 avg 0.63. Human baseline: 0.86 / 0.77 — machines still
   well below humans, which argues for the hybrid (Family 2) pattern.
   - **Dataset (ETDII)**: figshare 14935434, **CC BY 4.0**, 347 MB,
     ~264 km², Tucson AZ, Colwich KS, 5 NZ cities, fully annotated
     T&D towers+lines. VERIFIED via figshare API. Commercially usable
     with attribution. (Same family as Item 2.1's 6931088 deposit.)
   - **Code**: github.com/bohaohuang/transmission_grid — VERIFIED:
     TensorFlow OD API, 1 commit, abandoned, **NO LICENSE file →
     all-rights-reserved; do not copy code into our repo. Reimplement
     the (simple, well-described) pair-scoring algorithm from the
     paper instead.**
2. **RetinaNet + routing + corridor** — Haroun, Deros, Md Din,
   "Detection and Monitoring of Power Line Corridor From Satellite
   Imagery Using RetinaNet and K-Mean Clustering," *IEEE Access* 9
   (2021), DOI 10.1109/ACCESS.2021.3106550, gold OA (CC BY). VERIFIED
   citation+metrics via Semantic Scholar API. mAP **72.45% @IoU 0.5,
   85.21% @IoU 0.3**; routing connects each adjacent detected tower
   pair, then buffers into a corridor for vegetation monitoring
   (k-means in HSV). REPORTED (IEEE page snippet): "imagery resolution
   must reach at least 0.3 m to detect at least half of the power
   towers" — consistent with GridTracer and decisive for us: **~0.3 m
   yes; 0.6–1 m marginal for distribution poles, workable for large
   lattice towers only.** No public code found.
3. **PI-Detection** (Item 1 covers in depth) — MIT code, Detectron2;
   AP50 60.6%, F1 74.7% on WV3 Vietnam. Best license-clean modern
   starter code.
4. **YOLO line of work** — YOLOv9-GDV, Zhang et al., *Remote Sensing*
   17(13):2229 (2025). VERIFIED (mdpi.com/2072-4292/17/13/2229):
   mAP@0.5 **80.2%** on SRSPTD (sub-meter, US+NZ — a repackaging of
   the Duke ETDII imagery) and **94.6%** on 1 m GaoFen (proprietary
   GFTD). SRSPTD repo VERIFIED: github.com/ZX815/LSKF-YOLO (no
   explicit license — use the CC BY upstream figshare data instead).
   LSKF-YOLO citation: Shi et al., *IEEE TGRS* 62 (2024), DOI
   10.1109/TGRS.2024.3389056 (REPORTED).
5. **Least-cost-path corridor inference** — **gridfinder**, Arderne
   et al., *Scientific Data* 7:1 (2020) (REPORTED;
   s41597-019-0347-4). Code VERIFIED: github.com/carderne/gridfinder —
   **MIT**, active (v3.1.2, Apr 2024), VIIRS night-lights + OSM roads
   + many-to-many Dijkstra. Predicts *where MV/HV lines plausibly
   run*, not towers — ~1 km-scale prior, not evidence. Useful only as
   a search-space prior for B3.
6. TTPLA (Apache-2.0, VERIFIED) is drone **side-view** — not
   adaptable to nadir tiles. Ignore for detection.

**Adaptability:** directly on-target. Detection at 0.3 m works (mAP
0.6–0.85 across four independent groups); the weak link everywhere is
line/graph inference (F1 ~0.63) — which is exactly why our OSM-as-base
framing is right: with OSM providing topology, the detector only has
to confirm/deny towers along known bearings, a far easier problem
than GridTracer's blind graph. Compute: YOLO-class detector in ONNX
runs on CPU at a few 512px tiles/s/core — corridor-restricted
inference is CPU-feasible; nationwide blind sweeps are not (GPU
territory; Item 4 quantifies).

### 3.2 Family 2 — HOT / Development Seed human-in-the-loop hybrid (Pakistan, Nigeria, Zambia)

Primary sources: World Bank/ESMAP report "Machine Learning for High
Resolution High Voltage Grid Mapping" (2020) — VERIFIED, full PDF
read (documents1.worldbank.org/curated/en/614661605635613986/);
project docs VERIFIED developmentseed.org/ml-grid-docs/ and
developmentseed.org/projects/hv-grid/; dataset page REPORTED
(energydata.info).

**What was ML vs human (VERIFIED from report):**
- ML: **Xception** binary tile classifier (ImageNet transfer, Keras),
  P(HV tower) per **zoom-18 (~0.5 m/px) DigitalGlobe tile**. ~150
  model variants tuned with Hyperopt. Decision threshold **0.97**
  chosen from ROC / signal-detection framing. GPU inference:
  "hundreds of thousands of images per hour," 2–3 AWS GPU instances,
  countries split into 7–10 zones, scores shipped as GeoJSON
  overlays. Scale: Pakistan 50M z18 tiles, Nigeria 40M, Zambia 34M.
- Known false-positive classes (VERIFIED): wind turbines, dune
  shadows, gridded farmland, burned areas, and **imagery-quality
  boundaries** (sharp changes in prediction density where mosaic
  quality changes) — directly relevant to any heterogeneous mosaic.
- Human: 8 professional mappers in **JOSM with a customized To-Fix
  plugin** iterating through predicted-tower squares; humans traced
  tower-to-tower, added substations, fixed 50–100 m position errors;
  every OSM edit human-made; full validation second pass.

**Throughput (VERIFIED, Table 1 + Fig 17):** person-hours
mapping+validating at country scale — Pakistan 364.1+181.1, Nigeria
243.1+74.5, Zambia 167.2+29.5. Speedups vs manual: **33.4× km²/hour
scanned; 9.7× towers/hour; 15.9× substations/hour** (manual baseline
~120 km²/h).

**Open tooling:**
- Model/code: github.com/developmentseed/ml-hv-grid-pub — VERIFIED,
  **MIT**, includes training + pretrained weights (2018-era Keras;
  pattern reusable, code stale).
- **ml-enabler**: hotosm/ml-enabler VERIFIED — BSD-2-Clause,
  **archived 2025-03-07**; successor
  github.com/developmentseed/ml-enabler VERIFIED — BSD-2, model
  registry + prediction store + human feedback loop. License-clean to
  borrow architecture or code.
- **fAIr** (HOT's current AI-mapping service): github.com/hotosm/fAIr
  VERIFIED — **AGPL-3.0** ⚠️ copyleft: do NOT vendor or link into our
  product; YOLOv8-based, buildings/roads-focused, active (v2.2.19,
  Mar 2026). Workflow ideas (fAIrSwipe swipe-validation of
  predictions) are freely imitable; code is not.
- Tasking Manager: BSD-2 (REPORTED).

**Adaptability — highest of all four families.** The proven pattern
is exactly our architecture: cheap high-recall classifier → GeoJSON
candidate overlay → human verification UI → validated edits. Two
upgrades for 2026-us: (1) replace the human *tracing* step with the
Family-1 detector + OSM topology (humans only adjudicate flagged
segments via a /data web UI — minutes/week, not person-months); (2)
replace tile classification with corridor-restricted object
detection. The 0.97-threshold + ROC framing and the false-positive
taxonomy transfer verbatim.

### 3.3 Family 3 — SAR (Sentinel-1) tower detection / verification

- **High-resolution SAR works; 10 m Sentinel-1 blind detection is
  undemonstrated.** Li et al., "Hierarchical Transmission Tower
  Detection from High-Resolution SAR Image," *Remote Sensing*
  14(3):625 (2022) — VERIFIED: Gaofen-3 **3 m and 8 m**, SCR +
  density + convex-hull geometric filtering, 95.5% detection on a
  tiny 22-tower set, 1 false alarm. No claim about 10 m.
- Peng et al., "Power Transmission Tower Series Extraction in PolSAR
  Image Based on Time-Frequency Analysis and A-Contrario Theory,"
  *Sensors* 16(11):1862 (2016) — VERIFIED (PMC5134521): RADARSAT-2
  8 m / Pi-SAR 3 m / X-band 0.5 m; towers detected 85–100% **but
  false-alarm rates 58–76%** before exploiting the key prior: towers
  form **regularly-spaced collinear series** (a-contrario grouping).
  That prior is the transferable idea.
- P2Det prompt-learning oriented tower detection, arXiv:2404.01074 —
  VERIFIED abstract: detector conditioned on **point prompts** for
  high-res SAR. Conceptually "OSM-prompted verification" — but on
  high-res SAR, not S1.
- Sentinel-1 tower literature is **InSAR deformation monitoring, not
  detection**: Frontiers in Earth Science 14:1606062 (2026) —
  VERIFIED: explicitly states Sentinel-1's "relatively low spatial
  resolution hampers the precise detection of minor deformations at
  individual tower foundations." Matikainen et al., *ISPRS JPRS*
  119:10–31 (2016) (REPORTED): SAR line/pylon visibility is
  orientation-dependent, demonstrated mainly at high-res X-band.
- Encouraging analogy: offshore wind turbines routinely mapped in S1
  10 m GRD time series (arXiv:2604.20822, REPORTED) — but over a
  *dark ocean*. Land clutter is the whole problem.

**Honest assessment:** blind discovery at 10 m — no. **Verification
over known OSM corridors — plausible but unproven; cheap to test.**
The verification framing removes both hard parts: location prior (OSM
gives corridor + bearing + expected ~250–450 m spacing) and the FP
problem (only score contrast at predicted tower sites, both
ascending+descending passes, VV+VH, on a 50–100-scene temporal mean
that suppresses speckle ~√N). Expected to work for large rural
lattice towers (500 kV class, 40–55 m, strong corner-reflector
geometry) against field/rangeland background; expected to fail in
urban clutter and for small poles. Data free (Copernicus — our CDSE
pipeline), math is means and peak-finding — **fully CPU-compatible**,
and it is our only **cloud-independent, continuously-refreshing**
signal (useful later for change detection: tower removed/added). Run
as a ladder-gate-1 experiment: score S1 composite contrast at N known
tower sites vs N offset control points along the same corridors;
publish the ROC to experiments.md before believing anything.

### 3.4 Family 4 — Shadow-height inference / 3D

The method: height H = shadow length L × tan(solar elevation). Solar
elevation needs date + **time** + lat/lon.

**Key finding — unknown time-of-day does NOT kill the method:**
- **"Sundial: A method for inferring image acquisition time from
  shadow orientation"** — Bae, Legleiter, Yager, *Earth Surface
  Processes and Landforms* (2025), DOI 10.1002/esp.70157 — VERIFIED
  (USGS publications page): given date + location, measure shadow
  **azimuth**, invert solar geometry for time-of-capture; error
  **2.1 ± 3.4 min**, validated on 16 **WorldView** + 6 **NAIP**
  images — precisely our imagery classes. So: shadow direction → time
  → sun elevation → tower height, all from one image. The Esri
  identify contract gives capture DATE at high zoom (no time-of-day);
  tiles with absent metadata (99999) degrade to bounds.
- Fallback bounds without Sundial are loose: constellation local
  overpass times differ — GeoEye-1/WV-2 LTDN 10:30 (REPORTED), WV-3
  LTDN 13:30 (VERIFIED, eoPortal) — ±1.5 h ⇒ tens-of-% height error.
  Use Sundial, not assumptions.
- Supporting: Qureshi et al., "Building Height Estimation Using
  Shadow Length in Satellite Imagery," arXiv:2411.09411 (2024) —
  VERIFIED abstract: YOLOv7 building/shadow localization + ResNet18
  shadow-length regression, 42 cities; the pipeline automates.
- **Stereo/photogrammetry: dead on arrival** — basemap tiles and NAIP
  DOQQs are mono orthomosaics; no stereo pairs. Drop. (For US tower
  heights, USGS 3DEP lidar is the honest ground-truth substitute and
  is free — better than any shadow method where coverage exists; a
  data-root note, not an imagery method.)

**Adaptability:** feasible and CPU-trivial, but an **attribute
estimator, not a detector**: given a confirmed tower, estimate height
→ voltage class (345/500 kV lattice ~35–55 m vs 69/115 kV ~20–30 m)
as a verification/enrichment field. Error budget: ±1–2 px shadow
measurement, terrain slope (correct with 3DEP DEM), shadow tip on
vegetation, faint lattice shadows at 1 m NAIP. Label outputs as
estimates per the honesty clause.

### 3.5 License summary (commercial platform)

| Asset | License | Use |
|---|---|---|
| ETDII/Duke dataset (figshare 14935434) | CC BY 4.0 | OK — train/eval with attribution |
| PI-Detection (Mengqi-Ye) | MIT | OK — code reuse |
| ml-hv-grid-pub (DevSeed) | MIT | OK — pattern + code |
| ml-enabler (DevSeed/HOT) | BSD-2 | OK |
| gridfinder | MIT | OK |
| TTPLA | Apache-2.0 | OK but not applicable |
| bohaohuang/transmission_grid | none | NO code reuse; reimplement from paper |
| hotosm/fAIr | AGPL-3.0 | ideas only, never vendor/link |
| SRSPTD repackaging (ZX815) | none stated | use CC BY upstream instead |

### 3.6 Ranked recommendation

**Phase B1 — VERIFY over OSM corridors:**
1. **Corridor-restricted optical tower detector** (Family 1 detector
   + Family 2 workflow): YOLO/Faster R-CNN fine-tuned on ETDII
   (CC BY) + our own OSM-derived US chips, inference only inside
   buffered OSM corridors. CPU-viable at corridor scale; proven
   metrics; PI-Detection (MIT) as starter. *The workhorse.*
2. **Sentinel-1 temporal-composite tower-presence scoring** over the
   same corridors — free, all-weather, CPU-only, unproven at 10 m ⇒
   ship as a gated ladder experiment with stated prior (expect: works
   for ≥345 kV rural lattice, fails urban/small).
3. **Shadow+Sundial height estimation** on detector-confirmed towers
   → voltage-class attribute (labeled estimate). Cheap,
   differentiating product metadata.

**Phase B2 — EXTEND:** GridTracer-style pair-scoring REIMPLEMENTATION
(the repo is unlicensed) — detector proposals within 600 m of a known
line end + line-probability integration along candidate bearings; OSM
topology as seed collapses the hardest part of GridTracer's problem.
gridfinder (MIT) only as a weak routing prior. GPU becomes justified
here if extension areas are large.

**Phase B3 — DISCOVER:** full DevSeed-pattern sweep — high-recall
tile classifier (their exact recipe: ~0.5 m tiles, extreme threshold,
human-taxonomy of FPs) over rural US minus known corridors, feeding
the B1 detector for confirmation and the B2 grapher for topology. The
only phase that genuinely needs GPU-scale inference; defer until
B1/B2 metrics are logged.

**Drop:** SAR blind discovery at 10 m (no literature support,
clutter-dominated); stereo/photogrammetry on mosaic tiles (no stereo
exists); TTPLA/drone-view models (wrong geometry); fAIr code reuse
(AGPL); GridTracer code reuse (unlicensed — reimplement the paper's
algorithm).

**Cross-cutting caution to encode as a test:** prediction-density
artifacts at **imagery-mosaic quality boundaries** — any B1/B3 scorer
must be evaluated per-mosaic-source, not pooled (the
regime-conditioning rule applied to imagery).

---

## Item 4 — Imagery inventory + compute assessment (GV-A4)

Date: 2026-07-07. Session type: [RESEARCH]. VERIFIED = fetched and
read this session; REPORTED = secondary, not independently confirmed.

### 4.1 Esri World_Imagery — HARD WALL for ML use. Route around it.

License (VERIFIED, quoted from the actual legal documents):

Esri Master Agreement E204 (revised Aug 1, 2025), fetched from
esri.com/content/dam/esrisites/en-us/media/legal/ma-full/ma-full.pdf:

- **§3.2(c):** "Customer may take Online Services basemaps offline
  through Esri Content Packages and subsequently deliver (transfer)
  them to any device for use with licensed ArcGIS Runtime
  applications and ArcGIS Desktop. **Customer may not otherwise
  scrape, download, or store Data.**"
- **§3.3(h):** "**Customer may not use Data outside of the Software
  and Online Services to teach or train machine systems, models,
  software, databases, algorithms, and programs, including neural
  networks ('AI/ML Systems')** that learn from experience, adjust to
  new inputs, and perform humanlike tasks…"
- **§3.3(b):** "Customer may not use or allow third parties to use
  Data, for the purpose of compiling, enhancing, verifying,
  supplementing, adding to, or deleting from compilation of
  information that is **sold, rented, published, furnished, or in any
  manner provided to a third party**." — this independently kills
  using Esri imagery to build a grid dataset VolTradeAI sells.

E300 Product-Specific Terms (Nov 13, 2025), fetched from esri.com:

- **Footnote 96** (ArcGIS Online image services): "Customer may use
  ArcGIS Image services for interactive, non-programmatic access by
  Named Users only. **Programmatic use of the ArcGIS Image services
  (e.g., batch classification, deep learning, etc., or exporting
  volumes of data larger than 10MB at a time) are not permitted.**"
- **Footnote 10** (ArcGIS Location Platform): "Programmatic use of
  session tokens (**e.g., exporting volumes of basemap tiles**) is
  not permitted."

World Imagery item metadata (VERIFIED via arcgis.com sharing API,
item 10df2279f9684e4a9f6a7f08febac2a9): "This layer is not intended
to be used to export tiles for offline."

**Verdict: bulk tile scraping of World_Imagery for ML inference is
plainly forbidden — by three independent clauses, one of which names
deep learning explicitly.** Hard wall; route around, never through.
Continued use as a display basemap + the identify endpoint for
capture-date metadata (our current use) is the permitted
visualization/interactive category. Do not build the detector on
Esri tiles.

Resolution (reference only): service metadata (VERIFIED,
services.arcgisonline.com World_Imagery MapServer JSON) says Vantor
(ex-Maxar) 0.3 m in select metros, **0.5 m across the US** (rural
corridors included), community imagery 0.3–0.03 m in select areas;
tiles served to L23 (0.019 m/px) but that is upsampling — native is
~L19–20. Vivid Standard described as "30-cm HD across US." Rural
Texas corridors ~0.3–0.5 m there — but unusable for our purpose per
above.

### 4.2 NAIP — the workhorse. Public domain, 0.3–0.6 m, multiple free bulk paths.

**License:** USDA/FSA public domain with attribution requested
(VERIFIED via GEE catalog page quoting FSA). No ML restriction, no
resale restriction. The legal opposite of Esri.

**Resolution/cadence:** 0.6 m standard since ~2018; **2025
acquisition ~half the states at 30 cm, half at 60 cm** (REPORTED,
boydsmaps.com/mapinfo/naip2025.html). 4-band RGBN, leaf-on summer
collection, ~3-year state cycle (VERIFIED, GEE catalog:
USDA_NAIP_DOQQ — availability 2002-06-15 → 2023-11-17 in GEE).

**Access mechanics, in order of usefulness:**

1. **Microsoft Planetary Computer — best programmatic path, $0.**
   STAC API live (VERIFIED — fetched
   planetarycomputer.microsoft.com/api/stac/v1/collections/naip, got
   a valid STAC 1.0.0 collection): temporal extent 2010-01-01 →
   **2023-12-31**, GSD 0.3/0.6/1.0 m, COGs in Azure blob
   (`naipeuwest` account, container `naip`). The 2024 Hub retirement
   did NOT touch this — "the Planetary Computer Data and APIs remain
   available and unchanged" (VERIFIED via GitHub discussion 347).
   Query per state/year via STAC properties (`naip:state`,
   `naip:year` — REPORTED convention). Assets need short-lived SAS
   signing via the free token endpoint (`planetary_computer.sign()`);
   no account required. **Caveat: collection ends at 2023 — newest
   2024/2025 30-cm imagery not there yet.**
2. **AWS Open Data (VERIFIED, registry.opendata.aws/naip/):** buckets
   `naip-analytic` (4-band MRF+COG), `naip-source` (raw GeoTIFF),
   `naip-visualization` (3-band COG) — **all Requester-Pays,
   us-west-2, managed by Esri**. Registry states 2010–2023 coverage.
   Requester-pays = egress to internet ~$0.09/GB on us — **but reads
   from compute inside us-west-2 are transfer-free** (standard AWS;
   REPORTED). Docs: github.com/awslabs/open-data-docs naip.
3. **USDA direct, free, includes newest years:** NRCS Box
   (nrcs.app.box.com/v/naip, whole-state zips), Geospatial Data
   Gateway by state/county FIPS, USGS EarthExplorer / The National
   Map (JP2 CONUS) (all VERIFIED-as-existing; bulk ergonomics
   clunky). Which-year-covers-which-state: NAIP Coverage Map
   (fpacbc.usda.gov) + ArcGIS acquisition-status dashboards. Example
   recency: NJ/CT/RI have 2025 imagery; UT recollected 2024
   (REPORTED).
4. **Google Earth Engine** (VERIFIED catalog): `USDA/NAIP/DOQQ`, but
   **GEE free tier is research/education/nonprofit only; commercial
   use requires a paid license** — we are commercial, so GEE is not
   our free path.

**Honest caveats:** leaf-on = tree shadows/canopy can occlude
corridor edges; state/year radiometric seams (augment across
states/years in training); CONUS only; 1–3 years stale per state;
MPC/AWS mirrors lag USDA ~a year for newest states.

**Recommended architecture:** don't bulk-download at all — stream
512-px windows by HTTP range request from MPC COGs (free) for
corridor work; for full-state sweeps, run compute in AWS us-west-2
next to `naip-analytic` (requester-pays reads become ~$0 transfer +
trivial GET costs).

### 4.3 Sentinel-2 (10 m) — what's actually detectable

- **Individual towers: no.** Lattice footprint 3–10 m — sub-pixel at
  10 m. Every tower-detection paper uses sub-meter imagery
  (GridTracer VERIFIED; DevSeed×World Bank Pakistan/Nigeria/Zambia
  VERIFIED — Xception on high-res Mapbox/Maxar, 33× faster than
  manual; the Item-1 paper likewise).
- **Substations: yes, as segmentation.** Jindgar & Lindsay (VERIFIED
  abstract, arXiv:2409.17363) segment substations in Sentinel-2;
  SWIN > U-Net; multi-revisit latent fusion beats augmentation;
  benchmark ~27k substation locations (REPORTED; = Item 2.5 dataset).
  Large substations (50–500 m) span 5–50 S2 pixels.
- **Corridors: partially.** Cleared rights-of-way through forest
  (20–100 m swaths) visible as linear features
  (vegetation-encroachment literature, REPORTED). In open
  rangeland/desert (most of west Texas) there is NO clearing
  signature — corridors there are not reliably visible at 10 m.
- Precedent: **gridfinder** (Arderne et al., Sci Data 2020) predicts
  grid location from night-lights + OSM without imagery detection —
  useful prior/mask generator, not a detector.

**Use S2 for:** substation candidate generation + change monitoring
on our existing Copernicus pipeline; forested-corridor tracing. **Use
NAIP for:** towers, line vectorization, substation confirmation.

### 4.4 Texas-specific (Phase C first state) — TxGIO/StratMap

VERIFIED via tnris.org/stratmap/orthoimagery.html: publicly
downloadable statewide orthoimagery is **0.5 m or 1 m** (2004 1m;
2008/09 0.5m+1m; 2010 1m; **2014/15–2015/16 statewide 0.5 m
LEAF-OFF** — genuinely valuable: no canopy occlusion, complements
leaf-on NAIP; 2018/2020/2022 via NAIP). Higher-res **6-inch/1-foot
products are regional** StratMap acquisitions (urban), free via
DataHub (data.geographic.texas.gov). The statewide 6-inch **Texas
Imagery Service** (Hexagon-sourced) is government/university-only —
**not available to us**. Net: TxGIO adds a free 0.5 m leaf-off
statewide layer + 6-inch urban patches; NAIP 2022/2024 remains the
freshest statewide sub-meter source.

### 4.5 Other free sources (brief)

- USGS EarthExplorer "High Resolution Orthoimagery" legacy: 0.15–0.3 m
  urban orthos, mostly 2000s-era — training diversity, not current
  state (REPORTED).
- Several states publish statewide ≤1-ft orthos free (NJ, NC, NY, PA,
  VT, UT via UGRC etc.) — relevant past Texas (REPORTED).
- Microsoft Building Footprints / Google Open Buildings are
  *vectors*, no imagery rights — negative-class masks only.
- USGS `USGSNAIPImagery` ImageServer
  (imagery.nationalmap.gov) — free federal NAIP tile service; fine
  for display, not a bulk path.

### 4.6 Compute assessment

**Anchor figures.** ERCOT: "55,000+ miles of high-voltage
transmission lines" per ERCOT's fact sheet (VERIFIED via search) =
**88,500 km**. Texas area 695,662 km²; CONUS ~8.08M km². Corridor
buffer 200 m/side → 400 m swath → **~35,400 km² (5.1% of Texas)**.

**Chip counts (512-px chips, no overlap; ×1.56 for 20%-overlap
stride):**

| GSD | chip edge | TX corridor | TX full | CONUS full |
|---|---|---|---|---|
| 0.3 m | 153.6 m | 1.50 M | 29.5 M | 342 M |
| 0.6 m | 307.2 m | 0.38 M | 7.4 M | 85.6 M |
| 1.0 m | 512 m | 0.14 M | 2.7 M | 30.8 M |

**CPU throughput basis (VERIFIED benchmark):** Lenovo/Intel measured
YOLOv8x (~68M params, 640 px, INT8, OpenVINO) at **74.6 fps
single-stream / ~360 fps async on a 48-core Xeon 6740P**
(lenovopress.lenovo.com/lp2345). Scaled to 8–16 cores and 512-px
chips, honest planning number: **10–20 chips/s (use 15)** for a
RetinaNet/YOLO-class 30–60M-param INT8 model.

**GPU throughput (estimates, NOT benchmarked — flagged):**
RetinaNet-R50 @512 px, TensorRT/FP16 batched: **T4 ≈ 100 chips/s,
L4 ≈ 250 chips/s** (±2× error bars).

**Verified prices (as of 2026-07-07):** RunPod (runpod.io/pricing,
VERIFIED): **L4 $0.39/hr**, RTX 4090 $0.69/hr, RTX 3090 $0.46/hr,
A100 PCIe $1.39/hr. AWS g4dn.xlarge (T4): spot ~$0.22/hr, on-demand
$0.526 us-east-1 (REPORTED, instances.vantage.sh). AWS g6.xlarge
(L4, 24GB): spot ~$0.433/hr, on-demand $0.805 (REPORTED). GCP
g2-standard-4 (L4): ~$0.71/hr on-demand (REPORTED). Lambda has no
cheap small GPU (A10 $1.29/hr — VERIFIED); Vast.ai prices
dynamic/JS-rendered — couldn't verify exact numbers.

**The deliverable table (Texas; ×1.56 if 20% overlap):**

| Scenario | Chips | Wall-clock | Compute cost |
|---|---|---|---|
| **CPU corridor-verify @0.6m** | 0.38 M | **~7 h** (15/s, 16-core) | ~$0 (existing box) or ~$1 rented |
| CPU corridor-verify @0.3m | 1.5 M | ~28 h | ~$0–3 |
| CPU full-sweep @0.6m | 7.4 M | **~5.7 days** | marginal but slow |
| CPU full-sweep @0.3m | 29.5 M | ~23 days | not practical |
| GPU (L4) full-sweep @0.6m | 7.4 M | ~8 h | **~$3.20–3.60** (RunPod/g6 spot) |
| GPU (L4) full-sweep @0.3m | 29.5 M | ~33 h | **~$13–15** |
| GPU (T4 g4dn spot) full @0.3m | 29.5 M | ~82 h | ~$18 |

**All-50-states extrapolation (CONUS):** L4 @0.6m: 85.6M chips ≈ 95
GPU-h ≈ **$37–41**; @0.3m: 342M chips ≈ 380 GPU-h ≈ **$150–165**
(parallelize over 4–8 spot GPUs → ~2–4 days). With overlap, reruns,
failure margin: budget **~$100 (0.6m) / ~$400 (0.3m)** national.

**The real cost is data movement, not FLOPs:** Texas NAIP @0.6m ≈
3.9 TB compressed → ~$350 egress if pulled from the requester-pays
AWS buckets to an external box; @0.3m ≈ 15.5 TB → ~$1,400. CONUS
@0.3m ≈ $16k egress. **Avoid entirely** by (a) streaming COG windows
from Planetary Computer ($0, fine for corridor-scale), or (b) running
the GPU inside AWS us-west-2 next to `naip-analytic` (transfer-free
reads). Corridor-verify @0.6m is only ~200 GB — streamable from MPC
with no bulk download.

**Training:** fine-tuning a 30–60M detector on 2–5k labeled 512-px
chips ≈ 1–3 h on an RTX 4090 ($0.69/hr) → **$1–5/run, ~$50–100
total** with 10–20 experiment cycles. Labeling (drawing ~2–5k boxes
seeded from OSM/HIFLD tower+substation locations) dominates, not
compute.

### 4.7 What stays CPU-feasible NOW (no purchase)

**Corridor-verify over OSM/HIFLD 200-m buffers of ERCOT lines at
0.6 m NAIP = ~380k chips ≈ 7 h on a 16-core box (~28 h at 0.3 m).**
Substation-verify is even smaller: ~5–10k Texas candidate points ×
3×3 chips ≈ <100k chips ≈ ~2 h. Both stream from MPC free. This
covers the entire Phase B1 validation ladder (does our detector agree
with the known grid? where does it find undocumented assets inside
buffers?) without any GPU. NOTE: the Railway container is NOT this
box — this wants a one-off rented 8–16-core CPU instance
(~$0.30–0.70/hr, <$10 total) or any local machine.

### 4.8 Draft BLOCKED-FOR-MIKE purchase order (filed in wishlist.md)

GPU is NOT warranted for Phase B1 corridor-verify (CPU covers it). It
IS warranted (a) for detector TRAINING (mandatory regardless — Item 1:
no released weights exist; $50–100 total), and (b) the moment
full-state discovery sweeps begin (CPU 23 days/state @0.3m vs GPU
33 h/$15).

- **Service:** RunPod (simplest: per-second billing, no quota
  approval, prepaid credit) — runpod.io. Alternative for zero-egress
  NAIP reads: AWS g6.xlarge spot in us-west-2 (needs account +
  G-instance quota request, more setup).
- **Instance:** L4 ($0.39/hr) or RTX 4090 ($0.69/hr, ~1.5–2× faster).
- **Est. cost:** training/experiments $50–100; Texas full sweep
  $5–25; **all 50 states $100–400** depending on GSD mix. Suggested
  initial deposit: **$50**, expand after Texas validates.
- **Credential for Railway:** `RUNPOD_API_KEY` (RunPod console →
  Settings → API Keys). (AWS alternative:
  AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY scoped to EC2+S3.)
- **Capability unlocked:** detector training + full-state ML sweeps
  at ~$4–15/state/pass — repeatable national grid re-scans (change
  detection on every NAIP refresh), not just one-time mapping.

### 4.9 Key risks to log

1. GPU throughput figures (100/250 chips/s) are engineering
   estimates, not fetched benchmarks — first Texas run must measure;
   ±2× shift stays cheap, even ×4 worse stays cheap.
2. MPC NAIP ends at 2023; 2024/25 30-cm states require USDA Box/GDG
   or waiting for mirror updates.
3. The 15 chips/s CPU number assumes INT8 + async streams; naive FP32
   PyTorch on CPU is ~5–10× slower — quantization is mandatory for
   the CPU path.
4. The Esri wall is absolute for imagery, but the World_Imagery
   **identify/date-metadata** use we already ship remains legitimate
   and useful (e.g., to know imagery vintage per region).
