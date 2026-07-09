# GRID VISION — data-modality research (deep-research, 2026-07-09)

> Adversarially fact-checked (109 agents). Question: best free, accurate,
> cross-region-generalizable way to map US grid infrastructure.

## Bottom line

The fastest, most accurate national US grid coverage does NOT come from fixing cross-region optical detection — it comes from ingesting authoritative pre-built free vector data first, then using ML only for the tiers those datasets miss. HIFLD (compiled by Oak Ridge National Lab under DHS) already provides a national, free transmission-line layer (69–765 kV, conterminous US + Puerto Rico) AND an Electric Substations layer with zero generalization problem, and EIA officially defers to HIFLD as the authoritative source; this directly fills most of what OSM misses at transmission voltage. The genuine ML gap is DISTRIBUTION (poles/lines <69 kV) and tower point-locations, which no authoritative vector set covers. For those, street-view ML (Google Street View / Mapillary) is the most-proven modality (>80% precision/recall vs. PG&E ground truth; upward-facing CAM detectors hit F1 ~0.95 for lines, ~0.93 for poles) but is validated only in single California/Connecticut regions with coarse ~5–10 m geolocation and API costs, and NO street-view or SAR paper actually demonstrates the cross-biome generalization the question asks about. SAR tower detection rests on solid double-bounce physics but every strong result uses 3–8 m GaoFen-3, not free ~10 m Sentinel-1, and cross-region generalization is untested — making it a research bet, not a fast win. Recommended order: (1) HIFLD + EIA/HIFLD substations + OSM merge for transmission now; (2) general-purpose LiDAR extraction on free USGS 3DEP for towers/lines; (3) street-view ML for distribution; SAR last as exploratory.

## Findings (verified)

### 1. [high] HIFLD provides a free, authoritative, national pre-built vector dataset of US transmission lines (69–765 kV, conterminous US + Puerto Rico, compiled by Oak Ridge National Lab under DHS) plus Electric Substations, Power Plants, and Retail Service Territories — filling most of the transmission-tier grid without any ML and with no cross-region generalization problem. EIA itself publishes no line locations and officially defers to HIFLD.

**Evidence:** Merges claims 0,1,4,5,6. HIFLD Open Transmission Lines (ORNL/HARP QA, 69–765 kV, incl. underground where available) is the standard free US transmission layer per data.gov/CISA/EIA. EIA FAQ #567 states verbatim the shapefiles 'are available from HIFLD.' CISA enumerates HIFLD electric layers incl. Substations. Substations ARE covered (the 'substations are a gap' claim was refuted 0-3). Scope limits: it is transmission ≥69 kV only (no distribution, no tower point-locations); vintage is dated (2014–2022 snapshots); the public DHS portal was discontinued 2025-08-26 so it must be pulled from the Data Rescue Project / Source Cooperative / HIFLD Next archives, not a live feed.

- https://portal.datarescueproject.org/datasets/hifld-open-transmission-lines/
- https://www.eia.gov/tools/faqs/faq.php?id=567&t=3
- https://databasin.org/datasets/13bace6b70af4d2795785f42487c7fda/
- https://www.cisa.gov/resources-tools/resources/mapping-your-infrastructure-datasets-infrastructure-identification

### 2. [high] OSM alone cannot deliver complete national coverage: high-voltage transmission (345 kV+) is substantial but subtransmission (69–161 kV) is unevenly mapped and distribution (<69 kV) is largely absent; EIA circuit-mile totals by voltage class serve as an external completeness calibration (OSM diverges e.g. 0.5x at 765 kV, 1.8x at 345 kV, 1.5x at 69 kV).

**Evidence:** Merges claims 2 (3-0) and 3 (2-1). The 2026 arXiv paper 'Building Power Grid Models from Open Data' states OSM 345 kV+ is substantial, subtransmission uneven, distribution largely absent, and 15–30% of power=line ways lack voltage tags. This is exactly why HIFLD is needed to backfill the transmission tier. Caveat on claim 3 (medium, 2-1): the exact ratios are from a single non-peer-reviewed v1 preprint and the EIA source attribution (Electric Power Annual vs Form EIA-411) is imprecise; the calibration method is sound but ratios are estimates.

- https://arxiv.org/html/2605.04289v1

### 3. [high] General-purpose airborne LiDAR (not flown for powerline mapping) can detect and model powerlines of ANY voltage at the lower point densities typical of national surveys — directly relevant to extracting towers and lines from free USGS 3DEP LiDAR to fill gaps HIFLD/OSM miss.

**Evidence:** Merges claims 12 (3-0) and 13 (2-1). Peer-reviewed IEEE 2024 method works on 'point clouds whose density is usually lower than specific-purpose flights.' Target national surveys (PNOA 0.5–5 pts/m²) bracket USGS 3DEP QL2 (2 pts/m²); an independent Quebec study hit 98.6% on transmission towers at 2.6–13.8 pts/m². Caveats (claim 13 was 2-1): the paper did not confirm validation at ≤2 pts/m² specifically, and USGS notes QL2's 0.7 m spacing 'may present challenges' for thin DISTRIBUTION conductors — so LiDAR is strongest for towers/transmission, weaker for thin distribution lines. NOTE: the claim that 3DEP reached 99% national coverage was only 1-2 (partially refuted), so 3DEP completeness cannot be assumed nationwide — check actual tile availability per region.

- https://ieeexplore.ieee.org/document/10518086/

### 4. [high] Street-view ML (Google Street View / Mapillary) is the most-proven modality for the DISTRIBUTION tier that no authoritative dataset covers: a multi-modal framework using only street view + roads + buildings maps overhead AND underground distribution at >80% precision/recall vs. PG&E ground truth; upward-facing semi-supervised CNNs detect lines/poles at image-level F1 ~0.95/0.93; RetinaNet detects poles-with-crossarms at 0.95 precision / 0.81 recall.

**Evidence:** Merges claims 14,15,18 (all 3-0). Nature Comms 2023 explicitly avoids 'low-resolution or noisy remote sensing' and hits F1 0.83–0.91 across 6 CA areas. NeurIPS-2019 CCAI upward-view CAM detector: line F1 0.953, pole F1 0.925. Zhang & Witharana Sensors 2018 RetinaNet-101: pole precision 0.95/recall 0.81 at IoU>0.3 (falls to 0.73/0.62 at IoU>0.5). Applicability caveats: all target distribution poles/lines, NOT transmission towers or substations; street view carries API cost (so not strictly free — Mapillary is the free alternative to evaluate); none is a cross-region test.

- https://www.nature.com/articles/s41467-023-39647-3
- https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/neurips2019/31/paper.pdf
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6111250/

### 5. [high] Street-view geolocation of grid assets is COARSE — roughly 5–10 m error (only ~2.6% within 1 m, ~47% within 5 m, ~79% within 10 m via multi-view triangulation) — and region-level detection recall drops well below image-level scores (e.g. only 78% of actual poles detected in San Carlos).

**Evidence:** Merges claims 16,19 (3-0). Line-of-bearing triangulation from GSV yields 2.25 m avg error on detected poles but only 78% detected-within-4m. This matters for a map overlay: street-view-derived asset points will be ~5–10 m off, acceptable for a visual layer but not for precise topology.

- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6111250/
- https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/neurips2019/31/paper.pdf

### 6. [high] CROSS-REGION GENERALIZATION — the core problem the question raises — is NOT demonstrated by any street-view or SAR paper reviewed. Street-view models were evaluated only on single US regions (San Carlos CA, Mansfield CT), with 'extend to other regions' listed as future work; the SAR papers report no cross-region/cross-biome testing.

**Evidence:** Merges claims 7,17 (3-0). Image-level F1 scores are same-distribution random splits, not cross-region holdouts. The bold Nature-Comms claim that the CA model transfers to Sub-Saharan Africa WITHOUT fine-tuning was REFUTED 0-3, so cross-region transfer should not be assumed. Implication for the deliverable: no reviewed ML modality solves cross-region generalization out of the box — the reliable fix is to lean on authoritative vector data (HIFLD/LiDAR ground-truth) and use ML per-region with local fine-tuning, not to expect a single detector to generalize across biomes.

- https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/neurips2019/31/paper.pdf
- https://arxiv.org/abs/2404.01074

### 7. [high] SAR tower detection rests on solid physics (metal towers are bright elongated double-bounce scatterers, the brightest SAR return) but every strong accuracy result uses HIGH-RESOLUTION 3–8 m GaoFen-3, not free ~10 m Sentinel-1, and no cross-region generalization was tested — so SAR is a research bet, not a fast national-mapping win.

**Evidence:** Merges claims 8,9,10,11 (8/10/11 are 3-0; 9 is 2-1). Double-bounce physics confirmed by ASF/HyP3 and Brunner & Bruzzone. Hierarchical SCR detector hit 100% detection / 1 false alarm — but on N=4 GaoFen-3 scenes at 3 m/8 m, NOT Sentinel-1. Towers as point scatterers at free ~10 m Sentinel-1 GRD are much harder; buildings confound as strong scatterers; lattice towers are not solid walls so returns are more variable. The claim that P2Det proves SAR needs higher-than-Sentinel-1 resolution was refuted 0-3 (over-reach), but no evidence shows Sentinel-1 works either — the honest status is 'unproven on free radar.'

- https://www.mdpi.com/2072-4292/14/3/625
- https://hyp3-docs.asf.alaska.edu/guides/introduction_to_sar/
- https://arxiv.org/abs/2404.01074
