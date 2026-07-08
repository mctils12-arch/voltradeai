"""gridvision_train — GRID VISION tower-detector v0 ON-POD training package.

Everything the RunPod GPU pod needs to fine-tune the tower detector:
  - cog_reader.py     — NAIP COG windowed reader (pure geometry + rasterio-guarded read)
  - etdii_to_yolo.py  — ETDII US labels -> YOLO detector dataset (pure conversion)
  - weight_sink.py    — weight/metric persistence off the ephemeral pod
  - train.py          — the entrypoint (pip-bootstrap -> data -> train -> write-back)

DESIGN RULE (mirrors the three sibling scripts/gridvision_*.py): every pure
function is unit-testable OFFLINE with stdlib only. rasterio / ultralytics /
numpy / torch are ON-POD pip deps and are imported ONLY inside the functions
that need them, so the offline test battery and the --dry-run mode never import
them. See research/grid_vision_phaseb.md for the v0 = TOWER-detector decision and
the pre-stated prior (tower AP50 0.55-0.75 @0.6 m).
"""
