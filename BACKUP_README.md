# VolTradeAI Data Backups

This branch contains nightly snapshots of the bot's learning data.
Pushed automatically at 4am ET every night.

**Files backed up:**
- `voltrade_ml_model.pkl` — Trained LightGBM model
- `voltrade_fills.json` — Trade fill history
- `voltrade_weights.json` — Dynamic strategy weights
- `voltrade_trade_feedback.json` — Trade outcomes for ML training
- `voltrade_event_memory.json` — News event learning memory
- `voltrade_earnings_memory.json` — Earnings pattern memory
- `voltrade_blend_tracker.json` — Rule vs ML blend tracker
- `voltrade_insider_cache.json` — Insider trade cache
- `voltrade_health_log.json` — System health log

**To restore:** Copy files to `/data/voltrade/` on Railway volume.
