# Verification record — bot cockpit Stage 1 audit (2026-10-15)

Companion to `research/bot_cockpit_audit.md` §8. Status per claim unit, the evidence behind it, and how to finish the blocked independent pass.

## Method and what happened

1. Eight parallel stage readers produced `A…H_*.md` (every claim `file:line`).
2. Fifteen existence/caller claims were checked mechanically by grep in the main session (no `EventSource`/`bot/stream` client; no `/api/bot/clock` route; `.page-container` 920 px; no `client_order_id` writer; no `trade_updates`; hardcoded paper URL + `ALPACA_TRADING_URL` + literal `mode`; `CONVEXITY_OVERLAY.enabled` never read; `strategyWeights` only at `bot.ts:843–1015`; `runUpgradeCandidates` / `ml_attribution_summary` uncalled; `"fallback"` vs `"fallback_rules"`; `intel.get("vxx_ratio")`; `ml_toggle` readers = status reporters only; `check_weekly_loss` 1-element history; TIER2 `"complete"` never matched; no `tier1Running`).
3. An adversarial Workflow (`cockpit-audit-verify`, run `wf_fcd96993-30c`: 37 units × {code lens, live lens} + tiebreak + coverage critic = 112 agents) was launched at 03:38Z. One verifier (V01 code lens) completed; the other 111 failed to start with `You've hit your session limit · resets 6:40am (UTC)`.
4. The author then verified the headline units directly in-session by reading the cited code and the live probes (table below).

**To finish the independent pass** (after 06:40Z, from the same session or a new one with the script copied):
`Workflow({scriptPath: "/root/.claude/projects/-home-user-voltradeai/05845219-3cce-5839-bd8e-c3c9fbd41b8e/workflows/scripts/cockpit-audit-verify-wf_fcd96993-30c.js", resumeFromRunId: "wf_fcd96993-30c"})` — V01's result replays from cache. The script's claim list is the authoritative full text of each unit.

## Status per unit

| id | status |
|---|---|
| V01 | PARTIAL — independent verifier (code lens, confidence 0.86): the 10 s cap and the mutex bypass are real for all eleven cited commands, but most jobs complete under it live (manipulation, shadow backfill, COT proven by audit/archive rows); the 4 am block is dead code anyway; the Tier-3 retrain and the Tier-1 `manage_positions` one-liner are undecidable from live evidence |
| V02 | CONFIRMED — in-session code read (`bot.ts:7681` gate vs `:5209` dispatch; `floor(etHour)===4` at `:6199` unreachable; sole caller `:5211`); live BACKUP/RESEARCH/RETRAIN rows = 0 |
| V03 | reader-reported (A); (e) live-confirmed (24× "All open orders cancelled"); (g) `print('ok')` per A |
| V04 | reader-reported (A); absence of `tier1Running` [grep] |
| V05 | reader-reported (A, H); (b) weekly history length 1 [grep]; (d) live `DIAGNOSTIC` ×7 "Low win rate 27.8%" implies threshold 75 |
| V06 | reader-reported (A) |
| V07 | CONFIRMED — in-session: `current_positions` read at `be:3081`, first assigned `be:3186`, both inside `_scan_market_inner` (2571–4080), no `global` declaration → UnboundLocalError; `except` at 3157–3160 |
| V08 | CONFIRMED — in-session: `be:3921` `… if '_regime_ctx' in locals() else "NEUTRAL"`; `_regime_ctx` is assigned only inside `deep_score` (1451, 1458) |
| V09 | CONFIRMED — in-session: `be:1905–1915` excludes OCC symbols from `stock_tickers` then re-adds them via `pending_stock_tickers`; D reproduced the block with the live book |
| V10 | reader-reported (D); live counts consistent (8,372 `taken`, 0 `TRADE`) |
| V11 | slot gate CONFIRMED live (`OPTIONS-SLOT-FULL` ×158, `QUEUE` ×26 VXUS/KWEB); downstream hops (`should_use_options`, morning M3) reader-reported (D) |
| V12 | CONFIRMED — in-session: `be:192`=8 vs `bot:361`=6 vs `sc:110`=6; `sc:35` 30_000 vs 45 s; `sc:307` 5.0 vs `bot:350` −3 |
| V13 | reader-reported (D, E) |
| V14 | (a) CONFIRMED [grep `intel.get("vxx_ratio")`, `_spy_rets=[]`]; (b),(c) reader-reported (B, C, D) |
| V15 | reader-reported (C) |
| V16 | reader-reported (E) |
| V17 | (c) CONFIRMED — in-session `bot.ts:6390` reads `pos.created_at || pos.submitted_at` on an Alpaca Position (schema has neither; live probe shows neither); (a),(b),(d) reader-reported (E) |
| V18 | CONFIRMED — in-session: `_get_historical_stats` loads the whole feedback file (no `_seed` filter); stocks f* < 0 under the priors (.49/4.19/4.25 → −0.027) and under the seeded stats (44.6% WR); `_infer_strategy` defaults to `stocks` → `calculate_position` blocks; whether this is THE root cause is moot — B1 (NEUTRAL → 0 slots) also suffices |
| V19 | reader-reported (E) |
| V20 | CONFIRMED (code) — `enabled` never read [grep]; `side: "sell_to_close"` at `be:4531/4541`, `"buy_to_open"` at `be:4657/4668`; Alpaca rejection probable (side enum is buy|sell) but not observed live. D's "disabled by config" is wrong |
| V21 | reader-reported (E) |
| V22 | (a),(b) CONFIRMED — in-session: trade dict `be:3554–3597` has no `volume` key; `bot.ts:5660` `|| 1000000` → else branch 0.12–0.25%; (c) 2 s single poll reader-reported (F) |
| V23 | reader-reported (F, E) |
| V24 | reader-reported (F); live: 6 short puts held, `live_options_outcome_breakdown {orphan_exit 30}` |
| V25 | CONFIRMED — in-session: `bot.ts:1689` filter keeps `None`, `:1691` `None > 0` raises, catch `:1753` → `if (perf.totalTrades === 0)` fallback `:1758`; live `live_pnl_null_count 312` |
| V26 | CONFIRMED — in-session: `bot.ts:1034–1035` `unshift` then `if (length > 365) shift()` removes index 0 (newest); writer gating per A |
| V27 | (c) CONFIRMED — in-session: `pnl: 0` at `bot.ts:873` is the only assignment; `strategyWeights` read nowhere [grep]; (a),(b),(d) reader-reported (F, H) |
| V28 | reader-reported (F) |
| V29 | CONFIRMED — in-session: `bot.tsx:1039–1040` `qty*entry`/`qty*current`; `routes.ts:7070` `(price − matchBuy.price) × qty`; `bot.ts:2328–2330` literals 8/95/5; (c) account-route kill per `bot.ts:1235–1250` read in-session |
| V30 | backtest interpolation CONFIRMED — in-session: `bot.ts:3681–3697` builds `python3 "…" "${ticker}" "${strategy}" "${years}"` and passes it to a shell `exec` (owner-only); per-route auth table reader-reported (G) |
| V31 | reader-reported (G) |
| V32 | (c) CONFIRMED — `risk_kill_switch.py:738–760` only reads the two files; no writer repo-wide [grep]; (d) CONFIRMED (no `_seed` filter, see V18); (a),(b) reader-reported (H) |
| V33 | reader-reported (B) with sandbox probes for (c) EFTS schema and (d) USPTO 301 |
| V34 | reader-reported (B) |
| V35 | reader-reported (B, D); `feed_status()` zero callers per B |
| V36 | (a) CONFIRMED [grep]; rest reader-reported (C) |
| V37 | CONFIRMED — arithmetic −10.0355%, kill line $99,654.34; Tier-1 RTH gate and account-route trip site read in-session; `/api/health` computes but never kills (`bot.ts:1513`) |

## V01 — full verdict from the completed independent verifier (code lens)

### verdict

PARTIAL

### confidence

0.86

### mechanism

(a) GATE — CONFIRMED. server/bot.ts:233-237 isTrivialPython(cmd): `if (cmd.length > 150) return false; if (/import\s+(numpy|pandas|sklearn|lightgbm|scipy|torch)/.test(cmd)) return false; return true;`. Purely textual: a 26-char `python3 ml_retrain_safe.py` or `from bot_engine import manage_positions` (which imports the whole numpy/lightgbm stack transitively) is "trivial". The comment at 231-232 says the intent was "<100-byte pings".

(b) CAP + NO MUTEX + OVERRIDE — CONFIRMED. bot.ts:240 `maxWait = opts?.timeout || 30000`; :241 `hardTimeout = Math.min(opts?.timeout || 90000, 90000)`; :244-246 `if (isTrivialPython(cmd)) return execAsync(cmd, { ...opts, timeout: Math.min(hardTimeout, 10000), killSignal: 'SIGKILL' })` — returns BEFORE the `while (pythonRunning)` lock at :249-269, and `timeout:` sits after `...opts` so the caller's value is overridden. Effective cap = min(opts.timeout||90000, 10000): 900000→10000, 120000→10000, 15000→10000, 5000→5000. execAsync (bot.ts:53, :90-91) is promisify(child_process.exec); Node kills the child with killSignal after `timeout` ms.

(c) AFFECTED CALL SITES — CONFIRMED for every listed site (lengths measured with the real template text, interpolations resolved to Docker WORKDIR /app per Dockerfile:12/39): 4213 Tier-1 manage_positions = 120 chars (asks 15000 → 10000); 4285 shadow backfill = 116 (asks 120000 → 10000); 5739 and 6214-6216 `python3 ml_retrain_safe.py` = 26 (asks 900000 → 10000); 5802 manipulation = 123 (30000→10000); 5879 cftc_cot = 102 (45000→10000); 5900 sec_form4_bulk = 96 (120000→10000); 6021 `python3 -W ignore "/app/bot_engine.py" full 2>/dev/null` = 55 (300000→10000); 6149 backtest = 36 (120000→10000); 6167 analyze = 45 (30000→10000); 6186 `python3 "/app/bot_engine.py" scan` = 33 (180000→10000); 6254 off-site backup = 34 (60000→10000). NOT trivial (mutex + real timeout): 5714 model-age check (353 chars), 6226 local backup (856), 6030/6126 (606/291). Side finding: 5835 diagnostics is 149-150 chars while process.uptime()<1000 s, so it is capped at 10 s right after every boot (it completes anyway — 31 TIER3-DIAG rows live).

(d) "THESE DIE BY SIGKILL AT 10 s" — REFUTED as a blanket statement; holds only as "capped at 10 s":
  d1. Live-proven to COMPLETE under the cap on Oct 14: manipulation scan (150 MANIPULATION rows; the audit at :5810 is written only after JSON.parse of stdout); shadow backfill (3/3 SHADOW-BACKFILL stat rows at :4301, 0 SHADOW-BACKFILL-ERROR; each row requires the subprocess to have returned); cftc_cot (bot.ts:5879 is run_daily_update's ONLY caller; live /api/data/streams probe 2026-10-15T03:1xZ shows stream `cot` written_by cftc_cot.py save_archive, newest_file voltrade_cot_archive.json age_hours 15.7, health "live" — save_archive runs only after all 7 symbol fetches, cftc_cot.py:228-262, so the full network pull finished inside 10 s).
  d2. UNREACHABLE on the scheduler, so the cap is moot: runOvernightResearch is dispatched only from tier2Intelligence at bot.ts:5209-5211 when `!isMarketOpen && (_etHourNow >= 20 || _etHourNow < 4)`, but scheduleTier2 (:7681-7684) only runs tier2Intelligence when `clock.is_open || (etH >= 4 && etH < 20)`. The 4am block at :6199 `Math.floor(etHour) === 4` can never be true under a `<4 || >=20` dispatch (getETHour is fractional, server/orderParams.ts:2-7) → 6214-6216 retrain, 6226 backup and 6254 off-site backup are DEAD CODE. The 22-24/0-2/2-4 branches (6149, 6167, 6186) run only via the owner route POST /api/bot/run-now (:7955-7965, calls tier2Intelligence at night) ; the 20-22 branch (6021) additionally via a scan that starts in the last ~100 s before 20:00 ET (15-min cadence, :7620-7650) — zero RESEARCH/RESEARCH-ERROR rows in the window show neither happened Oct 14.
  d3. UNDECIDABLE live, cap plausibly binding: 5739 Tier-3 retrain — ml_retrain_safe.py:69 calls train_model(fast_mode=True) → ml_model_v2.py:238-293 (1 most-actives call + up to 8 sequential bars batches, timeout 20 each) → :300-309 up to 30 sequential Finnhub calls (timeout 5 each) → :1404 10k-row cap → 5-fold walk-forward CV → global + regime models → joblib.dump (:1527); finishing all of that plus interpreter start inside 10 s is implausible but not provable from code (network-bound). 4213 Tier-1 one-liner — cold `from bot_engine import manage_positions` + live Alpaca calls; pythonCall's own HEAVY_DAEMON_ONLY set (bot.ts:162-164) says manage_positions must not run as a subprocess, yet 4213 bypasses both the daemon and the mutex every 45 s during RTH; its failure is console-only (:4229-4231), so no audit row can confirm or refute. 5900 sec_form4_bulk — index fetch (timeout 20) + one quarter zip (timeout 180) with _save_last_check only at the end (sec_form4_bulk.py:279-299): a kill mid-download leaves no marker, so it retries every Tier-3 tick; no streams-inventory entry exists for this archive, so live state is unknown.

(e) OOM CLASSIFIER — CONFIRMED. Measured Node behaviour: exec timeout with killSignal SIGKILL yields {code:null, signal:"SIGKILL", killed:true, stderr:""}; bot.ts:5776-5786 then prints `code=?` and, because `_signal === "SIGKILL" && !_stderr.trim()`, appends " [likely OOM kill during import lightgbm/ml_model_v2]" — the exact production text quoted in ml_retrain_safe.py's docstring (lines 5-8), which that 2026-05-04 fix attributed to the 300 s timeout and "fixed" by raising the asked timeout to 900 s (bot.ts:5727-5739) — a no-op under the gate. ml_retrain_safe.py writes nothing to stderr itself, so stderr is empty unless a library warning lands there.

### evidence

bot.ts:233-237 (gate), :239-246 (trivial branch, `{ ...opts, timeout: Math.min(hardTimeout, 10000), killSignal: 'SIGKILL' }`), :53/:90-91 (execAsync = promisify(exec)). Measured command lengths (node, interpolations → /app): 4213=120, 4285=116, 5739=26, 5802=123, 5835=152 (149 when uptime<1000 s & tier2_ran_since_boot=true), 5879=102, 5900=96, 6021=55, 6149=36, 6167=45, 6186=33, 6215=26, 6254=34; non-trivial: 5714=353, 6226=856. Node probe: `promisify(exec)("sleep 3",{timeout:200,killSignal:"SIGKILL"})` → {"code":null,"signal":"SIGKILL","killed":true,"stderr":""}. Scheduler: bot.ts:7681 `isAnyWindow = clock.is_open || (etH >= 4 && etH < 20)`; :5209-5211 overnight dispatch `_etHourNow >= 20 || _etHourNow < 4`; :6199 `Math.floor(etHour) === 4`; :7955-7965 /api/bot/run-now; orderParams.ts:2-7 fractional hours. Retrain cost chain: ml_retrain_safe.py:48-71; ml_model_v2.py:238-293, :300-309, :1404-1410, :1527. Form4/COT guards: sec_form4_bulk.py:266-299 (_save_last_check after loop, zip timeout=180 at :131); cftc_cot.py:228-262 (save_archive after 7-symbol loop; sole caller bot.ts:5882). LIVE: scratchpad/audit/live/ml.json captured 03:12:18Z → model_age_hours 19.4 → mtime 07:45-07:51Z Oct 14 (= 03:45-03:51 ET, NOT 08:00Z/4am ET), corroborated by TIER3 "ML model fresh (6.6h old)" at 14:25Z; window has 150 MANIPULATION rows, 3 SHADOW-BACKFILL rows (97/74/83) and 0 SHADOW-BACKFILL-ERROR, 0 RESEARCH/RESEARCH-ERROR/RETRAIN/RETRAIN-ERROR/TIER3-ML-ERROR/BACKUP rows; new read-only probe GET /api/data/streams (HTTP 200, saved at scratchpad/audit/live/streams_probe/streams.json): stream `cot` age_hours 15.7 health "live", no entry for the sec_form4_bulk archive; GET /api/debug/ml-diagnostics → HTTP 403 (DEBUG_ENABLED unset), and ml_diagnostics.py:253-257 reads ml_retrain_status.json / ml_last_retrain.json, never the voltrade_last_retrain.json that ml_retrain_safe.py:91-101 writes. routes.ts:7162-7191 POST /api/ml/retrain: own promisify(exec) with timeout 300000, no mutex, sole writer of ml_status.json (live last_steps ends training_done, samples 10000 = fast_mode cap). git: shallow clone (51 commits) — `git log -S isTrivialPython` only reaches c4fdafc, so the gate's introduction date vs. the 2026-05-04 timeout bump cannot be established.

### corrected_claim

server/bot.ts:233-237 isTrivialPython() returns true for any command string ≤150 chars that lacks a literal `import numpy|pandas|sklearn|lightgbm|scipy|torch`; execPythonSerialized (:239-246) then bypasses the Python mutex and runs it under `{ ...opts, timeout: Math.min(hardTimeout, 10000), killSignal: 'SIGKILL' }` with hardTimeout = min(opts.timeout||90000, 90000) (:241), so every such caller's asked timeout is silently cut to min(asked, 10 s). All eleven cited commands measure as "trivial": Tier-1 manage_positions one-liner (4213, 120 chars, asks 15 s), shadow backfill (4285, 116, asks 120 s), both `python3 ml_retrain_safe.py` sites (5739, 6215; 26 chars, ask 900 s), manipulation scan (5802, 123, 30 s), cftc_cot (5879, 102, 45 s), sec_form4_bulk (5900, 96, 120 s), overnight bot_engine full/scan (6021, 6186; 55/33, 300/180 s), backtest (6149, 36, 120 s), analyze (6167, 45, 30 s) and off-site backup (6254, 34, 60 s). They are CAPPED at 10 s, not uniformly killed: the manipulation scan, shadow backfill and COT update are live-proven to complete inside the cap (150 MANIPULATION rows, 3/3 SHADOW-BACKFILL stat rows with 0 errors, COT archive written 15.7 h before the probe with health "live"); the 4am retrain/backup/off-site-backup block (6199-6273) is unreachable by construction (dispatched only when etHour<4 or ≥20, but gated on floor(etHour)===4) and the rest of the overnight engine runs only via owner POST /api/bot/run-now at night or a ~100 s scan straddle of 20:00 ET, neither of which occurred Oct 14; the Tier-3 retrain (5739), the Tier-1 bot_engine sync (4213) and the Form4 bulk download (5900) cannot be confirmed either way from live data — a <10 s fast_mode retrain is implausible given its serial HTTP chain + 5-fold CV + 4 models, and the Tier-1 sync and Form4 failures are console-only with no audit trail. When the cap does fire, Node reports code=null/signal=SIGKILL/stderr="" and the TIER3-ML-ERROR classifier at 5785-5786 labels it "[likely OOM kill during import lightgbm/ml_model_v2]" — the same text the 2026-05-04 fix tried to cure by raising the asked timeout to 900 s, which the gate makes a no-op. Honest live statement: the model file was last written ≈07:45-07:51Z on 2026-10-14 (03:45-03:51 ET) — not by the 4am path (dead code; 08:00Z) — by either the 10 s-capped Tier-3 subprocess (requires <10 s training) or the owner's POST /api/ml/retrain (300 s, own exec, sole writer of the ml_status.json that shows training_done); the captures cannot distinguish them and voltrade_last_retrain.json, which could, is exposed by no route. Not filed in research/.

### filed_in_research

NEW. Not in open_questions.md KNOWN BROKEN (grep for isTrivialPython / trivial gate / 10 s cap returns nothing). Nearest existing filings are adjacent, not the same finding: research/experiments.md:73086-73094 traces both `python3 ml_retrain_safe.py` call sites and assumes they run "hourly Tier-3 + 4am daily" at their asked timeouts; research/open_questions.md:4547-4556 and experiments.md:88758-88768 record that `isAnyWindow` keeps Tier 2 structurally dark 8pm-4am ET without noting that this makes runOvernightResearch (and the 4am retrain/backup block, by its own floor(etHour)===4 test) unreachable. Secondary side findings also unfiled: ml_diagnostics.py:253-257 reads ml_retrain_status.json / ml_last_retrain.json while ml_retrain_safe.py:91-101 writes voltrade_last_retrain.json (name mismatch, so the one artifact that names the retrain path is never surfaced); Tier-1 at bot.ts:4213 runs manage_positions as a cold bot_engine subprocess every 45 s despite pythonCall's HEAVY_DAEMON_ONLY rule at :162-164.

### cockpit_consequence

(1) Never render a job's asked timeout (900 s / 300 s / 120 s / 45 s / 30 s / 15 s) as its budget: for every execPythonSerialized call whose command string is ≤150 chars without a literal heavy import, the real budget is min(asked, 10 s) and the job runs outside the Python mutex — show the effective value and the mutex-bypass flag per job. (2) Model freshness (retrain_needed=false, "ML model fresh Nh old") is NOT evidence that the Tier-3 retrain path works; the cockpit must show the retrain path as UNVERIFIED and the last-write source as unknown (Tier-3 10 s-capped subprocess vs owner POST /api/ml/retrain) until voltrade_last_retrain.json or a TIER3 "ML retrain complete" audit row is surfaced. (3) The 4am ET daily retrain, nightly local backup and GitHub off-site backup must be displayed as NEVER RUNS (dead code), and the overnight research engine (8pm-4am) as owner-triggered-only, not as scheduled jobs. (4) A TIER3-ML-ERROR carrying "[likely OOM kill...]" must be rendered as "SIGKILL with empty stderr — exec timeout OR OOM", never as a memory diagnosis. (5) The Tier-1 bot_engine stop-state sync has no success/failure telemetry (console-only catch) — display as UNKNOWN, not healthy; likewise the Form4 bulk archive (no streams-inventory entry). (6) COT update, manipulation scan and shadow backfill may be shown as completing under the cap (live-proven), but the shadow backfill's designed 3000-job/120 s budget is really 10 s per run and must be labelled so.

## Coverage critic

Not run (session limit). The eight reports' own §"gaps" sections and the audit §4–§6 are the coverage statement until it is.
