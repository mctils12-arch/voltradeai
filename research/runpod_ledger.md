# RunPod GPU — spend plan, cost-cap protocol, and ledger

Funded by Mike 2026-07-07: **$50.00** balance. `RUNPOD_API_KEY` is in
**Railway only** (not this session — presence-checked 0 here). The
cost-cap CALCULATOR + ledger live in `scripts/runpod_budget.py`
(pure, tested: `test_runpod_budget.py`). The ledger data file is
`datacore/runpod/ledger.jsonl` (append-only).

## THE PLAN (which GPU workload runs, in what order, and why)

The two candidate GPU workloads were (1) grid-vision detector training
and (2) satellite design-model 4D-splatting. **Priority resolved:**

1. **GRID VISION detector training — the ONLY GPU workload.** Highest
   value per the standing weighting (irreplaceable archive / validated
   signal — the detector is the moat that maps the US grid where OSM is
   blank). This gets the whole $50.
2. **Satellite splatting — CANCELLED, $0 GPU spend.** The design-class
   model-library research (`research/orbital_models.md`, 2026-07-07)
   found **zero** true splat candidates from existing free imagery:
   Starlink has only flat-pack press photos (no 360° ground survey);
   ISS/Hubble imagery is sparse-angle and NASA already ships clean
   public-domain glTF; splat assets are 20–150 MB vs 0.5–5 MB glTF
   (disqualifying for mobile-390px). **glTF everywhere gets ~90% of the
   fidelity with no GPU cost.** So the earlier "1–3 marquee splats"
   scope is retired — no satellite job touches RunPod.

Net: **the $50 is reserved entirely for grid-vision**, and even that
spends only AFTER the non-GPU data-prep lands (ETDII download +
OSM-seeded NAIP chips). No GPU dollar is spent until there is training
data to spend it on.

## THE HARD RULE (mandate: never unbounded)

Any routine that launches a RunPod job MUST, in order:

1. Call `authorize_job(rows, gpu, hourly, max_hours)` and launch **only**
   if it returns `authorized: true`. It refuses on three grounds:
   - `unbounded` — `max_hours` missing / non-finite / ≤ 0 (the mandate's
     hard rule: no time cap → no launch),
   - `bad_rate` — `hourly` ≤ 0 or absurd (fat-finger guard),
   - `insufficient` — worst-case cost would leave < $5 floor buffer.
2. Enforce the returned **`max_runtime_seconds`**. CORRECTION (verified vs
   docs.runpod.io 2026-07-07): **RunPod pods have NO native auto-terminate /
   TTL field** — a pod bills until it is DELETEd or exits. So the cap is NOT
   a value handed to RunPod; it is enforced two ways, belt-and-suspenders:
   (a) a **wall-clock watchdog** that polls the pod and DELETEs it at the cap
   (`scripts/runpod_launch.py`), and (b) the training command wrapped in an
   in-pod `timeout <max_seconds>s` so the training PROCESS self-kills even if
   the watchdog disconnects. A cost cap that only lives in our head is not a
   cost cap — it has to actually terminate the pod.

Worst-case accounting is conservative: an OPEN job reserves
`hourly × max_hours` against the balance until it is CLOSED with its
actual cost, so we can never authorize a second job on money the first
might still spend.

## LEDGER PROTOCOL (session-side CLI, stdlib only)

```
python3 scripts/runpod_budget.py status
python3 scripts/runpod_budget.py authorize --gpu RTX4090 --hourly 0.34 --max-hours 6 --workload grid-detector
python3 scripts/runpod_budget.py open  --job gd-2026-07-08 --gpu RTX4090 --hourly 0.34 --max-hours 6 --workload grid-detector --note "first fine-tune"
python3 scripts/runpod_budget.py close --job gd-2026-07-08 --actual 1.83
```

- `open` reserves the worst case and prints the `max_runtime_seconds`
  to hand to RunPod.
- `close` records the actual spend (releases the reservation) and prints
  the new balance.
- `status` prints balance, open jobs, and — below $10 — the exact
  wishlist top-up purchase-order line.

## REFERENCE RATES (2026-07-07 research; live rate is read at launch)

| GPU       | ~$/hr (community/secure) | $50 buys |
|-----------|--------------------------|----------|
| RTX 4090  | 0.34                     | ~147 hr  |
| A40       | 0.44                     | ~114 hr  |
| A100      | 1.39–1.64                | ~30–36 hr|
| H100      | 2.39–2.89                | ~17–21 hr|

Per-second billing; serverless scale-to-zero; spot 50–80% cheaper.
Grid-vision fine-tuning of a mid-size detector fits comfortably on a
single RTX 4090 / A40 for a few hours — a first run is a **few dollars**,
not tens, so the $50 covers the initial train + several sweep iterations
with wide margin.

## LAUNCH PATH — RESOLVED: OPTION A (key in session), 2026-07-08

Mike chose **Option A**: add `RUNPOD_API_KEY` to the Claude Code **session
environment** (done). The launcher `scripts/runpod_launch.py` runs from the
session: gate → create pod → wall-clock watchdog → terminate → ledger, with
the in-pod `timeout` wrapper as backup (see the HARD RULE correction above).

IMPORTANT — env vars load at SESSION START. The session in which the key was
added was already running, so it did NOT see the key. **The next FRESH session
will have it** (presence-checked at boot). Then:
- `python3 scripts/runpod_launch.py dry-run --cmd "..."` — gate + create body,
  no key needed (already verified locally).
- `python3 scripts/runpod_launch.py smoke` — cheap `nvidia-smi` run (~$0.01),
  proves the whole path (create → watchdog → terminate → ledger) end-to-end
  BEFORE the real training script exists. Run this first.
- `python3 scripts/runpod_launch.py launch --job gd-v0 --cmd "<train>"` — real.

OPTION A CAVEAT (accepted): the watchdog lives in the session — keep it alive
during a run; the in-pod `timeout` still bounds the training process if it dies.
Option B (server-side watchdog, key stays in Railway) removes the caveat and is
the better long-term home if GPU jobs become recurring/automated — not built.

STILL NEEDED before the REAL fine-tune (independent of the launch path): the
training container/script that runs ON the pod (pulls ETDII US + builds NAIP
chips, runs the tower-detector fine-tune, writes weights back), plus the two
Phase-B data gaps (`build_power_tiles.sh` `power=tower`; Duke-US zips). The
tower-only v0 can train on ETDII US without the gaps.

## LEDGER (human-readable mirror; source of truth is the JSONL)

_No jobs opened yet. First entry will be the grid-vision detector fine-tune
once Phase B data-prep lands._
