# VolTradeAI memory audit — where the Railway bill goes (2026-11-03)

Read-only investigation. No code changed. Numbers come from the live server (its own
/api/health and diagnostics), a week of stored health samples, and a line-by-line
survey of the server (Node) and engine (Python) code. Dollar figures use your
rate of about $10 per GB-month.

## Here's where your memory is going

The container runs at about 3.7–4.0 GB most of the day. Right after each deploy it
starts at about 0.7 GB and climbs to 3.5+ GB within 15–40 minutes, then stays there
until the next deploy (5–12 deploys a day). It is not a slow leak: it is a set of
things that get loaded or started minutes after boot and are kept.

| # | What | Typical size | ~Cost/mo | Confidence |
|---|------|-------------:|---------:|------------|
| 1 | The web/orchestrator process (Node), of which: | 2.7 GB | $27 | measured |
| 1a | A list of every DTCC swap-report ID ever seen (6.56 million IDs), kept in memory to avoid archiving duplicates; grows ~5–8 MB a day | 0.4–0.5 GB | $4–5 | high |
| 1b | Slack the JavaScript engine keeps because it is allowed a 6 GB ceiling: it cleans up late and holds the freed space | 0.3–0.5 GB | $3–5 | medium-high |
| 1c | Four background re-reads of the position archives every 10–15 minutes (shadow-fleet 3 days, port-dwell 7 days, GNSS-integrity 2×21 days, entity graph): each run re-parses ~20 million lines from compressed files; about 2.5 minutes of every 10 are spent doing it, and the operating system keeps the re-read files cached in RAM | 0.1–0.3 GB working + 0.3–0.5 GB file cache | $3–6 | medium |
| 1d | Dashboard/map caches with no time limit (aircraft map windows, world aircraft bodies, vessel snapshots, flight-trip filters, methane/airport datasets loaded on first visit and never dropped) | 0.1–0.3 GB | $1.5–3 | medium-low |
| 1e | Program floor: code, libraries, including an image-processing library that is loaded at start but only used for aviation chart overviews | 0.15–0.25 GB | $0.5–1 (the avoidable part) | medium |
| 1f | About 60 live feeds, each keeping its latest payload plus a capped list of IDs already archived | 0.06–0.2 GB | small | medium |
| 1g | Live aircraft and vessel positions being served (bounded, necessary) | 0.06–0.08 GB | 0 | high |
| 2 | The Python trading daemon | 0.3–0.46 GB | $3–4.6 | measured |
| 2a | The shadow-portfolio log (a ~45 MB JSON file) is read whole, modified and rewritten for every candidate, 10–30 times per scan; it is the daemon's size spike | +0.1–0.15 GB peak | $1–1.5 | medium-high |
| 2b | A volatility-surface cache that is never trimmed (grows with every ticker looked at while the daemon is up) | 0.04–0.14 GB | $0.5–1 | medium |
| 2c | Libraries the engine must load (numpy, pandas, LightGBM, scikit-learn) | 0.25 GB | 0 | high |
| 3 | Extra Python interpreters started by the orchestrator: a fresh one every 45 seconds for position management (50–300 MB each, ~1,900 a day), one per scan candidate for research (125–150 MB each), the hourly model retrain (250–600 MB), overnight research jobs (150–500 MB, hourly 8 pm–4 am ET), and two per 10-second dashboard poll when the cockpit is open | 0.05–0.3 GB average, 1+ GB peaks | $0.5–3 | medium |

Measured anchors: Node heap live set 1.9–2.0 GB (floor of 30-second samples over an
hour, and the stored week of samples); Node RSS 2.65–2.8 GB; daemon peak 0.46 GB;
container 3.7–3.95 GB; port-dwell fold measured live at 724k points in 4.5 s for a
12-hour window (so ~63 s for its 7-day window, every 10 minutes).

## Each finding in plain English

**1a. The DTCC ID list (biggest single item, ~$4–5/month, growing).** The DTCC feed
re-publishes every swap event since inception every day, so the server keeps a list of
every ID it has already archived to skip the repeats. That list is now 6.56 million IDs
and lives entirely in RAM, like keeping every receipt you've ever filed spread out on
the desk so you can check whether a new one is a duplicate. It was 1.95 million in
September; it alone explains a third of the growth since then. Fix: keep the IDs in the
database on disk (or as a compact numeric index) and check duplicates there; the
archive's contents stay byte-for-byte identical.

**1b. The engine's ceiling is too high (~$3–5/month).** The start script gives the
JavaScript engine permission to use up to 6 GB. With that much headroom it tidies up
lazily and holds on to freed space, like a tenant who never takes the trash out
because the bin is huge. Capping it at roughly 1.5× what the program actually needs
makes it clean up promptly. This is one environment variable (no frozen file touched);
it must be set only after 1a shrinks the real need, or the July crash-loop repeats.

**1c. Re-reading the archives every 10 minutes (~$3–6/month, plus CPU).** Four
background jobs each re-read days of compressed position history from scratch every
10–15 minutes to recompute dashboards, like re-counting the whole warehouse every ten
minutes instead of counting only the new deliveries. The working set is modest, but
the operating system caches the re-read files in RAM and the engine churns through
garbage while it runs. Fix: fold each hour of the archive once, save that hour's
partial result, and combine the partials each cycle; the output is identical (the
same points aggregated the same way, and the repository already has
identical-output tests for these folds).

**1d. Caches that never expire (~$1.5–3/month).** Several map and dashboard caches
drop entries only when they hit a count limit, never by age, and some datasets are
loaded on the first visit and kept forever. Fix: add an idle timeout so they are
dropped after ~10–15 minutes without use and reloaded from disk on demand; same
answers, just not held overnight.

**1e. An image library loaded at start (~$0.5–1/month).** The server loads an image
processing library at boot because one feature (aviation chart overviews) can use it.
Fix: load it the first time that feature runs.

**2a. The shadow-portfolio log (~$1–1.5/month).** During each scan the daemon reads a
45 MB JSON file whole, adds one record, and writes it all back, for every candidate.
Fix: read it once per scan (or append records to a line-per-record file); same
records, same results, and the daemon's restart-causing spikes go away.

**2b. The volatility-surface cache (~$0.5–1/month).** Every ticker's option surface is
cached and never removed. Fix: keep the most recent 100–200 and drop the oldest.

**3. Too many fresh Python interpreters (~$0.5–3/month, mostly CPU).** Starting a
Python interpreter with the trading libraries is like booting a laptop to answer one
question. The position-management call does it every 45 seconds even though a
long-running daemon exists for exactly this job; research calls do it per candidate.
Fix: route those calls through the daemon; same functions, same output.

## Outside market hours

Nothing here changes what the system produces. The 45-second position spawn and
the cockpit-driven spawns already only matter during market hours or while the
dashboard is open. The overnight research jobs (8 pm–4 am ET) and the 4 am model
retrain can be serialized so they never overlap each other or the daemon, which
lowers the peak rather than the average. The 10-minute dashboard folds feed /data
pages, not trading signals; making them incremental (1c) keeps their freshness exactly
as it is, so I do not propose slowing them.

## Proposed fixes, in order, with estimated savings

| Order | Fix | Keeps output identical because | Saves (GB) | ~$/month |
|---|---|---|---|---|
| 1 | DTCC IDs to the on-disk database / compact index | same duplicate check, same archive | 0.4–0.5 now, more later | 4–5 |
| 2 | Cap the Node heap via VOLTRADE_NODE_HEAP_MB after fix 1 | cleanup timing only | 0.3–0.5 | 3–5 |
| 3 | Incremental archive folds (hour partials) | same aggregation, ratchet-tested | 0.3–0.7 incl. file cache | 3–6 |
| 4 | Idle timeouts on dashboard caches and lazy datasets | caches only | 0.1–0.3 | 1.5–3 |
| 5 | Shadow log: one read per scan | same records | 0.1–0.15 | 1–1.5 |
| 6 | Bound the vol-surface cache | TTL already applies | 0.05–0.1 | 0.5–1 |
| 7 | Route the 45-second and per-candidate Python calls through the daemon | same functions | 0.05–0.1 (+CPU) | 0.5–1 |
| 8 | Lazy-load the image library and the inlined JSON tables | load timing only | 0.05–0.1 | 0.5–1 |
| | **Total** | | **1.3–2.4 GB** | **$13–23** |

One optional step 0, if you want certainty before fix 3 and 4: a read-only
diagnostic that reports the heap's composition (sizes of each cache and the DTCC
set) from the live server. It is a small code change, so it also waits for your
approval.

Nothing is changed until you say which of these to do.
