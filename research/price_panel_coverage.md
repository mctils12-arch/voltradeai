# Price panel coverage — Alpaca free daily bars (2026-10-28)

Human order (2026-10-28): "Check whether Alpaca's free bars include delisted
tickers since 2010. If they do, build the price panel from them. No paid
data." Answer: **delisted names yes, from 2016 on; 2010-2015 no.** The panel
is built from 2016-01-04 (`server/pricePanel.ts`) and labelled
survivorship-REDUCED, not survivorship-free.

## Facts (live `bars_coverage` probe, prod v1.0.1203, 2026-10-28 23:20Z)

| Question | Result |
|---|---|
| Earliest daily bar, SPY/AAPL/IWM | sip **2016-01-04**, iex 2018-11-01. `history_reaches_2010`: false / false |
| Delisted tickers probed (2012-2023) | 17. 14 return bars, 12 reach their delisting event (within 10 days) |
| Pre-2016 delistings (MHS 2012, HNZ 2013, old DELL 2013, RSH 2015) | nothing: invisible by construction |
| Post-2016 (EMC, LNKD, MON, CELG, S, XLNX, CTXS, TWTR, SIVB, FRC, BBBY, ATVI) | all reach their event |
| By symbol mapping | default `asof` 12, unmapped `asof=-` 13, dated `asof` best |
| Rename (YHOO -> AABA) | dated asof: 944 bars to 2019-10-02 (AABA continuation); default asof stops 2017-06-16 |
| Reuse: S (Sprint -> SentinelOne) | default asof: 1338 bars, stops 2020-03-31 (old company); `asof=-`: 2720 bars to 2026 (both) |
| Reuse: DELL | default asof = the new Dell from 2018-12-28 |
| Past-event continuation | EMC, MON, BBBY default-asof series run past the event (successor / OTC) |
| /v2/assets | inactive 18,676, active 14,403; 236 inactive symbols are also active (reuse) |
| Assets as a delisting list | only **3 of 17** probed delisted names appear in the inactive list |

## What was built (commit B)

`server/pricePanel.ts` runs a nightly job (`price_panel_build` in the heartbeat registry). It works only 20:15-04:00 ET and makes one request at a time, at most 100/min. On a 429 it backs off. Each batch is checkpointed, so the job resumes the next night. The disk guard keeps 1.5 GiB free so the archives keep priority. Flag `VOLTRADE_PRICE_PANEL`: on by default, and when the env var is present it decides (`0` or `off` disables). Status is at `/api/diag/price_panel`.

- **Universe:** `/v2/assets` active plus inactive, OTC excluded, plus every symbol in Alpaca's corporate-actions feed from 2016 (merger acquirees, name changes, worthless removals). The supplement exists because the asset list alone under-enumerates delistings. The manifest records how many corporate-action events came back. If that number is zero or the call failed (`corp_actions_http: null`), the universe is assets-only and misses most delistings. A future session must check this on the first live build.
- **Bars:** multi-symbol `/v2/stocks/bars`, 1Day, `adjustment=all` plus a raw-close pass, default `asof`, from 2016-01-01.
  - The unmapped (`asof=-`) series is added for (a) reuse suspects (symbol both active and inactive, more than one asset id, or an active symbol with a dated event) and (b) active symbols whose default series ended more than 30 days before as_of. Case (b) is the S case: Sprint mapped onto today's symbol.
- **Identity:** asset id(s) plus date segment. A series splits wherever consecutive bars are more than 30 days apart.
  - **Clipping:** a default-mapping series of a delisted or reused name is clipped after its corporate-action event date.
  - **Flagging:** without a dated event, the series is kept and flagged `inactive_no_event_date`. The flag means the post-delisting OTC tail may still be in it, which is the BBBY case.
- **Files:** `DATA_DIR/price_panel/<build_id>/`, which Python reaches as `storage_config.PRICE_PANEL_DIR`.
  - `universe.jsonl`
  - `bars/bNNNNN.jsonl.gz`
  - `manifest.json`
  - `../current.json`, a pointer to the last complete build.

### Line format (one JSON object per symbol x mapping mode x segment)

`s, mode ("default"|"raw"), seg, seg_start, seg_end, ids[], st ("active"|"inactive"|"corp_action"), ev ({kind,date}|null), flags[], d0, n, k, dd[], c[], o[], h[], l[], v[], f[]`

Decoding (mirrors `decodeSegment`):
- **Dates:** day_i = d0 + cumulative sum of `dd`, in calendar days.
- **Closes:** close_i = (cumulative sum of `c`) / 10^k.
- **Open, high, low:** open/high/low_i = (close_int_i + o/h/l_i) / 10^k.
- **Volume:** `v` is shares, rounded.
- **Raw close:** raw_close_i = close_i x ratio, where ratio comes from the last `f` entry [index, ratio] with index <= i. If no `f` entry applies, the raw close is unknown.

Prices are back-adjusted for splits and dividends (`adjustment=all`). Use them for returns, and use raw close for point-in-time price filters.

## Caveats (also in every manifest)

1. **No 2010-2015.** The panel starts 2016-01-04. A test that needs 2009-2015 (the Lazy Prices gate 2 asks for 2009-2026) cannot be run honestly on it before 2016. Restrict the test window, or see below.
2. **Survivorship-reduced, not free.** A delisted name that appears in neither the asset list nor the corporate-actions feed is missing. The coverage rate is measurable: 12/13 post-2016 probes reached their event, but that is 13 hand-picked names, not a census.
3. **Delisting returns.** Post-delisting OTC prices are usually absent. The backtest must apply and state a delisting-return assumption, e.g. Shumway (1997): about -30% for performance delistings, 0 for cash mergers at the deal price.
4. **Ticker reuse and renames.** Never join on ticker alone. The Lazy Prices CIK join needs the filing-date ticker plus an EDGAR former-name check. Within a reuse-suspect symbol, prefer the `raw` segments and the event date.
5. **Licence.** Alpaca market data is for internal research only. Never redistribute it through /data or /api/v1; derived statistics only.
6. **Volume headroom.** On 2026-10-12 the Railway volume had about 1.39 GB free (wishlist "VOLUME CAPACITY"). If that still holds, the 1.5 GiB floor blocks the build before its first request, and `/api/diag/price_panel` shows `blocked: disk ...`. The floor is deliberate: an archive gap never refills, and the panel can be rebuilt any night. Unblocking is the existing human decision: grow the volume, or authorise vessel-archive thinning. A lower floor via `VOLTRADE_PRICE_PANEL_MIN_FREE_BYTES` is possible but eats the archives' margin. Expected panel size is about 150-300 MB compressed, with a hard cap of 600 MiB.

## What 2010-2015 would need (build-first, CLAUDE.md order)

1. **Raw material free?** Not from Alpaca (history starts 2016). Stooq offers free bulk daily US prices, but its US coverage of delisted names is poor and its terms forbid redistribution. Usable for internal research at best; survivorship coverage unverified. Yahoo serves no delisted tickers. Tiingo's free tier is for personal use with request caps; delisted coverage is claimed but unverified, and the licence is not commercial.
2. **Accumulation?** No. The gap is the past.
3. **Inference?** No honest free substitute for realised returns of delisted firms.
4. **Paid** (from the 2026-10-18 wishlist entry, prices unverified): EODHD about $20 for one month (delisted flag, EOD history back to the 1990s), with a 25-CIK delisted spot-check first. Escalate to Polygon/Massive Advanced (about $199 for one month) only if EODHD fails the spot-check.

The human said "no paid data", so 2010-2015 stays out of scope. Any test needing pre-2016 delisted coverage stays not-run and is recorded as such.
