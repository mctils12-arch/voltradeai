# Aircraft data contract for the app

Written 2026-11-16 for the session that builds the mobile app. It lists the
server endpoints an aircraft app needs, what each field is, its unit, how
fresh it is, which values are measured and which are estimated, and the
limits. All endpoints are `GET`, return JSON and need no key today.

Base URL: the site's own origin (production: the Railway deployment).

## Rules that hold for every endpoint

- **Community ADS-B only.** Positions come from volunteer ground receivers
  through adsb.lol (and two fallbacks, see Licensing). An aircraft outside
  receiver range is not seen. Oceans, polar regions and low altitudes far from
  cities have holes.
- **A missing value is `null`.** Never `0`. A reported `0` (level flight, wings
  level, calm wind) is a real value.
- **Nothing is interpolated on the server.** Where data stops, the response
  says so (a gap, an age, a `null`). If the app smooths, predicts or derives a
  value, the app must label it as an estimate.
- **Times.** Seconds since the Unix epoch (UTC) unless a field name ends in
  `Ms` or is documented as milliseconds. The older live endpoints use
  milliseconds (`at`, `seenAt`); the two new ones use seconds for fix times and
  milliseconds for `at`.
- **Units.** Storage and these new endpoints use SI (metres, metres per
  second). The older live endpoints use aviation units (feet, knots, ft/min).
  Each table below states the unit. Convert for display on the client.

## Which altitude is which

| Name | What it is | What it is not |
| --- | --- | --- |
| geometric altitude | GNSS height above the WGS84 ellipsoid, broadcast by the aircraft | Not height above mean sea level: the two differ by the local geoid height (about -105 m to +85 m). Not height above the ground. |
| barometric altitude | Pressure altitude on the 1013.25 hPa standard setting, broadcast by the aircraft | Not corrected for local pressure or temperature. Below the transition level it can differ from the real altitude by hundreds of feet. |

No field in these endpoints is "true altitude". `alt_kind` names the kind on
every point of a track.

## 1. Live position of one aircraft

`GET /api/data/aircraft/live/:hex`

- `hex`: ICAO 24-bit address, 6 hex digits.
- Polled by the web client every 2.5 s for the selected aircraft. Use the same
  cadence, not faster: the server allows about 1 upstream request per second
  in total and answers from its snapshot above that (`source: "limited"`).
- `Cache-Control: no-store`.

| Field | Meaning |
| --- | --- |
| `at` | server time, ms |
| `source` | `snapshot` \| `cache` \| `upstream` \| `fallback` \| `limited` \| `error` \| `absent`: where this answer came from. `fallback` means adsb.lol failed and the second provider answered (the row's `src` is then `adsbfi`, which is non-commercial: see Licensing). `error` means no provider had anything newer and the row is the last one held. |
| `fields`, `rows` | positional: `rows[0]` is one array in `fields` order. `rows` is empty when nothing holds the aircraft. |
| `intent` | autopilot/selected-altitude and derived state values, each with its age; `null` when never seen |

Row fields: `hex`, `lon`, `lat`, `altFt` (barometric feet, or geometric when no
barometric was sent; `null` on the ground), `gsKt` (ground speed, knots),
`trk` (track over ground, degrees, 0.1), `callsign`, `type`, `seenAt` (the
fix's own time, ms), `cat`, `gnd` (on ground), `reg`, `src` (provider),
`baroRate` (the fix's broadcast vertical rate, ft/min: barometric, or geometric
when no barometric rate was sent; `null` when none came with this fix).

Freshness: compare `seenAt` with `at`. A row older than about 2 minutes is a
last known position, not a live one. Rows older than 10 minutes are not served.

`intent.values` can carry `alt_geom` (ft), `alt_baro` (ft), `baro_rate` and
`geom_rate` (ft/min), `roll` (degrees), `track_rate`, `true_heading`,
`mag_heading`, wind, airspeeds and autopilot modes, only for aircraft that
send them and only while under 30 s old. `intent.age_s` gives each value's age.

## 2. Every aircraft, worldwide

`GET /api/data/aircraft/global`

Optional: `lamin`, `lamax`, `lomin`, `lomax` (all four: a bounding box),
`changed=<previous at>` (only rows changed since), `lawful=1` (adsb.lol rows
only). `Cache-Control: public, max-age=10`.

Same positional row as endpoint 1 without `baroRate`. One row per aircraft: its
freshest fix. `coverage` states what the sweep covered and how old it is;
`coverage.archive` and `coverage.archive.fastlane` state what is being written
to the archive and what is being skipped, with counts.

Not every aircraft on Earth: see `honesty` in the response.

## 3. Flown track of one aircraft (new)

`GET /api/data/aircraft/track/:hex?from=&to=&max=`

- `from`, `to`: whole epoch seconds. Neither: the last 24 hours. `from` alone:
  up to now. The window may not exceed 48 hours (HTTP 400 beyond that); ask for
  one flight at a time using the `start_t`/`end_t` from endpoint 4.
- `max`: 2 to 5000 points, default 2000.
- `Cache-Control: public, max-age=10`. The archive part is re-read at most
  every 30 s per aircraft and window; the live fix is added fresh every time.

The response is positional: `rows` is an array of arrays in `fields` order.
`units` and `enums` in the response repeat the table below.

| Field | Unit / values | Measured or estimated |
| --- | --- | --- |
| `t` | epoch s, the time of the fix | see `t_src` |
| `lat`, `lon` | degrees, 4 decimals (about 11 m) | broadcast |
| `alt_m` | metres: `alt_geom_m` if present, else `alt_baro_m`, else the archive's unlabelled altitude; `null` on the ground or unknown | broadcast |
| `alt_kind` | `geometric` \| `barometric` \| `unrecorded` \| `null` | label for `alt_m`. `unrecorded`: the archive line does not say which; it is barometric when the aircraft sent one, else geometric |
| `alt_geom_m` | metres above the WGS84 ellipsoid | broadcast |
| `alt_baro_m` | metres, pressure altitude | broadcast |
| `gs_ms` | ground speed, m/s, whole (about +-0.5 m/s) | broadcast |
| `track_deg` | track over the ground, degrees true, whole | broadcast. Not the nose heading |
| `vr_ms` | vertical rate, m/s, 0.1, + is climbing | see `vr_source` |
| `vr_source` | `broadcast` \| `derived` \| `null` | `derived` = computed on the server: (altitude of this fix - altitude of the previous fix) / `vr_dt_s`, both the same kind of altitude, the two fixes at most 120 s apart, neither on the ground. An average over `vr_dt_s`, from altitudes rounded to 1 m and times to 1 s |
| `vr_kind` | `barometric` \| `geometric` \| `null` | which rate was broadcast, or which altitude a derived rate used |
| `vr_dt_s` | seconds between the two fixes of a derived rate; `null` otherwise | |
| `on_ground` | boolean | broadcast |
| `roll_deg` | bank, degrees, + right wing down | broadcast, and only by a minority of aircraft; `null` otherwise |
| `t_src` | `fix` \| `receive_minus_age` \| `unadjusted` | `fix`: written at the fix's own time. `receive_minus_age`: the line was stamped when the server received it and the feed's position age was subtracted (good to 1-2 s). `unadjusted`: no age was recorded, the time may be late |
| `src` | `adsblol` \| `adsbfi` \| `airplaneslive` \| `opensky` \| `null` | provider of the fix |
| `pos_type` | `adsb_icao`, `adsr_icao`, `mlat`, `tisb_*`, `mode_s`, `null` | `mlat` and `tisb_*` positions are computed on the ground, not by the aircraft, and are noisier |
| `origin` | `archive` \| `live` | `live` = the newest fix in the live snapshot, appended when it is newer than the last archived line |

Other parts of the response:

- `live`: whether a live fix was appended, its time and age, and the reason
  when it was not.
- `gaps`: `threshold_s` (300), `count`, and `listed` (`from_t`, `to_t`,
  `seconds`; at most 200 listed). **Do not draw a line across a gap.** The
  points on both sides are real; nothing is known in between.
- `decimation`: when the window holds more fixes than `max`, every k-th fix is
  returned plus the first fix, the last fix and the fix on each side of every
  gap. `stride` is k. `vr_ms` was computed before thinning. If there are more
  gap end points than `max`, `gap_endpoints_kept` is `false`.
- `coverage`: first and last fix time, counts of broadcast / derived / missing
  vertical rates and of each altitude kind, the hour files read, and `archive`
  (see "How far back history goes").
- `attitude`: plain statements that pitch is not in the data and that bank is
  only in `roll_deg` where broadcast.

### Pitch and bank on the client

- **Pitch is never broadcast.** The app can show the flight path angle,
  `atan(vr_ms / gs_ms)`, labelled as an estimate. It is not the nose pitch.
- **Bank**: use `roll_deg` when it is not `null`. Otherwise estimate it from
  the turn rate: `bank = atan(gs_ms * turn_rate / 9.81)`, with `turn_rate` in
  radians per second from the change of `track_deg` between two fixes. Label it
  as an estimate, and do not compute it across a gap or across thinned points
  that are far apart.

### How dense the track is

The archive keeps a sample, not every broadcast:

| Aircraft state | One fix kept per |
| --- | --- |
| on the ground | 5 min |
| within 150 km of a listed strategic site | 30 s |
| below 3000 m | 60 s |
| otherwise (cruise) | 75 s |
| a turn of more than 5 degrees, a 500 ft altitude change, or a ground/air change | as soon as 5 s after the last kept fix |
| a tail on the server's tracked list | every 30 s poll, plus the provider's full day trace every 15 min |
| the aircraft a viewer has selected | every distinct fix the live endpoint receives (about every 2.5 s), since 2026-11-16 |

A fix is archived only when some path received it: a viewer's map covered it,
the tail is tracked, or the worldwide sweep was archiving (the sweep is paused
whenever the volume has under 2 GiB free).

## 4. Fleet history: past flights for saved tail numbers (new)

`GET /api/data/aircraft/fleet?tails=N843S,N123AB&hexes=ab8c8e`

- `tails`: comma-separated registrations. `hexes`: comma-separated ICAO hexes.
  Up to 20 aircraft per request in total. Anything malformed is HTTP 400.
- Neither parameter: the server's tracked list (endpoint 6) is used.
- `Cache-Control: public, max-age=30`. Each aircraft's flight list is
  recomputed at most every 5 minutes (`history_as_of`).
- The first request for an aircraft reads every local hour file once and can
  take from seconds to more than a minute on a full archive. Later requests are
  served from the 5-minute cache. Show a loading state.

Per aircraft:

| Field | Meaning |
| --- | --- |
| `tail` | the registration asked for, or `null` when only a hex was given |
| `hex` | the resolved ICAO hex, or `null` when it could not be resolved. Never guessed |
| `resolved_by` | `given` \| `tracked_registry` \| `live_feed` \| `faa_registry` \| `archive_registration` \| `null` |
| `resolution.basis` | one sentence: how the hex was found, or why it was not |
| `resolution.conflicts` | present when another source names a different hex |
| `tracked` | the tail's entry on the server's tracked list, or `null` |
| `flights` | newest first, at most 200 listed; `flight_count` is the total |
| `archive` | `fixes`, `first_fix_t` (the first fix the archive holds for this aircraft), `last_fix_t`, `fixes_dropped_over_cap` (not zero means older flights exist and were not read), `registrations_seen` |
| `live` | the newest live fix (`seen_t`, `age_s`, `fresh`, `lat`, `lon`, `alt_m`, `on_ground`, `gs_ms`, `track_deg`, `callsign`, `src`) or `null` |
| `live_note` | `"no signal now: ..."` when `live` is `null`; `"last known position, N s old"` when the fix is over 2 minutes old |

How a tail is resolved, in order: the tracked list's learned hex; a live row
carrying that registration; the FAA registry's N-number to Mode S pairing (a
2026-07-05 build covering only aircraft the archive held then); the
registration on an archived fix in the newest 72 hour files. If none match,
`hex` is `null` and `resolution.basis` says so. Pass `hexes=` when the app
already knows the hex.

Per flight (the same object as endpoint 5):

| Field | Meaning |
| --- | --- |
| `start_t`, `end_t` | first and last fix of the flight, epoch s |
| `duration_s` | `end_t - start_t`: a lower bound, because the archive is thinned |
| `fixes` | archived fixes in the flight |
| `from`, `to` | first and last position `{la, lo}`, degrees |
| `from_airport`, `to_airport` | nearest fixed-wing airport within 6 km (`id`, `n` name, `dist_km`, `el` elevation m) or `null` |
| `max_alt_m` | highest archived altitude, metres: barometric where sent, else geometric |
| `callsigns` | callsigns seen |
| `quality` | `complete` \| `taxi_only` \| `partial_start` \| `signal_lost_airborne` \| `partial_both` |
| `quality_basis` | why |
| `is_flight` | `false` = ground movement only |
| `verified` | both ends agree with an airport's position and elevation |
| `bbox` | `[minLon, minLat, maxLon, maxLat]` |

A new flight starts after 45 minutes without a fix, or after 15 minutes on the
ground between two airborne stretches. `signal_lost_airborne` means coverage
ended while the aircraft was in the air. It says nothing else.

To draw a flight, pass its `start_t` and `end_t` to endpoint 3 as `from` and
`to`. Both endpoints use the same clock.

This endpoint contains no FAA flight-plan or SWIM field.

## 5. Flights of one aircraft (existing)

`GET /api/data/aircraft/trips/:hex`

The same flight objects as endpoint 4 for one hex, cached 5 minutes. Its times
are the archive's stamps, which can be later than the fix times of endpoints 3
and 4 by the feed's position age (under a minute). Prefer endpoint 4 when the
result feeds endpoint 3.

## 6. The server's tracked tail list (existing)

`GET /api/data/aircraft/tracked` returns `planes` (`reg`, `hex` once learned,
`last_seen` ms, `last_pos`), `cap` (20) and `poll_s` (30).
`POST /api/data/aircraft/tracked/:reg` adds a tail,
`DELETE /api/data/aircraft/tracked/:reg` removes one. A tracked tail is
archived at every poll whether or not anyone is looking at it.

## 7. Flight plan (existing)

`GET /api/data/aircraft/plan/:hex?callsign=&lat=&lon=&alt=&trk=`

Returns `source` (`FILED_FAA` \| `HISTORY_PREDICTED` \| `ROUTE_DB_PREDICTED` \|
`NONE`), `label`, `origin`, `destination`, `points`, `cruiseAltFt`,
`cruiseAltEstimated`, `pathEstimated`, `deviation`, `events`, `ageSec` and
`honesty`. Everything except a `FILED_FAA` plan is a prediction and is labelled
so; any altitude not filed or observed has `altEstimated: true`. For an
aircraft on the FAA's LADD list the FAA-sourced plan is withheld and
`privacy: "LADD"` is set.

## 8. Estimated position when the last fix is old (new)

`GET /api/data/aircraft/estimate/:hex`

Use it beside `/live/:hex` when the live row's fix is getting old, about every
5 s. It makes no provider request of its own. `Cache-Control: no-store`.

The answer keeps the observed and the estimated strictly apart:

| Field | Meaning |
| --- | --- |
| `fix` | the last position actually received: `lat`, `lon`, `altFt`, `gsKt`, `trkDeg`, `onGround`, `fixMs` (the fix's own time), `ageSec`, `source` (`snapshot` \| `archive`), `provider`, `callsign`. With `source: "archive"` the speed and track were worked out from the last two archived positions and `motionDerived` is `true`. `null` when no position is held (see `reason`). |
| `estimate` | NOT an observed position. `estimated` is always `true`. |
| `estimate.method` | `dead_reckoning` (straight along the last track at the last ground speed: up to 120 s, or 30 s when a route can take over), `along_route` (advanced along the route the aircraft was on, stopping at its last point), or `none` (no position is estimated; `reason` says why) |
| `estimate.lat`, `lon`, `trkDeg`, `gsKt`, `distanceNm` | the estimate; `lat`/`lon` are `null` with method `none` |
| `estimate.altFt`, `altBasis` | the last reported altitude, held (`held`); the vertical rate is not applied |
| `estimate.basisFixMs`, `ageSec`, `at` | the fix it started from, its age, and the time the estimate is for |
| `estimate.withinNm` | a circle around the LAST FIX the aircraft is inside unless it sped up by more than 15%. It is not an error bar around the estimate. |
| `estimate.route` | `along_route` only: `source` (the plan's source, so a predicted route is never shown as filed), `crossTrackAtFixNm`, `remainingNm`, `reachedEnd` |
| `estimate.path` | `[lat, lon, epoch ms]` points starting at the real fix: the line to draw DASHED. It runs to 10 minutes past the estimated position along the route, or only as far as dead reckoning is allowed. |
| `estimate.assumptions`, `reason` | plain sentences the app can show |
| `plan` | `source`, `label`, `privacy`, `used` (whether the estimate ran along it), `note` |

Rules for the app:
- Draw `estimate` as estimated (dashed path, a distinct marker), never as the aircraft's position.
- Show the fix's real age (`fix.ageSec`) whenever an estimate is on screen.
- When `method` is `none`, show the last fix and its age, nothing else.
- NOT VALIDATED: the estimate's accuracy against real tracks has not been measured yet. An aircraft vectored off its route after the last fix is drawn on the route.
- The same arithmetic is in `shared/aircraftEstimate.ts` (pure TypeScript) for per-frame motion between answers.

## 9. Flight plan versions, amendments and off-route stretches (new)

`GET /api/data/aircraft/plan-history/:hex[?callsign=][&geometry=all][&archive=1]`

Only FAA-filed plans have versions (`source: "FILED_FAA"`); otherwise the lists
are empty and `source` says so. For an aircraft on the FAA's LADD list
everything is withheld (`source: "NONE"`, `privacy: "LADD"`).

| Field | Meaning |
| --- | --- |
| `original`, `current` | the first version we hold and the newest, each with `version`, `receivedAtMs` (our clock), `swimMessageMs` (the message's own time, when it carried one), `messageType`, `departure`, `arrival`, `cruiseAltFt`, `routeText`, `fixes`, `points` (`lat`, `lon`, `name`) |
| `plan.originalAsFiled` / `plan.originalIsFirstSeen` | `originalAsFiled` is true only when version 1 came from the filing message itself. Otherwise the "original" is the first version WE saw (for example after a server restart) and must be labelled that way. |
| `versions[]` | every version held, same shape; `points` only on `original`/`current` unless `geometry=all` |
| `amendments[]` | one per change: `at` and `atBasis` (message time or our receive time), `kinds` (`REROUTE`, `FIXES_ADDED`, `FIXES_REMOVED`, `SHORTCUT`, `DIRECT_TO`, `DESTINATION_CHANGE`, `ALTITUDE_CHANGE`, `ROUTE_TEXT_ONLY`), `change` (fix names added/removed, old and new destination or cruise altitude, the first differing leg, `maxLateralNm`), and `position` |
| `amendments[].position` | where the aircraft was when the amendment took effect: `method` `EXACT` \| `NEAREST` (a real fix within 30 s, with `fixTimeMs` and `gapSec`) \| `INTERPOLATED` (`interpolated: true`: between two fixes, an estimate). `null` with `positionReason` when no fix is near enough. Draw the marker only when it is not `null`; mark interpolated ones as estimated. |
| `offRoute.episodes[]` | stretches flown off the version in force: start and end time and position, `maxCrossTrackNm`, `classification` (`AMENDMENT_FOLLOWED` with the linked version and the lag, `TERMINAL_VECTORING`, `NO_AMENDMENT_RECEIVED`), `endReason`. The cause is never guessed: `NO_AMENDMENT_RECEIVED` says "not determined". |
| `offRoute.assessed`, `reason` | `false` with a reason when the route cannot be judged (for example a coarse or estimated path) |
| `flownTrack.url` | where to get the flown track (use section 3 for the full per-point data) |

To draw the three lines the owner asked for: `original.points` greyed,
`current.points` as the active route, the flown track from section 3, and one
marker per `amendments[].position`.

Limits to show honestly:
- Positions and off-route episodes need our ADS-B archive for that aircraft; for one nobody tracked or viewed they are often empty.
- Versions are held in memory for flights the server currently knows, and on disk for 7 days for flights whose plan CHANGED. A never-amended flight's plan is not written to disk.
- The plan is matched to the aircraft by callsign.
- NOT YET RUN AGAINST THE LIVE FEED at the time of writing: verified by unit tests only.

## How far back history goes

- Raw fixes are kept 30 days in total, then deleted everywhere.
- Endpoints 3, 4 and 5 read the hour files on the server's local disk only.
  With the cold tier (R2) configured, days older than about 22 days may already
  be off the local disk and are then not read by these endpoints.
- `coverage.archive.local_oldest_hour_t` is the start of the oldest hour file
  on the local disk at the time of the request. Nothing before it is served.
- Past the raw window only one coarse line per aircraft per day survives, with
  no altitude and no flights. It is not served by these endpoints.
- Vertical rate and geometric altitude as broadcast are in the archive from
  2026-10-14 onward. Older lines can only give a derived rate and an
  `unrecorded` altitude kind.
- A flight nobody received is not in the archive. "All past flights" means all
  flights the archive holds inside the window above.

## Licensing

| `src` | Provider | Terms |
| --- | --- | --- |
| `adsblol` | adsb.lol | ODbL 1.0. May be used commercially with attribution and share-alike of the database. |
| `adsbfi` | adsb.fi | Non-commercial. Used only as a fallback and to fill coverage holes. |
| `airplaneslive` | airplanes.live | Non-commercial. Viewport fallback only. |
| `opensky` | OpenSky Network | Non-commercial. Off unless credentials are configured. |

Every live row and every track point carries `src`. A paid or ad-supported app
must show only `adsblol` rows (`lawful=1` on endpoint 2; filter on `src`
elsewhere) and must credit adsb.lol. Airport names come from OurAirports
(public domain). The FAA registry pairing used to resolve a tail is public
domain.

## Limits

| Endpoint | Limit |
| --- | --- |
| live | about 1 provider request per second in total across all viewers; 20 distinct aircraft per minute |
| track | 48 h window, 2 to 5000 points, 50,000 fixes read per request (newest kept, the rest counted) |
| fleet | 20 aircraft, 200 flights listed per aircraft, 1,000,000 fixes per request split across the aircraft asked for (newest kept, the rest counted) |
| tracked list | 20 tails |
