# Data Licensing Audit — resell vs display, per source

Monetization readiness item 2 (human-approved 2026-07-04 as pre-revenue
prep; items 1 and 4 of the checklist wait for the charge decision).
This is the standing register the API's LICENSE_MARKS derive from.
RE-VERIFY EVERY ROW AT SWITCH — terms change; each row names its
source-of-truth URL. "Resell" = include in a paid data API response.
"Display" = show on our site with attribution only.

| Source | License | Resell in paid API? | Conditions |
|---|---|---|---|
| SEC EDGAR (Form 4, 8-K) | US public domain | YES | none; courtesy attribution |
| NOAA/NWS (radar) | US public domain | YES (as display tiles we don't — see OWM row for why tiles differ) | none |
| USGS (3DEP, Landsat) | US public domain | YES | none |
| EIA (crude storage) | US public domain | YES | none |
| USDA (CDL) | US public domain | YES | none |
| NASA FIRMS (fires) | Open, attribution | YES | credit NASA FIRMS/LANCE; safety-of-life disclaimer travels with every response |
| Digitraffic FI (trains) | CC BY 4.0 | YES | attribution required |
| Entur NO (trains) | NLOD | YES | attribution required |
| adsb.lol (aircraft) | ODbL 1.0 | YES, share-alike | derived DATABASES must be offered under ODbL + attribution; constrains exclusivity claims; positions + aircraft-derived stats endpoints carry the mark |
| airplanes.live, adsb.fi (fallbacks) | Non-commercial | NO | must leave the chain at switch (checklist item 1; runtime guard already enforces) |
| aisstream.io (vessels) | ToS-conditional | CONDITIONAL | re-read redistribution terms at switch; vessel + AIS-derived endpoints marked conditional until then |
| OpenWeatherMap (temp/wind tiles) | Display product | NO | excluded from the data API entirely; on-map display with attribution only |
| Copernicus S1/S2 (imagery) | CC BY-SA-like open license (free incl. commercial) | YES | attribution "Contains modified Copernicus Sentinel data [year]" |
| WRI GPPD (power plants) | CC BY 4.0 | YES | attribution required |
| Natural Earth (borders) | Public domain | YES | none (courtesy credit shown) |
| ESA WorldCover / JRC GFC2020 / JRC GSW (atlas) | CC BY 4.0 / free-with-attribution | YES (as derived stats; tile passthrough NO — serve from origin) | attribution per layer; GFW named as tile service where used |
| OUR DERIVED datasets (port dwell, shadow stats, transit counts, tank-fill readings, entity timelines) | Our work product | YES | ODbL inputs taint aircraft-derived DATABASES share-alike; AIS-derived stats inherit the aisstream condition; imagery-derived readings carry Copernicus attribution |
| Waitlist emails / any PII | First-party PII | NEVER | never exposed via any API (manifested) |

Enforcement today: server/apiProduct.ts LICENSE_MARKS carries the
per-endpoint marks; the providerCompliance.ts tripwire degrades
/api/health if billing signals appear with a non-commercial provider in
the aircraft chain. At switch: re-verify every row from its primary
source, then update LICENSE_MARKS in the same PR as the verification
note here.
