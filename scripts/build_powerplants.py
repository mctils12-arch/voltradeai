#!/usr/bin/env python3
"""build_powerplants.py — compile the US power-plant reference layer.

Sources (per DESIGN.md REFERENCE DATA ACCURACY):
 1. WRI Global Power Plant Database CSV (v1.3.0, CC BY 4.0) — the plant
    universe (name/fuel/capacity/owner + 2021-vintage coordinates).
 2. EIA-860 Plant file (xlsx, latest annual) — authoritative US registry;
    its coordinates are preferred wherever it knows the plant (fresher).
    NOTE (Hardeeville lesson, 2026-07-04): the registries share
    self-reported geocodes, so agreement is NOT verification.
 3. datacore/powerplants/imagery_verified.json — plant ids whose position
    was verified against imagery (facility visibly at the point). Top 100
    by MW verified 2026-07-04 (100/100 after EIA correction).
 4. datacore/powerplants/position_overrides.json — OSM-derived coordinate
    fixes for registry-mispositioned plants (position audit 2026-07-18,
    research/position_audit_2026-07-18.md). Applied AFTER the GPPD/EIA
    choice, keyed by gppd_idnr, only while the registry value still equals
    the recorded bad coordinate — an upstream fix disables the override.

Row format: [name, capacity_mw, fuel, owner, lat, lon, verified]
  verified: 1 = imagery-verified · 0 = approximate (registry-reported)

Re-run:
    curl -L -o /tmp/gppd.csv https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv
    curl -L -o /tmp/eia860.zip https://www.eia.gov/electricity/data/eia860/xls/eia860<YEAR>.zip && unzip /tmp/eia860.zip -d /tmp/eia860
    python3 scripts/build_powerplants.py /tmp/gppd.csv /tmp/eia860/2___Plant_Y<YEAR>.xlsx
"""
import csv, json, os, sys

FUEL_CODE = {
    "Nuclear": "nuclear", "Coal": "coal", "Gas": "gas", "Oil": "oil",
    "Hydro": "hydro", "Wind": "wind", "Solar": "solar",
    "Biomass": "other", "Waste": "other", "Geothermal": "other",
    "Storage": "other", "Cogeneration": "gas", "Petcoke": "coal",
    "Other": "other", "Wave and Tidal": "other",
}

HERE = os.path.dirname(os.path.abspath(__file__))
DST_DIR = os.path.join(HERE, "..", "datacore", "powerplants")


def load_eia(xlsx: str) -> dict:
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, read_only=True)
    ws = wb.active
    hdr = None
    out = {}
    for r in ws.iter_rows(values_only=True):
        if hdr is None:
            if r and any(str(c).strip() == "Plant Code" for c in r if c):
                hdr = [str(c).strip() if c else "" for c in r]
                i_code, i_lat, i_lon = hdr.index("Plant Code"), hdr.index("Latitude"), hdr.index("Longitude")
            continue
        try:
            out[int(r[i_code])] = (float(r[i_lat]), float(r[i_lon]))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def main(src: str, eia_xlsx: str | None) -> None:
    eia = load_eia(eia_xlsx) if eia_xlsx else {}
    verified_path = os.path.join(DST_DIR, "imagery_verified.json")
    verified = set(json.load(open(verified_path))["ids"]) if os.path.exists(verified_path) else set()
    overrides_path = os.path.join(DST_DIR, "position_overrides.json")
    overrides = {}
    if os.path.exists(overrides_path):
        overrides = {o["gppd_idnr"]: o for o in json.load(open(overrides_path))["overrides"]}

    plants, eia_used, overrides_used = [], 0, 0
    with open(src, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["country"] != "USA":
                continue
            try:
                lat, lon = float(row["latitude"]), float(row["longitude"])
                mw = round(float(row["capacity_mw"]), 1)
            except ValueError:
                continue
            idnr = row["gppd_idnr"]
            try:
                code = int(idnr.replace("USA", ""))
            except ValueError:
                code = None
            if code is not None and code in eia:
                lat, lon = eia[code]
                eia_used += 1
            ov = overrides.get(idnr)
            if ov and abs(lat - ov["from"][0]) < 0.02 and abs(lon - ov["from"][1]) < 0.02:
                # registry still carries the audited-bad coordinate -> apply the
                # OSM-derived fix; if upstream moved it, the override self-disables.
                lat, lon = ov["to"]
                overrides_used += 1
            fuel = FUEL_CODE.get(row["primary_fuel"], "other")
            plants.append([row["name"].strip()[:60], mw, fuel,
                           (row.get("owner") or "").strip()[:60],
                           round(lat, 4), round(lon, 4),
                           1 if idnr in verified else 0])
    plants.sort(key=lambda p: -p[1])
    out = {
        "_doc": "US power plants (RAW reference layer). Universe: WRI GPPD v1.3.0 "
                "(CC BY 4.0); coordinates preferentially from EIA-860 (authoritative "
                "US registry); row[6]=1 means position imagery-verified (facility "
                "visible at point), 0 means approximate (registry-reported). Built by "
                "scripts/build_powerplants.py.",
        "source": "Global Power Plant Database v1.3.0 (WRI, CC BY 4.0) + EIA-860",
        "fuels": ["nuclear", "coal", "gas", "oil", "hydro", "wind", "solar", "other"],
        "count": len(plants),
        "verified_count": sum(p[6] for p in plants),
        "plants": plants,
    }
    os.makedirs(DST_DIR, exist_ok=True)
    dst = os.path.join(DST_DIR, "us_power_plants.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"{len(plants)} plants ({eia_used} EIA-located, {out['verified_count']} imagery-verified, {overrides_used} position-overridden) -> {dst} ({os.path.getsize(dst)//1024} KB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
