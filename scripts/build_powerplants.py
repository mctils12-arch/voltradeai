#!/usr/bin/env python3
"""build_powerplants.py — compile the WRI Global Power Plant Database CSV
into the compact US-only JSON served at /api/data/powerplants.

Source: https://github.com/wri/global-power-plant-database (v1.3.0,
CC BY 4.0 — attribution shipped with the layer). Re-run when WRI releases
a new version:

    curl -L -o /tmp/gppd.csv https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv
    python3 scripts/build_powerplants.py /tmp/gppd.csv

Output: datacore/powerplants/us_power_plants.json
Compact row format: [name, capacity_mw, fuel_code, owner, lat, lon]
"""
import csv, json, os, sys

FUEL_CODE = {
    "Nuclear": "nuclear", "Coal": "coal", "Gas": "gas", "Oil": "oil",
    "Hydro": "hydro", "Wind": "wind", "Solar": "solar",
    "Biomass": "other", "Waste": "other", "Geothermal": "other",
    "Storage": "other", "Cogeneration": "gas", "Petcoke": "coal",
    "Other": "other", "Wave and Tidal": "other",
}

def main(src: str) -> None:
    plants = []
    with open(src, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["country"] != "USA":
                continue
            try:
                lat, lon = float(row["latitude"]), float(row["longitude"])
                mw = round(float(row["capacity_mw"]), 1)
            except ValueError:
                continue
            fuel = FUEL_CODE.get(row["primary_fuel"], "other")
            owner = (row.get("owner") or "").strip()[:60]
            name = row["name"].strip()[:60]
            plants.append([name, mw, fuel, owner, round(lat, 4), round(lon, 4)])
    plants.sort(key=lambda p: -p[1])  # biggest first (nicer at low zoom if capped)
    out = {
        "_doc": "US power plants (RAW reference layer). Compiled from the WRI "
                "Global Power Plant Database v1.3.0 (CC BY 4.0) by "
                "scripts/build_powerplants.py. Row: [name, capacity_mw, fuel, owner, lat, lon].",
        "source": "Global Power Plant Database v1.3.0 (WRI, CC BY 4.0)",
        "fuels": ["nuclear", "coal", "gas", "oil", "hydro", "wind", "solar", "other"],
        "count": len(plants),
        "plants": plants,
    }
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "datacore", "powerplants", "us_power_plants.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    by_fuel = {}
    for p in plants:
        by_fuel[p[2]] = by_fuel.get(p[2], 0) + 1
    print(f"{len(plants)} US plants -> {dst} ({os.path.getsize(dst)//1024} KB)")
    print("by fuel:", dict(sorted(by_fuel.items(), key=lambda kv: -kv[1])))

if __name__ == "__main__":
    main(sys.argv[1])
