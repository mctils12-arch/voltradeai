#!/usr/bin/env node
// QC-2 (human directive 2026-08-11: "the plane need to take off from an
// airport at its height and the land at an airport at the height we need
// software for that"): prune OurAirports airports.csv (public domain /
// CC0-style DPL — https://ourairports.com/data/) into a compact runtime
// index datacore/aircraft/airports_min.json used by server/airportsIndex.
//
// Usage: node scripts/build_airports.mjs <path/to/airports.csv>
// Keeps every open airport/seaplane base/heliport with valid coordinates;
// drops closed + balloonports. Elevation stored in METERS (archive
// convention); missing elevation stays null — never fabricated.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";

const src = process.argv[2];
if (!src) { console.error("usage: node scripts/build_airports.mjs <airports.csv>"); process.exit(1); }

// minimal RFC-4180 CSV row parser (quoted fields with embedded commas/quotes)
function parseCsvLine(line) {
  const out = [];
  let cur = "", inQ = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQ) {
      if (ch === '"') {
        if (line[i + 1] === '"') { cur += '"'; i++; }
        else inQ = false;
      } else cur += ch;
    } else if (ch === '"') inQ = true;
    else if (ch === ",") { out.push(cur); cur = ""; }
    else cur += ch;
  }
  out.push(cur);
  return out;
}

const TYPE_CODE = {
  large_airport: "L", medium_airport: "M", small_airport: "S",
  seaplane_base: "W", heliport: "H",
};

const lines = readFileSync(src, "utf-8").split("\n");
const header = parseCsvLine(lines[0]);
const col = Object.fromEntries(header.map((h, i) => [h.replace(/^"|"$/g, ""), i]));
const rows = [];
for (let i = 1; i < lines.length; i++) {
  if (!lines[i].trim()) continue;
  const f = parseCsvLine(lines[i]);
  const ty = TYPE_CODE[f[col.type]];
  if (!ty) continue; // closed / balloonport
  const la = parseFloat(f[col.latitude_deg]);
  const lo = parseFloat(f[col.longitude_deg]);
  if (!Number.isFinite(la) || !Number.isFinite(lo)) continue;
  const elFt = parseFloat(f[col.elevation_ft]);
  const ident = f[col.ident];
  if (!ident) continue;
  rows.push({
    id: ident,
    n: f[col.name] || ident,
    la: +la.toFixed(4), lo: +lo.toFixed(4),
    el: Number.isFinite(elFt) ? Math.round(elFt * 0.3048) : null,
    ty,
  });
}
const out = {
  source: "OurAirports (ourairports.com/data, public-domain dedication)",
  built: new Date().toISOString().slice(0, 10),
  count: rows.length,
  airports: rows,
};
const dest = path.join(process.cwd(), "datacore", "aircraft", "airports_min.json");
mkdirSync(path.dirname(dest), { recursive: true });
writeFileSync(dest, JSON.stringify(out));
console.log(`wrote ${dest}: ${rows.length} airports, ${(JSON.stringify(out).length / 1e6).toFixed(1)}MB`);
const byTy = {};
for (const r of rows) byTy[r.ty] = (byTy[r.ty] || 0) + 1;
console.log("by type:", byTy);
