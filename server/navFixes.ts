// FAA NASR fix + navaid gazetteer: ident -> position. SFDPS routePoints carry
// fix NAMES only (live 2026-09-30 route-shape sample: point[fix=...], no
// coordinates), so placing a filed route needs this table. Source: FAA NASR
// FIX_BASE + NAV_BASE (public domain), built by scripts/build_nasr_fixes.py.
// Unknown idents return null — a fix we cannot resolve is never guessed.
import fs from "fs";
import path from "path";
import { repoDataPath } from "./repoFiles";

let table: Map<string, [number, number]> | null = null;
let cycle: string | null = null;

export function loadNavFixes(jsonPath?: string): number {
  if (table) return table.size;
  const p = jsonPath || repoDataPath(path.join("datacore", "aircraft", "nasr_fixes.json"));
  try {
    const d = JSON.parse(fs.readFileSync(p, "utf-8"));
    table = new Map(Object.entries(d.fixes || {}) as [string, [number, number]][]);
    cycle = d.cycle ?? null;
  } catch (e: any) {
    console.error("[navFixes] load:", e?.message || e);
    table = new Map(); // degrade to no-matches, never throw at call sites
  }
  return table.size;
}

export function lookupFix(ident: string | null | undefined): { lat: number; lon: number } | null {
  if (!ident) return null;
  if (!table) loadNavFixes();
  const v = table!.get(ident.trim().toUpperCase());
  return v ? { lat: v[0], lon: v[1] } : null;
}

export const navFixesCycle = (): string | null => cycle;
/** test hook */
export function resetNavFixes(): void { table = null; cycle = null; }
