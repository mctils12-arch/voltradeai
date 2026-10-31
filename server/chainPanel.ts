/**
 * chainPanel.ts — the FIXED options-chain panel for Edge P3 (Cremers-
 * Weinbaum IV spread; research/edge_preregistration.md "Universe").
 *
 * WHY: the daily chain archive rotates with the CSP candidate list (1,717
 * names in 76 days), so a per-name time series of VS does not exist. P3's
 * gate-2 test needs the SAME names every day. The panel is frozen ONCE
 * (write-once file, never re-ranked — re-ranking would be a lookahead /
 * survivorship leak) from the CSP universe cache's dollar-volume ranking
 * and then archived first, every day, ahead of the rotating candidates.
 *
 * HONEST LIMIT: the pre-registration names "60-day median option-chain
 * quote count" as the liquidity rank; no such history exists at formation,
 * so equity dollar volume of the optionable CSP universe is the stand-in.
 * The file records `method` so the substitution is auditable. Not read by
 * any trading path.
 */
import fs from "fs";
import path from "path";
import { archiveBaseDir } from "./datacoreArchive";

export const PANEL_SIZE = 300;
/** Freeze only from a cache big enough to rank — a thin cache would freeze a
 *  short panel for good. */
export const PANEL_MIN_CANDIDATES = 250;

export interface ChainPanel { formed: string; method: string; size: number; names: string[] }

export function panelPath(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "optionchains", "panel", "p3_panel.json");
}

/** Top-N by dollar volume (tuple index 3), ties by symbol; plain tickers only. */
export function rankPanel(cacheJson: any, n = PANEL_SIZE): string[] {
  const cands: any[] = Array.isArray(cacheJson?.candidates) ? cacheJson.candidates : [];
  const rows = cands
    .map((c) => ({ s: String(c?.[0] || "").toUpperCase(), dv: Number(c?.[3]) }))
    .filter((r) => /^[A-Z]{1,5}$/.test(r.s) && Number.isFinite(r.dv) && r.dv > 0);
  const seen = new Set<string>();
  const uniq = rows.filter((r) => (seen.has(r.s) ? false : (seen.add(r.s), true)));
  uniq.sort((a, b) => b.dv - a.dv || (a.s < b.s ? -1 : 1));
  return uniq.slice(0, n).map((r) => r.s);
}

/** The frozen panel, or null before formation. Corrupt file → null (never
 *  silently re-freezes over it: the write below is exclusive). */
export function loadPanel(baseDir?: string): ChainPanel | null {
  try {
    const j = JSON.parse(fs.readFileSync(panelPath(baseDir), "utf8"));
    return Array.isArray(j?.names) && j.names.length ? (j as ChainPanel) : null;
  } catch { return null; }
}

/** Load the panel; if none exists and the cache can support one, freeze it
 *  (exclusive create, so two racing writers cannot disagree). */
export function ensurePanel(cacheJson: any, day: string, baseDir?: string): ChainPanel | null {
  const have = loadPanel(baseDir);
  if (have) return have;
  const names = rankPanel(cacheJson);
  const cands = Array.isArray(cacheJson?.candidates) ? cacheJson.candidates.length : 0;
  if (cands < PANEL_MIN_CANDIDATES || names.length < PANEL_MIN_CANDIDATES) return null;
  const panel: ChainPanel = {
    formed: day, size: names.length, names,
    method: "top dollar-volume of the CSP universe cache at formation (stand-in for 60d median option quote count)",
  };
  const fp = panelPath(baseDir);
  fs.mkdirSync(path.dirname(fp), { recursive: true });
  try { fs.writeFileSync(fp, JSON.stringify(panel), { flag: "wx" }); } catch { return loadPanel(baseDir); }
  return panel;
}
