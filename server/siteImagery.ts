/**
 * siteImagery.ts — attaches the latest-pulled Sentinel-2 chip metadata
 * (scripts/cdse_site_chips.py, DATACORE MAXIMUS Phase 3b) to each
 * strategic-site record served by /api/data/sites. Pure merge, no IO:
 * server/routes.ts imports the committed JSON index statically (esbuild
 * bakes it into dist/index.cjs — the same R14-lesson pattern every other
 * datacore JSON in routes.ts already follows) and passes it in here.
 */

export interface SiteChipInfo {
  file: string;
  scene: string;
  date: string;
  cloud_pct: number | null;
  bbox: [number, number, number, number];
  attribution: string;
  pulled_at: string;
}

/** A site never pulled yet gets no `imagery` field — never a fabricated
 *  placeholder (RAW OVERLAYS honesty rule). */
export function mergeSiteImagery(
  sites: any[],
  chips: Record<string, SiteChipInfo>,
): any[] {
  return sites.map((s) => {
    const chip = chips[s.id];
    return chip ? { ...s, imagery: chip } : s;
  });
}
