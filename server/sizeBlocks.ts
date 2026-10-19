// REPAIR 2026-10-16 (bot cockpit audit S5 gate B13 / §8 V18): Python's
// sizing engine (position_sizing.calculate_position) refuses to size a
// candidate whose bucket has non-positive Kelly expectancy — correct, but
// bot_engine's stock loop discarded the block_reason and the audit log
// showed nothing. With live stock stats at WR 27.8 % the "stocks" bucket
// has been blocked for every candidate for days and the page had no way to
// say so. scan_market now returns `sizing_blocks`; this formats the audit
// rows (capped so a wide scan cannot crowd the 2000-row log).
export interface SizingBlock { ticker?: string; reason?: string; score?: number | null; strategy?: string }

export const SIZE_BLOCK_AUDIT_CAP = 5;

export function formatSizeBlocks(blocks: unknown, cap: number = SIZE_BLOCK_AUDIT_CAP): string[] {
  if (!Array.isArray(blocks) || blocks.length === 0) return [];
  const rows = blocks.filter((b): b is SizingBlock => !!b && typeof b === "object");
  const out = rows.slice(0, cap).map((b) => {
    const score = typeof b.score === "number" && Number.isFinite(b.score) ? ` (score ${b.score.toFixed(1)})` : "";
    const strat = b.strategy ? ` [${b.strategy}]` : "";
    return `${String(b.ticker || "?").toUpperCase()}${strat}: ${String(b.reason || "blocked").slice(0, 200)}${score}`;
  });
  if (rows.length > cap) out.push(`… ${rows.length - cap} more candidates blocked by sizing this scan`);
  return out;
}
