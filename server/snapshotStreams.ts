/**
 * snapshotStreams.ts — build stamps of the git-versioned, bundled datasets
 * (streams whose archive is a whole-file JSON snapshot in datacore/, NOT a
 * volume JSONL dir). The /data Streams inventory scans only the volume, so
 * these streams read "no-data — key-gated, warming up" while their data is
 * in fact being served (2026-10-12 audit). The stamp is the dataset's own
 * embedded build time — honest age, not a deploy mtime. Statically imported
 * (esbuild inlines; the same JSON the serving modules already bundle).
 */
import eiaWeekly from "../datacore/eia/weekly_series.json";
import cpcDegreeDays from "../datacore/cpc/degree_days.json";
import jodiStocks from "../datacore/jodi/primary_stocks.json";
import railCarloads from "../datacore/rail/ep724_carloads.json";
import unComtrade from "../datacore/un_comtrade/bilateral_trade.json";
import aircraftSpine from "../datacore/aircraft/entity_spine.json";

export interface SnapshotStamp {
  artifact: string; // repo-relative path shown as the "newest file"
  built: string | null; // ISO build time embedded in the dataset
}

function stamp(artifact: string, doc: any): SnapshotStamp {
  const b = doc?.built ?? doc?.built_at;
  return { artifact, built: typeof b === "string" && !Number.isNaN(Date.parse(b)) ? b : null };
}

export const SNAPSHOT_STREAMS: Record<string, SnapshotStamp> = {
  eiaweekly: stamp("datacore/eia/weekly_series.json", eiaWeekly),
  cpcdegreedays: stamp("datacore/cpc/degree_days.json", cpcDegreeDays),
  jodi: stamp("datacore/jodi/primary_stocks.json", jodiStocks),
  railep724: stamp("datacore/rail/ep724_carloads.json", railCarloads),
  uncomtrade: stamp("datacore/un_comtrade/bilateral_trade.json", unComtrade),
  aircraftspine: stamp("datacore/aircraft/entity_spine.json", aircraftSpine),
};
