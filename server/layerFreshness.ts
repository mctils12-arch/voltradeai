/**
 * layerFreshness.ts — DATACORE MAXIMUS Phase 5 remaining item ("per-layer
 * freshness chips", wishlist.md). Joins the already-computed streams
 * inventory health (server/streamsInventory.ts: health/age_hours derived
 * from newest-archived-file age vs. a cadence heuristic) onto the /data
 * map's layer registry response, so the layer panel can show HOW FRESH
 * each layer's underlying data is — the PREMIUM EXPERIENCE STANDARD's
 * honesty-machinery clause ("every number visibly carries freshness,
 * provenance, and confidence") applied to the layer panel specifically.
 *
 * LAYER_TO_STREAM is a hand-verified mapping, NOT a guess or a naming
 * convention — each entry was traced layer id -> client fetch -> server
 * route -> the exact archive*() call -> the manifest whose `written_by`
 * names that call (see research/experiments.md for the full per-row
 * trace). A layer with no entry here, or whose stream isn't in the
 * inventory cache yet, gets NO `freshness` field — never a fabricated
 * one. Deliberately partial: several routes read STATIC curated
 * reference data (superfund, pfas, nuclear-incident registries, GEM
 * coal/methane — both trace to the same "gem" manifest, ambiguous
 * between the two layers, so neither is mapped) or DERIVED joins over
 * another stream's archive (portdwell/shadowstats over vessels) — none
 * of those have their own honest single-stream freshness reading.
 */
import type { StreamInventoryEntry } from "./streamsInventory";

export const LAYER_TO_STREAM: Readonly<Record<string, string>> = Object.freeze({
  aircraft: "aircraft",
  vessels: "vessels",
  trains: "trains",
  insider: "filings",
  earnings: "earnings8k",
  shortvol: "finrashortvol",
  attention: "wikiattention",
  // deliberately NOT the "cot" manifest (that one is cftc_cot.py, the
  // Python bot-side pipeline, unreachable from this Express route) — the
  // /api/data/cot route reads server/cftcCot.ts, which writes "cftccot".
  cot: "cftccot",
  fires: "fires",
  earthquakes: "earthquakes",
  buoys: "buoys",
  rivergauges: "usgswater",
  alerts: "nwsalerts",
  // spaceweather polls two streams (swpckindex 3-hourly, swpcxray ~1-min);
  // this join can only carry one — swpcxray is the faster-cadence, more
  // failure-sensitive of the pair, so a stall shows up here soonest.
  spaceweather: "swpcxray",
  plant_operations: "epacamd",
  faa_airports: "faastatus",
  border_waits: "cbpborderwait",
});

export interface LayerFreshness {
  stream: string;
  health: StreamInventoryEntry["health"];
  age_hours: number | null;
  health_note: string;
}

/**
 * Pure join: never mutates `layers`, never fabricates a freshness value
 * for a layer this session can't honestly source one for.
 */
export function attachLayerFreshness<T extends { id: string }>(
  layers: T[],
  streams: readonly Pick<StreamInventoryEntry, "stream" | "health" | "age_hours" | "health_note">[],
): (T & { freshness?: LayerFreshness })[] {
  const byStream = new Map(streams.map((s) => [s.stream, s]));
  return layers.map((l) => {
    const streamName = LAYER_TO_STREAM[l.id];
    if (!streamName) return l;
    const s = byStream.get(streamName);
    if (!s) return l;
    return {
      ...l,
      freshness: {
        stream: s.stream,
        health: s.health,
        age_hours: s.age_hours,
        health_note: s.health_note,
      },
    };
  });
}
