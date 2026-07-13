// Pure formatter for the port-dwell click card (hermetic — no DOM/map; unit
// tested with `npx tsx --test`). Input is the clicked feature's properties as
// maplibre delivers them (geojson properties: numbers may arrive as numbers,
// nested arrays arrive JSON-stringified). The caveat is the SERVER's own
// honesty text (portDwell.ts), passed through verbatim — never rewritten here.

export interface PortAnomalyExample {
  mmsi: string;
  name?: string;
  dwell_h: number;
  median_h: number;
}

export interface PortCardText {
  title: string;
  subtitle: string;
  body: string;
}

export function formatPortDetail(p: Record<string, any>): PortCardText {
  const days = p.window_hours != null ? Math.round(Number(p.window_hours) / 24) : null;
  let anomalies: PortAnomalyExample[] = [];
  try { anomalies = JSON.parse(p.anomaly_examples || "[]"); } catch { /* malformed → show count only */ }
  const anomalyCount = Number(p.anomaly_count) || 0;
  return {
    title: p.name || "Port",
    subtitle: `${p.in_port_now ?? 0} vessels in port now` + (days ? ` · ${days}d window` : ""),
    body: [
      `Completed calls: ${p.visits_completed ?? 0}` +
        (p.unique_vessels != null ? ` (${p.unique_vessels} unique vessels)` : ""),
      p.dwell_median_h != null
        ? `Dwell: median ${p.dwell_median_h}h` +
          (p.dwell_p90_h != null ? ` · p90 ${p.dwell_p90_h}h` : "") +
          (p.dwell_max_h != null ? ` · max ${p.dwell_max_h}h` : "")
        : "Dwell: not enough completed calls yet for a median",
      anomalyCount > 0
        ? `${anomalyCount} anomaly flag${anomalyCount === 1 ? "" : "s"} (dwell ≥ 3× this port's own median):\n` +
          anomalies.slice(0, 2).map((a) =>
            `  ${a.name || `MMSI ${a.mmsi}`}: ${a.dwell_h}h vs ${a.median_h}h median`).join("\n")
        : "No dwell anomalies flagged in the window",
      p.caveat ? `\n${p.caveat}` : null,
    ].filter(Boolean).join("\n"),
  };
}
