// MethaneHotspotsView — repeat satellite methane-plume detections grouped
// by nearest catalogued GEM asset (#/data/methane-hotspots). Gate-2(b) of
// the GEM METHANE-PLUME × EXTRACTION-REGISTRY PROXIMITY hypothesis
// (research/open_questions.md, server/gemMethaneProximity.ts's
// computeAssetPlumeStats): a repeat-detection count/rate per asset.
// STILL NOT A SIGNAL — no base rate or disclosed-intensity comparison
// exists yet (gate 2(c)/(d)); this is a descriptive stat only, ranked, not
// a ranked BUY/AVOID list. Reuses the generic .vt-filings-*/.vt-shortvol-*
// CSS — no new styles needed.
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, Cloud, ExternalLink } from "lucide-react";

interface AssetStat {
  assetId: string;
  kind: "oil_gas_extraction" | "coal_mine";
  name: string | null;
  operator: string | null;
  owner: string | null;
  parent: string | null;
  detectionCount: number;
  firstObservedAt: string | null;
  lastObservedAt: string | null;
  spanDays: number | null;
  ratePerYear: number | null;
  meanEmissionsKgHr: number | null;
}

type SortKey = "detectionCount" | "ratePerYear" | "meanEmissionsKgHr";
const SORT_OPTIONS: Array<{ key: SortKey; label: string }> = [
  { key: "detectionCount", label: "Most detections" },
  { key: "ratePerYear", label: "Highest repeat rate" },
  { key: "meanEmissionsKgHr", label: "Highest mean emissions" },
];

const KIND_LABEL: Record<AssetStat["kind"], string> = { oil_gas_extraction: "Oil/gas extraction", coal_mine: "Coal mine" };
const day = (iso: string | null) => (iso ? iso.slice(0, 10) : "—");
const rate = (n: number | null) => (n == null ? "—" : `${n < 10 ? n.toFixed(2) : n.toFixed(1)}/yr`);
const kgHr = (n: number | null) => (n == null ? "—" : `${n.toLocaleString(undefined, { maximumFractionDigits: 1 })} kg/hr`);

export default function MethaneHotspotsView({ onBack }: { onBack: () => void }) {
  const [stats, setStats] = useState<AssetStat[] | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey>("detectionCount");

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/methane-plumes/asset-stats");
        const d = await r.json();
        if (stop) return;
        setStats(Array.isArray(d.assetStats) ? d.assetStats : []);
        setNote(d.note ?? null);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const ranked = useMemo(() => {
    if (!stats) return [];
    // nulls always sort last regardless of direction — an asset with no
    // computable rate/emissions isn't "worse," it's unmeasured.
    return [...stats].sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey];
      if (av == null && bv == null) return b.detectionCount - a.detectionCount;
      if (av == null) return 1;
      if (bv == null) return -1;
      return bv - av;
    });
  }, [stats, sortKey]);

  return (
    <div className="vt-filings-page" role="region" aria-label="Methane plume repeat-detection hotspots">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Cloud size={16} />
        <div>
          <div className="vt-filings-title">Methane hotspots — repeat detections by asset</div>
          <div className="vt-filings-sub">
            satellite plume detections grouped by nearest catalogued asset (STILL NOT A SIGNAL — a descriptive count/rate, no base-rate or disclosure comparison yet) ·{" "}
            {stats ? `${stats.length} asset${stats.length === 1 ? "" : "s"} with a repeat match` : "loading…"} ·{" "}
            <a href="https://globalenergymonitor.org/projects/global-oil-gas-plant-tracker/" target="_blank" rel="noreferrer">Global Energy Monitor <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}

      {!error && (
        <div className="vt-shortvol-body">
          {note && <div className="vt-filings-sub">{note}</div>}

          <div className="vt-filings-filters" role="group" aria-label="Sort assets by">
            {SORT_OPTIONS.map((o) => (
              <button
                key={o.key}
                type="button"
                className="vt-filings-filter"
                aria-pressed={sortKey === o.key}
                style={sortKey === o.key ? { borderColor: "var(--accent-bright)", color: "var(--accent-bright)" } : undefined}
                onClick={() => setSortKey(o.key)}
              >
                {o.label}
              </button>
            ))}
          </div>

          {!stats && <div className="vt-filings-state">Loading repeat-detection stats…</div>}
          {stats && ranked.length === 0 && <div className="vt-filings-state">No asset has more than one unambiguous nearby detection yet.</div>}
          {stats && ranked.length > 0 && (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr>
                    <th>Asset</th>
                    <th>Kind</th>
                    <th>Operator / owner</th>
                    <th className="num">Detections</th>
                    <th>First seen</th>
                    <th>Last seen</th>
                    <th className="num">Repeat rate</th>
                    <th className="num">Mean emissions</th>
                  </tr>
                </thead>
                <tbody>
                  {ranked.map((s) => (
                    <tr key={s.assetId}>
                      <td data-l="Asset">{s.name || s.assetId}</td>
                      <td data-l="Kind">{KIND_LABEL[s.kind]}</td>
                      <td data-l="Operator / owner">{s.operator || s.owner || s.parent || "—"}</td>
                      <td data-l="Detections" className="num">{s.detectionCount}</td>
                      <td data-l="First seen">{day(s.firstObservedAt)}</td>
                      <td data-l="Last seen">{day(s.lastObservedAt)}</td>
                      <td data-l="Repeat rate" className="num">{rate(s.ratePerYear)}</td>
                      <td data-l="Mean emissions" className="num">{kgHr(s.meanEmissionsKgHr)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
