// FLIGHT PROCEDURES — "say they filed the ILS: pull up that plate and show it
// on the map with the path on the plate" (human request 2026-09-30), built
// around how IFR actually works:
//   * the filed route names the departure procedure (DP/SID) and the arrival
//     (STAR) — those are listed as FILED and the STAR (else the DP) is
//     auto-selected when the row is switched on;
//   * approaches are ASSIGNED by ATC, never filed — they are listed as
//     SUGGESTED (destination runways vs the latest METAR wind) and the user
//     picks one.
// Selecting a procedure draws its FAA CIFP path (lib/air/procedureLayer.ts)
// and, when the server georeferenced the chart's plan view, lays the plate
// under it with an opacity slider. A chart that did not georeference opens
// in the inline viewer here, with the reason — never force-fitted onto the map.
//
// Zero cost when off (Law IV): no fetch, no layer, no pdf.js. Every request
// is abortable and aborted on deselect / re-select. The CIFP cycle (the
// procedure data's age) and NOT FOR NAVIGATION are always visible.

import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { fmtKm, subscribeUnits } from "@/lib/units";
import { HEX_RE } from "@/lib/air/flightPlan";
import type { PlanRouteStore } from "@/lib/air/planRouteController";
import { frameCore } from "@/render/frameCore";
import { registerIcons, iconDataURL } from "@/lib/mapIcons";
import {
  cycleBadge, fetchFlightProcedures, fetchPlateGeoref, fetchProcedurePath, filedLabel, flightProceduresUrl,
  plateCornersUsable, plateGeorefUrl, procedurePathUrl, windText,
  type ChartRef, type FlightProcedures as FlightProceduresData, type PlateGeoref,
} from "@/lib/air/procedures";
import { PLATE_MAX_PX, ProcedureLayer, type ProcMapLike } from "@/lib/air/procedureLayer";
import { canvasToDecodedUrl, renderPlate } from "@/lib/air/pdfPlate";

const NM_TO_KM = 1.852;
const isPhone = () => typeof window !== "undefined" && !!window.matchMedia?.("(max-width: 639px)").matches;
const cssVar = (name: string) => {
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || "silver";
  } catch (e: unknown) {
    void e;
    return "silver";
  }
};
let errCount = 0;
const report = (where: string, e: unknown) => {
  if ((e as { name?: string })?.name === "AbortError") return;
  errCount++;
  if (errCount <= 5) console.warn(`[procedures] ${where}:`, e instanceof Error ? e.message : e);
};

export interface UseFlightProceduresOpts {
  mapRef: { current: unknown };
  mapReady: boolean;
  hex: string | null | undefined;
  /** the selected aircraft's broadcast callsign */
  getCallsign: () => string | null;
  /** the planned-route store (origin/destination airports) */
  planStore: PlanRouteStore | null;
  suppressed?: boolean;
}

interface Selection {
  airport: string;
  id: string;
  name: string;
  kind: "SID" | "STAR" | "IAP";
  transition: string | null;
  chart: ChartRef | null;
  origin: "filed" | "suggested";
}

type PlateState =
  | { s: "none" }
  | { s: "loading" }
  | { s: "overlay"; g: PlateGeoref }
  | { s: "viewer"; g: PlateGeoref | null; reason: string }
  | { s: "error"; msg: string };

export function useFlightProcedures(opts: UseFlightProceduresOpts): { row: JSX.Element | null } {
  const hex = opts.hex && HEX_RE.test(opts.hex) ? opts.hex.toLowerCase() : null;
  const [on, setOn] = useState(false);
  useEffect(() => { setOn(false); }, [hex]); // per selection, default off (compact card)
  const optsRef = useRef(opts);
  optsRef.current = opts;

  const planStore = opts.planStore;
  const plan = useSyncExternalStore(
    planStore ? planStore.subscribe : noopSubscribe,
    planStore ? () => planStore.get().plan : nullGet,
    planStore ? () => planStore.get().plan : nullGet,
  );
  const dep = plan?.origin?.icao ?? null;
  const arr = plan?.destination?.icao ?? null;
  const callsign = on && hex ? optsRef.current.getCallsign() : null;
  const url = callsign ? flightProceduresUrl(callsign, dep, arr) : null;

  const [data, setData] = useState<FlightProceduresData | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [sel, setSel] = useState<Selection | null>(null);
  const [plate, setPlate] = useState<PlateState>({ s: "none" });
  const [opacity, setOpacity] = useState(0.7);
  const [viewerUrl, setViewerUrl] = useState<string | null>(null);
  const layerRef = useRef<ProcedureLayer | null>(null);
  const viewerRef = useRef<HTMLDivElement | null>(null);
  const [, setTick] = useState(0);
  useEffect(() => subscribeUnits(() => setTick((t) => t + 1)), []);

  const active = !!hex && on && !opts.suppressed && opts.mapReady;

  // the layer lives exactly as long as the row is on for this aircraft
  useEffect(() => {
    if (!active) return;
    const map = optsRef.current.mapRef.current as (ProcMapLike & { hasImage?: (id: string) => boolean }) | null;
    if (!map) return;
    const layer = new ProcedureLayer(map, {
      loop: frameCore(), color: cssVar, onError: report,
      ensureIcons: () => registerIcons(map),
    });
    layerRef.current = layer;
    return () => { layer.dispose(); layerRef.current = null; setSel(null); setPlate({ s: "none" }); setData(null); };
  }, [active, hex]);

  // procedures for this flight (refetched when the plan's airports arrive)
  useEffect(() => {
    if (!active || !url) return;
    const ac = new AbortController();
    setLoadErr(null);
    fetchFlightProcedures(url, ac.signal).then((d) => {
      if (ac.signal.aborted) return;
      setData(d);
      // auto-select the FILED arrival (else departure) procedure
      setSel((cur) => {
        if (cur) return cur;
        const f = d.filed.star ?? d.filed.dp;
        return f ? { airport: f.airport, id: f.id, name: f.name, kind: f.kind, transition: f.transition, chart: f.charts[0] ?? null, origin: "filed" } : null;
      });
    }, (e: unknown) => { if (!ac.signal.aborted) { report("list", e); setLoadErr(e instanceof Error ? e.message : String(e)); } });
    return () => ac.abort();
  }, [active, url]);

  // selected procedure -> path on the map, then the plate
  useEffect(() => {
    const layer = layerRef.current;
    if (!active || !layer) return;
    if (!sel) { layer.setPath(null); layer.setPlate(null); setPlate({ s: "none" }); return; }
    const ac = new AbortController();
    fetchProcedurePath(procedurePathUrl(sel.airport, sel.id, sel.transition), ac.signal)
      .then((p) => { if (!ac.signal.aborted) layer.setPath(p); }, (e: unknown) => report("path", e));
    layer.setPlate(null);
    if (!sel.chart) { setPlate({ s: "viewer", g: null, reason: "no FAA chart matched this procedure in the current d-TPP cycle" }); return () => ac.abort(); }
    setPlate({ s: "loading" });
    const chart = sel.chart;
    fetchPlateGeoref(plateGeorefUrl(chart, sel.id), ac.signal).then(async (g) => {
      if (ac.signal.aborted) return;
      if (!plateCornersUsable(g)) { setPlate({ s: "viewer", g, reason: g.reason }); return; }
      const canvas = await renderPlate(chart.url, { crop: g.planView, maxPx: PLATE_MAX_PX, signal: ac.signal });
      const decoded = await canvasToDecodedUrl(canvas, ac.signal);
      if (ac.signal.aborted || layer.isDisposed()) { URL.revokeObjectURL(decoded); return; }
      layer.setPlate({ url: decoded, corners: g.corners });
      setPlate({ s: "overlay", g });
    }).catch((e: unknown) => {
      if (ac.signal.aborted) return;
      report("plate", e);
      setPlate({ s: "error", msg: e instanceof Error ? e.message : String(e) });
    });
    return () => ac.abort();
  }, [active, sel]);

  useEffect(() => { layerRef.current?.setOpacity(opacity); }, [opacity]);

  // inline viewer (charts that are not georeferenced, or on request)
  useEffect(() => {
    if (!viewerUrl) return;
    const ac = new AbortController();
    const host = viewerRef.current;
    renderPlate(viewerUrl, { maxPx: 1400, signal: ac.signal }).then((c) => {
      if (ac.signal.aborted || !host) return;
      c.style.cssText = "width:100%;height:auto;display:block;border-radius:6px;background:white;";
      host.replaceChildren(c);
    }, (e: unknown) => { if (!ac.signal.aborted) { report("viewer", e); if (host) host.textContent = "plate could not be rendered — open the PDF instead"; } });
    return () => { ac.abort(); if (host) host.replaceChildren(); };
  }, [viewerUrl]);
  useEffect(() => { setViewerUrl(null); }, [sel, on, hex]);

  if (!hex) return { row: null };

  const btnH = isPhone() ? 44 : 30;
  const line = { whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" } as const;
  const itemStyle = (selected: boolean) => ({
    display: "block", width: "100%", textAlign: "left" as const, minHeight: btnH, padding: "4px 8px", margin: "2px 0",
    borderRadius: 6, cursor: "pointer", font: "inherit", color: "var(--flight-ink)",
    background: selected ? "var(--flight-accent-soft)" : "transparent",
    border: `1px solid ${selected ? "var(--flight-accent)" : "var(--flight-panel-border)"}`,
  });
  const pick = (s: Selection) => setSel((cur) => (cur && cur.id === s.id && cur.airport === s.airport && cur.transition === s.transition ? null : s));
  const isSel = (airport: string, id: string) => !!sel && sel.airport === airport && sel.id === id;

  const body = on && !opts.suppressed ? (
    <div data-vt-procedures-panel className="om-sb"
         style={{ marginTop: 6, maxHeight: "38vh", overflowY: "auto", fontFamily: "var(--font-mono)", fontSize: 10.5, lineHeight: 1.45, color: "var(--flight-ink-dim)" }}>
      <div style={line} title="FAA CIFP / d-TPP data cycle — the procedure data's age">
        {cycleBadge(data?.cycle)}{data?.dtppCycle ? ` · d-TPP ${data.dtppCycle.ident}` : ""} · <span style={{ color: "var(--accent-orange)", fontWeight: 700 }}>NOT FOR NAVIGATION</span>
      </div>
      {!data && !loadErr && <div aria-live="polite">loading procedures…</div>}
      {loadErr && <div style={{ color: "var(--accent-orange)" }}>procedures unavailable: {loadErr}</div>}
      {data && (
        <>
          <div style={{ marginTop: 6, fontWeight: 700, letterSpacing: ".06em", color: "var(--flight-ink)" }}>FILED</div>
          {data.filed.dp && (
            <button style={itemStyle(isSel(data.filed.dp.airport, data.filed.dp.id))} data-vt-proc-item
                    onClick={() => pick({ airport: data.filed.dp!.airport, id: data.filed.dp!.id, name: data.filed.dp!.name, kind: "SID", transition: data.filed.dp!.transition, chart: data.filed.dp!.charts[0] ?? null, origin: "filed" })}>
              <div style={line}>{filedLabel("DP", data.filed.dp)} · {data.filed.dp.airport}</div>
            </button>
          )}
          {data.filed.star && (
            <button style={itemStyle(isSel(data.filed.star.airport, data.filed.star.id))} data-vt-proc-item
                    onClick={() => pick({ airport: data.filed.star!.airport, id: data.filed.star!.id, name: data.filed.star!.name, kind: "STAR", transition: data.filed.star!.transition, chart: data.filed.star!.charts[0] ?? null, origin: "filed" })}>
              <div style={line}>{filedLabel("STAR", data.filed.star)} · {data.filed.star.airport}</div>
            </button>
          )}
          {!data.filed.dp && !data.filed.star && <div>{data.filed.note}</div>}
          {(data.filed.dp?.versionMismatch || data.filed.star?.versionMismatch) && (
            <div style={{ color: "var(--accent-orange)" }}>filed version differs from the current cycle — the current procedure is drawn</div>
          )}
          <div style={{ marginTop: 6, fontWeight: 700, letterSpacing: ".06em", color: "var(--flight-ink)" }} title={data.approachHonesty}>
            APPROACHES · suggested — ATC assigns the approach
          </div>
          {data.wind && <div style={line}>{data.arrival?.icao} {windText(data.wind)}{data.wind.obsTime ? ` · obs ${data.wind.obsTime.slice(11, 16)}Z` : ""}</div>}
          {data.suggestedApproaches.length === 0 && <div>{data.arrival ? "no approach suggestion (no into-wind runway with a coded approach)" : "destination unknown — no approaches to suggest"}</div>}
          {data.suggestedApproaches.map((a) => (
            <button key={a.id} style={itemStyle(isSel(data.arrival?.icao ?? "", a.id))} data-vt-proc-item title={a.reason}
                    onClick={() => data.arrival && pick({ airport: data.arrival.icao, id: a.id, name: a.name, kind: "IAP", transition: null, chart: a.charts[0] ?? null, origin: "suggested" })}>
              <div style={line}>{a.name}</div>
              <div style={{ ...line, color: "var(--flight-ink-dim)" }}>{a.reason}</div>
            </button>
          ))}
          {sel && (
            <div data-vt-proc-selected style={{ marginTop: 6, paddingTop: 6, borderTop: "1px solid var(--flight-panel-border)" }}>
              <div style={{ ...line, color: "var(--flight-ink)", fontWeight: 600 }}>
                {sel.name}{sel.transition ? ` · ${sel.transition}` : ""} · {sel.origin === "filed" ? "FILED" : "SUGGESTED"}
              </div>
              <ProcLegend />
              {plate.s === "loading" && <div aria-live="polite">placing plate…</div>}
              {plate.s === "overlay" && (
                <>
                  <div style={line} title={plate.g.reason}>
                    plate on map · fit RMS {fmtKm((plate.g.rmsNm ?? 0) * NM_TO_KM, 2)} over {plate.g.controlPoints.length} fixes
                    {plate.g.embeddedAgreementNm != null ? ` · GeoPDF Δ ${fmtKm(plate.g.embeddedAgreementNm * NM_TO_KM, 2)}` : ""}
                  </div>
                  <label style={{ display: "flex", alignItems: "center", gap: 8, minHeight: btnH }}>
                    <span>plate opacity</span>
                    <input type="range" min={0} max={100} value={Math.round(opacity * 100)} aria-label="plate opacity"
                           onChange={(e) => setOpacity(Number(e.target.value) / 100)} style={{ flex: "1 1 auto" }} />
                  </label>
                </>
              )}
              {plate.s === "viewer" && <div>plate not placed on the map: {plate.reason}</div>}
              {plate.s === "error" && <div style={{ color: "var(--accent-orange)" }}>plate unavailable: {plate.msg}</div>}
              {sel.chart && (
                <div style={{ display: "flex", gap: 8, marginTop: 4, flexWrap: "wrap" }}>
                  <button className="vt-flight-follow" style={{ margin: 0, minHeight: btnH, padding: "0 10px" }}
                          onClick={() => setViewerUrl((v) => (v ? null : sel.chart!.url))}>
                    {viewerUrl ? "Hide plate" : "View plate"}
                  </button>
                  <a className="vt-flight-follow" style={{ margin: 0, minHeight: btnH, padding: "0 10px", display: "inline-flex", alignItems: "center" }}
                     href={sel.chart.url} target="_blank" rel="noopener noreferrer">Open PDF</a>
                </div>
              )}
              {viewerUrl && <div ref={viewerRef} data-vt-plate-viewer style={{ marginTop: 6 }} aria-live="polite">rendering plate…</div>}
            </div>
          )}
        </>
      )}
    </div>
  ) : null;

  const row = (
    <div data-testid="flight-procedures-row" style={{ padding: "8px 14px 0", flex: "0 0 auto" }}>
      <button
        className={`vt-flight-follow${on ? " on" : ""}`}
        aria-pressed={on}
        aria-expanded={on}
        data-vt-procedures-toggle
        style={{ margin: 0, padding: "0 12px", minHeight: btnH }}
        title="Instrument procedures — the filed DP/STAR and suggested approaches (ATC assigns approaches), drawn from FAA CIFP with the FAA plate. Not for navigation."
        onClick={() => setOn((v) => !v)}
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden>
          <path d="M3 20 L10 13 L14 15 L21 4" /><path d="M17 4h4v4" />
        </svg>
        {on ? "Procedures" : "Procedures · off"}
      </button>
      {body}
    </div>
  );
  return { row };
}

/** Legend for the drawn procedure, from the SAME icon registry the map uses. */
function ProcLegend() {
  const accent = cssVar("--accent-purple");
  const muted = cssVar("--text-secondary");
  const item = (icon: string, color: string, label: string) => (
    <span key={label} style={{ display: "inline-flex", alignItems: "center", gap: 4, marginRight: 10 }}>
      <img src={iconDataURL(icon, color, 12)} width={12} height={12} alt="" />{label}
    </span>
  );
  return (
    <div style={{ display: "flex", flexWrap: "wrap", marginTop: 2 }} data-vt-proc-legend>
      {item("vt-fix-wpt", accent, "fix")}
      {item("vt-fix-nav", accent, "navaid")}
      {item("vt-fix-faf", accent, "FAF")}
      {item("vt-fix-wpt", muted, "missed appr.")}
      <span style={{ marginRight: 10 }}>– – approximated leg</span>
    </div>
  );
}

const noopSubscribe = () => () => {};
const nullGet = () => null;
