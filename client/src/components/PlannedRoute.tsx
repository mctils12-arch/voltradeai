// PLANNED ROUTE — the React surface of the gray planned-route curtain
// (FLIGHT PROGRAM 2026-09-28): one hook the map page calls once, and the
// flight-card row it renders (toggle + provenance).
//
// usePlannedRoute owns the lifecycle and nothing else: while an aircraft is
// selected AND the toggle is on AND no archived-trip replay owns the curtain,
// it runs lib/air/planRouteController (abortable fetch, 60 s refresh, >10 nm
// re-fetch, PlanCurtainLayer, destination label); any change of those
// tears it down completely (Law IV — zero work when off). The card-row state
// arrives through the controller's external store, so a refresh re-renders
// only this row — never the map page.
//
// PROVENANCE ROW (PREMIUM EXPERIENCE STANDARD c — every number carries
// freshness, provenance, confidence): FILED vs PREDICTED badge, origin →
// destination, the server's own label, deviation state (distances through
// units.ts), whether altitudes are estimated, and the plan data's age. The
// server's one-sentence honesty disclosure rides the row's tooltip.

import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { fmtKm, subscribeUnits } from "@/lib/units";
import {
  HEX_RE,
  deviationText,
  fmtAgeShort,
  planAgeSec,
  planKind,
  planRouteText,
  planVectoringText,
  wasReplanned,
} from "@/lib/air/flightPlan";
import {
  createPlanRouteStore,
  startPlanRoute,
  type PlanLive,
  type PlanMapLike,
  type PlanRouteStore,
} from "@/lib/air/planRouteController";
import type { PlanSeam } from "@/lib/air/planCurtainLayer";
import { groundElevationSync } from "@/lib/elevation";

export interface UsePlannedRouteOpts {
  mapRef: { current: unknown };
  mapReady: boolean;
  /** the selected aircraft's ICAO24 hex (anything else = not selected). */
  hex: string | null | undefined;
  /** true while an archived-trip replay owns the curtain — the CURRENT
   *  plan next to a past trip would mislead, so it is hidden. */
  suppressed?: boolean;
  /** the plane's latest real fix (query params, DEM radius centre). */
  getLive: () => PlanLive | null;
  /** where the live curtain currently ends — the seam. */
  getSeam: () => PlanSeam | null;
  /** datamap's context-restore registry (re-added after a GL restore). */
  registry?: Map<string, unknown> | null;
}

export function usePlannedRoute(opts: UsePlannedRouteOpts): { row: JSX.Element | null } {
  const hex = opts.hex && HEX_RE.test(opts.hex) ? opts.hex.toLowerCase() : null;
  // default ON per selection: the user's OFF applies to that plane only
  const [offFor, setOffFor] = useState<string | null>(null);
  useEffect(() => { setOffFor(null); }, [hex]);
  const on = !!hex && offFor !== hex;

  const storeRef = useRef<PlanRouteStore | null>(null);
  if (!storeRef.current) storeRef.current = createPlanRouteStore();
  const store = storeRef.current;
  // latest-closure ref: the controller reads the page's refs through these
  // without the effect re-running every render
  const optsRef = useRef(opts);
  optsRef.current = opts;

  const active = !!hex && on && !opts.suppressed && opts.mapReady;
  useEffect(() => {
    if (!active || !hex) return;
    const map = optsRef.current.mapRef.current as PlanMapLike | null;
    if (!map) return;
    const h = startPlanRoute({
      map,
      hex,
      store,
      getLive: () => optsRef.current.getLive(),
      getSeam: () => optsRef.current.getSeam(),
      doc: typeof document !== "undefined" ? document : null,
      registry: optsRef.current.registry ?? null,
      demGround: groundElevationSync,
    });
    return () => h.stop();
  }, [active, hex, store]);

  const row = hex ? (
    <PlannedRouteRow
      store={store}
      on={on}
      suppressed={!!opts.suppressed}
      onToggle={() => setOffFor(on ? hex : null)}
    />
  ) : null;
  return { row };
}

const isPhone = () => typeof window !== "undefined" && !!window.matchMedia?.("(max-width: 639px)").matches;

function PlannedRouteRow({ store, on, suppressed, onToggle }: {
  store: PlanRouteStore;
  on: boolean;
  suppressed: boolean;
  onToggle: () => void;
}) {
  const st = useSyncExternalStore(store.subscribe, store.get, store.get);
  const [, setTick] = useState(0);
  useEffect(() => subscribeUnits(() => setTick((t) => t + 1)), []);
  // age ticker — 1 Hz only while a plan is on screen (zero cost otherwise)
  useEffect(() => {
    if (!on || st.status !== "ok") return;
    const iv = window.setInterval(() => setTick((t) => t + 1), 1000);
    return () => window.clearInterval(iv);
  }, [on, st.status]);

  const plan = st.plan;
  const kind = plan ? planKind(plan.source) : null;
  // COMPACT by design: the card is a 60vh box whose always-visible chrome
  // must not grow enough to squeeze the archived-trips block — at most three
  // one-line readouts beside the toggle; the server's full label and its
  // honesty sentence ride the tooltip (the lines ellipsize, never wrap).
  const line = { whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" } as const;
  let lines: JSX.Element[] = [];
  let tip: string | undefined;
  if (on && suppressed) {
    lines = [<div key="s" style={line}>hidden during archived-trip replay</div>];
  } else if (on && (st.status === "idle" || st.status === "loading")) {
    lines = [<div key="l" style={line} aria-live="polite">loading planned route…</div>];
  } else if (on && st.status === "error") {
    lines = [<div key="e" style={line}>unavailable right now</div>, <div key="e2" style={line}>retrying every 60 s</div>];
  } else if (on && st.status === "none" && plan) {
    tip = [plan.label, plan.honesty].filter(Boolean).join(" — ");
    lines = [
      <div key="n" style={line}>{plan.label || "No flight plan available"}</div>,
      <div key="n2" style={line}>no filed or predicted route · nothing drawn</div>,
    ];
  } else if (on && st.status === "ok" && plan) {
    const route = planRouteText(plan);
    const age = planAgeSec(plan, st.receivedAtMs, Date.now());
    const estAlt = plan.cruiseAltEstimated || plan.points.some((p) => p.altEstimated);
    const off = plan.deviation.state === "OFF_PLAN";
    // terminal-area vectoring: the connector from the plane is ATC vectors,
    // not the plan — said on the row, not only drawn dotted
    const vectors = planVectoringText(plan);
    tip = [plan.label, plan.honesty, estAlt ? "Dashed top edge = estimated altitude." : "",
      vectors ? "Faint dotted connector = ATC vectors, not the planned route." : ""]
      .filter(Boolean).join(" — ");
    lines = [
      <div key="k" style={line}>
        <span style={{
          fontWeight: 700, letterSpacing: ".06em", marginRight: 6,
          color: kind === "FILED" ? "var(--accent-green)" : "var(--accent-orange)",
        }}>{kind}</span>
        {route && <span style={{ color: "var(--flight-ink)", fontWeight: 600 }}>{route}</span>}
      </div>,
      <div key="d" style={line}>
        <span data-vt-plan-vectoring={vectors ? "" : undefined}
              style={{ color: off || vectors ? "var(--accent-orange)" : "var(--flight-ink)" }}>
          {vectors ?? deviationText(plan.deviation, (km, d) => fmtKm(km, d))}
        </span>
        {wasReplanned(plan) && " · re-planned"}
        {" · "}{fmtAgeShort(age)} old{st.refreshFailed && " · refresh failed"}
      </div>,
      <div key="b" style={line}>
        {estAlt ? "est. altitudes · " : ""}{plan.label}
      </div>,
    ];
  }

  return (
    <div data-testid="planned-route-row"
         style={{ padding: "10px 14px 0", flex: "0 0 auto", display: "flex", gap: 10, alignItems: "flex-start" }}>
      <button
        className={`vt-flight-follow${on ? " on" : ""}`}
        aria-pressed={on}
        data-vt-plan-toggle
        style={{ margin: 0, flex: "0 0 auto", padding: "0 12px", minHeight: isPhone() ? 44 : 32 }}
        title="Planned route — gray 3D curtain from where the aircraft is now to its destination (FILED = FAA flight plan · PREDICTED = estimated route)"
        onClick={onToggle}
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden>
          <path d="M4 18c4-1 5-8 9-9s5 3 7-3" strokeDasharray="3 3" /><circle cx="20" cy="6" r="2" />
        </svg>
        {on ? "Planned route" : "Planned route · off"}
      </button>
      {lines.length > 0 && (
        <div data-vt-plan-provenance title={tip}
             style={{ flex: "1 1 auto", minWidth: 0, fontFamily: "var(--font-mono)", fontSize: 10.5, lineHeight: 1.4, color: "var(--flight-ink-dim)" }}>
          {lines}
        </div>
      )}
    </div>
  );
}
