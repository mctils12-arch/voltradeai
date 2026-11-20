// PLAN CHANGES — the flight card's row for how the selected aircraft's FILED
// flight plan changed ([PRODUCT] 2026-11-20, UI coverage batch B): the
// versions the FAA feed sent, each real change in plain words with its time,
// and whether the aircraft left its route. Data: GET
// /api/data/aircraft/plan-history/:hex (lib/air/planHistory.ts, pure).
//
// COMPACT by the card's own rule (components/PlannedRoute.tsx): one line
// beside a toggle; the list opens inside the card with its own scroll, so the
// always-visible chrome grows by one row at most. The row appears ONLY when
// there is something to say (more than one version, or an off-route stretch):
// a plan that never changed, a predicted route and a withheld (LADD) aircraft
// are already stated by the planned-route row above, and add nothing here.
//
// Lifecycle (Law IV / definition of done): one abortable request per
// selection and one refresh every two minutes while the aircraft stays
// selected; deselecting aborts the request and clears the timer. No map layer,
// no per-frame work. Distances go through lib/units; times are Zulu, with the
// clock they come from (the FAA message's own, or our receive time) on hover.
import { useEffect, useRef, useState } from "react";
import { fmtKm, subscribeUnits } from "@/lib/units";
import {
  HEX6_RE, PLAN_HISTORY_REFRESH_MS, fetchPlanHistory, offRouteLine, originNote, planChangesHeadline, positionWords, zulu,
  type PlanHistoryView,
} from "@/lib/air/planHistory";

const NM_KM = 1.852;
const isPhone = () => typeof window !== "undefined" && !!window.matchMedia?.("(max-width: 639px)").matches;
const ageWords = (ms: number) => { const s = Math.max(0, Math.round(ms / 1000)); return s < 90 ? `${s} s` : `${Math.round(s / 60)} min`; };

export function usePlanChanges(opts: { hex: string | null | undefined; getCallsign?: () => string | null }): { row: JSX.Element | null } {
  const hex = opts.hex && HEX6_RE.test(opts.hex.toLowerCase()) ? opts.hex.toLowerCase() : null;
  const [got, setGot] = useState<{ hex: string; view: PlanHistoryView | null; at: number } | null>(null);
  const optsRef = useRef(opts);
  optsRef.current = opts;

  useEffect(() => {
    if (!hex) { setGot(null); return; }
    setGot(null);
    let stopped = false;
    let ac: AbortController | null = null;
    const load = () => {
      ac?.abort();
      ac = new AbortController();
      fetchPlanHistory(hex, optsRef.current.getCallsign?.() ?? null, (u, init) => fetch(u, init), ac.signal)
        .then((view) => { if (!stopped) setGot({ hex, view, at: Date.now() }); })
        // a failed or aborted read leaves the last good answer on the card; the next refresh asks again
        .catch((err: unknown) => { if (!stopped && (err as { name?: string } | null)?.name !== "AbortError") console.warn("[flight card] plan changes not read:", err); });
    };
    load();
    const iv = window.setInterval(load, PLAN_HISTORY_REFRESH_MS);
    return () => { stopped = true; window.clearInterval(iv); ac?.abort(); };
  }, [hex]);

  const view = got && got.hex === hex ? got.view : null;
  return { row: hex && view && view.state === "changes" ? <PlanChangesRow key={hex} v={view} fetchedAt={got!.at} /> : null };
}

function PlanChangesRow({ v, fetchedAt }: { v: PlanHistoryView; fetchedAt: number }) {
  const [open, setOpen] = useState(false);
  const [, setTick] = useState(0);
  useEffect(() => subscribeUnits(() => setTick((t) => t + 1)), []);
  // the "fetched N ago" line ticks only while the list is open (zero cost otherwise)
  useEffect(() => {
    if (!open) return;
    const iv = window.setInterval(() => setTick((t) => t + 1), 10_000);
    return () => window.clearInterval(iv);
  }, [open]);
  const fmtNm = (nm: number) => fmtKm(nm * NM_KM, nm * NM_KM < 20 ? 1 : 0);
  const headline = planChangesHeadline(v);
  const off = offRouteLine(v, fmtNm);
  const dim = "var(--flight-ink-dim)";
  const line = { whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" } as const;
  return (
    <div data-testid="plan-changes-row" data-open={open ? "true" : undefined} style={{ padding: "8px 14px 0", flex: "0 0 auto", display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
        <button
          className={`vt-flight-follow${open ? " on" : ""}`}
          aria-expanded={open}
          data-vt-plan-changes-toggle
          style={{ margin: 0, flex: "0 0 auto", padding: "0 12px", minHeight: isPhone() ? 44 : 32 }}
          title="How this aircraft's filed flight plan changed: each version the FAA feed sent, and each change to the route ahead, the destination or the cruise altitude"
          onClick={() => setOpen((o) => !o)}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden>
            <path d="M4 7h10M4 12h16M4 17h7" /><circle cx="18" cy="7" r="2" /><circle cx="15" cy="17" r="2" />
          </svg>
          Plan changes
        </button>
        <div data-vt-plan-changes-headline title={[v.label, v.honesty].filter(Boolean).join(" — ") || undefined}
             style={{ flex: "1 1 auto", minWidth: 0, fontFamily: "var(--font-mono)", fontSize: 10.5, lineHeight: 1.4, color: dim }}>
          <div style={open ? { overflowWrap: "anywhere" } : line}><span style={{ color: "var(--flight-ink)" }}>{headline}</span></div>
          {!open && off ? <div style={line}>{off}</div> : null}
        </div>
      </div>
      {open ? (
        <div data-vt-plan-changes-list role="list" aria-label={`Changes to the flight plan, newest first${v.callsign ? `, ${v.callsign}` : ""}`}
             style={{ maxHeight: 148, overflowY: "auto", overflowX: "hidden", display: "flex", flexDirection: "column", gap: 4, fontFamily: "var(--font-mono)", fontSize: 10.5, lineHeight: 1.4, color: dim, paddingRight: 2 }}>
          {v.changes.length ? v.changes.map((c) => (
            <div key={c.key} role="listitem" data-vt-plan-change={c.version} style={{ display: "grid", gridTemplateColumns: "50px minmax(0, 1fr)", columnGap: 8 }}>
              <span title={c.atBasis === "message" ? "the FAA message's own time (UTC)" : "the time WE received the message (UTC); the message carried none"}
                    style={{ color: "var(--flight-ink)", fontVariantNumeric: "tabular-nums" }}>{zulu(c.atMs)}{c.atBasis === "received" ? "*" : ""}</span>
              <span style={{ minWidth: 0, overflowWrap: "anywhere" }}>
                <span style={{ color: "var(--flight-ink)", fontWeight: 600 }}>{c.words}</span>
                {c.lateralNm !== null ? ` · route moved up to ${fmtNm(c.lateralNm)}${c.lateralEstimated ? " (estimated)" : ""}` : ""}
                {` · v${c.fromVersion}→v${c.version}`}{c.spansMissing ? " (versions between were not held)" : ""}
                <span title={c.positionReason ?? undefined}>{` · ${positionWords(c)}`}</span>
              </span>
            </div>
          )) : <div role="listitem">No change to the route ahead, the destination or the cruise altitude{v.restatements ? `: ${v.restatements} later version${v.restatements === 1 ? "" : "s"} restated the same route` : ""}.</div>}
          {off ? <div data-vt-plan-offroute>{off}</div> : null}
          <div>
            {originNote(v)}{v.changes.some((c) => c.atBasis === "received") ? " * = the time we received the message; it carried none of its own." : ""}
            {" "}FAA flight-plan messages as received; the change wording and the off-route judgement are derived by us. Fetched {ageWords(Date.now() - fetchedAt)} ago.
          </div>
        </div>
      ) : null}
    </div>
  );
}
