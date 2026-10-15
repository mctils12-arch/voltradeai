// AIRCRAFT INTENT C1 — the status line on the aircraft card: what the
// aircraft itself broadcasts about its autopilot, selected altitude/heading,
// altimeter setting and air data, in plain English. Only fresh values, only
// fields this aircraft sends; inferred states carry an "inferred" tag. Built
// by lib/air/intentStatus.ts (pure, unit-tested).

import { intentStatusParts, type IntentViewWire } from "@/lib/air/intentStatus";
import { getUnits } from "@/lib/units";

export function IntentStatusRow({ intent }: { intent: IntentViewWire | null | undefined }): JSX.Element | null {
  const parts = intentStatusParts(intent, getUnits());
  if (!parts.length) return null;
  return (
    <div className="vt-intent" data-vt-intent
         title="Broadcast by the aircraft (ADS-B / Mode S), last 30 s only. Inferred states are labeled.">
      <p className="vt-intent-head">
        <span className="vt-atc-label">Aircraft says</span>
        <span className="vt-runway-tag">broadcast · live</span>
      </p>
      <p className="vt-intent-line">
        {parts.map((p, i) => (
          <span key={p.key} className={`vt-intent-part${p.alert ? " vt-intent-alert" : ""}`}>
            {i > 0 && <span className="vt-intent-sep" aria-hidden> · </span>}
            {p.text}
            {p.inferred && <span className="vt-runway-tag vt-intent-inferred">inferred</span>}
          </span>
        ))}
      </p>
    </div>
  );
}
