// FacilityEventsView — world-media unrest/strike event mentions geocoded near
// our tracked facilities (#/data/facility-events). server/gdeltEvents.ts +
// server/routes.ts shipped the API-only route with no client view — the LAST
// of the "shipped-data-no-UI" gaps named in the 2026-08-20 port-imports
// session's own NEXT note, and the one that was held back because it needs a
// CAMEO decode table before it can be surfaced honestly. That table, plus the
// three things this payload will misrepresent if rendered naively, live in
// client/src/lib/cameoEvents.ts with their evidence.
//
// RAW display (gate-0 observation). No predictive claim is made or implied:
// the unrest-burst -> own-sensor-confirmation hypothesis is filed and
// ladder-gated, and nothing here is a validated signal. These are MEDIA
// MENTIONS — an article existing is not an incident occurring.
//
// Not a spatial map layer: the rows carry GDELT's own city/ADM-approximate
// geocoding, which is precise-looking and wrong at facility scale (see the
// distance column), so plotting them as points beside our metre-accurate
// layers would assert a precision the feed does not have. It launches from the
// panel-top list like air-quality/drought instead. Reuses the generic
// .vt-filings-* table CSS — no new styles.
import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { ArrowLeft, ExternalLink, Megaphone } from "lucide-react";
import {
  goldsteinAudit, groupByArticle, typeSummary, decodeCode,
  type GdeltEventRow,
} from "@/lib/cameoEvents";
import { fmtKm, getUnits, subscribeUnits } from "@/lib/units";

interface FacilityEventsPayload {
  kind?: string;
  warming_up?: boolean;
  source?: string;
  attribution?: string;
  time?: number;
  count?: number;
  note?: string;
  events?: GdeltEventRow[];
}

const tone = (t: number | null) => (t == null ? "—" : `${t > 0 ? "+" : ""}${t.toFixed(1)}`);
const dayLabel = (d: string) =>
  /^\d{8}$/.test(d) ? `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}` : d || "—";

export default function FacilityEventsView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<FacilityEventsPayload | null>(null);
  const [error, setError] = useState(false);
  // Live store, so the distance column follows the layers-panel unit toggle
  // without a reload (CLAUDE.md UNITS PREFERENCE).
  const units = useSyncExternalStore(subscribeUnits, getUnits, getUnits);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/facility-events");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const events = useMemo(() => data?.events ?? [], [data]);
  const incidents = useMemo(() => groupByArticle(events), [events]);
  const types = useMemo(() => typeSummary(events), [events]);
  const audit = useMemo(() => goldsteinAudit(events), [events]);
  const positiveTone = incidents.filter((i) => i.tone != null && i.tone > 0).length;
  const spread = useMemo(() => {
    const d = incidents.map((i) => i.nearestKm).filter((x): x is number => x != null);
    return d.length ? { min: Math.min(...d), max: Math.max(...d) } : null;
  }, [incidents]);
  const facilities = new Set(events.map((e) => e.site).filter(Boolean)).size;
  const km = (v: number | null) => (v == null ? "—" : fmtKm(v, 1, units));

  return (
    <div className="vt-filings-page" role="region" aria-label="Media events near tracked facilities">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Megaphone size={16} />
        <div>
          <div className="vt-filings-title">Media events near tracked facilities</div>
          <div className="vt-filings-sub">
            world-news unrest, strike and coercion mentions geocoded inside ±0.5° of our strategic sites — RAW
            observation, no predictive claim · {data?.attribution ?? "The GDELT Project"} ·{" "}
            <a href="https://www.gdeltproject.org/" target="_blank" rel="noreferrer">
              GDELT <ExternalLink size={11} />
            </a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && !data && <div className="vt-filings-state">Loading…</div>}
      {!error && data?.warming_up && (
        <div className="vt-filings-state">Warming up — the first 15-minute GDELT export is still in flight.</div>
      )}

      {!error && data && !data.warming_up && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub" style={{ color: "var(--accent-orange)" }}>
            Read these as <strong>verification prompts, not findings</strong>. Each row is a news article
            <em> mentioning</em> an event, classified by CAMEO — an actor–action political taxonomy. It cannot
            see industrial accidents (NASA FIRMS is our fire sensor), it classifies who-did-what-to-whom rather
            than the article's subject, and a match here means the article was geocoded into a box around a
            facility, not that anything happened at one.
          </div>

          <div className="vt-filings-sub">
            <strong>{events.length.toLocaleString()}</strong> event rows collapse to{" "}
            <strong>{incidents.length.toLocaleString()}</strong> distinct articles across{" "}
            <strong>{facilities}</strong> {facilities === 1 ? "facility" : "facilities"}. One article routinely
            produces several rows — GDELT emits a separate event ID per actor pairing and geocoding — so the row
            count is not an incident count, and neither number is a count of things that occurred.
          </div>

          <div className="vt-filings-sub">
            <strong>Goldstein check:</strong>{" "}
            {audit.checked === 0 ? (
              "no row on screen carries a Goldstein value to check."
            ) : (
              <>
                {audit.matched.toLocaleString()} of {audit.checked.toLocaleString()} rows carry exactly the value
                the published CAMEO table assigns their event code
                {audit.mismatches.length === 0 && audit.varyingCodes.length === 0
                  ? ", and no code shows two different values."
                  : "."}{" "}
                This column is a constant of the event <em>type</em>, not a severity someone assessed for this
                event, so it is shown once per type below rather than per row.
                {audit.mismatches.length > 0 && (
                  <>
                    {" "}
                    <strong>
                      {audit.mismatches.length} row{audit.mismatches.length === 1 ? "" : "s"} disagree with the
                      published table
                    </strong>{" "}
                    ({audit.mismatches.slice(0, 3).map((m) => `${m.id}: code ${m.code} sent ${m.sent}, table says ${m.published}`).join("; ")}
                    {audit.mismatches.length > 3 ? "; …" : ""}) — that is a change upstream, reported rather than
                    hidden.
                  </>
                )}
                {audit.unknownCodes.length > 0 && (
                  <> {audit.unknownCodes.length} code(s) present are not in our transcribed table ({audit.unknownCodes.join(", ")}); their rows show the raw code and no label.</>
                )}
              </>
            )}
          </div>

          {incidents.length > 0 && (
            <div className="vt-filings-sub">
              Distances below are measured from each article's own geocoded point to the facility it was filed
              under, using our site catalogue — the ±0.5° ingest box reaches roughly {fmtKm(70, 0, units)} at its corner, so
              "near" here can mean the same metro area rather than the same site.
              {spread && (
                <> On screen right now that spans {km(spread.min)} to {km(spread.max)}.</>
              )}
              {positiveTone > 0 && (
                <>
                  {" "}
                  {positiveTone} of these {incidents.length} articles carry a <strong>positive</strong> tone:
                  CAMEO codes the actor–action pair, so a favourably-written article can still land under a
                  coercion code. Expect false positives at this gate; that is what the confirmation step is for.
                </>
              )}
            </div>
          )}

          {data.note && <div className="vt-filings-sub">{data.note}</div>}

          {incidents.length > 0 ? (
            // .vt-filings-tablewrap is `flex: 1` because every page before this
            // one had exactly ONE table in the flex-column body. With two, the
            // shared space is split and each table scrolls inside its own
            // squeezed box — the first render of this page showed 2 of its 4
            // incidents with no indication the rest existed. Both wraps size to
            // content instead and the page body does the scrolling.
            <div className="vt-filings-tablewrap" style={{ flex: "0 0 auto" }}>
              <table className="vt-filings-table">
                <thead>
                  <tr>
                    {/* Day rides with the article and distance with the
                        facility it measures: eight columns wrapped every
                        numeric cell onto two lines at 768px. */}
                    {/* The shared phone stylesheet suppresses the stacked-card
                        label on the LAST cell, so the last column has to be one
                        that reads without one — a bare signed "-4.6" under no
                        label is not a tone, it is a mystery. Event types goes
                        last for that reason, not for its width. */}
                    <th>Article</th>
                    <th>Facility matched · distance</th>
                    <th className="num">Rows</th>
                    <th className="num">Mentions</th>
                    <th className="num">Tone</th>
                    <th>Event types</th>
                  </tr>
                </thead>
                <tbody>
                  {incidents.map((i) => (
                    <tr key={i.key}>
                      <td data-l="Article">
                        <div>
                        {i.url ? (
                          // Padded to a 44px tap target: these are the only
                          // per-row controls on the page and they are the
                          // whole point of it on a phone.
                          <a href={i.url} target="_blank" rel="noreferrer" style={{ display: "inline-block", padding: "7px 0" }}>
                            {i.host ?? "source"} <ExternalLink size={11} />
                          </a>
                        ) : (
                          <span>no article URL in source</span>
                        )}
                        <span className="vt-filings-role">{dayLabel(i.day)}</span>
                        </div>
                      </td>
                      {/* Multi-value cells wrap their lines in ONE element: at
                          <=639px the td itself becomes a flex row, so bare
                          sibling divs would be laid out side by side instead
                          of stacked. */}
                      <td data-l="Facility matched">
                        <div>
                          {i.sites.map((s) => (
                            <div key={s.id}>
                              {s.name} <span className="vt-filings-role" style={{ display: "inline" }}>{km(s.distanceKm)}</span>
                            </div>
                          ))}
                        </div>
                      </td>
                      <td data-l="Rows" className="num">{i.rows}</td>
                      <td data-l="Mentions" className="num" title="highest mention count among this article's rows">
                        {i.maxMentions ?? "—"}
                      </td>
                      {/* nowrap: a signed one-decimal number in a narrow mono
                          column otherwise breaks after the point ("-7." / "5"). */}
                      <td data-l="Tone" className="num" style={{ whiteSpace: "nowrap" }} title="GDELT AvgTone of the article, −100…+100">
                        {tone(i.tone)}
                      </td>
                      <td data-l="Event types">
                        <div>
                          {i.codes.map((c) => {
                            const d = decodeCode(c);
                            return (
                              <div key={c} title={d.rootLabel ? `${d.rootLabel} (root ${d.root})` : `root ${d.root}`}>
                                {d.label ?? `code ${c} — not in our decode table`}
                              </div>
                            );
                          })}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="vt-filings-state">
              No matching media event in the current 48-hour window. Silence here is a real reading, not a gap —
              the ingest filter is narrow by design.
            </div>
          )}

          {types.length > 0 && (
            <>
              <div className="vt-filings-sub" style={{ marginTop: 14 }}>
                <strong>Event types present.</strong> Goldstein is the published CAMEO constant for the code —
                the same number for every event of that type, everywhere, always. It ranks event <em>types</em> on
                a conflict–cooperation scale; it measures nothing about the events on this page.
              </div>
              <div className="vt-filings-tablewrap" style={{ flex: "0 0 auto" }}>
                <table className="vt-filings-table">
                  <thead>
                    <tr>
                      {/* Same last-cell rule as the table above: the phone
                          stylesheet drops the final label, so the self-
                          describing text column goes last. */}
                      <th className="num">Code</th>
                      <th>Root</th>
                      <th className="num">Goldstein (type constant)</th>
                      <th className="num">Rows</th>
                      <th>CAMEO type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {types.map((t) => (
                      <tr key={t.code}>
                        <td data-l="Code" className="num">{t.code}</td>
                        <td data-l="Root">{t.rootLabel ?? `root ${t.root}`}</td>
                        <td data-l="Goldstein" className="num">{t.goldstein ?? "—"}</td>
                        <td data-l="Rows" className="num">{t.count}</td>
                        <td data-l="CAMEO type">{t.label ?? "not in our decode table"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          <div className="vt-filings-sub" style={{ marginTop: 12 }}>
            Source: {data.source ?? "The GDELT Project"}. Rolling 48-hour window over the 15-minute exports;
            every matched row is archived permanently, so this window is the live view and not the record.
            GDELT is free for unlimited use including commercial, with attribution to The GDELT Project.
          </div>
        </div>
      )}
    </div>
  );
}
