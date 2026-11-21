// AIRPORT TRAFFIC — the FAA airport card's row for what the FAA's
// traffic-management feed says about the selected airport ([PRODUCT]
// 2026-11-20, UI coverage batch B, part 3): a ground stop or delay program
// with its reason and end time, the restrictions that name the airport, and
// the runways and hourly rates its tower last reported. Data: GET
// /api/data/swim/tfms/active (lib/air/airportTraffic.ts, pure). RAW, as
// received: no reading here is derived by us.
//
// COMPACT, like components/PlanChanges.tsx: two lines beside a toggle; the
// list opens inside the card with its own scroll, so the card's always-visible
// chrome grows by one row. The row is shown for every FAA airport card once
// the feed has answered: "nothing for this airport" is a sentence, not a gap.
//
// Lifecycle (Law IV / definition of done): one abortable request per
// selection and one refresh a minute while the airport stays selected;
// deselecting aborts the request and clears the timer. No map layer, no
// per-frame work. Times are Zulu, as the FAA publishes them.
import { useEffect, useState } from "react";
import {
  AIRPORT_CODE_RE, AIRPORT_TRAFFIC_REFRESH_MS, PROGRAM_WORDS, configLine, fetchAirportTraffic, periodWords, restrictionLine, trafficHeadline,
  type AirportTrafficView,
} from "@/lib/air/airportTraffic";

const isPhone = () => typeof window !== "undefined" && !!window.matchMedia?.("(max-width: 639px)").matches;
const ageWords = (ms: number) => { const s = Math.max(0, Math.round(ms / 1000)); return s < 90 ? `${s} s` : `${Math.round(s / 60)} min`; };

export function useAirportTraffic(opts: { code: string | null | undefined }): { row: JSX.Element | null } {
  const code = opts.code && AIRPORT_CODE_RE.test(opts.code.toUpperCase()) ? opts.code.toUpperCase() : null;
  const [got, setGot] = useState<{ code: string; view: AirportTrafficView | null; at: number; failed: string | null } | null>(null);

  useEffect(() => {
    if (!code) { setGot(null); return; }
    setGot(null);
    let stopped = false;
    let ac: AbortController | null = null;
    const load = () => {
      ac?.abort();
      ac = new AbortController();
      fetchAirportTraffic(code, (u, init) => fetch(u, init), ac.signal)
        .then((view) => { if (!stopped) setGot({ code, view, at: Date.now(), failed: null }); })
        // a failed refresh keeps the last good answer on the card and says so; a first failure is said in the row
        .catch((err: unknown) => {
          if (stopped || (err as { name?: string } | null)?.name === "AbortError") return;
          const why = String(err instanceof Error ? err.message : err).slice(0, 80);
          setGot((prev) => (prev && prev.code === code ? { ...prev, failed: why } : { code, view: null, at: Date.now(), failed: why }));
        });
    };
    load();
    const iv = window.setInterval(load, AIRPORT_TRAFFIC_REFRESH_MS);
    return () => { stopped = true; window.clearInterval(iv); ac?.abort(); };
  }, [code]);

  if (!code || !got || got.code !== code) return { row: null };
  return { row: <AirportTrafficRow key={code} code={code} v={got.view} fetchedAt={got.at} failed={got.failed} /> };
}

function AirportTrafficRow({ code, v, fetchedAt, failed }: { code: string; v: AirportTrafficView | null; fetchedAt: number; failed: string | null }) {
  const [open, setOpen] = useState(false);
  const [, setTick] = useState(0);
  // the "read N ago" line ticks only while the list is open (zero cost otherwise)
  useEffect(() => {
    if (!open) return;
    const iv = window.setInterval(() => setTick((t) => t + 1), 10_000);
    return () => window.clearInterval(iv);
  }, [open]);
  const now = Date.now();
  const dim = "var(--flight-ink-dim)";
  const line = { whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" } as const;
  const headline = v ? trafficHeadline(v, v.atMs ?? now) : failed ? `FAA traffic management not read for ${code} (${failed})` : `FAA traffic management not read for ${code}: the answer was not in the expected form`;
  const config = v ? configLine(v, v.atMs ?? now) : null;
  const hasList = !!v && (v.programs.length > 0 || v.restrictions.length > 0 || !!config);
  return (
    <div data-testid="airport-traffic-row" data-open={open ? "true" : undefined} data-airport={code} style={{ padding: "8px 14px 0", flex: "0 0 auto", display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
        <button
          className={`vt-flight-follow${open ? " on" : ""}`}
          aria-expanded={open}
          disabled={!hasList}
          data-vt-airport-traffic-toggle
          style={{ margin: 0, flex: "0 0 auto", padding: "0 12px", minHeight: isPhone() ? 44 : 32, opacity: hasList ? 1 : 0.6, cursor: hasList ? "pointer" : "default" }}
          title="What the FAA's traffic-management feed lists for this airport: ground stops and delay programs, restrictions, and the runways and hourly rates the tower last reported"
          onClick={() => setOpen((o) => !o)}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden>
            <path d="M12 3v18M5 8l7-5 7 5M7 21h10" />
          </svg>
          Traffic
        </button>
        <div data-vt-airport-traffic-headline style={{ flex: "1 1 auto", minWidth: 0, fontFamily: "var(--font-mono)", fontSize: 10.5, lineHeight: 1.4, color: dim }}>
          {/* the first line wraps, never cuts: its end is the program's end time (first run: "… volume · u…" at 352 px) */}
          <div style={{ overflowWrap: "anywhere" }}><span style={{ color: "var(--flight-ink)" }}>{headline}</span></div>
          {!open && config ? <div style={line} title={config}>{config}</div> : null}
        </div>
      </div>
      {open && v ? (
        <div data-vt-airport-traffic-list role="list" aria-label={`FAA traffic management for ${code}`}
             style={{ maxHeight: 148, overflowY: "auto", overflowX: "hidden", display: "flex", flexDirection: "column", gap: 4, fontFamily: "var(--font-mono)", fontSize: 10.5, lineHeight: 1.4, color: dim, paddingRight: 2 }}>
          {v.programs.map((p) => (
            <div key={p.key} role="listitem" data-vt-traffic-program={p.kind} style={{ overflowWrap: "anywhere" }}>
              <span style={{ color: "var(--flight-ink)", fontWeight: 600 }}>{PROGRAM_WORDS[p.kind]}</span>
              {p.reason ? ` · ${p.reason}` : ""}{periodWords(p.startMs, p.endMs, v.atMs ?? now) ? ` · ${periodWords(p.startMs, p.endMs, v.atMs ?? now)}` : ""}
              {p.advisory ? <span title="the FAA Command Center advisory that names this program">{` · ${p.advisory}`}</span> : null}
              {!p.fromProgram ? " · from the advisory only: the program message itself is not in the feed" : ""}
            </div>
          ))}
          {config ? <div role="listitem" data-vt-traffic-config style={{ overflowWrap: "anywhere" }}><span style={{ color: "var(--flight-ink)", fontWeight: 600 }}>Runways and rates</span>{` · ${config}`}</div> : null}
          {v.restrictions.map((r) => (
            <div key={r.key} role="listitem" data-vt-traffic-restriction style={{ overflowWrap: "anywhere" }}>{restrictionLine(r, v.atMs ?? now)}</div>
          ))}
          <div data-vt-airport-traffic-source>
            FAA traffic-management feed (SWIM TFMS), as received: {v.searched} programs and restrictions searched for {code}.
            {v.restrictions.length ? " A restriction is listed until its published end; one cancelled early can still be listed." : ""}
            {" "}The list starts again when the server restarts: a program sent before then is missing until the FAA sends it again.
            {" "}Times are Zulu. Not for operational use. Read {ageWords(now - fetchedAt)} ago{failed ? `; the last refresh failed (${failed})` : ""}.
          </div>
        </div>
      ) : null}
    </div>
  );
}
