// Satellite find & group panel — EARTH TWIN O6-3. Lives inside the legend's
// Orbital section (mobile-safe: the panel scrolls). Search jumps focus to a
// specific object (the "you can't find the ISS" fix); group chips filter the
// sky to one constellation (name-prefix decode of the catalog's own naming);
// the orbits toggle plots the group's real SGP4 tracks, RAAN-colored so
// planes read apart, CAP DISCLOSED when it bites.
import { memo, useMemo, useState } from "react";
import { searchSats, SAT_GROUPS, groupMemberHits, type SatSearchHit } from "@/lib/orbital/satFind";
import type { GpRecord } from "@/lib/orbital/tle";

export interface SatFinderProps {
  /** live catalog (null until loaded); gpVersion bumps force re-render. */
  gp: GpRecord[] | null;
  gpVersion: number;
  activeGroup: string | null;
  groupCount: number | null;
  orbitsOn: boolean;
  /** arcs actually plotted vs group size (cap disclosure), null = orbits off. */
  arcInfo: { shown: number; total: number } | null;
  onFind: (index: number) => void;
  onGroup: (key: string | null) => void;
  onOrbits: (on: boolean) => void;
}

export const SatFinder = memo(function SatFinder(props: SatFinderProps) {
  const [q, setQ] = useState("");
  const hits: SatSearchHit[] = useMemo(
    () => searchSats(props.gp, q),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [props.gp, props.gpVersion, q],
  );
  // O8 (live report 2026-07-18): search always covers the FULL catalog, even
  // while a group chip narrows the sky — clicking an outside hit clears the
  // filter (the parent's findSat), and the hit row SAYS so up front.
  const activeGrp = props.activeGroup
    ? SAT_GROUPS.find((g) => g.key === props.activeGroup) ?? null
    : null;
  // O8 (live report 2026-07-18: "it's hard to find"): while a chip filters
  // the sky, the finder lists the group's members as a direct click-to-focus
  // path — a filter must never leave the user hunting dots.
  const members = useMemo(
    () => (props.activeGroup ? groupMemberHits(props.gp, props.activeGroup) : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [props.gp, props.gpVersion, props.activeGroup],
  );
  return (
    <div className="vt-satfinder" data-vt-satfinder>
      <input
        className="vt-satfinder-input"
        placeholder="Find a satellite (name or NORAD id)…"
        value={q}
        aria-label="Find a satellite"
        onChange={(e) => setQ(e.target.value)}
      />
      {q.trim().length >= 2 && (
        <div className="vt-satfinder-hits om-sb">
          {hits.length === 0 && <span className="vt-legend-note">no match in the live catalog</span>}
          {hits.map((h) => (
            <button key={h.noradId} className="vt-satfinder-hit"
                    onClick={() => { props.onFind(h.index); setQ(""); }}>
              {h.name} <span className="vt-satfinder-id">#{h.noradId}</span>
              {activeGrp && !activeGrp.test(h.name.toUpperCase()) && (
                <span className="vt-satfinder-id"> · outside {activeGrp.label} — focusing clears the filter</span>
              )}
            </button>
          ))}
        </div>
      )}
      <div className="vt-satfinder-groups">
        {SAT_GROUPS.map((g) => (
          <button key={g.key}
                  className={`vt-satfinder-chip${props.activeGroup === g.key ? " vt-satfinder-chip-on" : ""}`}
                  onClick={() => props.onGroup(props.activeGroup === g.key ? null : g.key)}>
            {g.label}
          </button>
        ))}
        {props.activeGroup && (
          <button className={`vt-satfinder-chip${props.orbitsOn ? " vt-satfinder-chip-on" : ""}`}
                  onClick={() => props.onOrbits(!props.orbitsOn)}>
            orbits
          </button>
        )}
      </div>
      {activeGrp && q.trim().length < 2 && members && members.total > 0 && (
        <div className="vt-satfinder-hits om-sb">
          {members.hits.map((h) => (
            <button key={h.noradId} className="vt-satfinder-hit"
                    onClick={() => props.onFind(h.index)}>
              {h.name} <span className="vt-satfinder-id">#{h.noradId}</span>
            </button>
          ))}
          {members.total > members.hits.length && (
            <span className="vt-legend-note">
              first {members.hits.length} of {members.total.toLocaleString()} {activeGrp.label} members — type above to narrow
            </span>
          )}
        </div>
      )}
      {props.activeGroup && (
        <span className="vt-legend-note">
          {`showing only ${SAT_GROUPS.find((g) => g.key === props.activeGroup)?.label ?? props.activeGroup}`}
          {props.groupCount != null ? ` — ${props.groupCount.toLocaleString()} in the live catalog` : ""}
          {props.orbitsOn && props.arcInfo
            ? props.arcInfo.shown < props.arcInfo.total
              ? ` · orbits: ${props.arcInfo.shown} of ${props.arcInfo.total} plotted (evenly sampled — the full set is unreadable)`
              : ` · orbits: all ${props.arcInfo.shown} plotted`
            : ""}
          {props.orbitsOn ? " · color = orbital plane (RAAN)" : ""}
        </span>
      )}
    </div>
  );
});
