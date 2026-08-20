// Display decodes for the GDELT facility-events payload (/api/data/
// facility-events, server/gdeltEvents.ts). Kept here rather than inline in the
// page because every function below exists to stop the page asserting
// something the feed does not actually say.
//
// THREE PROPERTIES OF THIS FEED, each verified against the live payload on
// 2026-08-20 before any of this was written:
//
// 1. GoldsteinScale is a CONSTANT OF THE EVENT CODE, not a measurement of the
//    event. Every one of the 15 live rows carried exactly the value the
//    published CAMEO Goldstein table assigns its code (173 -> -5.0, 172 ->
//    -5.0, 192 -> -9.5, 1821 -> -9.0, 1712 -> -9.2, 141 -> -6.5), and no code
//    ever showed two values. Rendered per row it reads as a severity score
//    someone assessed for this incident; it is a restatement of the type
//    column. So the page shows it ONCE PER TYPE, and `goldsteinAudit()` checks
//    the claim live on whatever rows are on screen rather than trusting this
//    comment.
// 2. ONE ARTICLE PRODUCES SEVERAL ROWS. 15 live rows came from 9 distinct
//    article URLs — one article alone produced 4 rows, each a distinct
//    GlobalEventID with its own geocoding. Counting rows as incidents
//    overstates how much happened, so `groupByArticle()` collapses them and
//    the page shows both numbers.
// 3. "NEAR A FACILITY" MEANS THE SAME METRO AREA. server/gdeltEvents.ts
//    matches on a +/-0.5 degree BOX around the site, which is ~55 km of
//    latitude and up to ~55 km of longitude — a corner of that box is ~70 km
//    from the facility. GDELT's own geocoding is city/ADM-approximate on top
//    of that. `eventDistanceKm()` computes the real separation from our own
//    catalogued site coordinates so the reader sees the actual number instead
//    of the word "near".
//
// The label and Goldstein tables below are transcribed verbatim from GDELT's
// published lookups (CAMEO.eventcodes.txt / CAMEO.goldsteinscale.txt), limited
// to the root codes server/gdeltEvents.ts actually ingests (14/17/18/19/20 plus
// the 143x strike codes). Two upstream typos are preserved deliberately
// ("protest for  change" double space, "radiologicalweapons") — this is the
// source's text, and silently correcting a source table is how a decode starts
// drifting from what it decodes. An unknown code is NEVER guessed at from its
// root; it returns null and the page shows the raw code.

import strategicSites from "../../../datacore/sites/strategic_sites.json";

export interface GdeltEventRow {
  id: string;
  day: string;
  code: string;
  root: string;
  gold: number | null;
  tone: number | null;
  mentions: number | null;
  lat: number;
  lon: number;
  site: string;
  url: string | null;
  rt: string;
}

/** CAMEO EventRootCode -> label. Upstream ships these in caps; title-cased
 *  here for display, which changes no word. */
export const CAMEO_ROOT_LABEL: Record<string, string> = {
  "14": "Protest",
  "17": "Coerce",
  "18": "Assault",
  "19": "Fight",
  "20": "Use unconventional mass violence",
};

/** CAMEO EventCode -> published description, verbatim. */
export const CAMEO_EVENT_LABEL: Record<string, string> = {
  "140": "Engage in political dissent, not specified below",
  "141": "Demonstrate or rally",
  "1411": "Demonstrate for leadership change",
  "1412": "Demonstrate for policy change",
  "1413": "Demonstrate for rights",
  "1414": "Demonstrate for change in institutions, regime",
  "142": "Conduct hunger strike, not specified below",
  "1421": "Conduct hunger strike for leadership change",
  "1422": "Conduct hunger strike for policy change",
  "1423": "Conduct hunger strike for rights",
  "1424": "Conduct hunger strike for change in institutions, regime",
  "143": "Conduct strike or boycott, not specified below",
  "1431": "Conduct strike or boycott for leadership change",
  "1432": "Conduct strike or boycott for policy change",
  "1433": "Conduct strike or boycott for rights",
  "1434": "Conduct strike or boycott for change in institutions, regime",
  "144": "Obstruct passage, block",
  "1441": "Obstruct passage to demand leadership change",
  "1442": "Obstruct passage to demand policy change",
  "1443": "Obstruct passage to demand rights",
  "1444": "Obstruct passage to demand change in institutions, regime",
  "145": "Protest violently, riot",
  "1451": "Engage in violent protest for leadership change",
  "1452": "Engage in violent protest for policy change",
  "1453": "Engage in violent protest for rights",
  "1454": "Engage in violent protest for  change in institutions, regime",
  "170": "Coerce, not specified below",
  "171": "Seize or damage property, not specified below",
  "1711": "Confiscate property",
  "1712": "Destroy property",
  "172": "Impose administrative sanctions, not specified below",
  "1721": "Impose restrictions on political freedoms",
  "1722": "Ban political parties or politicians",
  "1723": "Impose curfew",
  "1724": "Impose state of emergency or martial law",
  "173": "Arrest, detain, or charge with legal action",
  "174": "Expel or deport individuals",
  "175": "Use tactics of violent repression",
  "180": "Use unconventional violence, not specified below",
  "181": "Abduct, hijack, or take hostage",
  "182": "Physically assault, not specified below",
  "1821": "Sexually assault",
  "1822": "Torture",
  "1823": "Kill by physical assault",
  "183": "Conduct suicide, car, or other non-military bombing, not spec below",
  "1831": "Carry out suicide bombing",
  "1832": "Carry out car bombing",
  "1833": "Carry out roadside bombing",
  "184": "Use as human shield",
  "185": "Attempt to assassinate",
  "186": "Assassinate",
  "190": "Use conventional military force, not specified below",
  "191": "Impose blockade, restrict movement",
  "192": "Occupy territory",
  "193": "Fight with small arms and light weapons",
  "194": "Fight with artillery and tanks",
  "195": "Employ aerial weapons",
  "196": "Violate ceasefire",
  "200": "Use unconventional mass violence, not specified below",
  "201": "Engage in mass expulsion",
  "202": "Engage in mass killings",
  "203": "Engage in ethnic cleansing",
  "204": "Use weapons of mass destruction, not specified below",
  "2041": "Use chemical, biological, or radiologicalweapons",
  "2042": "Detonate nuclear weapons",
};

/** CAMEO EventCode -> published GoldsteinScale value, verbatim. A property of
 *  the CODE. Two codes sharing a root can differ (1712 is -9.2 while 173 is
 *  -5.0), so this is a real table and not a per-root constant. */
export const CAMEO_GOLDSTEIN: Record<string, number> = {
  "140": -6.5, "141": -6.5, "1411": -6.5, "1412": -6.5, "1413": -6.5, "1414": -6.5,
  "142": -6.5, "1421": -6.5, "1422": -6.5, "1423": -6.5, "1424": -6.5,
  "143": -6.5, "1431": -6.5, "1432": -6.5, "1433": -6.5, "1434": -6.5,
  "144": -7.5, "1441": -7.5, "1442": -7.5, "1443": -7.5, "1444": -7.5,
  "145": -7.5, "1451": -7.5, "1452": -7.5, "1453": -7.5, "1454": -7.5,
  "170": -7.0, "171": -9.2, "1711": -9.2, "1712": -9.2,
  "172": -5.0, "1721": -5.0, "1722": -5.0, "1723": -5.0, "1724": -5.0,
  "173": -5.0, "174": -5.0, "175": -9.0,
  "180": -9.0, "181": -9.0, "182": -9.5, "1821": -9.0, "1822": -9.0, "1823": -10.0,
  "183": -10.0, "1831": -10.0, "1832": -10.0, "1833": -10.0,
  "184": -8.0, "185": -8.0, "186": -10.0,
  "190": -10.0, "191": -9.5, "192": -9.5, "193": -10.0, "194": -10.0, "195": -10.0, "196": -9.5,
  "200": -10.0, "201": -9.5, "202": -10.0, "203": -10.0, "204": -10.0, "2041": -10.0, "2042": -10.0,
};

export interface DecodedCode {
  code: string;
  /** null when upstream added a code our transcribed table predates. */
  label: string | null;
  root: string;
  rootLabel: string | null;
  /** The published constant for this code, null for an unknown code. */
  goldstein: number | null;
}

export function decodeCode(code: string | null | undefined, root?: string | null): DecodedCode {
  const c = String(code ?? "").trim();
  // The root the feed sent is authoritative; fall back to the code's own first
  // two digits only when the row omitted it.
  const r = String(root ?? "").trim() || c.slice(0, 2);
  return {
    code: c,
    label: CAMEO_EVENT_LABEL[c] ?? null,
    root: r,
    rootLabel: CAMEO_ROOT_LABEL[r] ?? null,
    goldstein: c in CAMEO_GOLDSTEIN ? CAMEO_GOLDSTEIN[c] : null,
  };
}

// ── the Goldstein claim, checked live rather than asserted ───────────────────

export interface GoldsteinAudit {
  /** Rows carrying both a code we know and a non-null value. */
  checked: number;
  matched: number;
  mismatches: Array<{ id: string; code: string; sent: number; published: number }>;
  /** Codes present in the data but absent from our transcribed table. */
  unknownCodes: string[];
  /** Codes whose rows did not all carry the same value — would falsify the
   *  "constant of the code" reading outright. */
  varyingCodes: string[];
}

export function goldsteinAudit(events: readonly GdeltEventRow[]): GoldsteinAudit {
  const seen = new Map<string, Set<number>>();
  const unknown = new Set<string>();
  const mismatches: GoldsteinAudit["mismatches"] = [];
  let checked = 0;
  let matched = 0;
  for (const e of events) {
    const code = String(e?.code ?? "").trim();
    if (!(code in CAMEO_EVENT_LABEL)) unknown.add(code);
    if (typeof e?.gold !== "number" || !Number.isFinite(e.gold)) continue;
    if (!seen.has(code)) seen.set(code, new Set());
    seen.get(code)!.add(e.gold);
    const published = CAMEO_GOLDSTEIN[code];
    if (published === undefined) continue;
    checked++;
    // Both sides come from a one-decimal published scale; compare with a
    // tolerance so a float round-trip never reads as a data defect.
    if (Math.abs(e.gold - published) < 1e-9) matched++;
    else mismatches.push({ id: e.id, code, sent: e.gold, published });
  }
  const varying = [...seen.entries()].filter(([, v]) => v.size > 1).map(([k]) => k);
  return {
    checked,
    matched,
    mismatches,
    unknownCodes: [...unknown].filter(Boolean).sort(),
    varyingCodes: varying.sort(),
  };
}

// ── facility geometry — our own catalogue, not the feed's ────────────────────

export interface SiteInfo { id: string; name: string; category?: string; operator?: string; lat: number; lon: number }

const SITE_BY_ID: Map<string, SiteInfo> = new Map(
  ((strategicSites as { sites?: SiteInfo[] }).sites ?? [])
    .filter((s) => Number.isFinite(s?.lat) && Number.isFinite(s?.lon))
    .map((s) => [s.id, s]),
);

export function siteInfo(id: string | null | undefined): SiteInfo | null {
  return SITE_BY_ID.get(String(id ?? "")) ?? null;
}

const R_EARTH_KM = 6371.0088;

export function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const rad = Math.PI / 180;
  const dLat = (lat2 - lat1) * rad;
  const dLon = (lon2 - lon1) * rad;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1 * rad) * Math.cos(lat2 * rad) * Math.sin(dLon / 2) ** 2;
  return 2 * R_EARTH_KM * Math.asin(Math.min(1, Math.sqrt(a)));
}

/** Separation between an event's own geocoded point and the facility it was
 *  matched to. null when the site id is not in our catalogue or the row has no
 *  usable coordinates — never 0, which would read as "at the facility". */
export function eventDistanceKm(ev: GdeltEventRow): number | null {
  const s = siteInfo(ev?.site);
  if (!s) return null;
  if (!Number.isFinite(ev?.lat) || !Number.isFinite(ev?.lon)) return null;
  return haversineKm(ev.lat, ev.lon, s.lat, s.lon);
}

// ── rows -> incidents ────────────────────────────────────────────────────────

export interface IncidentSite { id: string; name: string; distanceKm: number | null }

export interface Incident {
  /** Article URL, or the row id when the row carried no URL. */
  key: string;
  url: string | null;
  host: string | null;
  day: string;
  /** AvgTone is a property of the article; verified constant across every
   *  multi-row article in the live payload. Null if the rows disagree. */
  tone: number | null;
  /** Highest mention count among this article's rows — GDELT counts mentions
   *  per EVENT, so rows of one article legitimately differ. */
  maxMentions: number | null;
  /** How many GlobalEventIDs collapsed into this incident. */
  rows: number;
  sites: IncidentSite[];
  codes: string[];
  /** Closest of this incident's rows to its matched facility. */
  nearestKm: number | null;
}

export function hostOf(url: string | null | undefined): string | null {
  const raw = String(url ?? "").trim();
  if (!raw) return null;
  try {
    return new URL(raw).hostname.replace(/^www\./, "") || null;
  } catch {
    return null;
  }
}

/** Collapses rows sharing an article URL into one incident. Sorted by mentions
 *  desc, then row count desc, then key — a total order, so the table does not
 *  reshuffle between renders of identical data. */
export function groupByArticle(events: readonly GdeltEventRow[]): Incident[] {
  const byKey = new Map<string, GdeltEventRow[]>();
  for (const e of events) {
    if (!e) continue;
    const key = String(e.url ?? "").trim() || `id:${e.id}`;
    if (!byKey.has(key)) byKey.set(key, []);
    byKey.get(key)!.push(e);
  }
  const out: Incident[] = [];
  for (const [key, rows] of byKey) {
    const tones = new Set(rows.map((r) => r.tone).filter((t): t is number => typeof t === "number"));
    const mentions = rows.map((r) => r.mentions).filter((m): m is number => typeof m === "number");
    const dists = rows.map(eventDistanceKm).filter((d): d is number => d != null);
    const siteIds = [...new Set(rows.map((r) => r.site).filter(Boolean))];
    out.push({
      key,
      url: rows[0].url ?? null,
      host: hostOf(rows[0].url),
      day: rows[0].day,
      tone: tones.size === 1 ? [...tones][0] : null,
      maxMentions: mentions.length ? Math.max(...mentions) : null,
      rows: rows.length,
      sites: siteIds.map((id) => {
        const s = siteInfo(id);
        const d = rows
          .filter((r) => r.site === id)
          .map(eventDistanceKm)
          .filter((x): x is number => x != null);
        return { id, name: s?.name ?? id, distanceKm: d.length ? Math.min(...d) : null };
      }),
      codes: [...new Set(rows.map((r) => String(r.code ?? "").trim()).filter(Boolean))].sort(),
      nearestKm: dists.length ? Math.min(...dists) : null,
    });
  }
  out.sort(
    (a, b) =>
      (b.maxMentions ?? -1) - (a.maxMentions ?? -1) ||
      b.rows - a.rows ||
      a.key.localeCompare(b.key),
  );
  return out;
}

export interface TypeSummaryRow extends DecodedCode { count: number }

/** One row per distinct event code present, most frequent first — the honest
 *  home for the Goldstein constant. */
export function typeSummary(events: readonly GdeltEventRow[]): TypeSummaryRow[] {
  const counts = new Map<string, number>();
  for (const e of events) {
    const c = String(e?.code ?? "").trim();
    if (!c) continue;
    counts.set(c, (counts.get(c) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([code, count]) => {
      const root = events.find((e) => String(e?.code ?? "").trim() === code)?.root ?? null;
      return { ...decodeCode(code, root), count };
    })
    .sort((a, b) => b.count - a.count || a.code.localeCompare(b.code));
}
