// fundamentalsView.ts — every word and number the Company fundamentals page
// (#/data/xbrl-fundamentals) shows, as pure functions of what the server
// sent (server/xbrlFrames.ts xbrlPublicView / xbrlCompanyView). The page
// only lays these out.

export interface FundamentalsRun {
  day: string; started: string; finished: string | null; status: string; detail: string;
  frames: { roster: number; read: number; not_published: number; failed: number };
  rows: { first: number; revised: number; dropped: number; refiled: number; backfill: number };
  companies: number;
}
export interface FundamentalsCoverage { tag: string; period: string; state: string; held: number; first: number; revised: number; dropped: number; backfill: boolean }
export interface FundamentalsRevision {
  seen: string; tag: string; period: string; cik: number; name: string | null; ticker: string | null; end: string;
  val: number | null; prev_val: number | null; unit: string; filing: string; prev_filing: string | null;
}
export interface FundamentalsConcept { tag: string; label: string; unit: string; kind: string }
export interface FundamentalsPayload {
  kind?: string; source?: string; attribution?: string; note?: string; warming_up?: boolean;
  concepts?: FundamentalsConcept[]; last_run?: FundamentalsRun | null; nights?: number;
  coverage?: FundamentalsCoverage[]; revisions?: FundamentalsRevision[];
}
export interface CompanyFact { tag: string; label: string; unit: string; period: string; end: string; val: number; filing: string }
export interface CompanyPayload { cik: number; name: string | null; loc: string | null; ticker: string | null; facts: CompanyFact[]; error?: string }

/** `CY2026Q3` -> "Q3 2026"; `CY2026Q3I` -> "end of Q3 2026"; `CY2025` -> "year 2025". */
export function periodLabel(period: string): string {
  const q = /^CY(\d{4})Q([1-4])(I?)$/.exec(period);
  if (q) return `${q[3] ? "end of " : ""}Q${q[2]} ${q[1]}`;
  const y = /^CY(\d{4})$/.exec(period);
  return y ? `year ${y[1]}` : period;
}

/** Newest period first; within a quarter the three-month figure before the balance. */
export function periodOrder(period: string): number {
  const q = /^CY(\d{4})Q([1-4])(I?)$/.exec(period);
  if (q) return Number(q[1]) * 10 + Number(q[2]);
  const y = /^CY(\d{4})$/.exec(period);
  return y ? Number(y[1]) * 10 + 4.5 : 0; // a full year sorts just after its own fourth quarter
}

const STEPS: ReadonlyArray<[number, string]> = [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "K"]];

/** 404548000000 USD -> "$404.5B"; -3531000 -> "-$3.53M"; shares -> "14.53B shares". */
export function fmtAmount(val: number | null, unit: string): string {
  if (val === null || !Number.isFinite(val)) return "—";
  const abs = Math.abs(val);
  let body = abs.toLocaleString("en-US", { maximumFractionDigits: 0 });
  for (const [size, letter] of STEPS) {
    if (abs >= size) {
      const n = abs / size;
      body = `${n >= 100 ? n.toFixed(1) : n.toFixed(2)}${letter}`;
      break;
    }
  }
  if (unit === "USD") return `${val < 0 ? "-" : ""}$${body}`;
  return `${val < 0 ? "-" : ""}${body} ${unit}`;
}

/** The change from the number first held to the number held now, in percent
 *  of the first; null when the first was zero or either is missing. */
export function revisionPct(prev: number | null, now: number | null): number | null {
  if (prev === null || now === null || prev === 0) return null;
  return ((now - prev) / Math.abs(prev)) * 100;
}

export function fmtPct(p: number | null): string {
  if (p === null) return "—";
  const a = Math.abs(p);
  const digits = a >= 10 ? 0 : a >= 1 ? 1 : 2;
  return `${p > 0 ? "+" : p < 0 ? "-" : ""}${a.toFixed(digits)}%`;
}

/** "3 h ago", "2 d ago": how old the last read is. */
export function ageLabel(iso: string | null | undefined, nowMs: number): string {
  const t = iso ? Date.parse(iso) : NaN;
  if (!Number.isFinite(t)) return "age unknown";
  const h = (nowMs - t) / 3_600_000;
  if (h < 1) return "under an hour ago";
  if (h < 48) return `${Math.floor(h)} h ago`;
  return `${Math.floor(h / 24)} d ago`;
}

/** One sentence on the last read, with its age. Says so when it was not a full read. */
export function runLine(p: FundamentalsPayload, nowMs: number): string {
  const r = p.last_run;
  if (!r) return "No read has run yet. The first one runs tonight between 02:00 and 05:00 ET.";
  const when = `Last read ${r.day} (${ageLabel(r.finished ?? r.started, nowMs)})`;
  if (r.status === "skipped_low_disk") return `${when} did not run: the storage volume was under its free-space floor.`;
  if (r.status === "refused") return `${when} was stopped: SEC refused the request. ${r.frames.read} of ${r.frames.roster} tables were read.`;
  const tables = `${r.frames.read} of ${r.frames.roster} tables read`
    + (r.frames.not_published ? `, ${r.frames.not_published} not published by SEC yet` : "")
    + (r.frames.failed ? `, ${r.frames.failed} failed and will be read again tonight` : "");
  return `${when}: ${tables}. ${r.companies.toLocaleString("en-US")} companies on record.`;
}

/** What the last night added, in words. */
export function changeLine(p: FundamentalsPayload): string {
  const r = p.last_run;
  if (!r) return "";
  const live = r.rows.first - r.rows.backfill;
  if (r.rows.backfill > 0 && live === 0 && r.rows.revised === 0) {
    return `That read was the starting copy: ${r.rows.backfill.toLocaleString("en-US")} numbers already published before recording began. Their dates say when we started, not when they were filed.`;
  }
  const parts = [`${live.toLocaleString("en-US")} new numbers`, `${r.rows.revised.toLocaleString("en-US")} changed`];
  if (r.rows.dropped) parts.push(`${r.rows.dropped.toLocaleString("en-US")} removed by SEC`);
  return `That night: ${parts.join(", ")}.`;
}

export interface ConceptRow { tag: string; label: string; newestPeriod: string | null; companies: number; added: number; revised: number }

/** One line per concept: companies in its newest published quarter, and what
 *  last night added or changed across all its periods (the starting copy is
 *  not counted as added). */
export function conceptRows(p: FundamentalsPayload): ConceptRow[] {
  const cov = p.coverage ?? [];
  return (p.concepts ?? []).map((c) => {
    const mine = cov.filter((f) => f.tag === c.tag);
    const quarters = mine.filter((f) => /Q[1-4]I?$/.test(f.period) && f.state === "read" && f.held > 0)
      .sort((a, b) => periodOrder(b.period) - periodOrder(a.period));
    // the quarter in progress holds a handful of early reporters; show the newest quarter that is at least half as full as the fullest
    const fullest = quarters.reduce((m, f) => Math.max(m, f.held), 0);
    const shown = quarters.find((f) => f.held >= fullest / 2) ?? null;
    return {
      tag: c.tag, label: c.label, newestPeriod: shown ? shown.period : null, companies: shown ? shown.held : 0,
      added: mine.reduce((n, f) => n + (f.backfill ? 0 : f.first), 0),
      revised: mine.reduce((n, f) => n + f.revised, 0),
    };
  });
}

export interface FactRow { key: string; label: string; period: string; end: string; amount: string; filing: string }

/** A company's numbers, grouped by concept in the roster's order, newest period first. */
export function factRows(c: CompanyPayload, concepts: readonly FundamentalsConcept[]): FactRow[] {
  const order = new Map(concepts.map((k, i) => [k.tag, i] as const));
  return c.facts.slice().sort((a, b) => (order.get(a.tag) ?? 99) - (order.get(b.tag) ?? 99) || periodOrder(b.period) - periodOrder(a.period))
    .map((f) => ({ key: `${f.tag}/${f.period}`, label: f.label, period: periodLabel(f.period), end: f.end, amount: fmtAmount(f.val, f.unit), filing: f.filing }));
}

export function companyTitle(c: CompanyPayload): string {
  const name = c.name ?? `CIK ${c.cik}`;
  return c.ticker ? `${name} (${c.ticker})` : name;
}

export function conceptLabel(tag: string, concepts: readonly FundamentalsConcept[]): string {
  return concepts.find((c) => c.tag === tag)?.label ?? tag;
}
