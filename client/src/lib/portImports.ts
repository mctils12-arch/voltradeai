// portImports — pure decode/aggregation for /api/data/imports (US Census
// FT920 monthly port-level import values). Extracted out of the page for the
// same reason aqIndex.ts was: every function here can silently misstate a
// number if it gets the edge case wrong, and misstating a number is the one
// thing a data product may never do.
//
// THREE TRAPS THIS MODULE EXISTS TO CLOSE, all found in the live 2026-08-20
// payload (1,059 rows, 358 ports, 3 months) rather than assumed:
//
// 1. The feed carries Census's OWN national aggregate as a row — port code
//    "-", name "TOTAL FOR ALL PORTS". Left in a leaderboard it outranks every
//    real port by ~13x; summed with the rest it double-counts the country.
//    splitNational() removes it exactly once, by code, and hands it back
//    separately so the page can show it as what it is.
// 2. Not every port appears in every month (11 of 358 did not, live). A
//    month-over-month delta against a missing prior month is undefined, not
//    zero — joinMonths() emits null and the page renders "—".
// 3. cnt_val/cnt_wgt 0 and null mean different things: 0 is a figure Census
//    published for that port-month (768 of 1,059 live rows — inland and land-
//    border ports genuinely move no containerized vessel cargo), null means
//    the active query variant did not return the column at all (see
//    server/censusImports.ts QUERY_VARIANTS). They must never render alike.
//
// reconcile() is the payoff of (1): because the per-port rows are a complete
// partition of the national total, summing them is a free integrity check on
// every render — the same "the source ships its own checksum, so use it"
// discipline as secFtd.ts's trailer lines. Verified exact (ratio 1.000000) for
// all three live months at build; the page reports whatever it actually
// computes, including a mismatch.

export interface ImportObs {
  port: string;
  port_name: string | null;
  month: string;
  gen_val: number | null;
  cnt_val: number | null;
  cnt_wgt: number | null;
  rt: string;
}

/** Census's national-aggregate row is published under this port code. */
export const NATIONAL_PORT_CODE = "-";

export interface SplitRows {
  ports: ImportObs[];
  national: ImportObs[];
}

/** Separate the national aggregate rows from the per-port rows. */
export function splitNational(rows: ImportObs[] | null | undefined): SplitRows {
  const ports: ImportObs[] = [];
  const national: ImportObs[] = [];
  for (const r of rows ?? []) {
    if (!r || typeof r.port !== "string") continue;
    (r.port === NATIONAL_PORT_CODE ? national : ports).push(r);
  }
  return { ports, national };
}

/** Distinct months present, newest first. */
export function monthsOf(rows: ImportObs[] | null | undefined): string[] {
  const seen = new Set<string>();
  for (const r of rows ?? []) if (r?.month) seen.add(r.month);
  return [...seen].sort().reverse();
}

export interface PortRow {
  port: string;
  /** Census's published name, or null — never a name inferred from the code. */
  port_name: string | null;
  gen_val: number | null;
  cnt_val: number | null;
  cnt_wgt: number | null;
  /** Prior-month general-imports value, or null when the port has no prior row. */
  prev_gen_val: number | null;
  /** Fractional change vs. prior month, or null when it is undefined. */
  delta: number | null;
}

/**
 * Rows for `month`, each joined to its own prior-month observation.
 *
 * `delta` is null — never 0 — when the port has no prior row, when either
 * value is null, or when the prior value is 0 (a division that would report
 * an infinite move).
 */
export function joinMonths(rows: ImportObs[], month: string, prevMonth: string | null): PortRow[] {
  const prev = new Map<string, ImportObs>();
  if (prevMonth) for (const r of rows) if (r.month === prevMonth) prev.set(r.port, r);
  const out: PortRow[] = [];
  for (const r of rows) {
    if (r.month !== month) continue;
    const p = prev.get(r.port) ?? null;
    const prev_gen_val = p?.gen_val ?? null;
    const delta =
      r.gen_val == null || prev_gen_val == null || prev_gen_val === 0
        ? null
        : (r.gen_val - prev_gen_val) / prev_gen_val;
    out.push({
      port: r.port,
      port_name: r.port_name ?? null,
      gen_val: r.gen_val,
      cnt_val: r.cnt_val,
      cnt_wgt: r.cnt_wgt,
      prev_gen_val,
      delta,
    });
  }
  return out;
}

export interface Reconciliation {
  /** Sum of the per-port general-import values for the month. */
  sum: number;
  /** Census's own published all-ports total, or null if it sent no such row. */
  published: number | null;
  /** sum - published, or null when there is nothing to compare against. */
  diff: number | null;
  /** True only when the two agree to within `tolerance` of the published total. */
  exact: boolean;
}

/**
 * Live integrity check: do the per-port rows add up to the figure Census
 * itself published for the whole country that month?
 *
 * Tolerance is relative and tiny (1e-9) — these are integer dollar figures, so
 * anything above float noise is a real discrepancy worth showing the user.
 */
export function reconcile(
  ports: PortRow[] | ImportObs[],
  nationalRows: ImportObs[],
  month: string,
  tolerance = 1e-9,
): Reconciliation {
  let sum = 0;
  for (const r of ports) sum += r.gen_val ?? 0;
  const nat = nationalRows.find((r) => r.month === month) ?? null;
  const published = nat?.gen_val ?? null;
  const diff = published == null ? null : sum - published;
  const exact = published != null && published !== 0 && Math.abs(diff as number) <= Math.abs(published) * tolerance;
  return { sum, published, diff, exact };
}

export type SortKey = "gen_val" | "cnt_val" | "cnt_wgt" | "delta" | "port_name";

/**
 * Sort for display. Numeric keys sort descending with nulls LAST in every
 * case — a port with no published figure must never lead a leaderboard by
 * sorting as if it were zero or infinite.
 */
export function sortRows(rows: PortRow[], key: SortKey): PortRow[] {
  const out = rows.slice();
  if (key === "port_name") {
    out.sort((a, b) => (a.port_name ?? a.port).localeCompare(b.port_name ?? b.port));
    return out;
  }
  out.sort((a, b) => {
    const av = a[key];
    const bv = b[key];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    return bv - av;
  });
  return out;
}

/** Case-insensitive match on the published name or the Schedule D code. */
export function filterRows(rows: PortRow[], q: string): PortRow[] {
  const needle = q.trim().toLowerCase();
  if (!needle) return rows;
  return rows.filter(
    (r) => r.port.toLowerCase().includes(needle) || (r.port_name ?? "").toLowerCase().includes(needle),
  );
}
