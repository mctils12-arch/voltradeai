// useFilingHistory — one loader for the /data pages that read a `*-history`
// filing route ([PRODUCT] 2026-11-15): first page, the coverage it states,
// and the walk to older days through the server's own `next` query string.
// The rules (what counts as loaded, when the walk stops) live in
// filingCoverage.ts; this file only fetches and holds state.
import { useCallback, useEffect, useRef, useState } from "react";
import { appendFiledDays, appendOlderFilings, extendCoverage, readCoverage, readFiledDays, type FiledDayStatus, type FilingCoverage, type FilingHistoryMeta } from "./filingCoverage";

export interface FilingHistoryState<T, M> {
  /** the first page's body (source, warming_up, archive_start, …); null until it lands */
  first: M | null;
  filings: T[];
  coverage: FilingCoverage | null;
  /** filed-day pages only: the loaded filed days, newest first, each against EDGAR's count; [] otherwise */
  filedDays: FiledDayStatus[];
  error: boolean;
  loadingOlder: boolean;
  /** the last "load older" request failed; the loaded rows are unchanged and the control stays */
  olderError: boolean;
  loadOlder: () => void;
}

export function useFilingHistory<T extends { accession: string }, M extends FilingHistoryMeta & { filings?: T[] }>(
  /** route path without a query string, e.g. "/api/data/insider/history" */
  route: string,
  /** first page's query string, e.g. "?days=30" */
  firstQuery: string,
): FilingHistoryState<T, M> {
  const [first, setFirst] = useState<M | null>(null);
  const [filings, setFilings] = useState<T[]>([]);
  const [coverage, setCoverage] = useState<FilingCoverage | null>(null);
  const [filedDays, setFiledDays] = useState<FiledDayStatus[]>([]);
  const [error, setError] = useState(false);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [olderError, setOlderError] = useState(false);
  const alive = useRef(true);
  const busy = useRef(false);

  useEffect(() => {
    alive.current = true;
    const ctl = new AbortController();
    (async () => {
      try {
        const r = await fetch(route + firstQuery, { signal: ctl.signal });
        const d = (await r.json()) as M;
        if (!alive.current) return;
        setFirst(d);
        setFilings(Array.isArray(d?.filings) ? d.filings : []);
        setCoverage(readCoverage(d));
        setFiledDays(readFiledDays(d));
      } catch {
        if (alive.current) setError(true);
      }
    })();
    return () => { alive.current = false; ctl.abort(); };
  }, [route, firstQuery]);

  const loadOlder = useCallback(() => {
    if (busy.current || !coverage || !coverage.truncated || !coverage.next) return;
    busy.current = true;
    setLoadingOlder(true);
    setOlderError(false);
    const from = coverage;
    (async () => {
      try {
        const r = await fetch(route + from.next);
        if (!r.ok) throw new Error(String(r.status));
        const d = (await r.json()) as M;
        if (!alive.current) return;
        const next = extendCoverage(from, d);
        // rows are appended only when the older page continues the loaded run of days
        if (next.covered.from !== from.covered.from) {
          setFilings((have) => appendOlderFilings(have, d?.filings));
          setFiledDays((have) => appendFiledDays(have, readFiledDays(d)));
        }
        setCoverage(next);
      } catch {
        if (alive.current) setOlderError(true);
      } finally {
        busy.current = false;
        if (alive.current) setLoadingOlder(false);
      }
    })();
  }, [coverage, route]);

  return { first, filings, coverage, filedDays, error, loadingOlder, olderError, loadOlder };
}
