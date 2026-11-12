// useFilingHistory.ts — the one fetch for a /data filing archive page
// (2026-11-12 [PRODUCT]): reads the first page of a `*-history` route, keeps
// every older page the user loads, and hands the page the merged filings plus
// an honest coverage statement (filingHistory.ts). The three EDGAR pages used
// to each run their own effect and print the echoed `days`; this is that
// effect once, with the contract read.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { coverageLabel, mergeFilingPages, olderPageUrl, type FilingCoverage, type FilingHistoryPayload } from "./filingHistory";

export interface FilingHistoryState<T> {
  /** the first page's body as answered (pages read extra fields from it: warming_up, focused_cap…) */
  first: FilingHistoryPayload<T> | null;
  /** merged, deduplicated filings across every page read; null until the first page lands */
  filings: T[] | null;
  coverage: FilingCoverage | null;
  /** the sub-line words for the current coverage */
  label: string;
  /** the first page failed */
  error: boolean;
  /** an older page is being fetched */
  loadingOlder: boolean;
  /** the last older-page fetch failed (the pages already loaded stay) */
  olderError: boolean;
  /** true when the server says older days inside the window are still unread */
  hasOlder: boolean;
  loadOlder: () => void;
}

export function useFilingHistory<T extends { accession: string }>(firstUrl: string, fallbackDays: number, noun: string): FilingHistoryState<T> {
  const [pages, setPages] = useState<FilingHistoryPayload<T>[] | null>(null);
  const [error, setError] = useState(false);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [olderError, setOlderError] = useState(false);
  const alive = useRef(true);
  const inFlight = useRef(false);

  useEffect(() => {
    alive.current = true;
    setPages(null);
    setError(false);
    setOlderError(false);
    (async () => {
      try {
        const r = await fetch(firstUrl);
        const d = (await r.json()) as FilingHistoryPayload<T>;
        if (!alive.current) return;
        if (!r.ok || !d || typeof d !== "object") { setError(true); return; }
        setPages([d]);
      } catch {
        if (alive.current) setError(true);
      }
    })();
    return () => { alive.current = false; };
  }, [firstUrl]);

  const merged = useMemo(() => (pages ? mergeFilingPages(pages, fallbackDays) : null), [pages, fallbackDays]);
  const coverage = merged?.coverage ?? null;
  const nextUrl = olderPageUrl(firstUrl, coverage?.truncated ? coverage.next : null);

  const loadOlder = useCallback(() => {
    if (!nextUrl || inFlight.current) return;
    inFlight.current = true;
    setLoadingOlder(true);
    setOlderError(false);
    (async () => {
      try {
        const r = await fetch(nextUrl);
        const d = (await r.json()) as FilingHistoryPayload<T>;
        if (!alive.current) return;
        if (!r.ok || !d || typeof d !== "object") { setOlderError(true); return; }
        setPages((p) => (p ? [...p, d] : [d]));
      } catch {
        if (alive.current) setOlderError(true);
      } finally {
        inFlight.current = false;
        if (alive.current) setLoadingOlder(false);
      }
    })();
  }, [nextUrl]);

  return {
    first: pages?.[0] ?? null,
    filings: merged?.filings ?? null,
    coverage,
    label: coverageLabel(coverage, noun),
    error,
    loadingOlder,
    olderError,
    hasOlder: nextUrl !== null,
    loadOlder,
  };
}
