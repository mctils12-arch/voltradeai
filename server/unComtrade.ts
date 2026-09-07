/**
 * unComtrade.ts — UN Comtrade USA bilateral goods-trade archive, /data
 * client-agnostic view (API only this PR — no /data client page yet, same
 * incremental sequencing as EPA CAMD/FINRA/plant-operations: archive+API
 * first, dedicated client page a documented follow-up).
 *
 * Source: datacore/un_comtrade/bilateral_trade.json (scripts/
 * un_comtrade_ingest.py, session-run — keyless comtradeapi.un.org preview
 * API, never a live Railway poll, same "seeded pattern" as jodiOil.ts/
 * military_installations.json). USA (reporter 842) vs 6 major partners
 * (China/Mexico/Canada/Japan/Germany/South Korea), both flow directions,
 * cmdCode=TOTAL (UN-computed aggregate, not commodity-level detail — see
 * the ingest script's own docstring for why HS-6 detail is out of scope
 * for v1).
 *
 * LADDER STATUS (RAW OVERLAYS vs SIGNALS, CLAUDE.md): GATE 1 (DATA)
 * PASSED 2026-09-07 — every partner's CIF import value reconciles against
 * FRED's independent Census-Bureau-sourced customs-basis import series
 * within a stable, narrow-banded CIF/customs offset (means 1.01-1.06,
 * stdev <=0.0097, n=21 months each), computed with the exact functions
 * shipped in scripts/un_comtrade_gate1.py against real fetched FRED data;
 * see datacore/signal_ladder.json's un_comtrade_bilateral_trade entry and
 * research/experiments.md's 2026-09-07 entry for the exact pre-registered
 * bar, the full per-partner numbers, and an honest note on that script's
 * own end-to-end CLI reliability in this sandbox (its logic is verified
 * correct; a single automated run of its main() sometimes fails in this
 * sandbox specifically, unrelated to the data or the gate result).
 * GATE 2 (SIGNAL) NOT attempted this session — the census's own prior
 * ("too lagged for direct alpha, structural-thesis input only") stands
 * unchallenged; this view is RAW, self-reported bilateral trade levels
 * only, no predictive claim.
 */
import unComtradeArchive from "../datacore/un_comtrade/bilateral_trade.json";

export const UN_COMTRADE_GATE_NOTE =
  "GATE 1 (data) PASSED 2026-09-07 — every partner's CIF import value " +
  "reconciles against FRED's independent Census-Bureau customs-basis " +
  "import series within a stable CIF/customs-basis offset band (see " +
  "scripts/un_comtrade_gate1.py). GATE 2 (signal) not attempted — shown " +
  "here as RAW, self-reported bilateral trade levels only, no predictive claim.";

interface UnComtradeSeries { points: [string, number | null, number | null][]; n: number; first: string | null; last: string | null }
interface UnComtradeArchiveFile {
  source: string; attribution: string; license: string; built_at: string;
  reporter: { code: number; name: string };
  partners: Record<string, string>;
  flows: Record<string, string>;
  cmd_code: string;
  latest_period: string | null;
  series_count: number;
  series: Record<string, UnComtradeSeries>;
}

export interface UnComtradePartnerRow {
  partnerCode: number;
  partnerName: string;
  period: string;
  importsCifUsd: number | null;
  exportsFobUsd: number | null;
  tradeBalanceUsd: number | null; // exports.fob - imports.cif; null if either side is missing
  priorPeriod: string | null;
  priorImportsCifUsd: number | null;
  importsDeltaPct: number | null;
}

export interface UnComtradeView {
  kind: "raw";
  predictive: false;
  source: string;
  attribution: string;
  license: string;
  reporter: string;
  cmdCode: string;
  archiveLatestPeriod: string | null;
  partnerCount: number;
  rows: UnComtradePartnerRow[];
  note: string;
}

/** Builds one row per partner using the LATEST period each partner's
 *  M-flow (imports) series holds — never zero-fills a partner whose
 *  archive is momentarily behind another's (e.g. mid-backfill); a partner
 *  with no M-flow series at all is simply absent from `rows`, matching
 *  jodiOilStocksView's own "skip, never zero" convention. */
export function unComtradeView(
  doc: UnComtradeArchiveFile = unComtradeArchive as unknown as UnComtradeArchiveFile,
): UnComtradeView {
  const rows: UnComtradePartnerRow[] = [];
  for (const [codeStr, name] of Object.entries(doc.partners)) {
    const code = Number(codeStr);
    const imports = doc.series[`${codeStr}|M`];
    const exports_ = doc.series[`${codeStr}|X`];
    if (!imports || !imports.points.length) continue;
    const pts = imports.points;
    const last = pts[pts.length - 1];
    const prior = pts.length > 1 ? pts[pts.length - 2] : null;
    const period = last[0];
    const cif = last[1];
    const priorCif = prior ? prior[1] : null;
    const exportPoint = exports_?.points.find((p) => p[0] === period) ?? null;
    const fob = exportPoint ? exportPoint[2] : null;
    rows.push({
      partnerCode: code,
      partnerName: name,
      period,
      importsCifUsd: cif,
      exportsFobUsd: fob,
      tradeBalanceUsd: fob != null && cif != null ? Math.round(fob - cif) : null,
      priorPeriod: prior ? prior[0] : null,
      priorImportsCifUsd: priorCif,
      importsDeltaPct: cif != null && priorCif ? Math.round(((cif - priorCif) / priorCif) * 1000) / 10 : null,
    });
  }
  rows.sort((a, b) => (b.importsCifUsd ?? 0) - (a.importsCifUsd ?? 0));
  return {
    kind: "raw",
    predictive: false,
    source: doc.source,
    attribution: doc.attribution,
    license: doc.license,
    reporter: doc.reporter.name,
    cmdCode: doc.cmd_code,
    archiveLatestPeriod: doc.latest_period,
    partnerCount: rows.length,
    rows,
    note: UN_COMTRADE_GATE_NOTE,
  };
}
