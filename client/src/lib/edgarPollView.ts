// edgarPollView — the Pipeline health panel's "SEC filing pollers" section:
// what each of the three EDGAR pollers (insider trades, institutional
// holdings, earnings releases) read on its latest 15-minute poll
// (server/edgarPollHealth.ts publicEdgarPollView, carried in
// /api/data/pipeline-health-dashboard as `edgar`). Pure functions with tests;
// the page lays them out. Every word a reader sees is written here.
import { ago, type JobTone } from "./jobsView";

export type EdgarStreamId = "filings" | "filings13f" | "earnings8k";

export interface EdgarPollDocuments {
  at: string;
  candidates: number;
  returned: number;
  skipped: number;
  failed: number;
  first_error: string | null;
  /** of `returned`, filings read from their submission text because EDGAR had
   *  not yet published the filing's directory listing (Form 4 and 13F) */
  via_submission?: number;
}
export interface PublicEdgarPollStream {
  stream: EdgarStreamId | string;
  polls_ok: number;
  polls_failed: number;
  consecutive_failures: number;
  last_archive_write_at: string | null;
  documents: EdgarPollDocuments | null;
}
export interface PublicEdgarPollView {
  read_at: string;
  booted_at: string;
  poll_interval_min: number;
  streams: PublicEdgarPollStream[];
}

export interface EdgarPollRow {
  id: string;
  label: string;
  tone: JobTone;
  /** the chip: one or two words */
  state: string;
  /** what the latest poll could not read, in words; null when it read everything */
  cause: string | null;
  /** the latest poll's counts, the archive's last new row, and the polls since the server started */
  meta: string;
}

const LABEL: Record<string, string> = {
  filings: "Form 4: insider trades",
  filings13f: "13F: institutional holdings",
  earnings8k: "8-K: earnings releases",
};
/** what a fetched filing with nothing to keep is, per stream */
const SKIPPED: Record<string, string> = {
  filings: "named no issuer",
  filings13f: "had no primary document",
  earnings8k: "had no press-release exhibit",
};
/** two failed polls in a row is where the session health check warns */
export const EDGAR_FAILING_AFTER = 2;

const n = (v: unknown): number => (typeof v === "number" && Number.isFinite(v) && v > 0 ? Math.floor(v) : 0);
const filings = (k: number): string => `${k.toLocaleString("en-US")} ${k === 1 ? "filing" : "filings"}`;
const polls = (k: number): string => `${k.toLocaleString("en-US")} ${k === 1 ? "poll" : "polls"}`;

function row(s: PublicEdgarPollStream, bootedAt: string | null, nowMs: number): EdgarPollRow {
  const d = s.documents;
  const ok = n(s.polls_ok), bad = n(s.polls_failed), run = n(s.consecutive_failures);
  const failed = n(d?.failed), asked = n(d?.candidates), read = n(d?.returned), skipped = n(d?.skipped);
  const viaText = Math.min(read, n(d?.via_submission));

  let tone: JobTone = "ok", state = "reading";
  if (ok + bad === 0) { tone = "idle"; state = "no poll yet"; }
  else if (run >= EDGAR_FAILING_AFTER) { tone = "bad"; state = "failing"; }
  else if (failed > 0 && read === 0) { tone = "bad"; state = "read nothing"; }
  else if (failed > 0) { tone = "idle"; state = "read in part"; }

  let cause: string | null = null;
  if (failed > 0) {
    cause = `The latest poll could not read ${failed.toLocaleString("en-US")} of ${filings(asked)}.`
      + (d?.first_error ? ` First failure: ${d.first_error}.` : "");
  } else if (run >= EDGAR_FAILING_AFTER) {
    cause = `The last ${polls(run)} could not read EDGAR's list of new filings.`;
  }

  const parts: string[] = [];
  if (d) {
    const when = ago(d.at, nowMs);
    // EDGAR lists a filing before it publishes the filing's directory listing; such a
    // filing is read from its submission text, the same documents in one file
    const via = viaText > 0
      ? ` (${viaText === read ? (read === 1 ? "it" : "all") : viaText.toLocaleString("en-US")} from the submission text: EDGAR had not yet published ${viaText === 1 ? "its listing" : "their listings"})`
      : "";
    const skip = skipped > 0 ? `, ${skipped.toLocaleString("en-US")} ${SKIPPED[s.stream] ?? "had nothing to keep"}` : "";
    parts.push(asked === 0
      ? `Latest poll${when ? ` ${when}` : ""}: no filing of this kind on EDGAR's list.`
      : `Latest poll${when ? ` ${when}` : ""}: ${filings(asked)} asked for, ${read.toLocaleString("en-US")} read${via}${skip}.`);
  } else {
    parts.push("No poll has finished reading its list since the server started.");
  }
  const wrote = ago(s.last_archive_write_at, nowMs);
  parts.push(wrote ? `Last new filing archived ${wrote}.` : "No filing archived in the last two days.");
  const up = ago(bootedAt, nowMs);
  parts.push(`Since the server started${up ? ` ${up}` : ""}: ${polls(ok)} read, ${bad.toLocaleString("en-US")} failed.`);

  return { id: String(s.stream), label: LABEL[s.stream] ?? String(s.stream), tone, state, cause, meta: parts.join(" ") };
}

const ORDER: Record<JobTone, number> = { bad: 0, idle: 1, ok: 2 };

/** One row per poller, the ones that need reading first. */
export function edgarPollRows(view: PublicEdgarPollView | null | undefined, nowMs: number): EdgarPollRow[] {
  const list = Array.isArray(view?.streams) ? view!.streams : [];
  return list
    .filter((s) => s && typeof s.stream === "string")
    .map((s, i) => ({ r: row(s, view?.booted_at ?? null, nowMs), i }))
    .sort((a, b) => ORDER[a.r.tone] - ORDER[b.r.tone] || a.i - b.i)
    .map((x) => x.r);
}

/** "3 pollers, every 15 min: 2 reading, 1 read in part" */
export function edgarPollSummary(view: PublicEdgarPollView | null | undefined, nowMs: number): string {
  const rows = edgarPollRows(view, nowMs);
  if (!rows.length) return "no poller reported";
  const counts = new Map<string, number>();
  for (const r of rows) counts.set(r.state, (counts.get(r.state) ?? 0) + 1);
  const every = n(view?.poll_interval_min);
  const said = Array.from(counts.entries()).map(([state, k]) => `${k} ${state}`).join(", ");
  return `${rows.length} ${rows.length === 1 ? "poller" : "pollers"}${every ? `, every ${every} min` : ""}: ${said}`;
}
