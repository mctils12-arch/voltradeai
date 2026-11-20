// archiveView — the Data quality panel's archive table and off-site copy
// sentences (UI coverage batch C, second part, 2026-11-20). Reads two routes
// that existed with no page showing them in full: /api/data/archive/stats
// (every archive: files, bytes, first and newest file) and
// /api/data/archive/offload-status (the off-site copy). Pure functions with
// tests; the page lays them out.
//
// Why the newest file is a column: on 2026-11-20 the fires archive's newest
// file was 18 days old and no page showed it.

export interface ArchiveKindStat { files?: number; bytes?: number; oldest?: string | null; newest?: string | null }
export interface ArchiveStats { kinds?: Record<string, ArchiveKindStat>; totalBytes?: number }

interface TierKind { files?: number; objects?: number; bytes?: number; oldestDate?: string | null; newestDate?: string | null }
interface PermStream { r2Objects?: number; r2Bytes?: number; localFiles?: number; oldestDate?: string | null; newestDate?: string | null }
export interface OffloadStatus {
  configured?: boolean;
  retentionDays?: number;
  oldestReplayableDate?: string | null;
  lastRun?: string | null;
  lastRunSummary?: { uploaded?: number; uploadFailed?: number; pending?: number; permanent?: { uploaded?: number; uploadFailed?: number; pendingUploads?: number } } | null;
  lastError?: string | null;
  lastErrorAt?: string | null;
  tiers?: { r2?: { bytes?: number; objects?: number; byKind?: Record<string, TierKind> } };
  volume?: { freeBytes?: number; hotTierTargetFreeBytes?: number; underPressure?: boolean };
  permanentTier?: { enabled?: boolean; r2Objects?: number; r2Bytes?: number; byStream?: Record<string, PermStream>; conflicts?: unknown[]; missing?: unknown[]; lastReconcile?: string | null };
}

const n = (x: number) => x.toLocaleString("en-US");
const plural = (x: number, one: string, many = `${one}s`) => `${n(x)} ${x === 1 ? one : many}`;

export function fmtBytes(b: number | null | undefined): string {
  if (typeof b !== "number" || !Number.isFinite(b)) return "—";
  if (b >= 1e9) return `${(b / 1e9).toFixed(1)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(1)} MB`;
  if (b >= 1e3) return `${(b / 1e3).toFixed(0)} KB`;
  return `${b} B`;
}

const DAY_RE = /(\d{4}-\d{2}-\d{2})/;

/** What a file name says about its period: the day when it carries one
 *  ("2026-11-20-18.jsonl" -> 2026-11-20), otherwise the name without its
 *  folder and extension ("TX_2024Q3", "2026-11", "202605a"). A marker file
 *  (leading dot) or no name is null. */
export function fileLabel(name: string | null | undefined): string | null {
  if (!name) return null;
  const base = name.split("/").pop() || "";
  if (!base || base.startsWith(".")) return null;
  const m = base.match(DAY_RE);
  if (m) return m[1];
  return base.replace(/\.(jsonl|json)(\.gz)?$/, "") || null;
}

/** Age of the newest file by the UTC day in its name. A file dated after
 *  today holds day-ahead data; a name with no day gives no age. */
export function newestAge(name: string | null | undefined, nowMs: number): string {
  const m = (name || "").split("/").pop()?.match(DAY_RE);
  if (!m) return "";
  const t = Date.parse(`${m[1]}T00:00:00Z`);
  if (!Number.isFinite(t)) return "";
  const today = Math.floor(nowMs / 86400000) * 86400000;
  const days = Math.round((today - t) / 86400000);
  if (days < 0) return "dated ahead";
  if (days === 0) return "today";
  return `${n(days)} d ago`;
}

export interface ArchiveRow {
  kind: string;
  files: number;
  bytes: number;
  first: string;
  newest: string;
  age: string;
  /** what is held off-site for this archive, in words; "" when nothing is */
  offsite: string;
}

function offsiteFor(kind: string, o: OffloadStatus | null | undefined): string {
  const perm = o?.permanentTier?.byStream?.[kind];
  if (perm && (perm.r2Objects ?? 0) > 0) {
    const range = perm.oldestDate && perm.newestDate ? `, ${perm.oldestDate} to ${perm.newestDate}` : "";
    return `${plural(perm.r2Objects ?? 0, "day file")} kept for good${range}`;
  }
  const hot = o?.tiers?.r2?.byKind?.[kind];
  if (hot && (hot.objects ?? 0) > 0) {
    const range = hot.oldestDate && hot.newestDate ? `, ${hot.oldestDate} to ${hot.newestDate}` : "";
    return `${plural(hot.objects ?? 0, "file")} in the replay window${range}`;
  }
  return "";
}

/** Every archive, largest first; archives with no file on the volume last, by name. */
export function archiveRows(stats: ArchiveStats | null | undefined, offload: OffloadStatus | null | undefined, nowMs: number): ArchiveRow[] {
  const rows = Object.entries(stats?.kinds ?? {}).map(([kind, k]): ArchiveRow => {
    const files = typeof k.files === "number" ? k.files : 0;
    return {
      kind, files, bytes: typeof k.bytes === "number" ? k.bytes : 0,
      first: files ? fileLabel(k.oldest) ?? "—" : "—",
      newest: files ? fileLabel(k.newest) ?? "—" : "—",
      age: files ? newestAge(k.newest, nowMs) : "",
      offsite: offsiteFor(kind, offload),
    };
  });
  return rows.sort((a, b) => (b.bytes - a.bytes) || a.kind.localeCompare(b.kind));
}

function ago(iso: string | null | undefined, nowMs: number): string {
  const t = iso ? Date.parse(iso) : NaN;
  if (!Number.isFinite(t)) return "";
  const min = Math.max(0, Math.round((nowMs - t) / 60000));
  if (min < 1) return "under a minute ago";
  if (min < 60) return `${min} min ago`;
  const h = Math.floor(min / 60);
  return h < 48 ? `${h} h ${min % 60} min ago` : `${Math.floor(h / 24)} d ago`;
}

/** The off-site copy, one sentence per fact. */
export function offsiteLines(o: OffloadStatus | null | undefined, nowMs: number): string[] {
  if (!o) return [];
  if (!o.configured) return ["Off-site copy is not set up on this server: every archive file exists only on the volume."];
  const out: string[] = [];
  const s = o.lastRunSummary, p = s?.permanent;
  const when = ago(o.lastRun, nowMs);
  if (when && s) {
    const copied = (s.uploaded ?? 0) + (p?.uploaded ?? 0);
    const failed = (s.uploadFailed ?? 0) + (p?.uploadFailed ?? 0);
    const waiting = (s.pending ?? 0) + (p?.pendingUploads ?? 0);
    out.push(`Last copy run ${when}: ${plural(copied, "file")} copied, ${n(failed)} failed, ${n(waiting)} waiting.`);
  } else {
    out.push("No copy run has finished since the server started.");
  }
  const hot = o.tiers?.r2, perm = o.permanentTier;
  const bytes = (hot?.bytes ?? 0) + (perm?.r2Bytes ?? 0);
  const objects = (hot?.objects ?? 0) + (perm?.r2Objects ?? 0);
  if (objects > 0) {
    const parts: string[] = [];
    if ((perm?.r2Objects ?? 0) > 0) parts.push(`${fmtBytes(perm?.r2Bytes)} is day files that are never deleted`);
    if ((hot?.objects ?? 0) > 0) {
      parts.push(`${fmtBytes(hot?.bytes)} is the last ${n(o.retentionDays ?? 0)} days of hourly position logs${o.oldestReplayableDate ? `, replayable from ${o.oldestReplayableDate}` : ""}`);
    }
    out.push(`Held off-site: ${fmtBytes(bytes)} in ${plural(objects, "file")}. ${parts.join("; ")}.`);
  } else {
    out.push("Nothing is held off-site yet.");
  }
  if (perm?.enabled) {
    const differ = perm.conflicts?.length ?? 0, missing = perm.missing?.length ?? 0;
    const checked = ago(perm.lastReconcile, nowMs);
    out.push(`${checked ? `Compared with the off-site listing ${checked}` : "Not yet compared with the off-site listing"}: ${plural(differ, "file")} ${differ === 1 ? "differs" : "differ"} from the copy, ${n(missing)} missing off-site.`);
  }
  const v = o.volume;
  if (v && typeof v.freeBytes === "number") {
    const target = typeof v.hotTierTargetFreeBytes === "number" ? `; the server aims to keep ${fmtBytes(v.hotTierTargetFreeBytes)} free` : "";
    out.push(`The volume has ${fmtBytes(v.freeBytes)} free${target}${v.underPressure ? ". It is under that target, so days already verified off-site are removed from the volume first" : ""}.`);
  }
  if (o.lastError) out.push(`Last error${o.lastErrorAt ? ` ${ago(o.lastErrorAt, nowMs)}` : ""}: ${String(o.lastError).slice(0, 160)}`);
  return out;
}
