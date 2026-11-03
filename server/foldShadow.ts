/**
 * foldShadow.ts — the shared vocabulary for a fold that runs in three modes
 * while an incremental rewrite proves itself exact against the full
 * computation (memory fix A2b, 2026-11-03; used by gnssIntegrityFold.ts and
 * entityGraph.ts's managed poll).
 *
 *   full         today's computation, unchanged — the rollback position.
 *   shadow       compute BOTH, deep-compare with timing fields excluded,
 *                log any difference loudly (console.error + an audit row
 *                when a writer is wired) and SERVE THE FULL RESULT. The
 *                default: nothing a user sees changes until the incremental
 *                path has run mismatch-free for long enough to trust.
 *   incremental  serve the incremental result; the memory win.
 *
 * An unrecognised or unset flag resolves to `shadow`, never to the new path:
 * a typo in an env var must not switch production onto unproven code, and
 * must not silently disable the comparison either.
 *
 * firstDifference exists because a bare "deep-equal returned false" is not
 * actionable from a log line — the mismatch row names the first differing
 * path and both values, so a session reading the audit log can localise the
 * fault (ROOT VALIDATION LADDER spirit: fault AT the layer, not "the whole
 * pipeline differs").
 */
import { isDeepStrictEqual } from "util";

export type FoldMode = "full" | "shadow" | "incremental";

export type AuditWriter = (type: string, message: string) => void;

/** Parse a VOLTRADE_*_FOLD / _POLL env value. Case- and whitespace-
 *  insensitive; anything that is not exactly `full` or `incremental` is
 *  `shadow` (see the module header for why the default is the safe one). */
export function parseFoldMode(raw: string | undefined | null): FoldMode {
  const v = (raw ?? "").trim().toLowerCase();
  return v === "full" || v === "incremental" ? v : "shadow";
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** Path of the first difference between two JSON-like values, or null when
 *  they are deep-strict-equal. Arrays compare positionally; objects by key
 *  set (order-independent, a key present on one side only is itself the
 *  difference). The returned string is bounded so it fits an audit row. */
export function firstDifference(a: unknown, b: unknown, at = "$"): string | null {
  if (isDeepStrictEqual(a, b)) return null;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return `${at}.length: ${a.length} vs ${b.length}`;
    for (let i = 0; i < a.length; i++) {
      const d = firstDifference(a[i], b[i], `${at}[${i}]`);
      if (d) return d;
    }
  } else if (isRecord(a) && isRecord(b)) {
    for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
      if (!(k in a) || !(k in b)) return `${at}.${k}: present on one side only`;
      const d = firstDifference(a[k], b[k], `${at}.${k}`);
      if (d) return d;
    }
  }
  return `${at}: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`.slice(0, 400);
}
