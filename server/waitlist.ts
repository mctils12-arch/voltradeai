/**
 * waitlist.ts — API-product waitlist capture (/developers page). EMAIL
 * ONLY, no billing, no pricing enablement — "coming soon" per the
 * monetization readiness checklist (wishlist.md); the tripwire stays
 * untripped (no BILLING_ENABLED, no Stripe anywhere here).
 *
 * PII NOTE (manifested in datacore/manifests/waitlist.json): emails are
 * stored RAW on the volume day-files — they are the product's contact
 * list, needed as-is; they are never exposed through any API endpoint,
 * never committed to git, and never mixed into other streams.
 *
 * Pure module: fs only, base-injectable, no express/db imports.
 */
import fs from "fs";
import path from "path";
import { archiveBaseDir } from "./datacoreArchive";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;
export function validEmail(e: string): boolean {
  return typeof e === "string" && e.length <= 254 && EMAIL_RE.test(e);
}

let seen: Set<string> | null = null;

function seedSeen(dir: string): Set<string> {
  const s = new Set<string>();
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
      for (const line of fs.readFileSync(path.join(dir, f), "utf8").split("\n")) {
        if (!line) continue;
        try { s.add(JSON.parse(line).e); } catch {}
      }
    }
  } catch {}
  return s;
}

/** Records one email; returns "added" | "duplicate" | "invalid". */
export function addToWaitlist(email: string, source = "developers",
                              baseDir?: string, nowMs?: number): "added" | "duplicate" | "invalid" {
  const e = String(email || "").trim().toLowerCase();
  if (!validEmail(e)) return "invalid";
  const base = baseDir || archiveBaseDir();
  const dir = path.join(base, "waitlist");
  try { fs.mkdirSync(dir, { recursive: true }); } catch { return "invalid"; }
  if (!seen) seen = seedSeen(dir);
  if (seen.has(e)) return "duplicate";
  if (seen.size >= 100_000) return "invalid"; // abuse backstop
  seen.add(e);
  const now = nowMs ?? Date.now();
  try {
    fs.appendFileSync(path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`),
      JSON.stringify({ t: Math.floor(now / 1000), e, s: source }) + "\n");
  } catch { return "invalid"; }
  return "added";
}

/** test hook — resets the in-memory dedup so hermetic tmpdirs work */
export function _resetWaitlistSeen() { seen = null; }
