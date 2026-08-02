/**
 * githubOrgActivity.ts — GitHub org engineering-momentum archiver.
 * EDGE DOCTRINE axis (a) + NEW DATA ROOTS #5 (research/open_questions.md,
 * the last unbuilt item in that charter's build order — #1 8-K language,
 * #3 app-store both SHIPPED; #2 job postings KILLED at gate 0; #4 USPTO
 * BLOCKED on a human ID.me signup; this is #5, unblocked).
 *
 * HYPOTHESIS (gate 2, not attempted here): a develop-in-public company's
 * public-repo commit/merged-PR velocity leads or confirms engineering
 * headcount/investment trends the market later prices — small-cap
 * devtools names specifically (EDGE DOCTRINE #2, fish where whales can't:
 * this is too small and slow-moving a panel for institutional systematic
 * desks to bother scraping and maintaining).
 *
 * SOBER PRIOR, stated before building (REASONING STANDARD #10): the
 * HONEST CONFOUND this item's own research entry names is real and fatal
 * to most of the panel — public GitHub activity is a strategic, biased
 * slice that varies enormously by company culture. It is a live signal
 * for names that genuinely develop in public (their core product ships
 * through the watched org) and a rounding error or pure noise for
 * everyone else; private repos are invisible regardless. Expect the gate
 * 2 test to show real structure for maybe a third of this watchlist and
 * nothing for the rest — that split is itself the finding, not a reason
 * to not build. This module is GATE 1 (DATA) ONLY: an archiver, no
 * trading claim.
 *
 * SOURCE DEVIATION FROM THE FILED LADDER (logged honestly): the
 * research/open_questions.md spec named GH Archive (data.gharchive.org)
 * as the DATA source. This build uses the GitHub REST/Search API instead
 * — GH Archive ships the ENTIRE public GitHub firehose per hour (tens of
 * MB compressed each); filtering a ~15-org watchlist out of that stream
 * would mean downloading and streaming-parsing on the order of a
 * gigabyte a day for a nightly job, which conflicts with GOAL priority 1
 * (never break the loop) on a paper-trading container with a documented
 * 1GB RSS self-kill. The Search API gives the same weekly org-level
 * counts directly from GitHub's own index in ~2 keyless calls/org — the
 * same "few calls per cycle" shape every other archiver in this file
 * uses (fdaEvents.ts, appStoreRankings.ts). GH Archive remains the right
 * tool for a future full-firehose historical backfill; it is not needed
 * for an ongoing weekly counter.
 *
 * LICENSING (research/open_questions.md NEW DATA ROOTS #5, verified
 * 2026-07-04): GitHub REST/Search API is CONDITIONAL — aggregated,
 * non-personal metrics (counts, not per-user profiling beyond bot-
 * filtering a public commit's own author field) are an accepted use
 * under the GitHub API Terms; keyless calls are rate-limited (search:
 * 10 req/min unauthenticated) but sufficient at this watchlist's volume.
 * No token is configured or required — this stream must keep working
 * with zero GitHub credential in the environment.
 *
 * WATCHLIST: 15 develop-in-public SaaS/devtools names spanning small-cap
 * (PagerDuty, Braze, Asana, JFrog, Wix) through large-cap controls
 * (Cloudflare, Datadog, Snowflake, Palantir) — every ticker AND every
 * GitHub org login below was independently verified live this session
 * (2026-08-02) via web search against primary sources (SEC filings /
 * exchange listings for the ticker; the org's own public GitHub page for
 * the login) BEFORE being hardcoded — this sandbox's network proxy
 * blocks direct github.com/api.github.com calls (repo-scoped only), so
 * verification used external search rather than a live GitHub hit, but
 * every entry traces to a checked source, never guessed from a company
 * name (the job-postings item's failure mode, NEW DATA ROOTS #2, is
 * exactly what this avoids). One originally-considered name — Confluent
 * (CFLT) — was DROPPED after verification found IBM's acquisition
 * closed and the ticker went defunct 2026-03-18; HashiCorp, New Relic,
 * Sumo Logic, and Smartsheet were excluded up front for the same reason
 * (all confirmed taken private before this session).
 *
 * HONEST GAPS: merged-PR counts exclude GitHub-App bot authors via the
 * documented `-author:app/<slug>` search qualifier (dependabot, renovate,
 * github-actions) so dependency-bump PRs don't inflate "engineering
 * momentum." Commit totals are NOT bot-query-filtered (the commit-search
 * `author:` qualifier only matches a linked human/bot GitHub account
 * reliably for the exact login, and malformed exclusions would risk a
 * silent 422 that drops a whole org's week) — instead `uniqueActorsSample`
 * bot-filters using each commit's own `author.type === "Bot"` field from
 * the (up to 100) returned items, labeled `actorSampleCapped:true` when
 * the org's true weekly commit count exceeds that 100-item page, so a
 * high-volume org's undercounted sample is visible, never silently
 * passed off as a full count.
 *
 * Boundary: no imports from trading logic (SPINOUT-READY DATA LAYER).
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

export interface WatchlistOrg {
  ticker: string;
  company: string;
  org: string; // GitHub org login — verified live 2026-08-02, see header
}

// Verified live 2026-08-02 (web search against SEC/exchange + GitHub org
// pages — see header for method and the Confluent exclusion this caught).
export const WATCHLIST: WatchlistOrg[] = [
  { ticker: "MDB", company: "MongoDB, Inc.", org: "mongodb" },
  { ticker: "ESTC", company: "Elastic N.V.", org: "elastic" },
  { ticker: "NET", company: "Cloudflare, Inc.", org: "cloudflare" },
  { ticker: "DDOG", company: "Datadog, Inc.", org: "DataDog" },
  { ticker: "SNOW", company: "Snowflake Inc.", org: "snowflakedb" },
  { ticker: "TWLO", company: "Twilio Inc.", org: "twilio" },
  { ticker: "HUBS", company: "HubSpot, Inc.", org: "HubSpot" },
  { ticker: "PD", company: "PagerDuty, Inc.", org: "PagerDuty" },
  { ticker: "FROG", company: "JFrog Ltd.", org: "jfrog" },
  { ticker: "WIX", company: "Wix.com Ltd.", org: "wix" },
  { ticker: "U", company: "Unity Software Inc.", org: "Unity-Technologies" },
  { ticker: "ASAN", company: "Asana, Inc.", org: "Asana" },
  { ticker: "MNDY", company: "monday.com Ltd.", org: "mondaycom" },
  { ticker: "PLTR", company: "Palantir Technologies Inc.", org: "palantir" },
  { ticker: "BRZE", company: "Braze, Inc.", org: "braze-inc" },
];

// GitHub Apps whose merged PRs are dependency/infra noise, not human
// engineering momentum — excluded from the mergedPRs query via the
// documented `author:app/<slug>` qualifier.
const BOT_APP_SLUGS = ["dependabot", "renovate", "github-actions"];

export interface GithubActivityRecord {
  t: "org_week";
  key: string; // weekStart|org
  ticker: string;
  org: string;
  weekStart: string; // YYYY-MM-DD, Monday UTC
  weekEnd: string;   // YYYY-MM-DD, Sunday UTC (inclusive)
  mergedPRs: number | null;          // bot-app-excluded, see BOT_APP_SLUGS
  commits: number | null;            // NOT bot-filtered — see header HONEST GAPS
  uniqueActorsSample: number | null; // bot-filtered, first-page sample only
  actorSampleCapped: boolean;        // true when commits > the sampled 100 items
  rt: string; // as-seen UTC date
}

// ── week bounds ──────────────────────────────────────────────────────────────

/** Most recently FULLY COMPLETED Mon-Sun UTC week as of `nowMs` — never the
 *  in-progress current week, so every archived week's counts are final. */
export function lastCompletedWeek(nowMs: number): { weekStart: string; weekEnd: string } {
  const d = new Date(nowMs);
  const dow = d.getUTCDay(); // 0=Sun..6=Sat
  const daysSinceMonday = (dow + 6) % 7;
  const thisMonday = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate() - daysSinceMonday);
  const lastMonday = thisMonday - 7 * 86400_000;
  const lastSunday = thisMonday - 86400_000;
  return {
    weekStart: new Date(lastMonday).toISOString().slice(0, 10),
    weekEnd: new Date(lastSunday).toISOString().slice(0, 10),
  };
}

// ── query builders (pure — testable without network) ───────────────────────

export function mergedPrQuery(org: string, weekStart: string, weekEnd: string): string {
  const botExcl = BOT_APP_SLUGS.map((s) => `-author:app/${s}`).join(" ");
  return `org:${org} is:pr is:merged merged:${weekStart}..${weekEnd} ${botExcl}`;
}

export function commitsQuery(org: string, weekStart: string, weekEnd: string): string {
  return `org:${org} committer-date:${weekStart}..${weekEnd}`;
}

const KNOWN_BOT_NAMES = /^(dependabot|renovate|github-actions|greenkeeper|snyk-bot|allcontributors|semantic-release-bot)(\[bot\])?$/i;

function isBotCommitItem(item: any): boolean {
  if (item?.author?.type === "Bot") return true;
  const login = String(item?.author?.login || "");
  if (/\[bot\]$/i.test(login) || KNOWN_BOT_NAMES.test(login)) return true;
  const name = String(item?.commit?.author?.name || "");
  return /\[bot\]$/i.test(name) || KNOWN_BOT_NAMES.test(name);
}

/** GitHub search `total_count` is the true match count (not capped at the
 *  1,000-result pagination ceiling) for a scoped org+date-range query —
 *  safe to read directly, no pagination needed. */
export function parseSearchTotal(json: any): number | null {
  return typeof json?.total_count === "number" ? json.total_count : null;
}

export function parseCommitSample(json: any): { uniqueActorsSample: number | null; capped: boolean } {
  const total = typeof json?.total_count === "number" ? json.total_count : null;
  const items: any[] = Array.isArray(json?.items) ? json.items : [];
  if (!items.length && total === null) return { uniqueActorsSample: null, capped: false };
  const actors = new Set<string>();
  for (const item of items) {
    if (isBotCommitItem(item)) continue;
    const id = item?.author?.login || item?.commit?.author?.email || item?.sha;
    if (id) actors.add(String(id));
  }
  return { uniqueActorsSample: actors.size, capped: total !== null && total > items.length };
}

// ── fetch (injectable) ──────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
const HEADERS = {
  "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)",
  "Accept": "application/vnd.github+json",
  "X-GitHub-Api-Version": "2022-11-28",
};

async function getJson(fetchImpl: FetchFn, url: string): Promise<any> {
  const r = await fetchImpl(url, { headers: HEADERS, signal: AbortSignal.timeout(20000) as any });
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return JSON.parse(await r.text());
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** One org's week: 2 search calls (merged PRs, commits). Sequential across
 *  orgs with a delay between iterations to respect the unauthenticated
 *  search API's 10 req/min limit (2 calls/org * 15 orgs well under that
 *  at the default spacing). `skip(key)` lets the caller skip an org whose
 *  week is already archived — the common case on a frequently-redeployed
 *  container (see archiveGithubActivity's dedup), keeping repeat boots
 *  cheap. One org's failure never drops the rest (per-org try/catch,
 *  mirrors every other archiver in this file). */
export async function fetchGithubActivity(
  watchlist: WatchlistOrg[] = WATCHLIST,
  fetchImpl: FetchFn = fetch as any,
  nowMs?: number,
  opts: { delayMs?: number; skip?: (key: string) => boolean } = {},
): Promise<GithubActivityRecord[]> {
  const now = nowMs ?? Date.now();
  const rt = new Date(now).toISOString().slice(0, 10);
  const { weekStart, weekEnd } = lastCompletedWeek(now);
  const delayMs = opts.delayMs ?? 6500;
  const out: GithubActivityRecord[] = [];
  let calledOnce = false;
  for (let i = 0; i < watchlist.length; i++) {
    const w = watchlist[i];
    const key = `${weekStart}|${w.org}`;
    if (opts.skip?.(key)) continue;
    if (calledOnce) await sleep(delayMs);
    calledOnce = true;
    let mergedPRs: number | null = null;
    let commits: number | null = null;
    let uniqueActorsSample: number | null = null;
    let actorSampleCapped = false;
    try {
      const url = `https://api.github.com/search/issues?q=${encodeURIComponent(mergedPrQuery(w.org, weekStart, weekEnd))}&per_page=1`;
      mergedPRs = parseSearchTotal(await getJson(fetchImpl, url));
    } catch (e: any) {
      console.error(`[datacore] github_activity merged-PRs ${w.org}:`, e?.message || e);
    }
    await sleep(delayMs);
    try {
      const url = `https://api.github.com/search/commits?q=${encodeURIComponent(commitsQuery(w.org, weekStart, weekEnd))}&per_page=100`;
      const json = await getJson(fetchImpl, url);
      commits = parseSearchTotal(json);
      const sample = parseCommitSample(json);
      uniqueActorsSample = sample.uniqueActorsSample;
      actorSampleCapped = sample.capped;
    } catch (e: any) {
      console.error(`[datacore] github_activity commits ${w.org}:`, e?.message || e);
    }
    out.push({
      t: "org_week", key, ticker: w.ticker, org: w.org, weekStart, weekEnd,
      mergedPRs, commits, uniqueActorsSample, actorSampleCapped, rt,
    });
  }
  return out;
}

// ── archive ──────────────────────────────────────────────────────────────────

const archivedKeys = new Set<string>();
let seeded = false;

function githubActivityDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "github_activity");
}

function seedSeen(dir: string): void {
  let files: string[] = [];
  try { files = fs.readdirSync(dir); } catch { return; }
  for (const f of files) {
    if (!f.endsWith(".jsonl") && !f.endsWith(".jsonl.gz")) continue;
    let text: string | null = null;
    try {
      text = f.endsWith(".gz")
        ? zlib.gunzipSync(fs.readFileSync(path.join(dir, f))).toString("utf8")
        : fs.readFileSync(path.join(dir, f), "utf8");
    } catch { continue; }
    for (const line of text.split("\n")) {
      if (!line) continue;
      try { archivedKeys.add(JSON.parse(line).key); } catch {}
    }
  }
}

/** Weekly records are tiny in volume (<=15/week) — unlike the other
 *  archivers here, everything ever written stays in one growing file
 *  rather than a daily-file-per-poll layout; a full-history scan on boot
 *  is cheap at this row count and lets `isArchived` skip already-done
 *  weeks across restarts without re-deriving from many small files. */
export function isArchived(org: string, weekStart: string, baseDir?: string): boolean {
  if (!seeded) { seedSeen(githubActivityDir(baseDir)); seeded = true; }
  return archivedKeys.has(`${weekStart}|${org}`);
}

export function archiveGithubActivity(records: GithubActivityRecord[], baseDir?: string, nowMs?: number): number {
  const dir = githubActivityDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) { seedSeen(dir); seeded = true; }
  const fresh = records.filter((r) => r.key && (r.mergedPRs !== null || r.commits !== null) && !archivedKeys.has(r.key));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const fp = path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`);
    fs.appendFileSync(fp, fresh.map((r) => JSON.stringify(r)).join("\n") + "\n");
    fresh.forEach((r) => archivedKeys.add(r.key));
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] github_activity archive:", e?.message || e);
    return 0;
  }
}

// ── cache + poll ─────────────────────────────────────────────────────────────

let cache: { at: number; records: GithubActivityRecord[] } | null = null;
let polling = false;

export function latestGithubActivity(): { at: number; records: GithubActivityRecord[] } | null {
  return cache;
}

export async function refreshGithubActivityCache(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<void> {
  try {
    const records = await fetchGithubActivity(WATCHLIST, fetchImpl, nowMs, {
      skip: (key) => {
        const [weekStart, org] = key.split("|");
        return isArchived(org, weekStart);
      },
    });
    if (records.length) {
      const merged = cache ? [...cache.records.filter((r) => !records.some((n) => n.key === r.key)), ...records] : records;
      cache = { at: Date.now(), records: merged.slice(-WATCHLIST.length * 8) }; // ~8 weeks rolling
    } else if (!cache) {
      cache = { at: Date.now(), records: [] };
    }
    try { archiveGithubActivity(records, undefined, nowMs); } catch {}
  } catch (e: any) {
    console.error("[datacore] github_activity refresh:", e?.message || e);
  }
}

/** 24h check — cheap even at daily cadence because `isArchived` skips any
 *  org/week already written (see header): the real GitHub-API cost only
 *  fires once, in the week after it completes, however many times this
 *  container redeploys in between. A pure 7-day timer would instead risk
 *  losing a whole week to one transient failure with no retry for days. */
export function bootGithubActivityPoll(intervalMs = 24 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshGithubActivityCache();
  setInterval(() => { refreshGithubActivityCache(); }, intervalMs).unref?.();
}
