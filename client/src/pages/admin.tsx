import { useCallback, useEffect, useState } from "react";
import { Link } from "wouter";
import { apiRequest } from "@/lib/queryClient";

/**
 * /admin — owner-only "Signups & visits" (human-directed 2026-10-13).
 * Reads GET /api/admin/site-report (server/siteReport.ts), which sits
 * behind auth.ts requireOwner: account + waitlist emails exist ONLY on
 * this page. One request decides the state — 401 signed out, 403 not the
 * owner, 200 the report — so the page never needs a second auth call.
 */

interface AccountEntry {
  email: string;
  joinedAt: string;
  role: string;
  tier: string;
  subscriptionStatus: string | null;
  isOwner: boolean;
  activeApiKeys: number;
  apiKeyLastUsedAt: number | null;
  signedInLast24h: boolean;
}
interface WaitlistEntry { at: number; email: string; source: string }
interface VisitDay { day: string; views: number; uniques: number; browserViews: number; browserUniques: number; capped: boolean }
interface SiteReport {
  generatedAt: string;
  accounts: AccountEntry[];
  waitlist: WaitlistEntry[];
  visits: {
    /** "history": the persisted daily ledger; "memory": since the last restart. */
    source: "history" | "memory";
    since: string;
    totalPageViews: number;
    uniquesToday: number;
    uniquesByDay: Record<string, number>;
    topPages: Array<{ route: string; count: number }>;
    days?: VisitDay[];
    browserUniquesToday?: number;
  };
  notes: string[];
}

type LoadState =
  | { kind: "loading" }
  | { kind: "signed-out" }
  | { kind: "not-owner" }
  | { kind: "error" }
  | { kind: "ready"; report: SiteReport };

/** apiRequest() throws `"<status>: <body>"`; the status picks the state. */
function statusOf(err: unknown): number {
  const m = /^(\d{3}):/.exec(err instanceof Error ? err.message : "");
  return m ? Number(m[1]) : 0;
}

function fmtDate(v: string | number | null, opts?: Intl.DateTimeFormatOptions): string {
  if (v === null || v === "") return "—";
  const d = new Date(v);
  return Number.isNaN(d.getTime()) ? String(v) : d.toLocaleString(undefined, opts);
}
/** Full timestamp — headers and hover titles. */
const fmtWhen = (v: string | number | null) => fmtDate(v);
/** Table cells: "Aug 2, 2026" — the exact time is on hover. */
const fmtDay = (v: string | number | null) => fmtDate(v, { month: "short", day: "numeric", year: "numeric" });
/** Compact moment: "Oct 13, 3:40 AM". */
const fmtShort = (v: string | number | null) => fmtDate(v, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });

export default function AdminPage() {
  const [state, setState] = useState<LoadState>({ kind: "loading" });
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(() => {
    setRefreshing(true);
    apiRequest("GET", "/api/admin/site-report")
      .then((r) => r.json())
      .then((report: SiteReport) => setState({ kind: "ready", report }))
      .catch((err) => {
        const s = statusOf(err);
        setState(s === 401 ? { kind: "signed-out" } : s === 403 ? { kind: "not-owner" } : { kind: "error" });
      })
      .finally(() => setRefreshing(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  if (state.kind === "loading") {
    return (
      <div className="vt-dev-page vt-admin-page">
        <p className="vt-dev-caption">loading the signup report…</p>
      </div>
    );
  }

  if (state.kind !== "ready") {
    const copy = {
      "signed-out": "Log in with the owner account to see who has signed up and how many people visit.",
      "not-owner": "This page is only available to the site owner.",
      error: "Couldn't load the report — the server didn't answer. Try again in a moment.",
    }[state.kind];
    return (
      <div className="vt-dev-page vt-admin-page">
        <header className="vt-dev-hero">
          <h1>Signups &amp; visits</h1>
          <p>{copy}</p>
        </header>
        {state.kind === "signed-out" && <Link href="/login?next=/admin" className="vt-keys-login-link">Log in →</Link>}
        {state.kind === "error" && (
          <button type="button" className="vt-dev-copy-btn" onClick={load} disabled={refreshing}>
            {refreshing ? "Retrying…" : "Retry"}
          </button>
        )}
      </div>
    );
  }

  const { report } = state;
  const signups = report.accounts.filter((a) => !a.isOwner);
  const days = Object.keys(report.visits.uniquesByDay).sort();
  const maxDay = Math.max(1, ...days.map((d) => report.visits.uniquesByDay[d]));
  const since = fmtShort(report.visits.since);
  const history = report.visits.source === "history";

  return (
    <div className="vt-dev-page vt-admin-page">
      <header className="vt-dev-hero">
        <h1>Signups &amp; visits</h1>
        <p>Who has created an account or joined the API waitlist, and how many people are loading the site. Only you can see this page.</p>
        <div className="vt-admin-meta">
          <span className="vt-dev-caption">Generated {fmtWhen(report.generatedAt)}</span>
          <button type="button" className="vt-dev-copy-btn" onClick={load} disabled={refreshing}>
            {refreshing ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </header>

      <div className="vt-dev-tiers vt-admin-stats">
        <div className="vt-dev-tier">
          <h3>Accounts</h3>
          <p className="vt-dev-price">{signups.length.toLocaleString()}</p>
          <p className="vt-dev-caption">not counting yours</p>
        </div>
        <div className="vt-dev-tier">
          <h3>API waitlist</h3>
          <p className="vt-dev-price">{report.waitlist.length.toLocaleString()}</p>
          <p className="vt-dev-caption">from /developers</p>
        </div>
        <div className="vt-dev-tier">
          <h3>Visitors today</h3>
          <p className="vt-dev-price">{report.visits.uniquesToday.toLocaleString()}</p>
          <p className="vt-dev-caption">
            {report.visits.browserUniquesToday !== undefined
              ? `${report.visits.browserUniquesToday.toLocaleString()} in browsers`
              : "unique, crawlers included"}
          </p>
        </div>
        <div className="vt-dev-tier">
          <h3>Page loads</h3>
          <p className="vt-dev-price">{report.visits.totalPageViews.toLocaleString()}</p>
          <p className="vt-dev-caption">since {since}</p>
        </div>
      </div>

      <section className="vt-dev-section">
        <h2>Accounts <span className="vt-dev-badge">{signups.length} signed up</span></h2>
        {signups.length === 0 && (
          <p className="vt-dev-caption">No one else has created an account yet — the only account is yours.</p>
        )}
        <div className="vt-dev-table-wrap">
          <table className="vt-dev-table">
            <thead>
              <tr><th>Email</th><th>Joined</th><th>Plan</th><th>API keys</th><th>Signed in (24h)</th></tr>
            </thead>
            <tbody>
              {report.accounts.map((a) => (
                <tr key={a.email}>
                  <td>{a.email}{a.isOwner && <span className="vt-dev-badge vt-admin-you">you</span>}</td>
                  <td title={fmtWhen(a.joinedAt)}>{fmtDay(a.joinedAt)}</td>
                  <td>{a.isOwner ? "owner" : a.subscriptionStatus ? `${a.tier} · ${a.subscriptionStatus}` : a.tier}</td>
                  <td title={a.apiKeyLastUsedAt ? `last used ${fmtWhen(a.apiKeyLastUsedAt)}` : undefined}>
                    {a.activeApiKeys > 0
                      ? `${a.activeApiKeys} active${a.apiKeyLastUsedAt ? ` · used ${fmtDay(a.apiKeyLastUsedAt)}` : ""}`
                      : "—"}
                  </td>
                  <td>{a.signedInLast24h ? "yes" : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="vt-dev-section">
        <h2>API waitlist <span className="vt-dev-badge">{report.waitlist.length} joined</span></h2>
        {report.waitlist.length === 0 ? (
          <p className="vt-dev-caption">No one has joined the /developers waitlist yet.</p>
        ) : (
          <div className="vt-dev-table-wrap">
            <table className="vt-dev-table">
              <thead><tr><th>Email</th><th>Joined</th><th>Form</th></tr></thead>
              <tbody>
                {report.waitlist.map((w) => (
                  <tr key={`${w.email}-${w.at}`}><td>{w.email}</td><td title={fmtWhen(w.at)}>{fmtDay(w.at)}</td><td>{w.source}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="vt-dev-section">
        <h2>Visits</h2>
        <p className="vt-dev-caption">
          {history
            ? <>Recorded daily since {since}; nothing earlier exists. </>
            : <>Counted since {since} — this counter resets whenever the site redeploys. </>}
          Visitors are anonymous (IP addresses are hashed before counting) and only full page loads
          count, not clicks inside the app.{" "}
          {history
            ? "“In browsers” leaves out declared bots, headless browsers and scripts — the closest measure of people."
            : "Crawlers are included."}
        </p>
        {report.visits.totalPageViews === 0 ? (
          <p className="vt-dev-caption">No page loads counted yet in this window.</p>
        ) : (
          <>
            <h3>Unique visitors by day</h3>
            <div className="vt-keys-usage-bars" role="img"
              aria-label={days.map((d) => `${d}: ${report.visits.uniquesByDay[d]}`).join(", ")}>
              {days.map((d) => (
                <div key={d} className="vt-keys-usage-bar-col" title={`${d}: ${report.visits.uniquesByDay[d].toLocaleString()} visitors`}>
                  <span className="vt-admin-bar-n">{report.visits.uniquesByDay[d].toLocaleString()}</span>
                  <div className="vt-admin-bar-track">
                    <div className="vt-keys-usage-bar" style={{ height: `${Math.max(2, (report.visits.uniquesByDay[d] / maxDay) * 100)}%` }} />
                  </div>
                  <span className="vt-keys-usage-day">{d.slice(5)}</span>
                </div>
              ))}
            </div>
            {history && report.visits.days && (
              <>
                <h3>By day</h3>
                <div className="vt-dev-table-wrap">
                  <table className="vt-dev-table">
                    <thead><tr><th>Day</th><th>Visitors</th><th>In browsers</th><th>Page loads</th></tr></thead>
                    <tbody>
                      {[...report.visits.days].reverse().slice(0, 14).map((d) => (
                        <tr key={d.day}>
                          <td>{d.day}</td>
                          <td>{d.capped ? "≥ " : ""}{d.uniques.toLocaleString()}</td>
                          <td>{d.capped ? "≥ " : ""}{d.browserUniques.toLocaleString()}</td>
                          <td>{d.views.toLocaleString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
            <h3>Most-loaded pages{history ? " (last 7 days)" : ""}</h3>
            <div className="vt-dev-table-wrap">
              <table className="vt-dev-table">
                <thead><tr><th>Page</th><th>Loads</th></tr></thead>
                <tbody>
                  {report.visits.topPages.map((p) => (
                    <tr key={p.route}><td><code>{p.route}</code></td><td>{p.count.toLocaleString()}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
