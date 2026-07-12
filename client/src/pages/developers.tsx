import { useEffect, useState } from "react";
import { Link } from "wouter";

/**
 * /developers — the API-product developer page (throughput/API directive
 * 2026-07-04; PLATFORM INTEGRATION P2 premium pass 2026-07-11). Honest docs
 * over the live /api/v1 surface: the endpoint reference and every "try it"
 * sample render straight off /api/v1/meta and the live cache each endpoint
 * actually serves — the page can't drift from reality because it never
 * hardcodes a response. Waitlist capture (email only) and the API pricing
 * draft stay clearly marked NOT FOR SALE YET.
 *
 * MONETIZATION TRIPWIRE: nothing here bills, prices, or gates payment —
 * tiers render as a live preview of /api/v1/meta's own limits for human
 * review; keys are invite-only. DESIGN.md applies: theme tokens, three
 * widths, designed empty/error states, no bare spinners.
 */

// Types + shape guard live in lib/apiMetaGuard.ts (pure, unit-tested): the page
// renders straight off the live meta document, so everything it trusts about
// that document's shape is enforced in ONE validated gate instead of scattered
// optional-chaining.
import { parseApiMetaDoc, tierLimit, type ApiMetaDoc, type MetaEndpoint } from "@/lib/apiMetaGuard";

type RunState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "done"; data: any }
  | { status: "error"; err: string };

const TIER_LABEL: Record<string, string> = { dev: "Developer", pro: "Pro", enterprise: "Enterprise" };
const TIER_PRICE: Record<string, string> = { dev: "Free", pro: "TBA", enterprise: "Contact" };

function freshnessLabel(iso: unknown): string {
  const t = typeof iso === "string" ? Date.parse(iso) : NaN;
  if (!Number.isFinite(t)) return "freshness unknown";
  const ms = Date.now() - t;
  if (ms < 0) return "just now";
  const s = Math.floor(ms / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function curlFor(e: MetaEndpoint): string {
  let examplePath = e.path;
  if (e.path.includes(":kind")) examplePath = examplePath.replace(":kind", "aircraft").replace(":id", "a1b2c3") + "?hours=24";
  if (e.path === "/api/v1/meta") return `curl https://voltradeai.com${examplePath}`;
  return `curl -H "x-api-key: YOUR_KEY" \\\n  https://voltradeai.com${examplePath}`;
}

export default function DevelopersPage() {
  const [meta, setMeta] = useState<ApiMetaDoc | null>(null);
  const [metaErr, setMetaErr] = useState<string | null>(null);
  const [runs, setRuns] = useState<Record<string, RunState>>({});
  const [copied, setCopied] = useState<string | null>(null);
  const [email, setEmail] = useState("");
  const [wlState, setWlState] = useState<"idle" | "sending" | "done" | "dup" | "error">("idle");

  useEffect(() => {
    // parseApiMetaDoc rejects non-2xx responses and wrong shapes (JSON error
    // bodies, version-skewed older meta) so they render the designed
    // "reference unavailable" state instead of crashing the page at the first
    // meta.limits access (the pre-v1.0.290 failure mode).
    fetch("/api/v1/meta")
      .then(async (r) => parseApiMetaDoc(r.ok, await r.json()))
      .then((doc) => {
        if (doc) setMeta(doc);
        else setMetaErr("API reference unavailable — retrying on next visit");
      })
      .catch(() => setMetaErr("API reference unavailable — retrying on next visit"));
  }, []);

  const runEndpoint = async (e: MetaEndpoint) => {
    if (!e.preview) return;
    setRuns((r) => ({ ...r, [e.path]: { status: "loading" } }));
    try {
      const res = await fetch(e.preview);
      if (!res.ok) throw new Error(String(res.status)); // an error body is not a sample
      const data = await res.json();
      setRuns((r) => ({ ...r, [e.path]: { status: "done", data } }));
    } catch {
      setRuns((r) => ({ ...r, [e.path]: { status: "error", err: "request failed — the preview route may be warming up" } }));
    }
  };

  const copy = (text: string, id: string) => {
    navigator.clipboard?.writeText(text).then(() => {
      setCopied(id);
      setTimeout(() => setCopied((c) => (c === id ? null : c)), 1600);
    }).catch(() => {});
  };

  const joinWaitlist = async (e: React.FormEvent) => {
    e.preventDefault();
    setWlState("sending");
    try {
      const r = await fetch("/api/waitlist", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ email, source: "developers" }),
      });
      const d = await r.json();
      if (!r.ok) setWlState("error");
      else setWlState(d.status === "duplicate" ? "dup" : "done");
    } catch {
      setWlState("error");
    }
  };

  return (
    <div className="vt-dev-page">
      <header className="vt-dev-hero">
        <h1>The physical-economy archive, as an API</h1>
        <p>
          VolTradeAI records aircraft, vessel, and train positions, port dwell,
          SEC filings, fire detections, and satellite readings — continuously,
          from day one. The archive nobody can back-buy is opening as a data
          API. <strong>Access is invite-only during the preview</strong> — join
          the waitlist below.
        </p>
        <p className="vt-dev-positioning">
          We are not a basemap competitor: the atlas layers rest on the same
          open geospatial foundation as any Earth viewer, and we name every
          source. What a static atlas can&apos;t offer is what this API is
          for — live movement, entity fusion, market-validated signals, and
          programmatic access to all of it. Live vs. coming is stated per
          endpoint below; no claim is made to proprietary imagery.
        </p>
      </header>

      <section className="vt-dev-section">
        <h2>Endpoint explorer <span className="vt-dev-badge">v1 · preview</span></h2>
        {metaErr && <p className="vt-dev-caption">{metaErr}</p>}
        {!meta && !metaErr && <p className="vt-dev-caption">loading the live reference from /api/v1/meta…</p>}
        {meta && (
          <>
            <p className="vt-dev-caption">{meta.auth}</p>
            <div className="vt-dev-endpoints">
              {meta.endpoints.map((e) => {
                const run = runs[e.path] || { status: "idle" as const };
                const curl = curlFor(e);
                return (
                  <div className="vt-dev-endpoint" key={e.path}>
                    <div className="vt-dev-endpoint-head">
                      <span className="vt-dev-method">GET</span>
                      <code>{e.path}</code>
                    </div>
                    <p className="vt-dev-caption">{e.desc}</p>
                    {e.params !== "-" && <p className="vt-dev-caption">params: <code>{e.params}</code></p>}
                    <div className="vt-dev-code-row">
                      <pre className="vt-dev-code vt-dev-code-inline">{curl}</pre>
                      <button type="button" className="vt-dev-copy-btn" onClick={() => copy(curl, e.path)}>
                        {copied === e.path ? "Copied" : "Copy"}
                      </button>
                    </div>
                    {e.preview ? (
                      <div className="vt-dev-try">
                        <button
                          type="button"
                          className="vt-dev-try-btn"
                          disabled={run.status === "loading"}
                          onClick={() => runEndpoint(e)}
                        >
                          {run.status === "loading" ? "Running…" : run.status === "idle" ? "Run live example" : "Run again"}
                        </button>
                        {run.status === "done" && (
                          <>
                            <span className="vt-dev-freshness">{freshnessLabel(run.data?.generated_at)}</span>
                            <pre className="vt-dev-code vt-dev-sample">{JSON.stringify(run.data, null, 2).slice(0, 900)}</pre>
                          </>
                        )}
                        {run.status === "error" && <p className="vt-dev-wl-err">{run.err}</p>}
                      </div>
                    ) : (
                      <p className="vt-dev-caption">
                        No static preview — this endpoint needs a real id from a live feed (e.g. an ICAO24 hex from
                        the Live Map&apos;s aircraft layer). The curl above shows the exact shape to call.
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
            <h3>Coming (honestly gated)</h3>
            <ul className="vt-dev-list">
              {meta.coming_gated.map((g) => <li key={g}>{g}</li>)}
            </ul>
            <h3>License marks travel with every response</h3>
            <div className="vt-dev-table-wrap">
              <table className="vt-dev-table">
                <thead><tr><th>Data family</th><th>License</th><th>Redistribution</th></tr></thead>
                <tbody>
                  {Object.entries(meta.license_marks).map(([k, v]) => (
                    <tr key={k}>
                      <td><code>{k}</code></td>
                      <td>{v.license}</td>
                      <td>{v.resell === "ok" ? "permitted with attribution" : v.resell === "share-alike" ? "ODbL share-alike terms" : "conditional — ask us"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="vt-dev-caption">{meta.disclaimer}</p>
          </>
        )}
      </section>

      <section className="vt-dev-section">
        <h2>Pricing <span className="vt-dev-badge vt-dev-badge-soon">preview — not for sale yet</span></h2>
        <p className="vt-dev-caption">
          Drafted for review. Nothing is purchasable today; no billing exists on
          this site for API access. Limits below are read live from the API's
          own configuration — not hardcoded copy — so this table can&apos;t drift
          from what the API actually enforces.
        </p>
        <div className="vt-dev-tiers">
          {(["dev", "pro", "enterprise"] as const).map((tier) => {
            const lim = meta ? tierLimit(meta, tier) : null;
            return (
              <div className={`vt-dev-tier${tier === "pro" ? " vt-dev-tier-hi" : ""}`} key={tier}>
                <h3>{TIER_LABEL[tier]}</h3>
                <p className="vt-dev-price">{TIER_PRICE[tier]}</p>
                {lim ? (
                  <ul>
                    <li>{lim.perMinute.toLocaleString()} requests/min</li>
                    <li>{lim.perDay.toLocaleString()} requests/day</li>
                    <li>{tier === "enterprise" ? "Bulk archive access + custom analytics" : tier === "pro" ? "All stats endpoints + email support" : "Position tracks + archive stats"}</li>
                    {tier === "dev" && <li>Attribution required</li>}
                  </ul>
                ) : (
                  // meta absent OR this tier missing from the live document —
                  // honest gap, never invented numbers (the copy above promises
                  // these are read live from the API's own configuration).
                  <p className="vt-dev-caption">{meta ? "limits unavailable for this tier" : "loading live limits…"}</p>
                )}
              </div>
            );
          })}
        </div>
      </section>

      <section className="vt-dev-section" id="waitlist">
        <h2>Join the waitlist — or get a preview key now</h2>
        <p className="vt-dev-caption">
          Already have an account? Skip the wait —{" "}
          <Link href="/apikeys">generate a free preview key instantly</Link>{" "}
          at the same limits shown above. No account yet? Leave an email below
          and we'll write when broader access opens. Nothing else is
          collected; no payment details exist here.
        </p>
        {wlState === "done" ? (
          <p className="vt-dev-wl-ok">You're on the list — we'll write when preview keys open up.</p>
        ) : wlState === "dup" ? (
          <p className="vt-dev-wl-ok">Already on the list — nothing more to do.</p>
        ) : (
          <form className="vt-dev-wl" onSubmit={joinWaitlist}>
            <input
              type="email" required placeholder="you@company.com" value={email}
              aria-label="Email for API waitlist"
              onChange={(e) => setEmail(e.target.value)}
            />
            <button type="submit" disabled={wlState === "sending"}>
              {wlState === "sending" ? "Joining…" : "Join waitlist"}
            </button>
          </form>
        )}
        {wlState === "error" && <p className="vt-dev-wl-err">Couldn't save that — try again in a moment.</p>}
      </section>

      <footer className="vt-dev-footer">
        <Link href="/app#/data">Explore the live map →</Link>
      </footer>
    </div>
  );
}
