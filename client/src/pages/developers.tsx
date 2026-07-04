import { useEffect, useState } from "react";
import { Link } from "wouter";

/**
 * /developers — the API-product developer page (throughput/API directive
 * 2026-07-04, part 2). Honest docs over the live /api/v1 surface: endpoint
 * reference rendered from /api/v1/meta (the API documents itself — the page
 * can't drift from reality), a live sample from real data, license marks
 * shown as they travel with responses, waitlist capture (email only), and
 * the API pricing draft clearly marked NOT FOR SALE YET.
 *
 * MONETIZATION TRIPWIRE: nothing here bills, prices, or gates payment —
 * tiers render as a preview for human review; keys are invite-only.
 * DESIGN.md applies: theme tokens, three widths, designed empty/error
 * states, no bare spinners.
 */

interface MetaEndpoint { path: string; params: string; desc: string }
interface ApiMetaDoc {
  version: string;
  auth: string;
  endpoints: MetaEndpoint[];
  coming_gated: string[];
  limits: Record<string, { perMinute: number; perDay: number }>;
  license_marks: Record<string, { license: string; attribution: string; resell: string }>;
  disclaimer: string;
}

export default function DevelopersPage() {
  const [meta, setMeta] = useState<ApiMetaDoc | null>(null);
  const [metaErr, setMetaErr] = useState<string | null>(null);
  const [sample, setSample] = useState<any>(null);
  const [email, setEmail] = useState("");
  const [wlState, setWlState] = useState<"idle" | "sending" | "done" | "dup" | "error">("idle");

  useEffect(() => {
    fetch("/api/v1/meta").then((r) => r.json()).then(setMeta)
      .catch(() => setMetaErr("API reference unavailable — retrying on next visit"));
    fetch("/api/data/archive/stats").then((r) => r.json()).then(setSample).catch(() => {});
  }, []);

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
      </header>

      <section className="vt-dev-section">
        <h2>Live right now</h2>
        {sample ? (
          <div className="vt-dev-live">
            {(sample.kinds || sample.streams || []).length > 0 || sample.aircraft || sample.vessels ? (
              <pre className="vt-dev-code">{JSON.stringify(sample, null, 2).slice(0, 1200)}</pre>
            ) : (
              <pre className="vt-dev-code">{JSON.stringify(sample, null, 2).slice(0, 1200)}</pre>
            )}
            <p className="vt-dev-caption">
              Real response from <code>/api/data/archive/stats</code> — the recording metadata of the position archive, fetched by this page just now.
            </p>
          </div>
        ) : (
          <p className="vt-dev-caption">archive stats loading — the feed keeps recording either way</p>
        )}
      </section>

      <section className="vt-dev-section">
        <h2>Endpoint reference <span className="vt-dev-badge">v1 · preview</span></h2>
        {metaErr && <p className="vt-dev-caption">{metaErr}</p>}
        {meta && (
          <>
            <p className="vt-dev-caption">{meta.auth}</p>
            <div className="vt-dev-table-wrap">
              <table className="vt-dev-table">
                <thead><tr><th>Endpoint</th><th>Params</th><th>What it returns</th></tr></thead>
                <tbody>
                  {meta.endpoints.map((e) => (
                    <tr key={e.path}>
                      <td><code>{e.path}</code></td>
                      <td>{e.params}</td>
                      <td>{e.desc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <h3>Coming (honestly gated)</h3>
            <ul className="vt-dev-list">
              {meta.coming_gated.map((g) => <li key={g}>{g}</li>)}
            </ul>
            <h3>Example</h3>
            <pre className="vt-dev-code">{`curl -H "x-api-key: YOUR_KEY" \\
  https://voltradeai.com/api/v1/stats/portdwell`}</pre>
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
          this site for API access. Numbers are placeholders until launch.
        </p>
        <div className="vt-dev-tiers">
          <div className="vt-dev-tier">
            <h3>Developer</h3>
            <p className="vt-dev-price">Free</p>
            <ul><li>60 requests/min</li><li>10k requests/day</li><li>Position tracks + archive stats</li><li>Attribution required</li></ul>
          </div>
          <div className="vt-dev-tier vt-dev-tier-hi">
            <h3>Pro</h3>
            <p className="vt-dev-price">TBA</p>
            <ul><li>600 requests/min</li><li>100k requests/day</li><li>All stats endpoints</li><li>Email support</li></ul>
          </div>
          <div className="vt-dev-tier">
            <h3>Enterprise</h3>
            <p className="vt-dev-price">Contact</p>
            <ul><li>Custom limits</li><li>Bulk archive access</li><li>Custom analytics</li></ul>
          </div>
        </div>
      </section>

      <section className="vt-dev-section" id="waitlist">
        <h2>Join the waitlist</h2>
        <p className="vt-dev-caption">
          The archive will open. API access to our live and historical data is
          coming — leave an email and you're first in line. Nothing else is
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
