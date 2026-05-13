import { useEffect, useState } from "react";
import { Link } from "wouter";
import MarketingPage from "./_MarketingPage";

const BEEHIIV_PUBLICATION_URL = "https://voltradeai.beehiiv.com";
const BEEHIIV_SUBSCRIBE_URL = `${BEEHIIV_PUBLICATION_URL}/subscribe`;

interface IssueMeta {
  slug: string;
  title: string;
  subtitle?: string;
  date: string;
  excerpt?: string;
  hero?: string;
}

export default function NewsletterPage() {
  const [issues, setIssues] = useState<IssueMeta[] | null>(null);
  const [archiveError, setArchiveError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/newsletter/issues")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => setIssues(data.issues || []))
      .catch((err) => {
        console.warn("[Newsletter] archive fetch failed:", err);
        setArchiveError("Archive temporarily unavailable.");
        setIssues([]);
      });
  }, []);

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = e.currentTarget as HTMLFormElement;
    const email = (form.elements.namedItem("email") as HTMLInputElement).value.trim();
    if (!email) return;
    window.location.href = `${BEEHIIV_SUBSCRIBE_URL}?email=${encodeURIComponent(email)}`;
  }

  return (
    <MarketingPage
      title="The VolTradeAI Newsletter"
      subtitle="Information, not advice. What's actually moving the market — with the math behind it."
    >
      {/* SUBSCRIBE BLOCK */}
      <section style={{
        padding: 32,
        background: "rgba(15,29,51,0.5)",
        border: "1px solid rgba(77,159,255,0.2)",
        borderRadius: 14,
        marginBottom: 56,
      }}>
        <div style={{
          display: "inline-block",
          padding: "5px 12px",
          background: "rgba(74,222,128,0.10)",
          border: "1px solid rgba(74,222,128,0.30)",
          borderRadius: 100,
          fontFamily: "Geist Mono, monospace",
          fontSize: 11,
          fontWeight: 600,
          letterSpacing: "0.16em",
          color: "#4ade80",
          marginBottom: 16,
        }}>
          FREE FOREVER
        </div>
        <h2 style={{ fontSize: 28, fontWeight: 700, letterSpacing: "-0.02em", marginBottom: 8 }}>
          Subscribe.
        </h2>
        <p style={{ fontSize: 16, color: "#b3c2d8", marginBottom: 24, maxWidth: 540 }}>
          No fluff, no spam. One click to unsubscribe.
        </p>

        <form onSubmit={handleSubmit} style={{ display: "flex", gap: 8, flexWrap: "wrap", maxWidth: 520 }}>
          <input
            type="email"
            name="email"
            required
            placeholder="you@email.com"
            style={inputStyle}
          />
          <button type="submit" style={subscribeBtn}>
            Subscribe →
          </button>
        </form>

        <div style={metaText}>UNSUBSCRIBE ANY TIME</div>
      </section>

      {/* WHAT YOU'LL GET */}
      <section style={{ marginBottom: 56 }}>
        <h2 style={sectionTitle}>What you'll get</h2>
        <p style={para}>
          Every issue is one short read covering what we're seeing in the
          market, the geopolitical and sector forces actually driving prices,
          and how specific companies are positioned. Built to keep you a step
          ahead — not by predicting the future, but by giving you the context
          most financial news leaves out.
        </p>
        <ul style={listStyle}>
          <li style={listItem}><strong style={strong}>Market read</strong> — the regime, sector flows, vol environment</li>
          <li style={listItem}><strong style={strong}>Context, not calls</strong> — what's happening to specific companies and why, in plain English</li>
          <li style={listItem}><strong style={strong}>What's worth watching</strong> — the macro print, earnings, or event that could move things</li>
          <li style={listItem}><strong style={strong}>Short</strong> — 5 minutes max. We respect your inbox.</li>
        </ul>
      </section>

      {/* WHAT IT ISN'T */}
      <section style={{ marginBottom: 72 }}>
        <h2 style={sectionTitle}>What you <em>won't</em> get</h2>
        <p style={para}>
          Stock picks. Buy/sell recommendations. Hot tips. "Three stocks you
          must own this week." That's not what we do.
        </p>
        <p style={para}>
          VolTradeAI is an information platform. The newsletter is information
          too. We're not a registered investment advisor. We don't manage your
          money. What we share is research and context — what you do with it
          is up to you.
        </p>
      </section>

      {/* ARCHIVE */}
      <section>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 24 }}>
          <h2 style={sectionTitle}>Past issues</h2>
          {issues && issues.length > 0 && (
            <span style={{ fontSize: 13, color: "#6680a0", fontFamily: "Geist Mono, monospace" }}>
              {issues.length} {issues.length === 1 ? "issue" : "issues"}
            </span>
          )}
        </div>

        {issues === null && (
          <p style={{ color: "#6680a0", fontSize: 14 }}>Loading archive…</p>
        )}

        {issues !== null && issues.length === 0 && !archiveError && (
          <p style={{ color: "#6680a0", fontSize: 14 }}>
            The first issue is coming soon. Subscribe above to be the first to see it.
          </p>
        )}

        {archiveError && (
          <p style={{ color: "#fbb24c", fontSize: 14 }}>{archiveError}</p>
        )}

        {issues && issues.length > 0 && (
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
            gap: 20,
          }}>
            {issues.map((issue) => (
              <IssueCard key={issue.slug} issue={issue} />
            ))}
          </div>
        )}
      </section>
    </MarketingPage>
  );
}

function IssueCard({ issue }: { issue: IssueMeta }) {
  const formattedDate = issue.date
    ? new Date(issue.date + "T00:00:00").toLocaleDateString("en-US", {
        month: "short", day: "numeric", year: "numeric",
      })
    : "";

  return (
    <Link
      href={`/newsletter/${issue.slug}`}
      style={{
        display: "block",
        textDecoration: "none",
        padding: 0,
        background: "rgba(15,29,51,0.4)",
        border: "1px solid rgba(120,165,220,0.16)",
        borderRadius: 12,
        overflow: "hidden",
        transition: "border-color 0.15s, transform 0.15s",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLAnchorElement).style.borderColor = "rgba(77,159,255,0.5)";
        (e.currentTarget as HTMLAnchorElement).style.transform = "translateY(-2px)";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLAnchorElement).style.borderColor = "rgba(120,165,220,0.16)";
        (e.currentTarget as HTMLAnchorElement).style.transform = "translateY(0)";
      }}
    >
      {issue.hero && (
        <div style={{
          aspectRatio: "2 / 1",
          background: `url(${issue.hero}) center/cover, #050a13`,
        }} />
      )}
      <div style={{ padding: 20 }}>
        {formattedDate && (
          <div style={{
            fontFamily: "Geist Mono, monospace",
            fontSize: 11,
            letterSpacing: "0.16em",
            color: "#6680a0",
            textTransform: "uppercase",
            marginBottom: 8,
          }}>
            {formattedDate}
          </div>
        )}
        <h3 style={{
          margin: 0,
          fontSize: 17,
          fontWeight: 600,
          color: "#eef3fb",
          letterSpacing: "-0.01em",
          lineHeight: 1.3,
        }}>
          {issue.title}
        </h3>
        {issue.excerpt && (
          <p style={{
            margin: "8px 0 0 0",
            fontSize: 13,
            color: "#b3c2d8",
            lineHeight: 1.55,
            display: "-webkit-box",
            WebkitLineClamp: 3,
            WebkitBoxOrient: "vertical",
            overflow: "hidden",
          }}>
            {issue.excerpt}
          </p>
        )}
        <div style={{
          marginTop: 14,
          fontSize: 13,
          color: "#4d9fff",
          fontWeight: 500,
        }}>
          Read issue →
        </div>
      </div>
    </Link>
  );
}

// ── Styles ──────────────────────────────────────────────────────────────
const inputStyle: React.CSSProperties = {
  flex: 1,
  minWidth: 220,
  padding: "12px 16px",
  background: "rgba(5,10,19,0.6)",
  border: "1px solid rgba(120,165,220,0.16)",
  borderRadius: 8,
  color: "#eef3fb",
  fontSize: 15,
  fontFamily: "inherit",
  outline: "none",
};
const subscribeBtn: React.CSSProperties = {
  padding: "12px 24px",
  background: "#4d9fff",
  color: "#0a1628",
  border: "none",
  borderRadius: 8,
  fontWeight: 600,
  fontSize: 15,
  cursor: "pointer",
  fontFamily: "inherit",
};
const metaText: React.CSSProperties = {
  marginTop: 16,
  fontSize: 12,
  color: "#6680a0",
  fontFamily: "Geist Mono, monospace",
  letterSpacing: "0.08em",
};
const sectionTitle: React.CSSProperties = {
  fontSize: 22,
  fontWeight: 600,
  marginBottom: 16,
  letterSpacing: "-0.01em",
};
const para: React.CSSProperties = {
  fontSize: 16,
  color: "#b3c2d8",
  lineHeight: 1.7,
  marginBottom: 12,
};
const strong: React.CSSProperties = { color: "#eef3fb", fontWeight: 600 };
const listStyle: React.CSSProperties = {
  listStyle: "none",
  padding: 0,
  color: "#b3c2d8",
  fontSize: 16,
  lineHeight: 1.7,
};
const listItem: React.CSSProperties = {
  paddingLeft: 20,
  position: "relative",
  marginBottom: 8,
};
