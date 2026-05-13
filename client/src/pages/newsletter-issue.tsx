import { useEffect, useState } from "react";
import { useRoute, Link } from "wouter";
import MarketingPage from "./_MarketingPage";

interface IssueFull {
  slug: string;
  title: string;
  subtitle?: string;
  date: string;
  excerpt?: string;
  hero?: string;
  html: string;
}

export default function NewsletterIssuePage() {
  const [, params] = useRoute("/newsletter/:slug");
  const slug = params?.slug;
  const [issue, setIssue] = useState<IssueFull | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!slug) return;
    setIssue(null);
    setError(null);
    fetch(`/api/newsletter/issues/${encodeURIComponent(slug)}`)
      .then((r) => {
        if (r.status === 404) throw new Error("not_found");
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: IssueFull) => {
        setIssue(data);
        // Update document title for SEO + better tab labels
        document.title = `${data.title} — VolTradeAI Newsletter`;
      })
      .catch((err) => {
        if (err.message === "not_found") setError("not_found");
        else setError("load_failed");
      });
  }, [slug]);

  // Loading state
  if (!issue && !error) {
    return (
      <MarketingPage title="Loading…" subtitle="">
        <p style={{ color: "#6680a0" }}>Loading issue…</p>
      </MarketingPage>
    );
  }

  // 404
  if (error === "not_found") {
    return (
      <MarketingPage title="Issue not found" subtitle="The newsletter you're looking for doesn't exist.">
        <p style={{ marginBottom: 24, color: "#b3c2d8" }}>
          It may have been moved or the link is wrong.
        </p>
        <Link href="/newsletter" style={{
          color: "#4d9fff",
          textDecoration: "none",
          fontWeight: 500,
        }}>
          ← Back to all issues
        </Link>
      </MarketingPage>
    );
  }

  // Error state
  if (error) {
    return (
      <MarketingPage title="Couldn't load this issue" subtitle="">
        <p style={{ color: "#fbb24c" }}>Something went wrong. Try refreshing.</p>
      </MarketingPage>
    );
  }

  if (!issue) return null;

  const formattedDate = issue.date
    ? new Date(issue.date + "T00:00:00").toLocaleDateString("en-US", {
        weekday: "long", year: "numeric", month: "long", day: "numeric",
      })
    : "";

  return (
    <MarketingPage title={issue.title} subtitle={issue.subtitle}>
      {/* Date / back link */}
      <div style={{ marginBottom: 32, display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
        <Link href="/newsletter" style={{
          fontSize: 14,
          color: "#4d9fff",
          textDecoration: "none",
        }}>
          ← All issues
        </Link>
        {formattedDate && (
          <span style={{
            fontFamily: "Geist Mono, monospace",
            fontSize: 12,
            letterSpacing: "0.12em",
            color: "#6680a0",
            textTransform: "uppercase",
          }}>
            {formattedDate}
          </span>
        )}
      </div>

      {/* Hero image */}
      {issue.hero && (
        <img
          src={issue.hero}
          alt=""
          style={{
            width: "100%",
            height: "auto",
            borderRadius: 12,
            marginBottom: 40,
            display: "block",
          }}
        />
      )}

      {/* Markdown HTML — scoped class for styling */}
      <article
        className="newsletter-prose"
        dangerouslySetInnerHTML={{ __html: issue.html }}
        style={proseStyle}
      />

      {/* CSS for markdown content rendered above */}
      <style>{proseCss}</style>

      {/* Bottom subscribe nudge for new visitors */}
      <section style={{
        marginTop: 56,
        padding: 28,
        background: "rgba(77,159,255,0.06)",
        border: "1px solid rgba(77,159,255,0.2)",
        borderRadius: 12,
      }}>
        <h3 style={{ margin: "0 0 8px 0", fontSize: 18, fontWeight: 600 }}>
          Get the next one in your inbox.
        </h3>
        <p style={{ margin: "0 0 16px 0", fontSize: 14, color: "#b3c2d8" }}>
          Free, no spam, unsubscribe in one click.
        </p>
        <Link href="/newsletter" style={{
          display: "inline-block",
          padding: "10px 20px",
          background: "#4d9fff",
          color: "#0a1628",
          fontWeight: 600,
          fontSize: 14,
          borderRadius: 8,
          textDecoration: "none",
        }}>
          Subscribe →
        </Link>
      </section>
    </MarketingPage>
  );
}

const proseStyle: React.CSSProperties = {
  fontSize: 17,
  lineHeight: 1.7,
  color: "#dde4f0",
};

// CSS for rendered markdown — scoped to .newsletter-prose
const proseCss = `
  .newsletter-prose h1 { font-size: 32px; font-weight: 700; letter-spacing: -0.02em; margin: 48px 0 16px; color: #eef3fb; }
  .newsletter-prose h2 { font-size: 26px; font-weight: 700; letter-spacing: -0.015em; margin: 44px 0 14px; color: #eef3fb; }
  .newsletter-prose h3 { font-size: 20px; font-weight: 600; letter-spacing: -0.01em; margin: 32px 0 12px; color: #eef3fb; }
  .newsletter-prose h4 { font-size: 16px; font-weight: 600; margin: 24px 0 10px; color: #4d9fff; font-family: 'Geist Mono', monospace; text-transform: uppercase; letter-spacing: 0.08em; }
  .newsletter-prose p { margin: 0 0 18px 0; color: #b3c2d8; }
  .newsletter-prose strong { color: #eef3fb; font-weight: 600; }
  .newsletter-prose em { color: #b3c2d8; font-style: italic; }
  .newsletter-prose a { color: #4d9fff; text-decoration: underline; text-underline-offset: 2px; }
  .newsletter-prose a:hover { color: #7cc4ff; }
  .newsletter-prose ul, .newsletter-prose ol { padding-left: 24px; margin: 0 0 18px; color: #b3c2d8; }
  .newsletter-prose li { margin-bottom: 8px; }
  .newsletter-prose blockquote { border-left: 3px solid #4d9fff; padding-left: 16px; margin: 24px 0; font-style: italic; color: #b3c2d8; }
  .newsletter-prose code { font-family: 'Geist Mono', monospace; font-size: 14px; padding: 2px 6px; background: rgba(77,159,255,0.1); border-radius: 4px; color: #7cc4ff; }
  .newsletter-prose pre { background: #0a1628; border: 1px solid rgba(120,165,220,0.16); border-radius: 8px; padding: 16px; overflow-x: auto; margin: 20px 0; }
  .newsletter-prose pre code { background: transparent; padding: 0; color: #dde4f0; }
  .newsletter-prose img { max-width: 100%; height: auto; border-radius: 8px; margin: 24px 0; display: block; }
  .newsletter-prose hr { border: none; border-top: 1px solid rgba(120,165,220,0.16); margin: 40px 0; }
  .newsletter-prose table { border-collapse: collapse; margin: 20px 0; width: 100%; }
  .newsletter-prose th, .newsletter-prose td { padding: 10px 14px; border-bottom: 1px solid rgba(120,165,220,0.16); text-align: left; }
  .newsletter-prose th { color: #eef3fb; font-weight: 600; font-size: 14px; letter-spacing: 0.04em; text-transform: uppercase; }
`;
