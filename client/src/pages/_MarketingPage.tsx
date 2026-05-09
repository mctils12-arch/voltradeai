import { ReactNode } from "react";
import { useLocation, Link } from "wouter";

interface MarketingPageProps {
  /** Page title shown in <title> + as H1 */
  title: string;
  /** Optional subtitle shown below the H1 */
  subtitle?: string;
  /** Page-specific content rendered between header and footer */
  children: ReactNode;
}

/**
 * Shared shell for the marketing pages (About, Newsletter, Careers, Contact).
 * Same dark navy theme as the landing, plus a top nav and a minimal footer
 * with links back to the rest of the site.
 *
 * Pages that wrap content in <MarketingPage title="..."> get consistent
 * spacing, colors, and chrome without duplicating navigation HTML.
 */
export default function MarketingPage({ title, subtitle, children }: MarketingPageProps) {
  const [location, navigate] = useLocation();

  return (
    <div style={{
      minHeight: "100vh",
      background: "var(--bg-primary, #050a13)",
      color: "var(--text-primary, #eef3fb)",
      fontFamily: "Geist, -apple-system, system-ui, sans-serif",
      display: "flex",
      flexDirection: "column",
    }}>
      {/* Top nav — keeps marketing pages consistent with the landing */}
      <nav style={{
        padding: "20px 24px",
        borderBottom: "1px solid rgba(120,165,220,0.08)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        position: "sticky",
        top: 0,
        background: "rgba(5,10,19,0.85)",
        backdropFilter: "blur(12px)",
        zIndex: 50,
      }}>
        <Link href="/" style={{ textDecoration: "none", display: "inline-flex", alignItems: "baseline", letterSpacing: "-0.02em" }}>
          <span style={{ color: "#eef3fb", fontWeight: 700, fontSize: 18 }}>vol</span>
          <span style={{ color: "#b3c2d8", fontWeight: 500, fontSize: 18 }}>trade</span>
          <span style={{ color: "#4d9fff", fontWeight: 600, fontSize: 13, marginLeft: 4, fontFamily: "Geist Mono, monospace" }}>/AI</span>
        </Link>
        <div style={{ display: "flex", alignItems: "center", gap: 24, fontSize: 14 }}>
          <Link href="/" style={{ color: "#b3c2d8", textDecoration: "none" }}>Home</Link>
          <Link href="/pricing" style={{ color: "#b3c2d8", textDecoration: "none" }}>Pricing</Link>
          <Link href="/login" style={{
            padding: "8px 16px",
            background: "#4d9fff",
            color: "#0a1628",
            fontWeight: 600,
            borderRadius: 6,
            textDecoration: "none",
          }}>Sign in</Link>
        </div>
      </nav>

      {/* Header band */}
      <header style={{
        padding: "72px 24px 48px",
        borderBottom: "1px solid rgba(120,165,220,0.08)",
        background: "linear-gradient(180deg, rgba(15,29,51,0.4), rgba(5,10,19,0))",
      }}>
        <div style={{ maxWidth: 880, margin: "0 auto" }}>
          <h1 style={{
            fontSize: "clamp(36px, 5.5vw, 64px)",
            fontWeight: 700,
            letterSpacing: "-0.04em",
            lineHeight: 1.05,
            marginBottom: subtitle ? 16 : 0,
          }}>
            {title}
          </h1>
          {subtitle && (
            <p style={{ fontSize: 19, color: "#b3c2d8", lineHeight: 1.55, maxWidth: 640 }}>
              {subtitle}
            </p>
          )}
        </div>
      </header>

      {/* Page content */}
      <main style={{ flex: 1, padding: "56px 24px 80px" }}>
        <div style={{ maxWidth: 880, margin: "0 auto" }}>
          {children}
        </div>
      </main>

      {/* Compact footer */}
      <footer style={{
        padding: "32px 24px",
        borderTop: "1px solid rgba(120,165,220,0.08)",
        fontSize: 13,
        color: "#6680a0",
      }}>
        <div style={{ maxWidth: 880, margin: "0 auto", display: "flex", flexWrap: "wrap", justifyContent: "space-between", gap: 16 }}>
          <span>© {new Date().getFullYear()} VolTradeAI</span>
          <div style={{ display: "flex", gap: 20 }}>
            <Link href="/about" style={{ color: "#6680a0", textDecoration: "none" }}>About</Link>
            <Link href="/newsletter" style={{ color: "#6680a0", textDecoration: "none" }}>Newsletter</Link>
            <Link href="/careers" style={{ color: "#6680a0", textDecoration: "none" }}>Careers</Link>
            <Link href="/contact" style={{ color: "#6680a0", textDecoration: "none" }}>Contact</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
