import MarketingPage from "./_MarketingPage";

export default function CareersPage() {
  return (
    <MarketingPage
      title="Careers"
      subtitle="Help us build decision intelligence for individual investors."
    >
      <section style={{ marginBottom: 48 }}>
        <h2 style={sectionTitle}>How we work</h2>
        <p style={para}>
          {/* TODO: replace with real values / culture content */}
          Small, distributed team. Async-first. Bias toward shipping. Strong opinions, loosely held — and the math has to back them up.
        </p>
        <p style={para}>
          We hire generalists who can move between research, engineering, and design without losing their mind.
        </p>
      </section>

      {/* Open roles list (empty for now — easy to extend later) */}
      <section style={{ marginBottom: 48 }}>
        <h2 style={sectionTitle}>Open roles</h2>
        <div style={{
          padding: 24,
          background: "rgba(15,29,51,0.4)",
          border: "1px dashed rgba(120,165,220,0.16)",
          borderRadius: 12,
          color: "#6680a0",
          fontSize: 15,
        }}>
          No open roles right now. Reach out anyway if you think you should be here.
        </div>
      </section>

      <section>
        <h2 style={sectionTitle}>Reach out</h2>
        <p style={para}>
          Send your background, work you're proud of, and what you'd want to do here.
        </p>
        <a
          href="mailto:careers@voltradeai.com"
          style={{
            display: "inline-flex",
            alignItems: "center",
            padding: "12px 20px",
            background: "#4d9fff",
            color: "#0a1628",
            borderRadius: 8,
            fontWeight: 600,
            fontSize: 15,
            textDecoration: "none",
            gap: 6,
          }}
        >
          careers@voltradeai.com →
        </a>
      </section>
    </MarketingPage>
  );
}

const sectionTitle: React.CSSProperties = {
  fontSize: 22,
  fontWeight: 600,
  marginBottom: 16,
  letterSpacing: "-0.01em",
};
const para: React.CSSProperties = {
  fontSize: 17,
  lineHeight: 1.65,
  color: "#b3c2d8",
  marginBottom: 16,
};
