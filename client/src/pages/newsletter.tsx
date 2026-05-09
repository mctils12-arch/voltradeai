import MarketingPage from "./_MarketingPage";

const BEEHIIV_PUBLICATION_URL = "https://voltradeai.beehiiv.com";
const BEEHIIV_SUBSCRIBE_URL = `${BEEHIIV_PUBLICATION_URL}/subscribe`;

export default function NewsletterPage() {
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
      subtitle="Sunday evenings. One short read. Three names worth watching, and the variables we're weighing on each. Free, forever."
    >
      {/* Subscribe block */}
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
          SUNDAY · 7PM ET
        </div>
        <h2 style={{ fontSize: 28, fontWeight: 700, letterSpacing: "-0.02em", marginBottom: 8 }}>
          Prep for the week ahead.
        </h2>
        <p style={{ fontSize: 16, color: "#b3c2d8", marginBottom: 24, maxWidth: 540 }}>
          Free, every Sunday. No fluff, no spam. One click to unsubscribe.
        </p>

        <form onSubmit={handleSubmit} style={{ display: "flex", gap: 8, flexWrap: "wrap", maxWidth: 520 }}>
          <input
            type="email"
            name="email"
            required
            placeholder="you@email.com"
            style={{
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
            }}
          />
          <button
            type="submit"
            style={{
              padding: "12px 24px",
              background: "#4d9fff",
              color: "#0a1628",
              border: "none",
              borderRadius: 8,
              fontWeight: 600,
              fontSize: 15,
              cursor: "pointer",
              fontFamily: "inherit",
            }}
          >
            Subscribe →
          </button>
        </form>

        <div style={{ marginTop: 16, fontSize: 12, color: "#6680a0", fontFamily: "Geist Mono, monospace", letterSpacing: "0.08em" }}>
          UNSUBSCRIBE ANY TIME
        </div>
      </section>

      {/* What's in each issue */}
      <section style={{ marginBottom: 56 }}>
        <h2 style={{ fontSize: 22, fontWeight: 600, marginBottom: 16, letterSpacing: "-0.01em" }}>
          What's in each issue
        </h2>
        <ul style={{ listStyle: "none", padding: 0, color: "#b3c2d8", fontSize: 16, lineHeight: 1.7 }}>
          <li style={listItem}><strong style={strong}>Market read</strong> — what we're seeing in the regime, sector flows, vol surface</li>
          <li style={listItem}><strong style={strong}>Three names</strong> — tickers we're watching, with the variables that flagged them</li>
          <li style={listItem}><strong style={strong}>One thing to watch</strong> — the macro print, earnings, or event that could move things</li>
          <li style={listItem}><strong style={strong}>Short</strong> — 5 minutes max. We respect your inbox.</li>
        </ul>
      </section>

      {/* Past issues link */}
      <section>
        <h2 style={{ fontSize: 22, fontWeight: 600, marginBottom: 16, letterSpacing: "-0.01em" }}>
          Past issues
        </h2>
        <p style={{ fontSize: 16, color: "#b3c2d8", marginBottom: 16 }}>
          Read what we've sent before — published openly. No subscription required.
        </p>
        <a
          href={BEEHIIV_PUBLICATION_URL}
          target="_blank"
          rel="noopener noreferrer"
          style={{
            color: "#4d9fff",
            textDecoration: "none",
            fontWeight: 500,
            fontSize: 15,
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          Read the archive →
        </a>
      </section>
    </MarketingPage>
  );
}

const listItem: React.CSSProperties = {
  paddingLeft: 20,
  position: "relative",
  marginBottom: 8,
};
const strong: React.CSSProperties = {
  color: "#eef3fb",
  fontWeight: 600,
};
