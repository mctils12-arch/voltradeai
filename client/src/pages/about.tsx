import MarketingPage from "./_MarketingPage";

export default function AboutPage() {
  return (
    <MarketingPage
      title="About VolTradeAI"
      subtitle="Decision intelligence for individual investors. The variables most platforms hide — surfaced before every trade."
    >
      <section style={{ marginBottom: 56 }}>
        <h2 style={sectionTitle}>What we do</h2>
        <p style={para}>
          Most investing platforms either show you the price (Yahoo, Google Finance) or let you trade it (Robinhood, Fidelity).
          Neither tells you whether you should — or what you're really risking when you do.
        </p>
        <p style={para}>
          VolTradeAI weighs every variable that matters — momentum, valuation, volatility regime, options flow, sentiment, sector context, tax exposure — and gives you a clear read on every ticker. Plus an algo that runs systematic strategies for those who want to automate.
        </p>
      </section>

      <section style={{ marginBottom: 56 }}>
        <h2 style={sectionTitle}>Why we built it</h2>
        <p style={para}>
          {/* TODO: replace with real founder story when ready */}
          Placeholder — your story goes here.
        </p>
      </section>

      <section style={{ marginBottom: 56 }}>
        <h2 style={sectionTitle}>How we make money</h2>
        <p style={para}>
          A flat monthly subscription. <strong style={strong}>$30/month</strong> or <strong style={strong}>$300/year</strong> for Pro. The free tier is genuinely free — no credit card, no upsell traps. We make money when our paid users get enough value to stay.
        </p>
      </section>

      <section>
        <h2 style={sectionTitle}>What we don't do</h2>
        <p style={para}>
          We are not a registered investment advisor. We do not manage your money. We do not give personalized financial advice. VolTradeAI is software that surfaces the data and runs the math. Decisions are yours.
        </p>
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
const strong: React.CSSProperties = {
  color: "#eef3fb",
  fontWeight: 600,
};
