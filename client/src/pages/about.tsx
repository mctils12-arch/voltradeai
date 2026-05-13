import MarketingPage from "./_MarketingPage";

export default function AboutPage() {
  return (
    <MarketingPage
      title="About VolTradeAI"
      subtitle="An information platform for individual investors. Not a recommendation engine, not a broker. The variables most platforms hide, surfaced before you make a decision."
    >
      {/* ===================================================================
          FOUNDER STORY — real content
      =================================================================== */}
      <section style={{ marginBottom: 64 }}>
        <h2 style={sectionTitle}>Why this exists</h2>
        <p style={para}>
          I built the bot for myself first. Self-taught quant. Years of trading my
          own account, made real money doing it, lost some too, and along the way
          figured out the thing nobody at retail talks about:{" "}
          <strong style={strong}>most trading is gambling — even when it works.</strong>{" "}
          People rip a good month, declare themselves smart, and then give it back
          over the next three. Not because they're dumb. Because the data they
          actually need to make a decision isn't in front of them, and the
          platforms they use are built to keep it that way.
        </p>
        <p style={para}>
          Everything you need to <em>actually</em> evaluate a position is online —
          volatility surfaces, options flow, sector rotation, options dealer
          positioning, macro context, valuation against history, tax exposure. The
          data exists. But it's spread across fifteen sites, costs hundreds of
          dollars a month at the professional level, and most people don't have
          the time, the math, or the patience to assemble it before clicking buy.
          So they don't. They click anyway. And the casino wins.
        </p>
        <p style={para}>
          Meanwhile, institutional desks have been running this kind of analysis
          for decades, and a new wave of hedge funds is running bots a hundred
          times more sophisticated than what you'd ever see on Wealthfront.{" "}
          <strong style={strong}>None of that gets offered to retail.</strong>{" "}
          The economics don't work — the fee model demands an account size most
          people don't have, and the regulatory cost of selling it to small
          investors is brutal. So the gap between what professionals see and what
          retail sees keeps widening. That gap is the casino edge.
        </p>
        <p style={para}>
          VolTradeAI is what I built to close that gap for myself. Fourteen
          variables, weighed against each other, on any ticker. The same kind of
          read a quant desk runs in five seconds — built so a person who's never
          opened a Bloomberg terminal can look at it and actually understand
          what's happening. After a couple of years running it on my own
          accounts, I started letting other people use it. That's the company.
        </p>
      </section>

      {/* ===================================================================
          THE BROADER POINT — the casino frame
      =================================================================== */}
      <section style={{ marginBottom: 64 }}>
        <h2 style={sectionTitle}>You're the product. We're not selling you.</h2>
        <p style={para}>
          Here's the thing nobody at Robinhood will tell you out loud: in today's
          retail brokerage model, <strong style={strong}>you're not the customer
          — you're the inventory.</strong> Your orders get sold to high-frequency
          trading firms (payment for order flow), who make money on the spread.
          The cleaner the UI, the more options you trade, the more 0DTE you buy,
          the more they make. None of those incentives line up with you doing
          well over time.
        </p>
        <p style={para}>
          The gamification — the confetti, the streak counters, the push
          notifications, the leaderboards — isn't an accident. It's casino
          architecture applied to securities. The SEC has been writing about it.
          Studies have been done on it. Nobody fixed it because the model
          prints money.
        </p>
        <p style={para}>
          VolTradeAI is the other side of that. <strong style={strong}>You pay
          for the software. That's the entire transaction.</strong> No order
          flow. No data resale. No "free" trades that cost you more than you
          realize. No streaks, no confetti, no nudges to trade more. The product
          works whether you trade once a month or once a year — and frankly, the
          users who do best with it tend to trade less.
        </p>
      </section>

      {/* ===================================================================
          INFORMATIONAL vs ADVISORY — explicit positioning, important legally
      =================================================================== */}
      <section style={{ marginBottom: 64 }}>
        <h2 style={sectionTitle}>What VolTradeAI is — and isn't</h2>
        <p style={para}>
          <strong style={strong}>VolTradeAI is software.</strong> It surfaces
          information: live prices, volatility surfaces, options flow,
          sentiment, sector and macro context, fundamental ratios, and the
          relationships between all of them. It does the math so you don't
          have to dig for the inputs every time you look at a ticker.
        </p>
        <p style={para}>
          <strong style={strong}>VolTradeAI is not</strong> a registered
          investment advisor. We do not manage your money. We do not tell you
          what to buy or sell. We don't take a percentage of your account. The
          decisions are yours — we just give you better inputs to make them with.
        </p>
        <p style={para}>
          The algorithm — the "bot" you'll see mentioned across the site — is
          an internal research tool we run on our own accounts. We do not
          currently offer automated trading on customer accounts. When and if
          we do, that capability will sit behind regulatory registration and a
          clear separate agreement.
        </p>
      </section>

      {/* ===================================================================
          HOW WE MAKE MONEY
      =================================================================== */}
      <section style={{ marginBottom: 64 }}>
        <h2 style={sectionTitle}>How we make money</h2>
        <p style={para}>
          A flat monthly subscription. <strong style={strong}>$30/month</strong> or
          <strong style={strong}> $300/year</strong> for Pro. The free tier
          is genuinely free — no credit card, no data resale, no upsell traps.
          We make money when our paid users get enough value to stay.
        </p>
        <p style={para}>
          We don't take payment for order flow. We don't sell user data. We
          don't promote tickers. Our incentives are aligned with one thing:
          building software that's worth what people pay for it.
        </p>
      </section>

      {/* ===================================================================
          THE NEWSLETTER
      =================================================================== */}
      <section style={{ marginBottom: 64 }}>
        <h2 style={sectionTitle}>The newsletter</h2>
        <p style={para}>
          Every Sunday evening, we send a free informational read on the week
          ahead — what we're watching in the market, the geopolitical and
          sector context that's actually driving prices, and how specific
          companies are positioned within that backdrop.
        </p>
        <p style={para}>
          It's information, not advice. We don't tell you what to buy. We
          tell you what's happening and why it matters, with the math behind
          it. Subscribe at <a href="/newsletter" style={link}>the newsletter page</a>.
        </p>
      </section>

      {/* ===================================================================
          DISCLAIMERS — important for legal posture, kept brief
      =================================================================== */}
      <section style={{ paddingTop: 16, borderTop: "1px solid rgba(120,165,220,0.16)" }}>
        <p style={{ ...para, fontSize: 13, color: "#6680a0" }}>
          Nothing on this site or in our communications constitutes investment
          advice or a recommendation to buy or sell any security. All content
          is informational and educational. Past performance does not predict
          future results. Markets carry risk; you can lose money. Talk to a
          licensed financial professional before making investment decisions
          based on anything you read here.
        </p>
      </section>
    </MarketingPage>
  );
}

const sectionTitle: React.CSSProperties = {
  fontSize: 24,
  fontWeight: 700,
  marginBottom: 18,
  letterSpacing: "-0.015em",
  color: "#eef3fb",
};
const para: React.CSSProperties = {
  fontSize: 17,
  lineHeight: 1.7,
  color: "#b3c2d8",
  marginBottom: 16,
};
const strong: React.CSSProperties = {
  color: "#eef3fb",
  fontWeight: 600,
};
const link: React.CSSProperties = {
  color: "#4d9fff",
  textDecoration: "none",
};
