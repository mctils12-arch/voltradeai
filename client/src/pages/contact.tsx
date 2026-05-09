import { useState } from "react";
import MarketingPage from "./_MarketingPage";

export default function ContactPage() {
  const [submitted, setSubmitted] = useState(false);

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = e.currentTarget;
    const name = (form.elements.namedItem("name") as HTMLInputElement).value;
    const email = (form.elements.namedItem("email") as HTMLInputElement).value;
    const message = (form.elements.namedItem("message") as HTMLTextAreaElement).value;
    // Until a /api/contact backend endpoint is wired, fall back to mailto.
    // This guarantees the message reaches you even with no server route in place.
    const subject = encodeURIComponent(`VolTradeAI contact form — ${name}`);
    const body = encodeURIComponent(`From: ${name} <${email}>\n\n${message}`);
    window.location.href = `mailto:hello@voltradeai.com?subject=${subject}&body=${body}`;
    setSubmitted(true);
  }

  return (
    <MarketingPage
      title="Contact"
      subtitle="Questions, feedback, partnerships, press — get in touch."
    >
      {/* Quick contact options */}
      <section style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
        gap: 16,
        marginBottom: 56,
      }}>
        <ContactCard
          label="General"
          email="hello@voltradeai.com"
          desc="Product questions, feedback, anything else."
        />
        <ContactCard
          label="Support"
          email="support@voltradeai.com"
          desc="Account or billing issues."
        />
        <ContactCard
          label="Press / partners"
          email="press@voltradeai.com"
          desc="Media, partnerships, integrations."
        />
      </section>

      {/* Optional message form (mailto-backed for now) */}
      <section style={{ marginBottom: 32 }}>
        <h2 style={{ fontSize: 22, fontWeight: 600, marginBottom: 16, letterSpacing: "-0.01em" }}>
          Or send a message
        </h2>
        {submitted ? (
          <div style={{
            padding: 24,
            background: "rgba(74,222,128,0.08)",
            border: "1px solid rgba(74,222,128,0.3)",
            borderRadius: 12,
            color: "#4ade80",
            fontSize: 15,
          }}>
            Thanks — your email client should be opening with your message ready to send.
          </div>
        ) : (
          <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 14, maxWidth: 600 }}>
            <input
              type="text"
              name="name"
              placeholder="Your name"
              required
              style={inputStyle}
            />
            <input
              type="email"
              name="email"
              placeholder="you@email.com"
              required
              style={inputStyle}
            />
            <textarea
              name="message"
              placeholder="What's on your mind?"
              required
              rows={6}
              style={{ ...inputStyle, resize: "vertical", fontFamily: "inherit" }}
            />
            <button
              type="submit"
              style={{
                alignSelf: "flex-start",
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
              Send →
            </button>
          </form>
        )}
      </section>
    </MarketingPage>
  );
}

function ContactCard({ label, email, desc }: { label: string; email: string; desc: string }) {
  return (
    <a href={`mailto:${email}`} style={{
      display: "block",
      padding: 20,
      background: "rgba(15,29,51,0.4)",
      border: "1px solid rgba(120,165,220,0.16)",
      borderRadius: 12,
      textDecoration: "none",
      transition: "border-color 0.15s, background 0.15s",
    }}
    onMouseEnter={(e) => {
      (e.currentTarget as HTMLAnchorElement).style.borderColor = "rgba(77,159,255,0.4)";
      (e.currentTarget as HTMLAnchorElement).style.background = "rgba(20,37,67,0.6)";
    }}
    onMouseLeave={(e) => {
      (e.currentTarget as HTMLAnchorElement).style.borderColor = "rgba(120,165,220,0.16)";
      (e.currentTarget as HTMLAnchorElement).style.background = "rgba(15,29,51,0.4)";
    }}>
      <div style={{ fontFamily: "Geist Mono, monospace", fontSize: 11, letterSpacing: "0.16em", color: "#6680a0", textTransform: "uppercase", marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ fontSize: 16, fontWeight: 600, color: "#4d9fff", marginBottom: 6 }}>
        {email}
      </div>
      <div style={{ fontSize: 13, color: "#b3c2d8" }}>
        {desc}
      </div>
    </a>
  );
}

const inputStyle: React.CSSProperties = {
  padding: "12px 16px",
  background: "rgba(5,10,19,0.6)",
  border: "1px solid rgba(120,165,220,0.16)",
  borderRadius: 8,
  color: "#eef3fb",
  fontSize: 15,
  outline: "none",
};
