import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Check, Loader2, ArrowRight, Lock } from "lucide-react";

interface AuthMe {
  authenticated: boolean;
  email?: string;
  tier?: "free" | "pro";
  isOwner?: boolean;
  subscriptionStatus?: string | null;
}

const FEATURES_FREE = [
  "Look up any of 8,000+ US-listed tickers",
  "Real-time price, volatility, options data",
  "Sentiment & news aggregation",
  "Personal watchlists",
  "Market scanner",
];

const FEATURES_PRO = [
  "Everything in Free",
  "Full decision read on every ticker (14 variables)",
  "Tax-aware buy/sell signals (long-term vs short-term)",
  "Portfolio risk monitoring & rebalance alerts",
  "Priority support",
  "Early access to new features",
];

export default function PricingPage() {
  const [, navigate] = useLocation();
  const [me, setMe] = useState<AuthMe | null>(null);
  const [loading, setLoading] = useState(true);
  const [checkoutLoading, setCheckoutLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Annual selected by default — pushes higher LTV plan first (industry standard)
  const [billingPeriod, setBillingPeriod] = useState<"monthly" | "annual">("annual");

  useEffect(() => {
    apiRequest("GET", "/api/auth/me")
      .then((r) => r.json())
      .then((data) => setMe(data))
      .catch(() => setMe({ authenticated: false }))
      .finally(() => setLoading(false));
  }, []);

  // If user just came back from a successful checkout, refresh auth
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("checkout") === "success") {
      queryClient.invalidateQueries({ queryKey: ["/api/auth/me"] });
      // Give Stripe webhook a moment to update tier, then bounce to app
      setTimeout(() => navigate("/app?welcome=pro"), 1500);
    }
  }, [navigate]);

  async function handleUpgrade() {
    if (!me?.authenticated) {
      navigate("/login?next=/pricing");
      return;
    }
    setCheckoutLoading(true);
    setError(null);
    try {
      const r = await apiRequest("POST", "/api/billing/checkout", { plan: billingPeriod });
      const data = await r.json();
      if (data.url) {
        window.location.href = data.url;
      } else {
        setError(data.message || data.error || "Failed to start checkout");
        setCheckoutLoading(false);
      }
    } catch (err: any) {
      setError(err.message || "Network error");
      setCheckoutLoading(false);
    }
  }

  async function handleManage() {
    setCheckoutLoading(true);
    setError(null);
    try {
      const r = await apiRequest("POST", "/api/billing/portal", {});
      const data = await r.json();
      if (data.url) {
        window.location.href = data.url;
      } else {
        setError(data.message || "Failed to open billing portal");
        setCheckoutLoading(false);
      }
    } catch (err: any) {
      setError(err.message || "Network error");
      setCheckoutLoading(false);
    }
  }

  const isPro = me?.tier === "pro";
  const isOwner = me?.isOwner;

  return (
    <div style={{
      minHeight: "100vh",
      background: "var(--bg-primary)",
      color: "var(--text-primary)",
      fontFamily: "var(--font-body)",
      padding: "80px 24px 64px",
    }}>
      <div style={{ maxWidth: 1080, margin: "0 auto" }}>
        {/* Back to home */}
        <a href="/" style={{ color: "#6680a0", fontSize: 14, textDecoration: "none", display: "inline-flex", alignItems: "center", gap: 6, marginBottom: 32 }}>
          ← Back to home
        </a>

        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 64 }}>
          <h1 style={{
            fontSize: "clamp(40px, 6vw, 72px)",
            fontWeight: 700,
            letterSpacing: "-0.04em",
            lineHeight: 1,
            marginBottom: 20,
          }}>
            Free for the read.<br/>
            <span style={{ color: "#4d9fff" }}>Pro for the depth.</span>
          </h1>
          <p style={{ fontSize: 18, color: "#b3c2d8", maxWidth: 600, margin: "0 auto", lineHeight: 1.5 }}>
            Look up any stock for free. Upgrade for full decision intelligence —
            tax math, risk monitoring, every variable that matters.
          </p>
        </div>

        {/* Monthly / Annual toggle */}
        <div style={{ display: "flex", justifyContent: "center", marginBottom: 32 }}>
          <div style={{
            display: "inline-flex",
            padding: 4,
            background: "rgba(15,29,51,0.6)",
            border: "1px solid rgba(120,165,220,0.16)",
            borderRadius: 100,
            gap: 4,
          }}>
            <button
              onClick={() => setBillingPeriod("monthly")}
              style={{
                padding: "8px 18px",
                borderRadius: 100,
                border: "none",
                background: billingPeriod === "monthly" ? "#4d9fff" : "transparent",
                color: billingPeriod === "monthly" ? "#0a1628" : "#b3c2d8",
                fontWeight: 600,
                fontSize: 13,
                cursor: "pointer",
                transition: "background 0.15s, color 0.15s",
              }}
            >
              Monthly
            </button>
            <button
              onClick={() => setBillingPeriod("annual")}
              style={{
                padding: "8px 18px",
                borderRadius: 100,
                border: "none",
                background: billingPeriod === "annual" ? "#4d9fff" : "transparent",
                color: billingPeriod === "annual" ? "#0a1628" : "#b3c2d8",
                fontWeight: 600,
                fontSize: 13,
                cursor: "pointer",
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                transition: "background 0.15s, color 0.15s",
              }}
            >
              Annual
              <span style={{
                padding: "2px 6px",
                borderRadius: 4,
                background: billingPeriod === "annual" ? "rgba(10,22,40,0.25)" : "rgba(74,222,128,0.15)",
                color: billingPeriod === "annual" ? "#0a1628" : "#4ade80",
                fontSize: 10,
                fontWeight: 700,
                letterSpacing: "0.04em",
              }}>
                2 MONTHS FREE
              </span>
            </button>
          </div>
        </div>

        {/* Two-tier grid */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
          gap: 24,
          maxWidth: 800,
          margin: "0 auto",
        }}>
          {/* Free tier */}
          <Tier
            name="Free"
            price="$0"
            period="forever"
            tagline="Everything you need to look up a stock."
            features={FEATURES_FREE}
            ctaLabel={me?.authenticated ? (isPro ? "Downgrade in portal →" : "Current plan") : "Sign up free →"}
            ctaDisabled={me?.authenticated && !isPro}
            ctaSecondary={!isPro}
            onCta={() => {
              if (!me?.authenticated) navigate("/login");
              else if (isPro) handleManage();
            }}
          />

          {/* Pro tier — price + period flip based on billing toggle */}
          <Tier
            name="Pro"
            price={billingPeriod === "annual" ? "$25" : "$30"}
            period="/ month"
            subPrice={billingPeriod === "annual" ? "$300 billed annually · save $60" : undefined}
            tagline="Full decision intelligence on every trade."
            features={FEATURES_PRO}
            highlighted
            ctaLabel={
              isOwner ? "Owner — auto-Pro" :
              isPro ? "Manage subscription →" :
              checkoutLoading ? "Opening Stripe…" :
              billingPeriod === "annual" ? "Get Pro — annual →" : "Get Pro — monthly →"
            }
            ctaDisabled={checkoutLoading || isOwner}
            ctaLoading={checkoutLoading}
            onCta={isPro ? handleManage : handleUpgrade}
          />
        </div>

        {/* Status callouts */}
        {error && (
          <div style={{
            marginTop: 24,
            maxWidth: 800,
            margin: "24px auto 0",
            padding: "12px 16px",
            background: "rgba(255,90,110,0.1)",
            border: "1px solid rgba(255,90,110,0.3)",
            borderRadius: 8,
            color: "#ff5a6e",
            fontSize: 14,
          }}>
            {error}
          </div>
        )}

        {isPro && me?.subscriptionStatus && me.subscriptionStatus !== "active" && (
          <div style={{
            marginTop: 24,
            maxWidth: 800,
            margin: "24px auto 0",
            padding: "12px 16px",
            background: "rgba(251,178,76,0.1)",
            border: "1px solid rgba(251,178,76,0.3)",
            borderRadius: 8,
            color: "#fbb24c",
            fontSize: 14,
          }}>
            Subscription status: <strong>{me.subscriptionStatus}</strong>. Manage in the billing portal.
          </div>
        )}

        {/* Footer note about RIA / ToS */}
        <div style={{
          marginTop: 64,
          padding: "24px 0",
          borderTop: "1px solid rgba(120,165,220,0.16)",
          fontSize: 13,
          color: "#6680a0",
          textAlign: "center",
          lineHeight: 1.65,
        }}>
          <p style={{ marginBottom: 8 }}>
            VolTradeAI provides software tools for individual investors. We are not a registered investment advisor and do not manage your money.
          </p>
          <p>
            Cancel anytime via the customer portal. No questions asked.
          </p>
        </div>
      </div>
    </div>
  );
}

interface TierProps {
  name: string;
  price: string;
  period: string;
  /** Optional small line under the big price ("$300 billed annually · save $60") */
  subPrice?: string;
  tagline: string;
  features: string[];
  highlighted?: boolean;
  ctaLabel: string;
  ctaDisabled?: boolean;
  ctaLoading?: boolean;
  ctaSecondary?: boolean;
  onCta: () => void;
}

function Tier({ name, price, period, subPrice, tagline, features, highlighted, ctaLabel, ctaDisabled, ctaLoading, ctaSecondary, onCta }: TierProps) {
  return (
    <div style={{
      padding: 32,
      background: highlighted ? "rgba(77,159,255,0.04)" : "rgba(15,29,51,0.4)",
      border: `1px solid ${highlighted ? "rgba(77,159,255,0.3)" : "rgba(120,165,220,0.16)"}`,
      borderRadius: 14,
      position: "relative",
      ...(highlighted ? {
        boxShadow: "0 0 60px rgba(77,159,255,0.08), 0 0 0 1px rgba(124,196,255,0.06) inset",
      } : {}),
    }}>
      {highlighted && (
        <div style={{
          position: "absolute",
          top: -10,
          right: 24,
          padding: "4px 10px",
          fontSize: 10,
          fontFamily: "var(--font-mono)",
          fontWeight: 600,
          letterSpacing: "0.16em",
          textTransform: "uppercase",
          color: "#0a1628",
          background: "#4d9fff",
          borderRadius: 4,
        }}>
          Most popular
        </div>
      )}

      <div style={{ marginBottom: 24 }}>
        <div style={{ fontSize: 14, color: "#b3c2d8", fontWeight: 500, marginBottom: 8 }}>{name}</div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
          <span style={{ fontSize: 48, fontWeight: 800, letterSpacing: "-0.03em", color: "#eef3fb" }}>{price}</span>
          <span style={{ fontSize: 14, color: "#6680a0" }}>{period}</span>
        </div>
        {subPrice && (
          <div style={{
            marginTop: 6,
            fontSize: 12,
            fontFamily: "var(--font-mono)",
            color: "#4ade80",
            letterSpacing: "0.04em",
          }}>
            {subPrice}
          </div>
        )}
        <p style={{ marginTop: 12, fontSize: 14, color: "#b3c2d8" }}>{tagline}</p>
      </div>

      <ul style={{ listStyle: "none", padding: 0, marginBottom: 28, display: "flex", flexDirection: "column", gap: 10 }}>
        {features.map((feat, i) => (
          <li key={i} style={{ display: "flex", alignItems: "flex-start", gap: 10, fontSize: 14, color: "#eef3fb" }}>
            <Check size={16} style={{ color: "#4ade80", flexShrink: 0, marginTop: 2 }} />
            <span>{feat}</span>
          </li>
        ))}
      </ul>

      <button
        onClick={onCta}
        disabled={ctaDisabled}
        style={{
          width: "100%",
          padding: "12px 18px",
          borderRadius: 8,
          border: ctaSecondary ? "1px solid rgba(120,165,220,0.16)" : "none",
          background: ctaSecondary ? "transparent" : (highlighted ? "#4d9fff" : "rgba(15,29,51,0.6)"),
          color: ctaSecondary ? "#b3c2d8" : (highlighted ? "#0a1628" : "#eef3fb"),
          fontWeight: 600,
          fontSize: 14,
          cursor: ctaDisabled ? "default" : "pointer",
          opacity: ctaDisabled ? 0.55 : 1,
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 8,
          transition: "transform 0.15s ease, box-shadow 0.2s ease",
        }}
        onMouseEnter={(e) => {
          if (!ctaDisabled && highlighted) {
            (e.currentTarget as HTMLButtonElement).style.transform = "translateY(-1px)";
            (e.currentTarget as HTMLButtonElement).style.boxShadow = "0 8px 24px rgba(77,159,255,0.3)";
          }
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLButtonElement).style.transform = "translateY(0)";
          (e.currentTarget as HTMLButtonElement).style.boxShadow = "none";
        }}
      >
        {ctaLoading && <Loader2 size={14} className="spin" />}
        {ctaLabel}
      </button>
    </div>
  );
}
