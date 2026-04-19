import { useState, useEffect } from "react";
import { Moon, Sun, BarChart2, ScanLine, Newspaper, Bookmark, Bot, LogOut, LogIn, X, Info } from "lucide-react";
import { apiRequest, queryClient } from "@/lib/queryClient";
import AnalyzePage from "./analyze";
import ScannerPage from "./scanner";
import NewsPage from "./news";
import WatchlistPage from "./watchlist";
import BotDashboard from "./bot";
import LoginPage from "./login";

type TabId = "analyze" | "scanner" | "news" | "watchlist" | "bot";

interface Tab {
  id: TabId;
  label: string;
  icon: React.ReactNode;
  mobileIcon: React.ReactNode;
  requiresAuth: boolean;
}

const TABS: Tab[] = [
  { id: "analyze",   label: "Analyze",   icon: <BarChart2 size={14} />, mobileIcon: <BarChart2 size={20} />, requiresAuth: false },
  { id: "scanner",   label: "Scanner",   icon: <ScanLine size={14} />,  mobileIcon: <ScanLine size={20} />,  requiresAuth: false },
  { id: "news",      label: "News",      icon: <Newspaper size={14} />, mobileIcon: <Newspaper size={20} />, requiresAuth: false },
  { id: "watchlist", label: "Watchlist", icon: <Bookmark size={14} />,  mobileIcon: <Bookmark size={20} />,  requiresAuth: false },
  { id: "bot",       label: "AI Engine",  icon: <Bot size={14} />,       mobileIcon: <Bot size={20} />,       requiresAuth: true },
];

function Logo() {
  return (
    <svg aria-label="VolTradeAI" width="28" height="28" viewBox="0 0 32 32" fill="none">
      <rect width="32" height="32" rx="2" fill="#00e5ff" fillOpacity="0.1" />
      <rect width="32" height="32" rx="2" stroke="#00e5ff" strokeWidth="1" fill="none" strokeOpacity="0.5" />
      <polyline points="4,22 10,14 16,18 22,8 28,12" stroke="#00e5ff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <circle cx="22" cy="8" r="2.5" fill="#00e5ff" />
    </svg>
  );
}

interface HomeProps {
  authenticated: boolean;
  authLoading: boolean;
  isMobile: boolean;
  isOwner?: boolean;
}

function getInitialTab(): TabId {
  const hash = window.location.hash.replace("#/", "");
  if (hash && TABS.some(t => t.id === hash)) return hash as TabId;
  return "analyze";
}

export default function Home({ authenticated, authLoading, isMobile, isOwner }: HomeProps) {
  const [activeTab, setActiveTab] = useState<TabId>(getInitialTab);
  const [dark, setDark] = useState(() => localStorage.getItem("theme") !== "light");
  const [analyzeTarget, setAnalyzeTarget] = useState<string | undefined>(undefined);
  const [showLogin, setShowLogin] = useState(false);
  const [pendingTab, setPendingTab] = useState<TabId | null>(null);

  // Sync hash with active tab (Bug 1)
  useEffect(() => {
    window.location.hash = "#/" + activeTab;
  }, [activeTab]);

  // Persist dark mode (Bug 16)
  useEffect(() => {
    localStorage.setItem("theme", dark ? "dark" : "light");
  }, [dark]);

  const handleSelectTicker = (ticker: string) => {
    setAnalyzeTarget(ticker);
    setActiveTab("analyze");
  };

  const handleTabClick = (tabId: TabId) => {
    const tab = TABS.find(t => t.id === tabId);
    if (tab?.requiresAuth && !authenticated) {
      setPendingTab(tabId);
      setShowLogin(true);
      return;
    }
    setActiveTab(tabId);
  };

  const handleLoginSuccess = () => {
    queryClient.invalidateQueries({ queryKey: ["/api/auth/me"] });
    setShowLogin(false);
    setActiveTab(pendingTab || "bot");
    setPendingTab(null);
  };

  const handleLogout = async () => {
    await apiRequest("POST", "/api/auth/logout");
    queryClient.invalidateQueries({ queryKey: ["/api/auth/me"] });
    setActiveTab("analyze");
  };

  // Login overlay
  if (showLogin) {
    return (
      <div className={dark ? "dark" : "light"} style={{ minHeight: "100dvh", background: dark ? "#050a12" : "#e8ecf1" }}>
        <div style={{ position: "fixed", top: 16, left: 16, zIndex: 9990 }}>
          <button
            onClick={() => setShowLogin(false)}
            style={{
              display: "flex", alignItems: "center", gap: 6,
              padding: "8px 16px", borderRadius: 3,
              background: dark ? "rgba(0, 20, 40, 0.9)" : "rgba(255, 255, 255, 0.9)",
              border: dark ? "1px solid rgba(0, 229, 255, 0.3)" : "1px solid rgba(0, 80, 120, 0.3)",
              color: dark ? "#00e5ff" : "#0a1628", fontSize: 12, fontWeight: 500,
              fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.05em",
              textTransform: "uppercase", cursor: "pointer",
              boxShadow: "0 0 12px rgba(0, 229, 255, 0.1)",
            }}
          >
            <X size={14} /> BACK
          </button>
        </div>
        <LoginPage onLogin={handleLoginSuccess} />
      </div>
    );
  }

  return (
    <div className={dark ? "dark" : "light"} style={{
      minHeight: "100dvh",
      background: dark ? "#050a12" : "#e8ecf1",
      color: dark ? "#c8d6e5" : "#0a1628",
    }}>

      {/* ── Desktop top nav bar (hidden on mobile) ── */}
      <nav className="tab-nav desktop-nav" style={dark ? {} : { background: 'rgba(232, 236, 241, 0.92)' }}>
        <div className="tab-nav-logo">
          <Logo />
          <span className="tab-nav-logo-text"><span style={{ color: "#d4a017" }}>VolTrade</span><span style={{ color: "#00e5ff" }}>AI</span></span>
        </div>

        <div className="tab-nav-tabs">
          {TABS.map(tab => (
            <button
              key={tab.id}
              className={`tab-btn ${activeTab === tab.id ? "active" : ""}`}
              onClick={() => handleTabClick(tab.id)}
              aria-current={activeTab === tab.id ? "page" : undefined}
            >
              {tab.icon}
              {tab.label}
              {tab.requiresAuth && !authenticated && (
                <span style={{ fontSize: 9, opacity: 0.5, marginLeft: 2 }}>🔒</span>
              )}
            </button>
          ))}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <a
            href="/bot"
            className="theme-toggle-btn"
            aria-label="About the Bot"
            title="About the Bot — performance, strategy, mechanics"
            style={{ color: "#00e5ff", display: "inline-flex", alignItems: "center", justifyContent: "center", textDecoration: "none" }}
          >
            <Info size={15} />
          </a>
          <button
            className="theme-toggle-btn"
            onClick={() => setDark(!dark)}
            aria-label="Toggle light/dark mode"
            title={dark ? "Switch to light mode" : "Switch to dark mode"}
          >
            {dark ? <Sun size={15} /> : <Moon size={15} />}
          </button>
          {authenticated ? (
            <button
              className="theme-toggle-btn"
              onClick={handleLogout}
              aria-label="Sign out"
              title="Sign out"
            >
              <LogOut size={15} />
            </button>
          ) : (
            <button
              className="theme-toggle-btn"
              onClick={() => setShowLogin(true)}
              aria-label="Sign in"
              title="Sign in"
              style={{ color: "#00e5ff" }}
            >
              <LogIn size={15} />
            </button>
          )}
        </div>
      </nav>

      {/* ── Mobile top bar (shown only on mobile) ── */}
      <nav className="mobile-top-bar" style={dark ? {} : { background: 'rgba(232, 236, 241, 0.92)', borderBottomColor: 'rgba(0, 80, 120, 0.12)' }}>
        <div className="tab-nav-logo">
          <Logo />
          <span className="tab-nav-logo-text"><span style={{ color: "#d4a017" }}>VolTrade</span><span style={{ color: "#00e5ff" }}>AI</span></span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginLeft: "auto" }}>
          <a
            href="/bot"
            className="theme-toggle-btn"
            aria-label="About the Bot"
            title="About the Bot"
            style={{ color: "#00e5ff", display: "inline-flex", alignItems: "center", justifyContent: "center", textDecoration: "none" }}
          >
            <Info size={15} />
          </a>
          <button
            className="theme-toggle-btn"
            onClick={() => setDark(!dark)}
            aria-label="Toggle light/dark mode"
          >
            {dark ? <Sun size={15} /> : <Moon size={15} />}
          </button>
          {authenticated ? (
            <button className="theme-toggle-btn" onClick={handleLogout} aria-label="Sign out">
              <LogOut size={15} />
            </button>
          ) : (
            <button className="theme-toggle-btn" onClick={() => setShowLogin(true)} aria-label="Sign in" style={{ color: "#00e5ff" }}>
              <LogIn size={15} />
            </button>
          )}
        </div>
      </nav>

      {/* ── Page content ── */}
      <main className="page-container tab-page">
        {activeTab === "analyze" && (
          <AnalyzePage key={analyzeTarget} initialTicker={analyzeTarget} />
        )}
        {activeTab === "scanner" && (
          <ScannerPage onSelectTicker={handleSelectTicker} />
        )}
        {activeTab === "news" && (
          <NewsPage onSelectTicker={handleSelectTicker} />
        )}
        {activeTab === "watchlist" && (
          <WatchlistPage onSelectTicker={handleSelectTicker} authenticated={authenticated} />
        )}
        {activeTab === "bot" && (
          authenticated && isOwner ? <BotDashboard /> : (
            <div style={{ textAlign: "center", paddingTop: "4rem", color: "var(--text-tertiary)" }}>
              <Bot size={48} style={{ margin: "0 auto 1rem", opacity: 0.3 }} />
              <p style={{ fontSize: "1.1rem", fontWeight: 600, color: "var(--text-secondary)" }}>
                {authenticated ? "AI Trading Engine access is restricted to the account owner" : "Sign in to access the AI Trading Engine"}
              </p>
              {!authenticated && (
                <>
                  <p style={{ fontSize: "0.85rem", marginTop: "0.5rem" }}>The AI engine scans the market, finds opportunities, and trades automatically.</p>
                  <button
                    onClick={() => setShowLogin(true)}
                    style={{ marginTop: "1.5rem", padding: "10px 24px", background: "rgba(0,229,255,0.08)", border: "1px solid rgba(0,229,255,0.3)", color: "#00e5ff", borderRadius: 4, fontSize: 14, fontWeight: 600, cursor: "pointer", fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.1em", textTransform: "uppercase" }}
                  >
                    Sign In
                  </button>
                </>
              )}
            </div>
          )
        )}
      </main>

      {/* ── Mobile bottom tab bar (shown only on mobile) ── */}
      <nav className="mobile-bottom-bar">
        {TABS.map(tab => (
          <button
            key={tab.id}
            className={`mobile-tab-btn ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => handleTabClick(tab.id)}
          >
            {tab.mobileIcon}
            <span className="mobile-tab-label">{tab.label}</span>
            {tab.requiresAuth && !authenticated && (
              <span className="mobile-tab-lock">🔒</span>
            )}
          </button>
        ))}
      </nav>
    </div>
  );
}
