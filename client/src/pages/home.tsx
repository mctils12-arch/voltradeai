import { useState, useEffect } from "react";
import { Moon, Sun, BarChart2, ScanLine, Newspaper, Bookmark, Bot, LogOut, LogIn, Menu, X } from "lucide-react";
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
  { id: "bot",       label: "Bot",       icon: <Bot size={14} />,       mobileIcon: <Bot size={20} />,       requiresAuth: true },
];

function Logo() {
  return (
    <svg aria-label="VolTradeAI" width="28" height="28" viewBox="0 0 32 32" fill="none">
      <rect width="32" height="32" rx="8" fill="#0a84ff" fillOpacity="0.15" />
      <rect width="32" height="32" rx="8" stroke="#0a84ff" strokeWidth="1" fill="none" strokeOpacity="0.4" />
      <polyline points="4,22 10,14 16,18 22,8 28,12" stroke="#0a84ff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <circle cx="22" cy="8" r="2.5" fill="#0a84ff" />
    </svg>
  );
}

interface HomeProps {
  authenticated: boolean;
  authLoading: boolean;
}

export default function Home({ authenticated, authLoading }: HomeProps) {
  const [activeTab, setActiveTab] = useState<TabId>("analyze");
  const [dark, setDark] = useState(true);
  const [analyzeTarget, setAnalyzeTarget] = useState<string | undefined>(undefined);
  const [showLogin, setShowLogin] = useState(false);

  const handleSelectTicker = (ticker: string) => {
    setAnalyzeTarget(ticker);
    setActiveTab("analyze");
  };

  const handleTabClick = (tabId: TabId) => {
    const tab = TABS.find(t => t.id === tabId);
    if (tab?.requiresAuth && !authenticated) {
      setShowLogin(true);
      return;
    }
    setActiveTab(tabId);
  };

  const handleLoginSuccess = () => {
    queryClient.invalidateQueries({ queryKey: ["/api/auth/me"] });
    setShowLogin(false);
    // After login, go to the tab they were trying to access
    setActiveTab("bot");
  };

  const handleLogout = async () => {
    await apiRequest("POST", "/api/auth/logout");
    queryClient.invalidateQueries({ queryKey: ["/api/auth/me"] });
    setActiveTab("analyze");
  };

  // Login overlay
  if (showLogin) {
    return (
      <div className={dark ? "dark" : "light"} style={{ minHeight: "100dvh", background: dark ? "#000" : "#f5f5f7" }}>
        <div style={{ position: "absolute", top: 16, left: 16, zIndex: 200 }}>
          <button
            onClick={() => setShowLogin(false)}
            style={{
              display: "flex", alignItems: "center", gap: 6,
              padding: "8px 16px", borderRadius: 10,
              background: "rgba(255,255,255,0.08)", border: "1px solid rgba(255,255,255,0.12)",
              color: "#a1a1a6", fontSize: 13, fontWeight: 500,
            }}
          >
            <X size={14} /> Back
          </button>
        </div>
        <LoginPage onLogin={handleLoginSuccess} />
      </div>
    );
  }

  return (
    <div className={dark ? "dark" : "light"} style={{
      minHeight: "100dvh",
      background: dark ? "#000000" : "#f5f5f7",
      color: dark ? "#f5f5f7" : "#1d1d1f",
    }}>

      {/* ── Desktop top nav bar (hidden on mobile) ── */}
      <nav className="tab-nav desktop-nav" style={dark ? {} : { background: 'rgba(255,255,255,0.85)' }}>
        <div className="tab-nav-logo">
          <Logo />
          <span className="tab-nav-logo-text">VolTradeAI</span>
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
              style={{ color: "#0a84ff" }}
            >
              <LogIn size={15} />
            </button>
          )}
        </div>
      </nav>

      {/* ── Mobile top bar (shown only on mobile) ── */}
      <nav className="mobile-top-bar" style={dark ? {} : { background: 'rgba(255,255,255,0.85)' }}>
        <div className="tab-nav-logo">
          <Logo />
          <span className="tab-nav-logo-text">VolTradeAI</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginLeft: "auto" }}>
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
            <button className="theme-toggle-btn" onClick={() => setShowLogin(true)} aria-label="Sign in" style={{ color: "#0a84ff" }}>
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
          <WatchlistPage onSelectTicker={handleSelectTicker} />
        )}
        {activeTab === "bot" && (
          authenticated ? <BotDashboard /> : null
        )}
      </main>

      {/* ── Mobile bottom tab bar (shown only on mobile) ── */}
      <nav className="mobile-bottom-bar" style={dark ? {} : { background: 'rgba(255,255,255,0.92)', borderTopColor: 'rgba(0,0,0,0.1)' }}>
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
