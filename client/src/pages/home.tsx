import { useState, useEffect, useRef } from "react";
import { BarChart2, ScanLine, Newspaper, Bookmark, Bot, LogOut, LogIn, X, Info, ChevronDown, Eye, Layers, Briefcase, Scale, Calculator, Crosshair, Globe } from "lucide-react";
import { apiRequest, queryClient } from "@/lib/queryClient";
import AnalyzePage, { AnalyzeSection } from "./analyze";
import ScannerPage from "./scanner";
import ResearchPage from "./research";
import NewsPage from "./news";
import WatchlistPage from "./watchlist";
import PlannerPage from "./planner";
import DataMapPage from "./datamap";
import TaxesPage from "./taxes";
import BotDashboard from "./bot";
import LoginPage from "./login";

type TabId = "analyze" | "research" | "scanner" | "news" | "watchlist" | "data" | "planner" | "taxes" | "bot";

interface Tab {
  id: TabId;
  label: string;
  icon: React.ReactNode;
  mobileIcon: React.ReactNode;
  requiresAuth: boolean;
  /** Tab is owner-only (e.g. live trading is restricted to platform owner pre-RIA). */
  requiresOwner?: boolean;
}

const TABS: Tab[] = [
  { id: "analyze",   label: "Analyze",   icon: <BarChart2 size={14} />, mobileIcon: <BarChart2 size={20} />, requiresAuth: false },
  { id: "research",  label: "Research",  icon: <Scale size={14} />,     mobileIcon: <Scale size={20} />,     requiresAuth: false },
  { id: "scanner",   label: "Scanner",   icon: <ScanLine size={14} />,  mobileIcon: <ScanLine size={20} />,  requiresAuth: false },
  { id: "news",      label: "News",      icon: <Newspaper size={14} />, mobileIcon: <Newspaper size={20} />, requiresAuth: false },
  { id: "watchlist", label: "Watchlist", icon: <Bookmark size={14} />,  mobileIcon: <Bookmark size={20} />,  requiresAuth: false },
  { id: "data",      label: "Data",      icon: <Globe size={14} />,     mobileIcon: <Globe size={20} />,     requiresAuth: false },
  { id: "planner",   label: "Planner",   icon: <Crosshair size={14} />, mobileIcon: <Crosshair size={20} />, requiresAuth: false },
  { id: "taxes",     label: "Taxes",     icon: <Calculator size={14} />, mobileIcon: <Calculator size={20} />, requiresAuth: false },
  { id: "bot",       label: "AI Engine", icon: <Bot size={14} />,       mobileIcon: <Bot size={20} />,       requiresAuth: true, requiresOwner: true },
];

function Logo() {
  return (
    <a href="/" style={{ display: "inline-flex", alignItems: "center", textDecoration: "none", letterSpacing: "-0.02em" }} aria-label="VolTradeAI home">
      <span style={{ color: "#eef3fb", fontWeight: 700, fontSize: 16 }}>vol</span>
      <span style={{ color: "#b3c2d8", fontWeight: 500, fontSize: 16 }}>trade</span>
      <span style={{ color: "#4d9fff", fontWeight: 600, fontSize: 13, marginLeft: 4, fontFamily: "Geist Mono, monospace" }}>/AI</span>
    </a>
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
  // The hash can be either "analyze" or "analyze/<section>" — strip the
  // subpath when matching against TabIds.
  const root = hash.split("/")[0];
  if (root && TABS.some(t => t.id === root)) return root as TabId;
  return "analyze";
}

function getInitialAnalyzeSection(): AnalyzeSection {
  const hash = window.location.hash.replace("#/", "");
  const parts = hash.split("/");
  if (parts[0] === "analyze" && parts[1]) {
    const sec = parts[1];
    if (sec === "options" || sec === "smart-money" || sec === "structure" || sec === "etf-builder") {
      return sec;
    }
  }
  return "options";
}

// Top-nav dropdown items for the Analyze section. Defined here so the
// component and the mobile bottom-bar can both reference them.
const ANALYZE_SECTIONS: { id: AnalyzeSection; label: string; icon: React.ReactNode; description: string; accent: string }[] = [
  { id: "options",     label: "Options & Volatility", icon: <BarChart2 size={14} />, description: "IV/HV, VRP, spreads, vol cone", accent: "#4d9fff" },
  { id: "smart-money", label: "Smart Money",          icon: <Eye size={14} />,       description: "Insiders, institutions, flow",   accent: "#fbb24c" },
  { id: "structure",   label: "Structure",            icon: <Layers size={14} />,    description: "Float, short, S/R levels",       accent: "#c084fc" },
  { id: "etf-builder", label: "ETF Builder",          icon: <Briefcase size={14} />, description: "Holdings, drag, replicate",      accent: "#4ade80" },
];

export default function Home({ authenticated, authLoading, isMobile, isOwner }: HomeProps) {
  const [activeTab, setActiveTab] = useState<TabId>(getInitialTab);
  const [analyzeSection, setAnalyzeSection] = useState<AnalyzeSection>(getInitialAnalyzeSection);
  // Desktop hover dropdown — open/close state with a small close delay so
  // the user can move from button → menu without it snapping shut.
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownCloseTimer = useRef<number | null>(null);

  const openDropdown = () => {
    if (dropdownCloseTimer.current) {
      window.clearTimeout(dropdownCloseTimer.current);
      dropdownCloseTimer.current = null;
    }
    setDropdownOpen(true);
  };
  const closeDropdown = (delay = 120) => {
    if (dropdownCloseTimer.current) window.clearTimeout(dropdownCloseTimer.current);
    dropdownCloseTimer.current = window.setTimeout(() => setDropdownOpen(false), delay);
  };
  // DARK-ONLY 2026-05-03: light theme removed because several components
  // didn't render cleanly in light mode. The `dark` variable is kept as
  // a constant `true` so per-theme conditional code throughout this file
  // and child components remains valid without rewriting it. localStorage
  // theme key is no longer read or written.
  const [dark] = useState(true);
  const [analyzeTarget, setAnalyzeTarget] = useState<string | undefined>(undefined);
  const [showLogin, setShowLogin] = useState(false);
  const [pendingTab, setPendingTab] = useState<TabId | null>(null);

  // Sync hash with active tab (Bug 1)
  // 2026-05-23: when on the analyze tab, also encode the active section so
  // links like /app#/analyze/etf-builder restore the dropdown selection.
  useEffect(() => {
    if (activeTab === "analyze") {
      window.location.hash = `#/analyze/${analyzeSection}`;
    } else {
      // Only rewrite when the hash ROOT disagrees with the tab — tabs may own
      // subpaths (e.g. #/data/filings) that this sync must not stomp.
      const curRoot = window.location.hash.replace("#/", "").split("/")[0];
      if (curRoot !== activeTab) window.location.hash = "#/" + activeTab;
    }
  }, [activeTab, analyzeSection]);

  // Dark-only 2026-05-03: theme persistence effect removed (always dark).

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
      <div className={dark ? "dark" : "light"} style={{ minHeight: "100dvh", background: dark ? "#050a13" : "#e8ecf1" }}>
        <div style={{ position: "fixed", top: 16, left: 16, zIndex: 9990 }}>
          <button
            onClick={() => setShowLogin(false)}
            style={{
              display: "flex", alignItems: "center", gap: 6,
              padding: "8px 16px", borderRadius: 3,
              background: dark ? "rgba(0, 20, 40, 0.9)" : "rgba(255, 255, 255, 0.9)",
              border: dark ? "1px solid rgba(77, 159, 255, 0.3)" : "1px solid rgba(0, 80, 120, 0.3)",
              color: dark ? "#4d9fff" : "#0a1628", fontSize: 12, fontWeight: 500,
              fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.05em",
              textTransform: "uppercase", cursor: "pointer",
              boxShadow: "0 0 12px rgba(77, 159, 255, 0.1)",
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
      background: dark ? "#050a13" : "#e8ecf1",
      color: dark ? "#c8d6e5" : "#0a1628",
    }}>

      {/* ── Desktop top nav bar (hidden on mobile) ── */}
      <nav className="tab-nav desktop-nav" style={dark ? {} : { background: 'rgba(232, 236, 241, 0.92)' }}>
        <div className="tab-nav-logo">
          <Logo />
        </div>

        <div className="tab-nav-tabs">
          {TABS.filter(tab => !tab.requiresOwner || isOwner).map(tab => {
            // ANALYZE DROPDOWN 2026-05-23: the Analyze tab is now a hover
            // dropdown exposing four sections (Options, Smart Money,
            // Structure, ETF Builder). Other tabs render as before.
            if (tab.id === "analyze") {
              return (
                <div
                  key={tab.id}
                  className="analyze-dropdown-wrapper"
                  onMouseEnter={openDropdown}
                  onMouseLeave={() => closeDropdown(120)}
                >
                  <button
                    className={`tab-btn ${activeTab === tab.id ? "active" : ""}`}
                    onClick={() => handleTabClick(tab.id)}
                    aria-current={activeTab === tab.id ? "page" : undefined}
                    aria-haspopup="menu"
                    aria-expanded={dropdownOpen}
                  >
                    {tab.icon}
                    {tab.label}
                    <ChevronDown
                      size={12}
                      style={{
                        opacity: 0.6,
                        marginLeft: 2,
                        transform: dropdownOpen ? "rotate(180deg)" : "rotate(0)",
                        transition: "transform 150ms ease",
                      }}
                    />
                  </button>
                  {dropdownOpen && (
                    <div
                      className="analyze-dropdown-menu"
                      role="menu"
                      onMouseEnter={openDropdown}
                      onMouseLeave={() => closeDropdown(80)}
                    >
                      {ANALYZE_SECTIONS.map(s => {
                        const isActive = activeTab === "analyze" && analyzeSection === s.id;
                        return (
                          <button
                            key={s.id}
                            role="menuitem"
                            className={`analyze-dropdown-item ${isActive ? "active" : ""}`}
                            onClick={() => {
                              setAnalyzeSection(s.id);
                              setActiveTab("analyze");
                              setDropdownOpen(false);
                            }}
                            style={{ ["--item-accent" as any]: s.accent }}
                          >
                            <span className="analyze-dropdown-icon" style={{ color: s.accent }}>
                              {s.icon}
                            </span>
                            <span className="analyze-dropdown-text">
                              <span className="analyze-dropdown-label">{s.label}</span>
                              <span className="analyze-dropdown-description">{s.description}</span>
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            }

            // Default tab button
            return (
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
            );
          })}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <a
            href="/#bot"
            className="theme-toggle-btn"
            aria-label="About the Bot"
            title="About the Bot — performance, strategy, mechanics"
            style={{ color: "#4d9fff", display: "inline-flex", alignItems: "center", justifyContent: "center", textDecoration: "none" }}
          >
            <Info size={15} />
          </a>
          {/* Light/dark theme toggle removed 2026-05-03: dark-only UI to
              eliminate per-theme styling drift. The light variant had
              several components that didn't render cleanly. Keeping the
              `dark` state variable as a constant `true` so per-theme
              conditional code below remains valid without rewriting it. */}
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
              style={{ color: "#4d9fff" }}
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
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginLeft: "auto" }}>
          <a
            href="/bot"
            className="theme-toggle-btn"
            aria-label="About the Bot"
            title="About the Bot"
            style={{ color: "#4d9fff", display: "inline-flex", alignItems: "center", justifyContent: "center", textDecoration: "none" }}
          >
            <Info size={15} />
          </a>
          {/* Light/dark toggle removed 2026-05-03 — dark-only UI */}
          {authenticated ? (
            <button className="theme-toggle-btn" onClick={handleLogout} aria-label="Sign out">
              <LogOut size={15} />
            </button>
          ) : (
            <button className="theme-toggle-btn" onClick={() => setShowLogin(true)} aria-label="Sign in" style={{ color: "#4d9fff" }}>
              <LogIn size={15} />
            </button>
          )}
        </div>
      </nav>

      {/* ── Page content ── */}
      <main className="page-container tab-page">
        {activeTab === "analyze" && (
          <AnalyzePage key={analyzeTarget} initialTicker={analyzeTarget} section={analyzeSection} />
        )}
        {activeTab === "research" && (
          <ResearchPage onSelectTicker={handleSelectTicker} />
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
        {activeTab === "data" && (
          <DataMapPage />
        )}
        {activeTab === "planner" && (
          <PlannerPage />
        )}
        {activeTab === "taxes" && (
          <TaxesPage />
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
                    style={{ marginTop: "1.5rem", padding: "10px 24px", background: "rgba(77,159,255,0.08)", border: "1px solid rgba(77,159,255,0.3)", color: "#4d9fff", borderRadius: 4, fontSize: 14, fontWeight: 600, cursor: "pointer", fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.1em", textTransform: "uppercase" }}
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
        {TABS.filter(tab => !tab.requiresOwner || isOwner).map(tab => (
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
