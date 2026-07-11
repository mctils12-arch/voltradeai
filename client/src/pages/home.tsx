import { useState, useEffect, useRef } from "react";
import { useLocation } from "wouter";
import { BarChart2, ScanLine, Newspaper, Bookmark, Bot, LogOut, LogIn, X, Info, ChevronDown, Eye, Layers, Briefcase, Scale, Calculator, Crosshair, Globe, TrendingUp, Database, Radio, Share2, Code2, Sparkles } from "lucide-react";
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

// ── TWO-WORLD NAV (PLATFORM INTEGRATION program P1, research/platform_program.md;
// human decision 2026-07-11) ─────────────────────────────────────────────────
// The nine flat tabs collapse into two grouped drop-downs — the STOCK/trading
// world ("Trade") and the DATA/intelligence world ("Data") — plus the owner-only
// AI Engine. Each menu leaf carries an ACTION: switch a top-level tab (optionally
// with an analyze section), open a datamap overlay via the URL hash (datamap's
// hashchange listener reacts), or navigate a wouter route (the /developers API
// page — this is how the API surface finally becomes reachable from inside the
// app). TABS (below) stays the source of truth for page content, the hash sync,
// and the flat mobile bottom bar; these menus are a desktop presentation over it.
type NavAction =
  | { kind: "tab"; tab: TabId; section?: AnalyzeSection }
  | { kind: "hash"; hash: string }   // datamap overlay; implies the "data" tab
  | { kind: "route"; route: string }; // wouter route (leaves the /app hash shell)

interface NavLeaf {
  key: string;
  label: string;
  icon: React.ReactNode;
  description: string;
  accent: string;
  action: NavAction;
  soon?: boolean; // rendered disabled with a "soon" tag (honest, not clickable)
}
interface NavSubgroup { heading: string; items: NavLeaf[]; }
interface NavMenu { id: "trade" | "data"; label: string; icon: React.ReactNode; groups: NavSubgroup[]; }

const NAV_MENUS: NavMenu[] = [
  {
    id: "trade", label: "Trade", icon: <TrendingUp size={14} />,
    groups: [
      {
        heading: "Analyze",
        items: ANALYZE_SECTIONS.map(s => ({
          key: `analyze-${s.id}`, label: s.label, icon: s.icon,
          description: s.description, accent: s.accent,
          action: { kind: "tab" as const, tab: "analyze" as TabId, section: s.id },
        })),
      },
      {
        heading: "Tools",
        items: [
          { key: "scanner",   label: "Scanner",   icon: <ScanLine size={14} />,  description: "Momentum & unusual activity", accent: "#4d9fff", action: { kind: "tab", tab: "scanner" } },
          { key: "watchlist", label: "Watchlist", icon: <Bookmark size={14} />,  description: "Your tracked tickers",        accent: "#4ade80", action: { kind: "tab", tab: "watchlist" } },
          { key: "research",  label: "Research",  icon: <Scale size={14} />,     description: "Deep-dive fundamentals",      accent: "#c084fc", action: { kind: "tab", tab: "research" } },
          { key: "news",      label: "News",      icon: <Newspaper size={14} />, description: "Market & ticker headlines",   accent: "#fbb24c", action: { kind: "tab", tab: "news" } },
          { key: "planner",   label: "Planner",   icon: <Crosshair size={14} />, description: "Position & risk planning",    accent: "#4d9fff", action: { kind: "tab", tab: "planner" } },
          { key: "taxes",     label: "Taxes",     icon: <Calculator size={14} />, description: "Realized P&L & lots",        accent: "#4ade80", action: { kind: "tab", tab: "taxes" } },
        ],
      },
    ],
  },
  {
    id: "data", label: "Data", icon: <Database size={14} />,
    groups: [
      {
        heading: "Map",
        items: [
          { key: "data-map", label: "Live Map", icon: <Globe size={14} />, description: "Aircraft, vessels, satellites, grid", accent: "#4d9fff", action: { kind: "hash", hash: "#/data" } },
        ],
      },
      {
        heading: "Intelligence",
        items: [
          { key: "data-streams", label: "Streams",          icon: <Radio size={14} />,  description: "Every live + archived stream", accent: "#4ade80", action: { kind: "hash", hash: "#/data/streams" } },
          { key: "data-graph",   label: "Everything Graph", icon: <Share2 size={14} />, description: "Entities & cross-stream links", accent: "#c084fc", action: { kind: "hash", hash: "#/data/graph" } },
        ],
      },
      {
        heading: "Build",
        items: [
          { key: "developers", label: "Developers & API", icon: <Code2 size={14} />,    description: "Endpoints, keys, docs",         accent: "#fbb24c", action: { kind: "route", route: "/developers" } },
          { key: "signals",    label: "Signals",          icon: <Sparkles size={14} />, description: "Validated signal products",     accent: "#4d9fff", action: { kind: "tab", tab: "data" }, soon: true },
        ],
      },
    ],
  },
];

// Which top-level tabs belong to which menu — drives the menu button's active
// underline (highlight "Data" while the map/overlays are open, "Trade" while any
// trading tab is open). Derived once from the menu actions so it can't drift.
const MENU_TABS: Record<"trade" | "data", Set<TabId>> = {
  trade: new Set<TabId>(["analyze", "scanner", "watchlist", "research", "news", "planner", "taxes"]),
  data: new Set<TabId>(["data"]),
};

export default function Home({ authenticated, authLoading, isMobile, isOwner }: HomeProps) {
  const [activeTab, setActiveTab] = useState<TabId>(getInitialTab);
  const [analyzeSection, setAnalyzeSection] = useState<AnalyzeSection>(getInitialAnalyzeSection);
  const [, navigate] = useLocation(); // wouter route nav (Developers/API page)
  // Desktop hover dropdowns — which top-nav menu ("trade" | "data") is open, with
  // a small close delay so the pointer can travel button → menu without it
  // snapping shut. Only one menu is open at a time.
  const [openMenuId, setOpenMenuId] = useState<null | "trade" | "data">(null);
  const menuCloseTimer = useRef<number | null>(null);

  const openMenu = (id: "trade" | "data") => {
    if (menuCloseTimer.current) {
      window.clearTimeout(menuCloseTimer.current);
      menuCloseTimer.current = null;
    }
    setOpenMenuId(id);
  };
  const closeMenu = (delay = 120) => {
    if (menuCloseTimer.current) window.clearTimeout(menuCloseTimer.current);
    menuCloseTimer.current = window.setTimeout(() => setOpenMenuId(null), delay);
  };
  const toggleMenu = (id: "trade" | "data") => setOpenMenuId(cur => (cur === id ? null : id));

  // Execute a nav-menu leaf's action. Tab actions honor auth gating via
  // handleTabClick; hash actions open a datamap overlay (its hashchange listener
  // reacts, and the hash-sync effect below won't stomp a "#/data/..." subpath);
  // route actions navigate a wouter page. Closes any open menu after.
  const runNavAction = (action: NavAction) => {
    setOpenMenuId(null);
    if (action.kind === "route") { navigate(action.route); return; }
    if (action.kind === "hash") {
      window.location.hash = action.hash; // fires hashchange → datamap opens overlay
      handleTabClick("data");
      return;
    }
    if (action.section) setAnalyzeSection(action.section);
    handleTabClick(action.tab);
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
          {/* TWO-WORLD NAV (P1, research/platform_program.md): Trade + Data
              hover/click drop-downs replace the nine flat tabs; the owner-only
              AI Engine stays a direct button. Menu leaves run NavActions. */}
          {NAV_MENUS.map(menu => {
            const menuActive = MENU_TABS[menu.id].has(activeTab);
            const isOpen = openMenuId === menu.id;
            return (
              <div
                key={menu.id}
                className="analyze-dropdown-wrapper"
                onMouseEnter={() => openMenu(menu.id)}
                onMouseLeave={() => closeMenu(120)}
              >
                <button
                  className={`tab-btn ${menuActive ? "active" : ""}`}
                  onClick={() => toggleMenu(menu.id)}
                  aria-current={menuActive ? "page" : undefined}
                  aria-haspopup="menu"
                  aria-expanded={isOpen}
                >
                  {menu.icon}
                  {menu.label}
                  <ChevronDown
                    size={12}
                    style={{
                      opacity: 0.6,
                      marginLeft: 2,
                      transform: isOpen ? "rotate(180deg)" : "rotate(0)",
                      transition: "transform 150ms ease",
                    }}
                  />
                </button>
                {isOpen && (
                  <div
                    className="analyze-dropdown-menu nav-dropdown-menu"
                    role="menu"
                    onMouseEnter={() => openMenu(menu.id)}
                    onMouseLeave={() => closeMenu(80)}
                  >
                    {menu.groups.map(group => (
                      <div key={group.heading} className="nav-dropdown-group">
                        <div className="nav-dropdown-heading">{group.heading}</div>
                        {group.items.map(item => {
                          const leafActive =
                            item.action.kind === "tab" &&
                            activeTab === item.action.tab &&
                            (item.action.section ? analyzeSection === item.action.section : true) &&
                            !item.soon;
                          return (
                            <button
                              key={item.key}
                              role="menuitem"
                              disabled={item.soon}
                              className={`analyze-dropdown-item ${leafActive ? "active" : ""} ${item.soon ? "soon" : ""}`}
                              onClick={() => { if (!item.soon) runNavAction(item.action); }}
                              style={{ ["--item-accent" as any]: item.accent }}
                            >
                              <span className="analyze-dropdown-icon" style={{ color: item.accent }}>
                                {item.icon}
                              </span>
                              <span className="analyze-dropdown-text">
                                <span className="analyze-dropdown-label">
                                  {item.label}
                                  {item.soon && <span className="nav-soon-tag">soon</span>}
                                </span>
                                <span className="analyze-dropdown-description">{item.description}</span>
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}

          {/* AI Engine — owner-only, direct button (not part of either world) */}
          {isOwner && (
            <button
              className={`tab-btn ${activeTab === "bot" ? "active" : ""}`}
              onClick={() => handleTabClick("bot")}
              aria-current={activeTab === "bot" ? "page" : undefined}
            >
              <Bot size={14} />
              AI Engine
            </button>
          )}
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
