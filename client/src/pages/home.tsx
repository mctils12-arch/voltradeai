import { useState, useEffect, useRef, useSyncExternalStore } from "react";
import { useLocation } from "wouter";
import { BarChart2, ScanLine, Newspaper, Bookmark, Bot, LogOut, LogIn, X, Info, ChevronDown, Eye, Layers, Briefcase, Scale, Calculator, Crosshair, Globe, TrendingUp, Database, Radio, Share2, Code2, Sparkles, LayoutGrid, Target, Brain, FileSearch } from "lucide-react";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { onSessionExpired, SESSION_EXPIRED_NOTICE } from "@/lib/authExpiry";
import type { AnalyzeSection } from "./analyze";
// The Trade world is one cockpit page (2026-10-29): Options & Volatility,
// Smart Money, Structure, Research, Scanner, News, Watchlist, Planner, Taxes
// and the ETF builder are its panels; the old page components stay as the
// panel bodies' wrappers only.
import TradePage from "@/trade/TradePage";
import DataMapPage from "./datamap";
import BotDashboard from "./bot";
import ForecastPage from "./forecast";
import BrainPage from "./brain";
import TickerDossierPage from "./tickerDossier";
import TradeBackdrop from "@/components/TradeBackdrop";
import { getActiveTicker, setActiveTicker, subscribeActiveTicker } from "@/lib/activeTicker";
import { paneIdFromHash } from "../bot/popout";
import LoginPage from "./login";

type TabId = "analyze" | "research" | "scanner" | "news" | "watchlist" | "data" | "planner" | "taxes" | "forecast" | "brain" | "dossier" | "bot";

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
  // Weekly forecast screen (2026-10-27, SHADOW): owner-only — its calls name the open positions
  { id: "forecast",  label: "Forecast",  icon: <Target size={14} />,    mobileIcon: <Target size={20} />,    requiresAuth: true, requiresOwner: true },
  // Brain page (2026-10-29): what the bot uses now (live/shadow/off) + short- vs long-term evidence — owner-only
  { id: "brain",     label: "Brain",     icon: <Brain size={14} />,     mobileIcon: <Brain size={20} />,     requiresAuth: true, requiresOwner: true },
  // Ticker dossier (2026-10-29): owner-only — opened from the AI Engine's look-up box (#/dossier/<SYM>); it folds the look-up in
  { id: "dossier",   label: "Dossier",   icon: <FileSearch size={14} />, mobileIcon: <FileSearch size={20} />, requiresAuth: true, requiresOwner: true },
  { id: "bot",       label: "AI Engine", icon: <Bot size={14} />,       mobileIcon: <Bot size={20} />,       requiresAuth: true, requiresOwner: true },
];

// Phone bottom bar (2026-09-24): 8-9 tabs squeezed into a 390px bar left
// ~43px per tab, so labels ran together ("AnalyzeResearchScanner") and tap
// targets fell under the 44px minimum. Four primary tabs + a More sheet is
// the standard phone pattern; everything stays one tap from the sheet.
const MOBILE_PRIMARY_TABS: TabId[] = ["analyze", "research", "news", "data"];

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

// "#/analyze/<section>/<SYMBOL>" (2026-10-29): a shareable link to one
// symbol's view. The third segment seeds lib/activeTicker before the Analyze
// page mounts, so it opens loaded; the state→hash sync below writes it back.
function tickerFromHash(hash: string): string | null {
  const parts = hash.replace("#/", "").split("/");
  return parts[0] === "analyze" && parts[2] ? parts[2] : null;
}

function getInitialTab(): TabId {
  const hash = window.location.hash.replace("#/", "");
  // The hash can be "analyze", "analyze/<section>" or
  // "analyze/<section>/<SYMBOL>" — strip the subpath when matching TabIds.
  const root = hash.split("/")[0];
  const sym = tickerFromHash(window.location.hash);
  if (sym) setActiveTicker(sym); // invalid symbols are ignored by the store
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
type NavMenuId = "trade" | "data" | "engine";
interface NavMenu { id: NavMenuId; label: string; icon: React.ReactNode; groups: NavSubgroup[]; }

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
          { key: "api-keys",   label: "My API Keys",      icon: <Code2 size={14} />,    description: "Generate & manage preview keys", accent: "#4ade80", action: { kind: "route", route: "/apikeys" } },
          { key: "signals",    label: "Signals",          icon: <Sparkles size={14} />, description: "Validated signal products",     accent: "#4d9fff", action: { kind: "tab", tab: "data" }, soon: true },
        ],
      },
    ],
  },
];

// Which top-level tabs belong to which menu — drives the menu button's active
// underline (highlight "Data" while the map/overlays are open, "Trade" while any
// trading tab is open). Derived once from the menu actions so it can't drift.
const MENU_TABS: Record<NavMenuId, Set<TabId>> = {
  trade: new Set<TabId>(["analyze", "scanner", "watchlist", "research", "news", "planner", "taxes"]),
  data: new Set<TabId>(["data"]),
  engine: new Set<TabId>(["bot", "dossier", "forecast", "brain"]),
};

// ONE AI ENGINE MENU (human 2026-10-29: "trade data forecast brain ai engine,
// what do we need, it seems like a lot"): the four owner-only direct buttons
// (Forecast, Brain, Dossier, AI Engine) collapse into one owner-only
// drop-down beside Trade and Data, built on the same NavMenu machinery.
// Every destination stays one hover + one click away; TABS, the hash
// routes and the phone bar are untouched, so deep links keep working.
const ENGINE_MENU: NavMenu = {
  id: "engine", label: "AI Engine", icon: <Bot size={14} />,
  groups: [
    {
      heading: "Operate",
      items: [
        { key: "engine-cockpit", label: "Cockpit", icon: <Bot size={14} />,        description: "Live bot: edge, book, orders, today",      accent: "var(--accent)",       action: { kind: "tab", tab: "bot" } },
        { key: "engine-dossier", label: "Dossier", icon: <FileSearch size={14} />, description: "One ticker: market → stock → options → plan", accent: "var(--accent-green)", action: { kind: "tab", tab: "dossier" } },
      ],
    },
    {
      heading: "Evidence",
      items: [
        { key: "engine-forecast", label: "Forecast", icon: <Target size={14} />, description: "Weekly calls in shadow, with calibration", accent: "var(--accent-orange)", action: { kind: "tab", tab: "forecast" } },
        { key: "engine-brain",    label: "Brain",    icon: <Brain size={14} />,  description: "What is live, shadow, off — and why",      accent: "var(--accent-purple)", action: { kind: "tab", tab: "brain" } },
      ],
    },
  ],
};

export default function Home({ authenticated, authLoading, isMobile, isOwner }: HomeProps) {
  const [activeTab, setActiveTab] = useState<TabId>(getInitialTab);
  // /app#/bot/pane/<id>: one bot panel popped out into its own browser window
  // (client/src/bot/popout.ts) — rendered alone, without the shell
  const [paneId] = useState(() => paneIdFromHash(window.location.hash));
  const [analyzeSection, setAnalyzeSection] = useState<AnalyzeSection>(getInitialAnalyzeSection);
  const [, navigate] = useLocation(); // wouter route nav (Developers/API page)
  // The Trade world's active symbol (lib/activeTicker): read here, above the
  // hash-sync effect that writes it into the URL. The ticker bar is below.
  const activeTicker = useSyncExternalStore(subscribeActiveTicker, getActiveTicker, getActiveTicker);
  // Desktop hover dropdowns — which top-nav menu ("trade" | "data") is open, with
  // a small close delay so the pointer can travel button → menu without it
  // snapping shut. Only one menu is open at a time.
  const [openMenuId, setOpenMenuId] = useState<null | NavMenuId>(null);
  const menuCloseTimer = useRef<number | null>(null);

  const openMenu = (id: NavMenuId) => {
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
  const toggleMenu = (id: NavMenuId) => setOpenMenuId(cur => (cur === id ? null : id));

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
  const [showLogin, setShowLogin] = useState(false);
  const [pendingTab, setPendingTab] = useState<TabId | null>(null);
  const [sessionNotice, setSessionNotice] = useState<string | null>(null);

  // Expired session (field bug 2026-10-28): any app API call answering 401
  // "Not authenticated"/"Session expired" (lib/authExpiry, reported from
  // queryClient) while this shell still believes it is signed in opens the
  // sign-in view with a notice and returns to the same tab after sign-in.
  // Guarded on the signed-in belief so a signed-out visitor is never bounced.
  const authRef = useRef(authenticated);
  authRef.current = authenticated;
  const tabRef = useRef(activeTab);
  tabRef.current = activeTab;
  useEffect(() => onSessionExpired(() => {
    if (!authRef.current) return;
    setPendingTab(tabRef.current);
    setSessionNotice(SESSION_EXPIRED_NOTICE);
    setShowLogin(true);
  }), []);

  // Sync hash with active tab (Bug 1)
  // 2026-05-23: when on the analyze tab, also encode the active section so
  // links like /app#/analyze/etf-builder restore the dropdown selection.
  useEffect(() => {
    if (activeTab === "analyze") {
      // Carry the active symbol so the URL is a shareable link to this view.
      const sym = getActiveTicker();
      window.location.hash = sym ? `#/analyze/${analyzeSection}/${sym}` : `#/analyze/${analyzeSection}`;
    } else {
      // Only rewrite when the hash ROOT disagrees with the tab — tabs may own
      // subpaths (e.g. #/data/filings) that this sync must not stomp.
      const curRoot = window.location.hash.replace("#/", "").split("/")[0];
      if (curRoot !== activeTab) window.location.hash = "#/" + activeTab;
    }
  }, [activeTab, analyzeSection, activeTicker]);

  // Dark-only 2026-05-03: theme persistence effect removed (always dark).

  const [mobileMoreOpen, setMobileMoreOpen] = useState(false);
  useEffect(() => {
    if (!mobileMoreOpen) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setMobileMoreOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mobileMoreOpen]);

  const handleTabClick = (tabId: TabId) => {
    setMobileMoreOpen(false);
    const tab = TABS.find(t => t.id === tabId);
    if (tab?.requiresAuth && !authenticated) {
      setPendingTab(tabId);
      setShowLogin(true);
      return;
    }
    setActiveTab(tabId);
  };

  // Hash → state (2026-07-20 audit repair): every tab change writes a
  // history entry via the state→hash sync effect above, but nothing read
  // the hash back — so the browser Back/Forward buttons (and any
  // same-document link to /app#/<tab>) changed the URL while the view
  // stayed frozen. datamap owns the #/data/* subpaths (its own hashchange
  // listener); this only moves the top-level tab and analyze section, and
  // routes through handleTabClick so auth gating still applies.
  useEffect(() => {
    const onHash = () => {
      const hash = window.location.hash.replace("#/", "");
      const [root, sub] = hash.split("/");
      const sym = tickerFromHash(window.location.hash); // "#/analyze/<section>/<SYMBOL>"
      if (sym) setActiveTicker(sym);
      if (root && root !== activeTab && TABS.some(t => t.id === root)) {
        handleTabClick(root as TabId);
      }
      if (root === "analyze" && (sub === "options" || sub === "smart-money" || sub === "structure" || sub === "etf-builder") && sub !== analyzeSection) {
        setAnalyzeSection(sub);
      }
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [activeTab, analyzeSection, authenticated]);

  const handleLoginSuccess = () => {
    queryClient.invalidateQueries({ queryKey: ["/api/auth/me"] });
    // after an expiry, every cached answer was a 401 — refetch the lot
    if (sessionNotice) void queryClient.invalidateQueries();
    setSessionNotice(null);
    setShowLogin(false);
    setActiveTab(pendingTab || "bot");
    setPendingTab(null);
  };

  const handleLogout = async () => {
    await apiRequest("POST", "/api/auth/logout");
    queryClient.invalidateQueries({ queryKey: ["/api/auth/me"] });
    setActiveTab("analyze");
  };

  // ── The Trade world (2026-10-29, human: "it needs to look like the bot UI
  // but have the functionality of what's there and no duplicates"): every
  // Trade tab renders the one cockpit page (trade/TradePage) over one moving
  // map (TradeBackdrop, mounted below) with one active symbol
  // (lib/activeTicker). The tab only says which panel to open.
  const isTradeTab = MENU_TABS.trade.has(activeTab);

  // Login overlay
  if (showLogin) {
    return (
      <div className={dark ? "dark" : "light"} style={{ minHeight: "100dvh", background: dark ? "#050a13" : "#e8ecf1" }}>
        <div style={{ position: "fixed", top: 16, left: 16, zIndex: 9990 }}>
          <button
            onClick={() => { setShowLogin(false); setSessionNotice(null); }}
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
        <LoginPage onLogin={handleLoginSuccess} notice={sessionNotice} />
      </div>
    );
  }

  if (paneId && authenticated && isOwner) {
    return <div className="dark" style={{ minHeight: "100dvh", background: "var(--bg-primary)" }}><BotDashboard /></div>;
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
              hover/click drop-downs replace the nine flat tabs. 2026-10-29:
              the owner-only AI Engine is a third drop-down of the same kind
              (ENGINE_MENU) instead of four direct buttons. Menu leaves run
              NavActions. */}
          {(isOwner ? [...NAV_MENUS, ENGINE_MENU] : NAV_MENUS).map(menu => {
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

          {/* Forecast, Brain, Dossier and the cockpit live in ENGINE_MENU above
              (owner-only third drop-down, 2026-10-29). The phone keeps them as
              tabs in the More sheet via TABS. */}
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

      {/* ── Moving-map backdrop: one mount for the whole Trade world, so the
          canvas and its particle field survive tab switches ── */}
      {isTradeTab && <TradeBackdrop />}

      {/* ── Page content ── */}
      <main className="page-container tab-page">
        {isTradeTab && (
          <TradePage tab={activeTab as "analyze" | "research" | "scanner" | "news" | "watchlist" | "planner" | "taxes"} section={analyzeSection} authenticated={authenticated} isOwner={!!isOwner} />
        )}
        {activeTab === "data" && (
          <DataMapPage />
        )}
        {activeTab === "forecast" && (
          authenticated && isOwner ? <ForecastPage /> : (
            <div style={{ textAlign: "center", paddingTop: "4rem", color: "var(--text-tertiary)" }}>
              <Target size={48} style={{ margin: "0 auto 1rem", opacity: 0.3 }} />
              <p style={{ fontSize: "1.1rem", fontWeight: 600, color: "var(--text-secondary)" }}>
                {authenticated ? "The weekly forecast is restricted to the account owner" : "Sign in to see the weekly forecast"}
              </p>
              {!authenticated && (
                <button className="vt-fc-btn" style={{ marginTop: "1.5rem" }} onClick={() => setShowLogin(true)}>Sign in</button>
              )}
            </div>
          )
        )}
        {activeTab === "brain" && (
          authenticated && isOwner ? <BrainPage /> : (
            <div style={{ textAlign: "center", paddingTop: "4rem", color: "var(--text-tertiary)" }}>
              <Brain size={48} style={{ margin: "0 auto 1rem", opacity: 0.3 }} />
              <p style={{ fontSize: "1.1rem", fontWeight: 600, color: "var(--text-secondary)" }}>
                {authenticated ? "The Brain page is restricted to the account owner" : "Sign in to see what the bot is using"}
              </p>
              {!authenticated && (
                <button className="vt-fc-btn" style={{ marginTop: "1.5rem" }} onClick={() => setShowLogin(true)}>Sign in</button>
              )}
            </div>
          )
        )}
        {activeTab === "dossier" && (
          authenticated && isOwner ? <TickerDossierPage /> : (
            <div style={{ textAlign: "center", paddingTop: "4rem", color: "var(--text-tertiary)" }}>
              <FileSearch size={48} style={{ margin: "0 auto 1rem", opacity: 0.3 }} />
              <p style={{ fontSize: "1.1rem", fontWeight: 600, color: "var(--text-secondary)" }}>
                {authenticated ? "The ticker dossier is restricted to the account owner" : "Sign in to open the ticker dossier"}
              </p>
              {!authenticated && (
                <button className="vt-fc-btn" style={{ marginTop: "1.5rem" }} onClick={() => setShowLogin(true)}>Sign in</button>
              )}
            </div>
          )
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
      {(() => {
        const visible = TABS.filter(tab => !tab.requiresOwner || isOwner);
        const primary = visible.filter(t => MOBILE_PRIMARY_TABS.includes(t.id));
        const overflow = visible.filter(t => !MOBILE_PRIMARY_TABS.includes(t.id));
        const activeOverflow = overflow.find(t => t.id === activeTab);
        return (
          <>
            {mobileMoreOpen && (
              <div className="mobile-more-backdrop" onClick={() => setMobileMoreOpen(false)}>
                <div className="mobile-more-sheet" role="dialog" aria-label="More sections"
                     onClick={(e) => e.stopPropagation()}>
                  <div className="mobile-more-grip" aria-hidden="true" />
                  <div className="mobile-more-grid">
                    {overflow.map(tab => (
                      <button key={tab.id}
                              className={`mobile-more-item ${activeTab === tab.id ? "active" : ""}`}
                              onClick={() => handleTabClick(tab.id)}>
                        {tab.mobileIcon}
                        <span>{tab.label}</span>
                        {tab.requiresAuth && !authenticated && <span className="mobile-tab-lock">🔒</span>}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}
            <nav className="mobile-bottom-bar">
              {primary.map(tab => (
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
              <button
                className={`mobile-tab-btn ${activeOverflow || mobileMoreOpen ? "active" : ""}`}
                aria-expanded={mobileMoreOpen}
                aria-label={activeOverflow ? `More sections (showing ${activeOverflow.label})` : "More sections"}
                onClick={() => setMobileMoreOpen(v => !v)}
                data-vt-mobile-more
              >
                {activeOverflow ? activeOverflow.mobileIcon : <LayoutGrid size={20} />}
                <span className="mobile-tab-label">{activeOverflow ? activeOverflow.label : "More"}</span>
              </button>
            </nav>
          </>
        );
      })()}
    </div>
  );
}
