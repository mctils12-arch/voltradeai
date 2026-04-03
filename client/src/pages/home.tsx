import { useState } from "react";
import { Moon, Sun, BarChart2, ScanLine, Newspaper, Bookmark } from "lucide-react";
import AnalyzePage from "./analyze";
import ScannerPage from "./scanner";
import NewsPage from "./news";
import WatchlistPage from "./watchlist";

// ── Tab types ─────────────────────────────────────────────────────────────────

type TabId = "analyze" | "scanner" | "news" | "watchlist";

interface Tab {
  id: TabId;
  label: string;
  icon: React.ReactNode;
}

const TABS: Tab[] = [
  { id: "analyze",   label: "Analyze",   icon: <BarChart2  size={14} /> },
  { id: "scanner",   label: "Scanner",   icon: <ScanLine   size={14} /> },
  { id: "news",      label: "News",      icon: <Newspaper  size={14} /> },
  { id: "watchlist", label: "Watchlist", icon: <Bookmark   size={14} /> },
];

// ── VolTradeAI Logo SVG ───────────────────────────────────────────────────────

function Logo() {
  return (
    <svg aria-label="VolTradeAI" width="28" height="28" viewBox="0 0 32 32" fill="none">
      <rect width="32" height="32" rx="8" fill="#0a84ff" fillOpacity="0.15" />
      <rect width="32" height="32" rx="8" stroke="#0a84ff" strokeWidth="1" fill="none" strokeOpacity="0.4" />
      <polyline
        points="4,22 10,14 16,18 22,8 28,12"
        stroke="#0a84ff"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
      <circle cx="22" cy="8" r="2.5" fill="#0a84ff" />
    </svg>
  );
}

// ── Home (Tab Shell) ──────────────────────────────────────────────────────────

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabId>("analyze");
  const [dark, setDark] = useState(true);
  const [analyzeTarget, setAnalyzeTarget] = useState<string | undefined>(undefined);

  // Called by Scanner, News, Watchlist when user clicks a ticker
  const handleSelectTicker = (ticker: string) => {
    setAnalyzeTarget(ticker);
    setActiveTab("analyze");
  };

  return (
    <div className={dark ? "dark" : "light"} style={{
      minHeight: "100dvh",
      background: dark ? "#000000" : "#f5f5f7",
      color: dark ? "#f5f5f7" : "#1d1d1f",
    }}>

      {/* ── Top nav bar ── */}
      <nav className="tab-nav" style={dark ? {} : { background: 'rgba(255,255,255,0.85)' }}>
        {/* Logo */}
        <div className="tab-nav-logo">
          <Logo />
          <span className="tab-nav-logo-text">VolTradeAI</span>
        </div>

        {/* Tabs */}
        <div className="tab-nav-tabs">
          {TABS.map(tab => (
            <button
              key={tab.id}
              className={`tab-btn ${activeTab === tab.id ? "active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
              aria-current={activeTab === tab.id ? "page" : undefined}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>

        {/* Theme toggle */}
        <button
          className="theme-toggle-btn"
          onClick={() => setDark(!dark)}
          aria-label="Toggle light/dark mode"
          title={dark ? "Switch to light mode" : "Switch to dark mode"}
        >
          {dark ? <Sun size={15} /> : <Moon size={15} />}
        </button>
      </nav>

      {/* ── Page content ── */}
      <main className="page-container tab-page">
        {activeTab === "analyze" && (
          <AnalyzePage
            key={analyzeTarget}
            initialTicker={analyzeTarget}
          />
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
      </main>
    </div>
  );
}
