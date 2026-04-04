import { useState, useEffect, useRef, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import {
  Search,
  TrendingUp,
  TrendingDown,
  ChevronDown,
  ChevronUp,
  Send,
  BarChart3,
  DollarSign,
  Activity,
  Filter,
  Star,
  Minus,
  ExternalLink,
} from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

interface IndexData {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePct: number;
  sparkline: number[];
}

interface NewsArticle {
  id: string;
  title: string;
  published_utc: string;
  article_url: string;
  image_url?: string;
  description?: string;
  publisher: {
    name: string;
    homepage_url?: string;
    logo_url?: string;
    favicon_url?: string;
  };
  tickers?: string[];
  keywords?: string[];
}

interface NewsResponse {
  results: NewsArticle[];
}

interface MarketTicker {
  ticker: string;
  close: number;
  open: number;
  high: number;
  low: number;
  volume: number;
  change_pct: number;
  vwap: number;
}

interface MarketSnapshotResponse {
  results: MarketTicker[];
  date: string;
  total: number;
}

type SubTab = "markets" | "crypto" | "earnings" | "screener";

// ── Popular tickers per sub-tab ──────────────────────────────────────────────

const TRENDING_TICKERS: Record<SubTab, string[]> = {
  markets: ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "AMD", "NFLX", "DIS"],
  crypto: ["COIN", "MSTR", "RIOT", "MARA", "HUT", "CLSK", "BTBT", "WULF"],
  earnings: ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD"],
  screener: ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV"],
};

const INDEX_SYMBOLS: Record<string, string> = {
  SPY: "S&P 500",
  QQQ: "NASDAQ",
  DIA: "Dow Jones",
  "^VIX": "VIX",
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function timeAgo(utcString: string): string {
  const diff = Date.now() - new Date(utcString).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

function deriveSentiment(article: NewsArticle): "Bullish" | "Bearish" | "Neutral" {
  const text = `${article.title} ${article.description ?? ""} ${(article.keywords ?? []).join(" ")}`.toLowerCase();
  const bullWords = ["surge", "rally", "soar", "gain", "jump", "rise", "beat", "record", "high", "profit", "growth", "buy", "bullish", "upgrade", "strong", "positive", "boost"];
  const bearWords = ["fall", "drop", "plunge", "decline", "loss", "miss", "cut", "lower", "bearish", "downgrade", "concern", "risk", "warn", "sell", "crash", "weak"];
  let bull = 0, bear = 0;
  for (const w of bullWords) if (text.includes(w)) bull++;
  for (const w of bearWords) if (text.includes(w)) bear++;
  if (bull > bear) return "Bullish";
  if (bear > bull) return "Bearish";
  return "Neutral";
}

// ── Sparkline SVG ────────────────────────────────────────────────────────────

function Sparkline({ data, positive, width = 80, height = 32 }: { data: number[]; positive: boolean; width?: number; height?: number }) {
  if (!data || data.length < 2) return <div style={{ width, height }} />;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * width;
      const y = height - 2 - ((v - min) / range) * (height - 4);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  const color = positive ? "#00ff41" : "#ff3333";

  // Create area fill path
  const firstPoint = `0,${height}`;
  const lastPoint = `${width},${height}`;
  const areaPath = `M${firstPoint} L${points.split(" ").join(" L")} L${lastPoint} Z`;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: "block" }}>
      <defs>
        <linearGradient id={`sparkGrad-${positive ? "g" : "r"}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.2} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={areaPath} fill={`url(#sparkGrad-${positive ? "g" : "r"})`} />
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ── Market Sentiment Badge ───────────────────────────────────────────────────

function MarketSentiment({ articles }: { articles: NewsArticle[] }) {
  const counts = { Bullish: 0, Bearish: 0, Neutral: 0 };
  for (const a of articles) counts[deriveSentiment(a)]++;

  let label: string;
  let color: string;
  if (counts.Bullish > counts.Bearish * 1.5) {
    label = "Bullish Sentiment";
    color = "#00ff41";
  } else if (counts.Bearish > counts.Bullish * 1.5) {
    label = "Bearish Sentiment";
    color = "#ff3333";
  } else {
    label = "Mixed Sentiment";
    color = "#d4a017";
  }

  return (
    <div style={{
      display: "inline-flex",
      alignItems: "center",
      gap: 6,
      padding: "4px 10px",
      borderRadius: 3,
      background: `${color}11`,
      border: `1px solid ${color}33`,
      fontSize: "0.72rem",
      fontWeight: 600,
      color,
      fontFamily: "var(--font-mono)",
      letterSpacing: "0.05em",
      textTransform: "uppercase",
    }}>
      <Activity size={12} />
      {label}
    </div>
  );
}

// ── Search Page Component ────────────────────────────────────────────────────

export default function SearchPage({ onSelectTicker }: { onSelectTicker: (ticker: string) => void }) {
  const [searchQuery, setSearchQuery] = useState("");
  const [subTab, setSubTab] = useState<SubTab>("markets");
  const [expandedNews, setExpandedNews] = useState<Set<string>>(new Set());
  const [askInput, setAskInput] = useState("");
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Fetch market snapshot for top assets & trending tickers
  const { data: snapshotData } = useQuery<MarketSnapshotResponse>({
    queryKey: ["/api/market-snapshot"],
    queryFn: async () => {
      const res = await apiRequest("GET", "/api/market-snapshot");
      return res.json();
    },
    staleTime: 5 * 60 * 1000,
    refetchInterval: 5 * 60 * 1000,
    retry: 2,
  });

  // Fetch news for market summary
  const { data: newsData } = useQuery<NewsResponse>({
    queryKey: ["/api/news"],
    queryFn: async () => {
      const res = await apiRequest("GET", "/api/news");
      return res.json();
    },
    staleTime: 5 * 60 * 1000,
    refetchInterval: 5 * 60 * 1000,
    retry: 2,
  });

  // Build index data from snapshot
  const indexData: IndexData[] = useMemo(() => {
    if (!snapshotData?.results) return [];
    const indices: IndexData[] = [];
    for (const [sym, name] of Object.entries(INDEX_SYMBOLS)) {
      const ticker = snapshotData.results.find((r) => r.ticker === sym);
      if (ticker) {
        // Generate sparkline from available data (high/low/open/close creates a mini trend)
        const sparkline = [ticker.open, (ticker.open + ticker.high) / 2, ticker.high, (ticker.high + ticker.low) / 2, ticker.low, (ticker.low + ticker.close) / 2, ticker.close];
        indices.push({
          symbol: sym,
          name,
          price: ticker.close,
          change: ticker.close - ticker.open,
          changePct: ticker.change_pct,
          sparkline,
        });
      }
    }
    return indices;
  }, [snapshotData]);

  // Build trending ticker data from snapshot
  const trendingData = useMemo(() => {
    if (!snapshotData?.results) return [];
    const tickers = TRENDING_TICKERS[subTab];
    return tickers
      .map((sym) => {
        const t = snapshotData.results.find((r) => r.ticker === sym);
        if (!t) return null;
        return { ticker: t.ticker, price: t.close, changePct: t.change_pct, volume: t.volume };
      })
      .filter(Boolean) as { ticker: string; price: number; changePct: number; volume: number }[];
  }, [snapshotData, subTab]);

  // Watchlist data for sidebar
  const { data: watchlistData } = useQuery<{ tickers: string[] }>({
    queryKey: ["/api/watchlist"],
    queryFn: async () => {
      const res = await apiRequest("GET", "/api/watchlist");
      return res.json();
    },
    staleTime: 60 * 1000,
    retry: 1,
  });

  const watchlistPrices = useMemo(() => {
    if (!snapshotData?.results || !watchlistData?.tickers) return [];
    return watchlistData.tickers.slice(0, 8).map((sym) => {
      const t = snapshotData.results.find((r) => r.ticker === sym);
      return {
        ticker: sym,
        price: t?.close ?? 0,
        changePct: t?.change_pct ?? 0,
      };
    });
  }, [snapshotData, watchlistData]);

  const articles = newsData?.results ?? [];

  // Handle search submission
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const q = searchQuery.trim().toUpperCase();
    if (q) {
      onSelectTicker(q);
    }
  };

  // Handle "ask anything" submission
  const handleAsk = (e: React.FormEvent) => {
    e.preventDefault();
    const q = askInput.trim().toUpperCase();
    if (q) {
      // Try to extract a ticker from the question
      const match = q.match(/\b([A-Z]{1,5})\b/);
      if (match) {
        onSelectTicker(match[1]);
      }
      setAskInput("");
    }
  };

  const toggleNewsExpand = (id: string) => {
    setExpandedNews((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const SUB_TABS: { id: SubTab; label: string; icon: React.ReactNode }[] = [
    { id: "markets", label: "Markets", icon: <BarChart3 size={13} /> },
    { id: "crypto", label: "Crypto", icon: <DollarSign size={13} /> },
    { id: "earnings", label: "Earnings", icon: <TrendingUp size={13} /> },
    { id: "screener", label: "Screener", icon: <Filter size={13} /> },
  ];

  return (
    <div className="search-page-layout">
      {/* ── Main Content Area ── */}
      <div className="search-main">
        {/* Search Bar */}
        <form onSubmit={handleSearch} className="search-hero-form">
          <div className="search-hero-wrapper">
            <Search size={18} className="search-hero-icon" />
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search for stocks, crypto, and more..."
              className="search-hero-input"
              autoComplete="off"
              spellCheck={false}
            />
          </div>
        </form>

        {/* Sub-tabs Row */}
        <div className="search-subtabs">
          {SUB_TABS.map((t) => (
            <button
              key={t.id}
              className={`search-subtab ${subTab === t.id ? "active" : ""}`}
              onClick={() => setSubTab(t.id)}
            >
              {t.icon}
              {t.label}
            </button>
          ))}
          <div style={{ marginLeft: "auto" }}>
            {articles.length > 0 && <MarketSentiment articles={articles} />}
          </div>
        </div>

        {/* Top Assets Cards */}
        {indexData.length > 0 && (
          <section className="search-section">
            <h3 className="search-section-title">Top Assets</h3>
            <div className="search-index-grid">
              {indexData.map((idx) => {
                const positive = idx.changePct >= 0;
                return (
                  <button
                    key={idx.symbol}
                    className="search-index-card glass-card"
                    onClick={() => onSelectTicker(idx.symbol)}
                  >
                    <div className="search-index-header">
                      <span className="search-index-name">{idx.name}</span>
                      <span className="search-index-symbol">{idx.symbol}</span>
                    </div>
                    <div className="search-index-price">
                      ${idx.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                    <div className="search-index-row">
                      <span
                        className="search-index-change"
                        style={{ color: positive ? "#00ff41" : "#ff3333" }}
                      >
                        {positive ? "+" : ""}
                        {idx.changePct.toFixed(2)}%
                        <span style={{ fontSize: "0.7rem", marginLeft: 4, opacity: 0.8 }}>
                          {positive ? "+" : ""}${idx.change.toFixed(2)}
                        </span>
                      </span>
                      <Sparkline data={idx.sparkline} positive={positive} width={72} height={28} />
                    </div>
                  </button>
                );
              })}
            </div>
          </section>
        )}

        {/* Market Summary */}
        {articles.length > 0 && (
          <section className="search-section">
            <h3 className="search-section-title">Market Summary</h3>
            <div className="search-news-list">
              {articles.slice(0, 6).map((article) => {
                const expanded = expandedNews.has(article.id);
                const sentiment = deriveSentiment(article);
                const sentColor = sentiment === "Bullish" ? "#00ff41" : sentiment === "Bearish" ? "#ff3333" : "#7a8ba0";
                return (
                  <div key={article.id} className="search-news-item glass-card">
                    <button
                      className="search-news-header"
                      onClick={() => toggleNewsExpand(article.id)}
                    >
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div className="search-news-title">{article.title}</div>
                        <div className="search-news-meta">
                          <span style={{ color: sentColor, fontWeight: 600 }}>
                            {sentiment === "Bullish" ? <TrendingUp size={11} style={{ display: "inline", verticalAlign: "-2px", marginRight: 3 }} /> : sentiment === "Bearish" ? <TrendingDown size={11} style={{ display: "inline", verticalAlign: "-2px", marginRight: 3 }} /> : <Minus size={11} style={{ display: "inline", verticalAlign: "-2px", marginRight: 3 }} />}
                            {sentiment}
                          </span>
                          <span style={{ color: "var(--text-tertiary)" }}>{article.publisher.name}</span>
                          <span style={{ color: "var(--text-tertiary)" }}>{timeAgo(article.published_utc)}</span>
                        </div>
                      </div>
                      {expanded ? <ChevronUp size={16} style={{ color: "var(--text-tertiary)", flexShrink: 0 }} /> : <ChevronDown size={16} style={{ color: "var(--text-tertiary)", flexShrink: 0 }} />}
                    </button>
                    {expanded && (
                      <div className="search-news-body">
                        {article.description && (
                          <p style={{ color: "var(--text-secondary)", fontSize: "0.82rem", lineHeight: 1.6, marginBottom: 8 }}>
                            {article.description}
                          </p>
                        )}
                        {article.tickers && article.tickers.length > 0 && (
                          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 8 }}>
                            {article.tickers.slice(0, 6).map((t) => (
                              <button
                                key={t}
                                onClick={() => onSelectTicker(t)}
                                className="chip"
                                style={{ cursor: "pointer" }}
                              >
                                {t}
                              </button>
                            ))}
                          </div>
                        )}
                        <a
                          href={article.article_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 4,
                            fontSize: "0.75rem",
                            color: "var(--accent-color)",
                            textDecoration: "none",
                          }}
                        >
                          Read full article <ExternalLink size={11} />
                        </a>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {/* Trending / Popular Tickers */}
        <section className="search-section">
          <h3 className="search-section-title">
            {subTab === "markets" ? "Popular Tickers" : subTab === "crypto" ? "Crypto Stocks" : subTab === "earnings" ? "Earnings Watch" : "ETF Screener"}
          </h3>
          {trendingData.length > 0 ? (
            <div className="search-trending-grid">
              {trendingData.map((t) => {
                const positive = t.changePct >= 0;
                return (
                  <button
                    key={t.ticker}
                    className="search-trending-card glass-card"
                    onClick={() => onSelectTicker(t.ticker)}
                  >
                    <div className="search-trending-header">
                      <span className="search-trending-ticker">{t.ticker}</span>
                      <span
                        className="search-trending-change"
                        style={{ color: positive ? "#00ff41" : "#ff3333" }}
                      >
                        {positive ? "+" : ""}{t.changePct.toFixed(2)}%
                      </span>
                    </div>
                    <div className="search-trending-price">
                      ${t.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                    <div style={{ fontSize: "0.68rem", color: "var(--text-tertiary)" }}>
                      Vol: {(t.volume / 1e6).toFixed(1)}M
                    </div>
                  </button>
                );
              })}
            </div>
          ) : (
            <div style={{ color: "var(--text-tertiary)", fontSize: "0.82rem", padding: "1rem 0" }}>
              Loading market data...
            </div>
          )}
        </section>

        {/* Ask Anything Input */}
        <section className="search-section">
          <form onSubmit={handleAsk} className="search-ask-form">
            <div className="search-ask-wrapper">
              <input
                type="text"
                value={askInput}
                onChange={(e) => setAskInput(e.target.value)}
                placeholder="Ask anything about markets..."
                className="search-ask-input"
                autoComplete="off"
              />
              <button type="submit" className="search-ask-btn" disabled={!askInput.trim()}>
                <Send size={16} />
              </button>
            </div>
          </form>
        </section>
      </div>

      {/* ── Right Sidebar (desktop only) ── */}
      <aside className="search-sidebar">
        {/* Watchlist Preview */}
        <div className="search-sidebar-section glass-card">
          <h4 className="search-sidebar-title">
            <Star size={13} style={{ color: "#d4a017" }} />
            Watchlist
          </h4>
          {watchlistPrices.length > 0 ? (
            <div className="search-watchlist-items">
              {watchlistPrices.map((w) => {
                const positive = w.changePct >= 0;
                return (
                  <button
                    key={w.ticker}
                    className="search-watchlist-row"
                    onClick={() => onSelectTicker(w.ticker)}
                  >
                    <span className="search-watchlist-ticker">{w.ticker}</span>
                    <span className="search-watchlist-price">
                      {w.price > 0 ? `$${w.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "—"}
                    </span>
                    <span
                      className="search-watchlist-change"
                      style={{ color: positive ? "#00ff41" : "#ff3333" }}
                    >
                      {positive ? "+" : ""}{w.changePct.toFixed(2)}%
                    </span>
                  </button>
                );
              })}
            </div>
          ) : (
            <div style={{ color: "var(--text-tertiary)", fontSize: "0.75rem", padding: "0.5rem 0" }}>
              Sign in to see your watchlist
            </div>
          )}
        </div>

        {/* Market Sentiment Card */}
        <div className="search-sidebar-section glass-card">
          <h4 className="search-sidebar-title">
            <Activity size={13} style={{ color: "#00e5ff" }} />
            Market Overview
          </h4>
          {indexData.length > 0 ? (
            <div className="search-sidebar-indices">
              {indexData.map((idx) => {
                const positive = idx.changePct >= 0;
                return (
                  <div key={idx.symbol} className="search-sidebar-index-row">
                    <span style={{ fontSize: "0.72rem", color: "var(--text-secondary)" }}>{idx.name}</span>
                    <span style={{ fontSize: "0.72rem", color: positive ? "#00ff41" : "#ff3333", fontWeight: 600, fontFamily: "var(--font-mono)" }}>
                      {positive ? "+" : ""}{idx.changePct.toFixed(2)}%
                    </span>
                  </div>
                );
              })}
            </div>
          ) : (
            <div style={{ color: "var(--text-tertiary)", fontSize: "0.75rem", padding: "0.5rem 0" }}>
              Loading...
            </div>
          )}
        </div>

        {/* Quick Tickers */}
        <div className="search-sidebar-section glass-card">
          <h4 className="search-sidebar-title">Quick Access</h4>
          <div className="search-sidebar-chips">
            {["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "MSFT"].map((t) => (
              <button key={t} className="chip" onClick={() => onSelectTicker(t)} style={{ cursor: "pointer" }}>
                {t}
              </button>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}
