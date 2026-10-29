import { useState, useEffect, useRef } from "react";
import { getActiveTicker } from "@/lib/activeTicker";
import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { Search, RefreshCw, ExternalLink, TrendingUp, TrendingDown, Minus } from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

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
  count?: number;
  next_url?: string;
}

type SentimentLabel = "Bullish" | "Bearish" | "Neutral";

// ── Helpers ───────────────────────────────────────────────────────────────────

const BULLISH_WORDS = [
  "surge", "rally", "soar", "gain", "jump", "rise", "beat", "record", "high",
  "profit", "growth", "buy", "bullish", "upgrade", "outperform", "exceed",
  "strong", "positive", "boost", "upside", "breakout", "milestone", "recover",
];
const BEARISH_WORDS = [
  "fall", "drop", "plunge", "decline", "loss", "miss", "cut", "lower",
  "bearish", "downgrade", "underperform", "concern", "risk", "warn", "sell",
  "crash", "collapse", "disappointing", "weak", "recession", "fear", "halt",
];

function deriveSentiment(article: NewsArticle): SentimentLabel {
  const text = `${article.title} ${article.description ?? ""} ${(article.keywords ?? []).join(" ")}`.toLowerCase();
  let bull = 0;
  let bear = 0;
  for (const w of BULLISH_WORDS) if (text.includes(w)) bull++;
  for (const w of BEARISH_WORDS) if (text.includes(w)) bear++;
  if (bull > bear) return "Bullish";
  if (bear > bull) return "Bearish";
  return "Neutral";
}

function timeAgo(utcString: string): string {
  const now = Date.now();
  const then = new Date(utcString).getTime();
  const diff = now - then;
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

// ── News Body ─────────────────────────────────────────────────────────────────
// 2026-10-29: the list is its own component so the cockpit can place it in a
// panel. A `ticker` from the host is the filter (the cockpit's one look-up
// box owns the symbol); `showFilter` adds the typed, debounced filter the
// page always had, seeded from `ticker`. The body owns the fetch, so the
// story count, data age and Refresh live in its status row.

export function NewsBody({ ticker, onSelectTicker, showFilter }: {
  ticker: string | null;
  onSelectTicker: (ticker: string) => void;
  showFilter?: boolean;
}) {
  const [tickerFilter, setTickerFilter] = useState(ticker ?? "");
  const [debouncedTicker, setDebouncedTicker] = useState(ticker ?? "");
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // A new symbol from the host replaces whatever was typed here.
  useEffect(() => {
    setTickerFilter(ticker ?? "");
    setDebouncedTicker(ticker ?? "");
  }, [ticker]);

  // Debounce ticker input (only the typed filter needs it)
  useEffect(() => {
    if (!showFilter) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      setDebouncedTicker(tickerFilter.trim().toUpperCase());
    }, 600);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [tickerFilter, showFilter]);

  const activeFilter = showFilter ? debouncedTicker : (ticker ?? "");

  const { data, isLoading, isError, refetch, isFetching, dataUpdatedAt } = useQuery<NewsResponse>({
    queryKey: ["/api/market/news", activeFilter],
    queryFn: async () => {
      const url = activeFilter
        ? `/api/market/news?ticker=${encodeURIComponent(activeFilter)}`
        : `/api/market/news`;
      const res = await apiRequest("GET", url);
      return res.json();
    },
    staleTime: 5 * 60 * 1000,
    refetchInterval: 5 * 60 * 1000,
    retry: 2,
  });

  const articles = data?.results ?? [];
  const lastUpdate = dataUpdatedAt ? new Date(dataUpdatedAt).toLocaleTimeString() : null;

  return (
    <div>
      {/* Status + refresh */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem', flexWrap: 'wrap' }}>
        <p style={{ flex: 1, margin: 0, fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
          {isLoading ? 'Loading…' : `${articles.length} stories`}
          {lastUpdate && ` • Updated ${lastUpdate}`}
          {' • Auto-refreshes every 5 min'}
        </p>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            padding: '0.4rem 0.875rem',
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: '8px',
            fontSize: '0.8rem',
            color: 'var(--text-secondary)',
          }}
        >
          <RefreshCw size={13} className={isFetching ? "animate-spin" : ""} />
          {isFetching ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {/* Ticker filter — only when this body owns the symbol */}
      {showFilter && (
      <div style={{ position: 'relative', maxWidth: '320px', marginBottom: '1.25rem' }}>
        <Search
          size={15}
          style={{
            position: 'absolute',
            left: '0.75rem',
            top: '50%',
            transform: 'translateY(-50%)',
            color: 'var(--text-tertiary)',
            pointerEvents: 'none'
          }}
        />
        <input
          type="text"
          value={tickerFilter}
          onChange={e => setTickerFilter(e.target.value.toUpperCase())}
          placeholder="Filter by ticker (e.g. AAPL)…"
          style={{
            width: '100%',
            height: '40px',
            padding: '0 0.875rem 0 2.25rem',
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: '10px',
            color: 'var(--text-primary)',
            fontSize: '0.85rem',
          }}
        />
        {tickerFilter && (
          <button
            onClick={() => setTickerFilter("")}
            style={{
              position: 'absolute',
              right: '0.625rem',
              top: '50%',
              transform: 'translateY(-50%)',
              color: 'var(--text-tertiary)',
              fontSize: '0.85rem',
            }}
          >
            ✕
          </button>
        )}
      </div>
      )}

      {/* Content */}
      {isLoading && (
        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
          <RefreshCw size={24} className="animate-spin" style={{ margin: '0 auto 0.75rem' }} />
          <p style={{ fontSize: '0.88rem' }}>Loading market news…</p>
        </div>
      )}

      {isError && !isLoading && (
        <div style={{
          padding: '1.5rem',
          background: 'rgba(255,69,58,0.07)',
          border: '1px solid rgba(255,69,58,0.2)',
          borderRadius: '12px',
          textAlign: 'center',
        }}>
          <p style={{ color: '#ff453a', fontSize: '0.88rem', marginBottom: '0.75rem' }}>
            Failed to load news.
          </p>
          <button
            onClick={() => refetch()}
            style={{
              padding: '0.4rem 1rem',
              background: 'var(--bg-card-hover)',
              border: '1px solid var(--border)',
              borderRadius: '8px',
              color: 'var(--text-primary)',
              fontSize: '0.82rem',
            }}
          >
            Try Again
          </button>
        </div>
      )}

      {!isLoading && !isError && articles.length === 0 && (
        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.88rem' }}>
          {activeFilter ? `No news found for ${activeFilter}.` : 'No news available at this time.'}
        </div>
      )}

      {!isLoading && articles.length > 0 && (
        <div className="news-grid">
          {articles.map((article) => {
            const sentiment = deriveSentiment(article);
            return (
              <NewsCard
                key={article.id}
                article={article}
                sentiment={sentiment}
                onSelectTicker={onSelectTicker}
              />
            );
          })}
        </div>
      )}

      <p style={{
        fontSize: '0.72rem',
        color: 'var(--text-tertiary)',
        marginTop: '1.25rem',
        textAlign: 'center',
      }}>
        News powered by Alpaca Markets · Sentiment derived from keywords · Not financial advice
      </p>
    </div>
  );
}

// ── News Page ─────────────────────────────────────────────────────────────────

export default function NewsPage({ onSelectTicker }: { onSelectTicker: (ticker: string) => void }) {
  // One symbol across the Trade world (lib/activeTicker): the filter opens
  // on the name the user was just analyzing; clearing it shows the market.
  const [seed] = useState(() => getActiveTicker() ?? "");

  return (
    <div>
      {/* Page header */}
      <div style={{ marginBottom: '1.25rem' }}>
        <h1 style={{
          fontSize: '1.4rem',
          fontWeight: 700,
          letterSpacing: '-0.02em',
          color: 'var(--text-primary)',
          marginBottom: '0.25rem'
        }}>
          Market News
        </h1>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
          Headlines across the market or for one symbol, with keyword-derived sentiment.
        </p>
      </div>
      <NewsBody ticker={seed || null} onSelectTicker={onSelectTicker} showFilter />
    </div>
  );
}

// ── News Card ─────────────────────────────────────────────────────────────────

function NewsCard({
  article,
  sentiment,
  onSelectTicker,
}: {
  article: NewsArticle;
  sentiment: SentimentLabel;
  onSelectTicker: (ticker: string) => void;
}) {
  const SentimentIcon = sentiment === "Bullish"
    ? <TrendingUp size={10} />
    : sentiment === "Bearish"
    ? <TrendingDown size={10} />
    : <Minus size={10} />;

  return (
    <div className="news-card">
      {/* Publisher + time */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        marginBottom: '0.5rem',
      }}>
        {article.publisher.favicon_url && (
          <img
            src={article.publisher.favicon_url}
            alt=""
            style={{ width: 14, height: 14, borderRadius: '3px', flexShrink: 0 }}
            onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
          />
        )}
        <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {article.publisher.name}
        </span>
        <span style={{ fontSize: '0.68rem', color: 'var(--text-tertiary)', flexShrink: 0 }}>
          {timeAgo(article.published_utc)}
        </span>
      </div>

      {/* Headline */}
      <a
        href={article.article_url}
        target="_blank"
        rel="noopener noreferrer"
        style={{ textDecoration: 'none' }}
      >
        <h3 style={{
          fontSize: '0.9rem',
          fontWeight: 600,
          color: 'var(--text-primary)',
          lineHeight: 1.4,
          marginBottom: '0.5rem',
          display: '-webkit-box',
          WebkitLineClamp: 3,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
        }}>
          {article.title}
        </h3>
      </a>

      {/* Description */}
      {article.description && (
        <p style={{
          fontSize: '0.78rem',
          color: 'var(--text-secondary)',
          lineHeight: 1.5,
          marginBottom: '0.625rem',
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
        }}>
          {article.description}
        </p>
      )}

      {/* Footer row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
        {/* Sentiment badge */}
        <span className={`sentiment-badge ${sentiment.toLowerCase()}`}>
          {SentimentIcon}
          {sentiment}
        </span>

        {/* Related tickers */}
        {article.tickers && article.tickers.length > 0 && (
          <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap', flex: 1 }}>
            {article.tickers.slice(0, 5).map(t => (
              <button
                key={t}
                onClick={(e) => { e.stopPropagation(); onSelectTicker(t); }}
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.68rem',
                  fontWeight: 700,
                  color: '#60a5fa',
                  background: 'rgba(10,132,255,0.08)',
                  border: '1px solid rgba(10,132,255,0.15)',
                  borderRadius: '4px',
                  padding: '0.1rem 0.35rem',
                  letterSpacing: '0.04em',
                  transition: 'background 120ms',
                }}
                onMouseEnter={e => (e.currentTarget.style.background = 'rgba(10,132,255,0.15)')}
                onMouseLeave={e => (e.currentTarget.style.background = 'rgba(10,132,255,0.08)')}
              >
                {t}
              </button>
            ))}
          </div>
        )}

        {/* External link */}
        <a
          href={article.article_url}
          target="_blank"
          rel="noopener noreferrer"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.25rem',
            fontSize: '0.72rem',
            color: 'var(--text-tertiary)',
            textDecoration: 'none',
            marginLeft: 'auto',
            flexShrink: 0,
          }}
          onMouseEnter={e => (e.currentTarget.style.color = 'var(--text-secondary)')}
          onMouseLeave={e => (e.currentTarget.style.color = 'var(--text-tertiary)')}
        >
          Read <ExternalLink size={10} />
        </a>
      </div>
    </div>
  );
}
