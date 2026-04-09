import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { Plus, Trash2, TrendingUp, TrendingDown, Search, RefreshCw } from "lucide-react";

// ── Watchlist Row ─────────────────────────────────────────────────────────────

function WatchlistRow({
  ticker,
  priceData,
  pricesLoading,
  onRemove,
  onAnalyze,
}: {
  ticker: string;
  priceData?: { price: number; change: number };
  pricesLoading: boolean;
  onRemove: () => void;
  onAnalyze: () => void;
}) {
  const isUp = priceData ? priceData.change >= 0 : true;

  return (
    <tr>
      <td>
        <div>
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontWeight: 700,
            fontSize: '0.92rem',
            color: 'var(--text-primary)',
          }}>
            {ticker}
          </span>
        </div>
      </td>
      <td style={{ textAlign: 'right' }}>
        {pricesLoading && !priceData ? (
          <span style={{ fontSize: '0.78rem', color: 'var(--text-tertiary)' }}>
            <RefreshCw size={12} className="animate-spin" style={{ display: 'inline' }} />
          </span>
        ) : priceData ? (
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontWeight: 600,
            fontSize: '0.9rem',
            color: 'var(--text-primary)',
          }}>
            ${Number(priceData.price ?? 0).toFixed(2)}
          </span>
        ) : (
          <span style={{ fontSize: '0.78rem', color: 'var(--text-tertiary)' }}>—</span>
        )}
      </td>
      <td style={{ textAlign: 'right' }}>
        {priceData ? (
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontWeight: 600,
            fontSize: '0.82rem',
            color: isUp ? '#30d158' : '#ff453a',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.2rem',
          }}>
            {isUp ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
            {isUp ? '+' : ''}{Number(priceData.change ?? 0).toFixed(2)}%
          </span>
        ) : (
          <span style={{ fontSize: '0.78rem', color: 'var(--text-tertiary)' }}>—</span>
        )}
      </td>
      <td style={{ textAlign: 'right' }}>
        <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end', alignItems: 'center' }}>
          <button
            onClick={onAnalyze}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.3rem',
              padding: '0.3rem 0.7rem',
              background: 'rgba(10,132,255,0.1)',
              border: '1px solid rgba(10,132,255,0.2)',
              borderRadius: '6px',
              fontSize: '0.75rem',
              fontWeight: 600,
              color: '#60a5fa',
              transition: 'background 120ms',
            }}
            onMouseEnter={e => (e.currentTarget.style.background = 'rgba(10,132,255,0.18)')}
            onMouseLeave={e => (e.currentTarget.style.background = 'rgba(10,132,255,0.1)')}
          >
            <TrendingUp size={11} />
            Analyze
          </button>
          <button
            onClick={onRemove}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              padding: '0.3rem',
              background: 'transparent',
              border: '1px solid var(--border-subtle)',
              borderRadius: '6px',
              color: 'var(--text-tertiary)',
              transition: 'all 120ms',
            }}
            onMouseEnter={e => {
              (e.currentTarget.style.color) = '#ff453a';
              (e.currentTarget.style.borderColor) = 'rgba(255,69,58,0.3)';
            }}
            onMouseLeave={e => {
              (e.currentTarget.style.color) = 'var(--text-tertiary)';
              (e.currentTarget.style.borderColor) = 'var(--border-subtle)';
            }}
            aria-label={`Remove ${ticker}`}
          >
            <Trash2 size={13} />
          </button>
        </div>
      </td>
    </tr>
  );
}

// ── Watchlist Page ────────────────────────────────────────────────────────────

export default function WatchlistPage({ onSelectTicker, authenticated }: { onSelectTicker: (ticker: string) => void; authenticated?: boolean }) {
  const [tickers, setTickers] = useState<string[]>([]);
  const [input, setInput] = useState("");
  const isLoggedIn = !!authenticated;

  // Load watchlist from API (Bug 7: use parent auth, Bug 9: use useQuery)
  const { data: watchlistData, isLoading: watchlistLoading } = useQuery<{ tickers: string[] }>({
    queryKey: ["/api/watchlist"],
    queryFn: async () => {
      const r = await apiRequest("GET", "/api/watchlist");
      return r.json();
    },
    enabled: isLoggedIn,
    staleTime: 60_000,
    retry: 1,
  });

  useEffect(() => {
    if (watchlistData?.tickers && Array.isArray(watchlistData.tickers)) {
      setTickers(watchlistData.tickers);
    }
  }, [watchlistData]);

  // Fetch prices using useQuery (Bug 8: error handling, Bug 9: use useQuery)
  const symbolsKey = tickers.join(",");
  const { data: priceData, isLoading: pricesLoading, isError: priceError } = useQuery<Record<string, { price: number; change: number }>>({
    queryKey: ["/api/market/snapshots", symbolsKey],
    queryFn: async () => {
      const res = await apiRequest("GET", `/api/market/snapshots?symbols=${symbolsKey}`);
      const data = await res.json();
      const newPrices: Record<string, { price: number; change: number }> = {};
      for (const [ticker, snap] of Object.entries(data) as any) {
        const bar = snap.dailyBar || {};
        const prev = snap.prevDailyBar || {};
        const c = bar.c || 0;
        const pc = prev.c || c;
        newPrices[ticker] = { price: c, change: pc > 0 ? ((c - pc) / pc) * 100 : 0 };
      }
      return newPrices;
    },
    enabled: tickers.length > 0,
    staleTime: 30_000,
    retry: 2,
  });

  const prices = priceData ?? {};

  const addTicker = async () => {
    const t = input.trim().toUpperCase();
    if (!t || tickers.includes(t)) return;
    setTickers(prev => [...prev, t]);
    setInput("");
    if (isLoggedIn) {
      try {
        await apiRequest("POST", "/api/watchlist/add", { ticker: t });
      } catch {}
    }
  };

  const removeTicker = async (t: string) => {
    setTickers(prev => prev.filter(x => x !== t));
    if (isLoggedIn) {
      try {
        await apiRequest("POST", "/api/watchlist/remove", { ticker: t });
      } catch {}
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") addTicker();
  };

  return (
    <div>
      {/* Page header */}
      <div style={{ marginBottom: '1.5rem' }}>
        <h1 style={{
          fontSize: '1.4rem',
          fontWeight: 700,
          letterSpacing: '-0.02em',
          color: 'var(--text-primary)',
          marginBottom: '0.25rem'
        }}>
          Watchlist
        </h1>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
          {tickers.length > 0
            ? `Tracking ${tickers.length} ticker${tickers.length !== 1 ? 's' : ''} · ${isLoggedIn ? 'Saved to your account' : 'Sign in to save'}`
            : `Add tickers to track · ${isLoggedIn ? 'Saved to your account' : 'Sign in to save'}`}
        </p>
      </div>

      {/* Sign up reminder */}
      {!isLoggedIn && (
        <div style={{
          padding: '12px 16px',
          marginBottom: '1rem',
          background: 'rgba(0, 229, 255, 0.06)',
          border: '1px solid rgba(0, 229, 255, 0.15)',
          borderRadius: 6,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '12px',
          flexWrap: 'wrap',
        }}>
          <span style={{ fontSize: 12, color: '#7a8ba0', fontFamily: "'JetBrains Mono', monospace" }}>
            🔒 Create a free account to save your watchlist across sessions
          </span>
        </div>
      )}

      {/* Price error display (Bug 8) */}
      {priceError && (
        <div style={{
          padding: '10px 14px',
          marginBottom: '1rem',
          background: 'rgba(255, 69, 58, 0.07)',
          border: '1px solid rgba(255, 69, 58, 0.2)',
          borderRadius: 6,
          fontSize: '0.82rem',
          color: '#ff453a',
        }}>
          Failed to load prices. Data will retry automatically.
        </div>
      )}

      {/* Add ticker */}
      <div style={{
        display: 'flex',
        gap: '0.625rem',
        marginBottom: '1.5rem',
        maxWidth: '480px',
      }}>
        <div style={{ position: 'relative', flex: 1 }}>
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
            value={input}
            onChange={e => setInput(e.target.value.toUpperCase())}
            onKeyDown={handleKeyDown}
            placeholder="Add ticker (e.g. AAPL)…"
            maxLength={10}
            style={{
              width: '100%',
              height: '44px',
              padding: '0 0.875rem 0 2.25rem',
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              borderRadius: '10px',
              color: 'var(--text-primary)',
              fontSize: '0.9rem',
              fontFamily: 'var(--font-mono)',
              letterSpacing: '0.05em',
            }}
          />
        </div>
        <button
          onClick={addTicker}
          disabled={!input.trim()}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            padding: '0 1rem',
            height: '44px',
            background: input.trim() ? 'var(--accent-color)' : 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: '10px',
            fontSize: '0.85rem',
            fontWeight: 600,
            color: input.trim() ? 'white' : 'var(--text-secondary)',
            flexShrink: 0,
            transition: 'all 150ms',
            cursor: input.trim() ? 'pointer' : 'not-allowed',
            opacity: input.trim() ? 1 : 0.5,
          }}
        >
          <Plus size={15} />
          Add
        </button>
      </div>

      {/* Quick adds */}
      {tickers.length === 0 && (
        <div style={{ marginBottom: '1.5rem' }}>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '0.5rem' }}>
            Quick add:
          </p>
          <div style={{ display: 'flex', gap: '0.375rem', flexWrap: 'wrap' }}>
            {["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMZN", "MSFT", "META"].map(t => (
              <button
                key={t}
                onClick={async () => {
                  if (tickers.includes(t)) return;
                  setTickers(prev => [...prev, t]);
                  if (isLoggedIn) {
                    try { await apiRequest("POST", "/api/watchlist/add", { ticker: t }); } catch {}
                  }
                }}
                className="chip"
              >
                {t}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Table */}
      {tickers.length === 0 ? (
        <div style={{
          padding: '3rem 1rem',
          textAlign: 'center',
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          borderRadius: '16px',
          backdropFilter: 'blur(20px)',
        }}>
          <div style={{
            width: 48,
            height: 48,
            borderRadius: '50%',
            background: 'var(--bg-card-hover)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 1rem',
          }}>
            <Search size={20} style={{ color: 'var(--text-tertiary)' }} />
          </div>
          <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
            Your watchlist is empty
          </h3>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
            Add tickers to track their prices and quickly jump to analysis.
          </p>
        </div>
      ) : (
        <div style={{
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          borderRadius: '16px',
          overflow: 'hidden',
          backdropFilter: 'blur(20px)',
        }}>
          <table className="watchlist-table" style={{ width: '100%' }}>
            <thead>
              <tr>
                <th>Ticker</th>
                <th style={{ textAlign: 'right' }}>Last Price</th>
                <th style={{ textAlign: 'right' }}>Change</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {tickers.map(t => (
                <WatchlistRow
                  key={t}
                  ticker={t}
                  priceData={prices[t]}
                  pricesLoading={pricesLoading}
                  onRemove={() => removeTicker(t)}
                  onAnalyze={() => onSelectTicker(t)}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p style={{
        fontSize: '0.72rem',
        color: 'var(--text-tertiary)',
        marginTop: '1.25rem',
        textAlign: 'center',
      }}>
        {isLoggedIn ? 'Watchlist saved to your account · Persists across sessions' : 'Sign in to save your watchlist · Click Analyze to open full analysis'}
      </p>
    </div>
  );
}
