import { useState, useEffect, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { Search, RefreshCw, TrendingUp, TrendingDown, BarChart2, ChevronUp, ChevronDown } from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

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
}

type SortKey = "ticker" | "close" | "change_pct" | "volume" | "vwap";
type SortDir = "asc" | "desc";
type FilterMode = "all" | "gainers" | "losers" | "high_volume";

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtVol(n: number) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return String(n);
}

function fmtPct(n: number) {
  const sign = n >= 0 ? "+" : "";
  return `${sign}${Number(n ?? 0).toFixed(2)}%`;
}

// ── Scanner Page ──────────────────────────────────────────────────────────────

export default function ScannerPage({ onSelectTicker }: { onSelectTicker: (ticker: string) => void }) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<FilterMode>("all");
  const [sortKey, setSortKey] = useState<SortKey>("volume");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const { data, isLoading, isError, refetch, isFetching, dataUpdatedAt } = useQuery<MarketSnapshotResponse>({
    queryKey: ["/api/market-snapshot"],
    queryFn: async () => {
      const res = await apiRequest("GET", "/api/market-snapshot");
      return res.json();
    },
    staleTime: 5 * 60 * 1000,
    retry: 2,
  });

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(d => d === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const SortIcon = ({ col }: { col: SortKey }) => {
    if (sortKey !== col) return <span style={{ opacity: 0.3 }}>↕</span>;
    return sortDir === "asc" ? <ChevronUp size={12} /> : <ChevronDown size={12} />;
  };

  const tickers = data?.results ?? [];
  const lastUpdate = dataUpdatedAt ? new Date(dataUpdatedAt).toLocaleTimeString() : null;

  // Apply filter
  const filtered = tickers.filter(t => {
    if (search && !t.ticker.toLowerCase().includes(search.toLowerCase())) return false;
    if (filter === "gainers") return t.change_pct > 1;
    if (filter === "losers") return t.change_pct < -1;
    if (filter === "high_volume") {
      const median = 2_000_000;
      return t.volume > median;
    }
    return true;
  });

  // Apply sort
  const sorted = [...filtered].sort((a, b) => {
    let va = a[sortKey];
    let vb = b[sortKey];
    if (sortKey === "ticker") {
      va = a.ticker as any;
      vb = b.ticker as any;
      return sortDir === "asc"
        ? (va as string).localeCompare(vb as string)
        : (vb as string).localeCompare(va as string);
    }
    return sortDir === "asc"
      ? (va as number) - (vb as number)
      : (vb as number) - (va as number);
  });

  return (
    <div>
      {/* Page header */}
      <div className="scanner-page-header">
        <div style={{ flex: 1 }}>
          <h1 style={{
            fontSize: '1.4rem',
            fontWeight: 700,
            letterSpacing: '-0.02em',
            color: 'var(--text-primary)',
            marginBottom: '0.25rem'
          }}>
            Market Scanner
          </h1>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
            {tickers.length > 0 ? `${tickers.length.toLocaleString()} stocks • ` : ''}
            {data?.date ? `Data for ${data.date}` : 'Loading market data…'}
            {lastUpdate && ` • Updated ${lastUpdate}`}
          </p>
        </div>
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

      {/* Search + filter row */}
      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '0.875rem', flexWrap: 'wrap', alignItems: 'center' }}>
        <div style={{ position: 'relative', flex: '1 1 200px', minWidth: 0 }}>
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
            value={search}
            onChange={e => setSearch(e.target.value.toUpperCase())}
            placeholder="Filter by ticker…"
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
        </div>
        <div className="scanner-filter-chips" style={{ margin: 0 }}>
          {(["all", "gainers", "losers", "high_volume"] as FilterMode[]).map(f => (
            <button
              key={f}
              className={`filter-chip ${filter === f ? "active" : ""}`}
              onClick={() => setFilter(f)}
            >
              {f === "all" ? "All" : f === "gainers" ? "Gainers" : f === "losers" ? "Losers" : "High Volume"}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        borderRadius: '16px',
        overflow: 'hidden',
        backdropFilter: 'blur(20px)',
      }}>
        {isLoading || (isFetching && tickers.length === 0) ? (
          <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
            <RefreshCw size={24} className="animate-spin" style={{ margin: '0 auto 0.75rem' }} />
            <p style={{ fontSize: '0.88rem' }}>Loading market data from Polygon.io…</p>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginTop: '0.5rem' }}>
              This may take a few seconds on first load.
            </p>
          </div>
        ) : isError ? (
          <div style={{ padding: '2rem', textAlign: 'center' }}>
            <p style={{ color: 'var(--accent-red)', fontSize: '0.88rem', marginBottom: '0.75rem' }}>
              Failed to load market data.
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
        ) : sorted.length === 0 ? (
          <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.88rem' }}>
            No tickers match your filter.
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="market-table" style={{ minWidth: '540px' }}>
              <thead>
                <tr>
                  <th className={sortKey === "ticker" ? "sorted" : ""} onClick={() => handleSort("ticker")}>
                    Ticker <SortIcon col="ticker" />
                  </th>
                  <th className={sortKey === "close" ? "sorted" : ""} onClick={() => handleSort("close")}>
                    Price <SortIcon col="close" />
                  </th>
                  <th className={sortKey === "change_pct" ? "sorted" : ""} onClick={() => handleSort("change_pct")}>
                    Change % <SortIcon col="change_pct" />
                  </th>
                  <th className={sortKey === "volume" ? "sorted" : ""} onClick={() => handleSort("volume")}>
                    Volume <SortIcon col="volume" />
                  </th>
                  <th className={sortKey === "vwap" ? "sorted" : ""} onClick={() => handleSort("vwap")}>
                    VWAP <SortIcon col="vwap" />
                  </th>
                </tr>
              </thead>
              <tbody>
                {sorted.slice(0, 500).map((row) => {
                  const isUp = row.change_pct >= 0;
                  return (
                    <tr
                      key={row.ticker}
                      onClick={() => onSelectTicker(row.ticker)}
                      style={{ cursor: 'pointer' }}
                    >
                      <td>
                        <span style={{
                          fontFamily: 'var(--font-mono)',
                          fontWeight: 700,
                          fontSize: '0.88rem',
                          color: 'var(--text-primary)',
                        }}>
                          {row.ticker}
                        </span>
                      </td>
                      <td>
                        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--text-primary)' }}>
                          ${Number(row.close ?? 0).toFixed(2)}
                        </span>
                      </td>
                      <td>
                        <span style={{
                          fontFamily: 'var(--font-mono)',
                          fontWeight: 600,
                          color: isUp ? '#30d158' : '#ff453a',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '0.2rem',
                        }}>
                          {isUp ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                          {fmtPct(row.change_pct)}
                        </span>
                      </td>
                      <td>
                        <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                          {fmtVol(row.volume)}
                        </span>
                      </td>
                      <td>
                        <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                          ${Number(row.vwap ?? 0).toFixed(2)}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {sorted.length > 200 && (
              <div style={{
                padding: '0.625rem',
                textAlign: 'center',
                fontSize: '0.75rem',
                color: 'var(--text-tertiary)',
                borderTop: '1px solid var(--border-subtle)',
              }}>
                Showing top 500 of {sorted.length.toLocaleString()} results. Use the search bar or filters to find any stock.
              </div>
            )}
          </div>
        )}
      </div>

      <p style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '0.875rem', textAlign: 'center' }}>
        Data from Polygon.io · Previous trading day · All US stocks with $1+ price and 50K+ daily volume · Click any row to deep-analyze with VRP algorithm
      </p>
    </div>
  );
}
