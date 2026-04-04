import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { Search, RefreshCw, TrendingUp, TrendingDown, BarChart2, ChevronUp, ChevronDown } from "lucide-react";
import SectorHeatmap from "@/components/SectorHeatmap";

// ── Types ─────────────────────────────────────────────────────────────────────

interface StockData {
  ticker: string;
  close: number;
  open: number;
  high: number;
  low: number;
  volume: number;
  change_pct: number;
  vwap: number;
}

// MarketTicker is an alias kept for compatibility
type MarketTicker = StockData;

interface MarketSnapshotResponse {
  results: StockData[];
  date: string;
}

type SortKey = "ticker" | "close" | "change_pct" | "volume" | "vwap";
type SortDir = "asc" | "desc";
type FilterMode = "all" | "gainers" | "losers" | "high_volume";
type ViewMode = "table" | "heatmap";

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

function getHeatColor(change: number): string {
  if (change >= 5) return "rgba(48, 209, 88, 0.85)";
  if (change >= 3) return "rgba(48, 209, 88, 0.65)";
  if (change >= 1.5) return "rgba(48, 209, 88, 0.45)";
  if (change >= 0.5) return "rgba(48, 209, 88, 0.25)";
  if (change >= 0) return "rgba(48, 209, 88, 0.1)";
  if (change >= -0.5) return "rgba(255, 69, 58, 0.1)";
  if (change >= -1.5) return "rgba(255, 69, 58, 0.25)";
  if (change >= -3) return "rgba(255, 69, 58, 0.45)";
  if (change >= -5) return "rgba(255, 69, 58, 0.65)";
  return "rgba(255, 69, 58, 0.85)";
}

function getTextColor(change: number): string {
  if (change >= 0.5) return "#30d158";
  if (change <= -0.5) return "#ff453a";
  return "#7a8ba0";
}

// ── Scanner Page ──────────────────────────────────────────────────────────────

export default function ScannerPage({ onSelectTicker }: { onSelectTicker: (ticker: string) => void }) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<FilterMode>("all");
  const [sortKey, setSortKey] = useState<SortKey>("volume");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [viewMode, setViewMode] = useState<ViewMode>("table");

  const { data, isLoading, isError, refetch, isFetching, dataUpdatedAt } = useQuery<MarketSnapshotResponse>({
    queryKey: ["/api/market/scanner"],
    queryFn: async () => {
      const res = await apiRequest("GET", "/api/market/scanner");
      return res.json();
    },
    staleTime: 120000, // 2 minutes
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
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.25rem" }}>
            <h1 style={{
              fontSize: '1.4rem',
              fontWeight: 700,
              letterSpacing: '-0.02em',
              color: 'var(--text-primary)',
              margin: 0,
            }}>
              Market Scanner
            </h1>
            <span style={{
              fontSize: 9, padding: "3px 8px", borderRadius: 3,
              background: "rgba(0, 229, 255, 0.1)", border: "1px solid rgba(0, 229, 255, 0.2)",
              color: "#00e5ff", fontFamily: "'JetBrains Mono', monospace",
              letterSpacing: "0.12em", textTransform: "uppercase", fontWeight: 600,
            }}>
              TERMINAL
            </span>
          </div>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', margin: 0 }}>
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
            cursor: isFetching ? 'not-allowed' : 'pointer',
          }}
        >
          <RefreshCw size={13} className={isFetching ? "animate-spin" : ""} />
          {isFetching ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {/* Sector Heatmap */}
      <div style={{
        marginBottom: "1.5rem",
        background: "rgba(0, 8, 20, 0.6)",
        border: "1px solid rgba(0, 229, 255, 0.1)",
        borderRadius: 8,
        padding: "1rem",
      }}>
        <SectorHeatmap />
      </div>

      {/* View toggle */}
      <div style={{ display: "flex", gap: "8px", marginBottom: "1rem", alignItems: "center" }}>
        <span style={{
          fontSize: 10, color: "#4a5c70", fontFamily: "'JetBrains Mono', monospace",
          letterSpacing: "0.08em", marginRight: 4,
        }}>VIEW:</span>
        <button
          onClick={() => setViewMode("table")}
          style={{
            padding: "6px 14px", borderRadius: 4, fontSize: 11, fontWeight: 600,
            fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.08em",
            textTransform: "uppercase", cursor: "pointer",
            background: viewMode === "table" ? "rgba(0, 229, 255, 0.15)" : "transparent",
            border: `1px solid ${viewMode === "table" ? "#00e5ff" : "rgba(0, 229, 255, 0.15)"}`,
            color: viewMode === "table" ? "#00e5ff" : "#4a5c70",
            transition: "all 150ms ease",
          }}
        >
          TABLE
        </button>
        <button
          onClick={() => setViewMode("heatmap")}
          style={{
            padding: "6px 14px", borderRadius: 4, fontSize: 11, fontWeight: 600,
            fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.08em",
            textTransform: "uppercase", cursor: "pointer",
            background: viewMode === "heatmap" ? "rgba(0, 229, 255, 0.15)" : "transparent",
            border: `1px solid ${viewMode === "heatmap" ? "#00e5ff" : "rgba(0, 229, 255, 0.15)"}`,
            color: viewMode === "heatmap" ? "#00e5ff" : "#4a5c70",
            transition: "all 150ms ease",
          }}
        >
          HEATMAP
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
              background: 'rgba(0, 8, 20, 0.8)',
              border: '1px solid rgba(0, 229, 255, 0.15)',
              borderRadius: '6px',
              color: 'var(--text-primary)',
              fontSize: '0.85rem',
              fontFamily: "'JetBrains Mono', monospace",
              outline: 'none',
            }}
          />
        </div>
        {/* Sector filter chips — tactical style */}
        <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
          {(["all", "gainers", "losers", "high_volume"] as FilterMode[]).map(f => {
            const labels: Record<FilterMode, string> = {
              all: "ALL", gainers: "GAINERS", losers: "LOSERS", high_volume: "HIGH VOL"
            };
            const active = filter === f;
            return (
              <button
                key={f}
                onClick={() => setFilter(f)}
                style={{
                  padding: "5px 12px",
                  borderRadius: 4,
                  fontSize: 10,
                  fontWeight: 600,
                  fontFamily: "'JetBrains Mono', monospace",
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  cursor: "pointer",
                  background: active ? "rgba(0, 229, 255, 0.12)" : "transparent",
                  border: `1px solid ${active ? "rgba(0, 229, 255, 0.5)" : "rgba(0, 229, 255, 0.12)"}`,
                  color: active ? "#00e5ff" : "#4a5c70",
                  transition: "all 150ms ease",
                }}
              >
                {labels[f]}
              </button>
            );
          })}
        </div>
      </div>

      {/* Heatmap view */}
      {viewMode === "heatmap" && (
        <div style={{
          background: "rgba(0, 8, 20, 0.6)",
          border: "1px solid rgba(0, 229, 255, 0.1)",
          borderRadius: 8,
          padding: "1rem",
        }}>
          <div style={{
            display: "flex", justifyContent: "space-between", alignItems: "center",
            marginBottom: "0.75rem",
          }}>
            <h3 style={{
              fontSize: 13, fontWeight: 700, color: "#c8d6e5", margin: 0,
              fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.1em",
              textTransform: "uppercase",
            }}>
              STOCK HEATMAP
            </h3>
            <span style={{ fontSize: 10, color: "#4a5c70", fontFamily: "'JetBrains Mono', monospace" }}>
              {sorted.length.toLocaleString()} TICKERS
            </span>
          </div>
          {isLoading || (isFetching && tickers.length === 0) ? (
            <div style={{ padding: "2rem", textAlign: "center", color: "#4a5c70", fontFamily: "'JetBrains Mono', monospace", fontSize: 12 }}>
              LOADING MARKET DATA...
            </div>
          ) : (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
              {sorted.slice(0, 200).map(stock => (
                <div
                  key={stock.ticker}
                  onClick={() => onSelectTicker(stock.ticker)}
                  style={{
                    flex: "0 0 calc(5% - 2px)",
                    minWidth: "64px",
                    height: "60px",
                    background: getHeatColor(stock.change_pct),
                    border: "1px solid rgba(0, 229, 255, 0.06)",
                    borderRadius: 4,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    padding: "4px 2px",
                    cursor: "pointer",
                    transition: "transform 120ms ease, box-shadow 120ms ease",
                  }}
                  onMouseOver={e => {
                    (e.currentTarget as HTMLElement).style.transform = "scale(1.06)";
                    (e.currentTarget as HTMLElement).style.boxShadow = "0 0 10px rgba(0, 229, 255, 0.2)";
                    (e.currentTarget as HTMLElement).style.zIndex = "10";
                  }}
                  onMouseOut={e => {
                    (e.currentTarget as HTMLElement).style.transform = "scale(1)";
                    (e.currentTarget as HTMLElement).style.boxShadow = "none";
                    (e.currentTarget as HTMLElement).style.zIndex = "1";
                  }}
                >
                  <div style={{
                    fontSize: 9, fontWeight: 700, color: "#c8d6e5",
                    fontFamily: "'JetBrains Mono', monospace",
                    letterSpacing: "0.04em",
                  }}>
                    {stock.ticker}
                  </div>
                  <div style={{
                    fontSize: 9, fontWeight: 600,
                    color: getTextColor(stock.change_pct),
                    fontFamily: "'JetBrains Mono', monospace",
                    marginTop: 2,
                  }}>
                    {fmtPct(stock.change_pct)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Table view */}
      {viewMode === "table" && (
        <div style={{
          background: 'rgba(0, 8, 20, 0.6)',
          border: '1px solid rgba(0, 229, 255, 0.1)',
          borderRadius: '8px',
          overflow: 'hidden',
          backdropFilter: 'blur(20px)',
        }}>
          {isLoading || (isFetching && tickers.length === 0) ? (
            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
              <RefreshCw size={24} className="animate-spin" style={{ margin: '0 auto 0.75rem' }} />
              <p style={{ fontSize: '0.88rem', fontFamily: "'JetBrains Mono', monospace" }}>LOADING MARKET DATA...</p>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginTop: '0.5rem', fontFamily: "'JetBrains Mono', monospace" }}>
                This may take a few seconds on first load.
              </p>
            </div>
          ) : isError ? (
            <div style={{ padding: '2rem', textAlign: 'center' }}>
              <p style={{ color: '#ff453a', fontSize: '0.88rem', marginBottom: '0.75rem', fontFamily: "'JetBrains Mono', monospace" }}>
                FAILED TO LOAD MARKET DATA
              </p>
              <button
                onClick={() => refetch()}
                style={{
                  padding: '0.4rem 1rem',
                  background: 'rgba(0, 8, 20, 0.8)',
                  border: '1px solid rgba(0, 229, 255, 0.2)',
                  borderRadius: '6px',
                  color: '#00e5ff',
                  fontSize: '0.82rem',
                  fontFamily: "'JetBrains Mono', monospace",
                  cursor: 'pointer',
                }}
              >
                TRY AGAIN
              </button>
            </div>
          ) : sorted.length === 0 ? (
            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.88rem', fontFamily: "'JetBrains Mono', monospace" }}>
              NO TICKERS MATCH FILTER
            </div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{
                minWidth: '540px',
                width: '100%',
                borderCollapse: 'collapse',
                fontFamily: "'JetBrains Mono', monospace",
              }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid rgba(0, 229, 255, 0.12)" }}>
                    {([
                      { key: "ticker", label: "TICKER" },
                      { key: "close", label: "PRICE" },
                      { key: "change_pct", label: "CHANGE %" },
                      { key: "volume", label: "VOLUME" },
                      { key: "vwap", label: "VWAP" },
                    ] as { key: SortKey; label: string }[]).map(col => (
                      <th
                        key={col.key}
                        onClick={() => handleSort(col.key)}
                        style={{
                          padding: "10px 14px",
                          textAlign: "left",
                          fontSize: 10,
                          fontWeight: 700,
                          color: sortKey === col.key ? "#00e5ff" : "#4a5c70",
                          letterSpacing: "0.1em",
                          cursor: "pointer",
                          userSelect: "none",
                          background: sortKey === col.key ? "rgba(0, 229, 255, 0.04)" : "transparent",
                          whiteSpace: "nowrap",
                          borderBottom: sortKey === col.key ? "2px solid rgba(0, 229, 255, 0.3)" : "2px solid transparent",
                        }}
                      >
                        <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                          {col.label} <SortIcon col={col.key} />
                        </span>
                      </th>
                    ))}
                    <th style={{
                      padding: "10px 14px",
                      textAlign: "left",
                      fontSize: 10,
                      fontWeight: 700,
                      color: "#4a5c70",
                      letterSpacing: "0.1em",
                    }}>
                      BAR
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
                        style={{
                          cursor: 'pointer',
                          borderBottom: "1px solid rgba(0, 229, 255, 0.04)",
                          transition: "background 120ms ease",
                        }}
                        onMouseOver={e => {
                          (e.currentTarget as HTMLElement).style.background = "rgba(0, 229, 255, 0.04)";
                        }}
                        onMouseOut={e => {
                          (e.currentTarget as HTMLElement).style.background = "transparent";
                        }}
                      >
                        <td style={{ padding: "9px 14px" }}>
                          <span style={{
                            fontFamily: "'JetBrains Mono', monospace",
                            fontWeight: 700,
                            fontSize: '0.88rem',
                            color: '#00e5ff',
                            letterSpacing: "0.04em",
                          }}>
                            {row.ticker}
                          </span>
                        </td>
                        <td style={{ padding: "9px 14px" }}>
                          <span style={{
                            fontFamily: "'JetBrains Mono', monospace",
                            fontWeight: 600,
                            color: '#c8d6e5',
                            fontSize: "0.85rem",
                          }}>
                            ${Number(row.close ?? 0).toFixed(2)}
                          </span>
                        </td>
                        <td style={{ padding: "9px 14px" }}>
                          <span style={{
                            fontFamily: "'JetBrains Mono', monospace",
                            fontWeight: 600,
                            fontSize: "0.85rem",
                            color: isUp ? '#30d158' : '#ff453a',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '0.2rem',
                          }}>
                            {isUp ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                            {fmtPct(row.change_pct)}
                          </span>
                        </td>
                        <td style={{ padding: "9px 14px" }}>
                          <span style={{
                            fontFamily: "'JetBrains Mono', monospace",
                            color: '#7a8ba0',
                            fontSize: "0.82rem",
                          }}>
                            {fmtVol(row.volume)}
                          </span>
                        </td>
                        <td style={{ padding: "9px 14px" }}>
                          <span style={{
                            fontFamily: "'JetBrains Mono', monospace",
                            color: '#7a8ba0',
                            fontSize: "0.82rem",
                          }}>
                            ${Number(row.vwap ?? 0).toFixed(2)}
                          </span>
                        </td>
                        <td style={{ padding: "9px 14px" }}>
                          {/* Mini change bar */}
                          <div style={{
                            width: "40px",
                            height: "4px",
                            borderRadius: 2,
                            background: isUp ? "#30d158" : "#ff453a",
                            opacity: Math.min(Math.abs(row.change_pct) / 5, 1),
                          }} />
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
                  fontSize: '0.72rem',
                  color: '#4a5c70',
                  borderTop: '1px solid rgba(0, 229, 255, 0.06)',
                  fontFamily: "'JetBrains Mono', monospace",
                  letterSpacing: "0.04em",
                }}>
                  SHOWING TOP 500 OF {sorted.length.toLocaleString()} RESULTS · USE SEARCH OR FILTERS TO NARROW
                </div>
              )}
            </div>
          )}
        </div>
      )}

      <p style={{
        fontSize: '0.72rem',
        color: '#4a5c70',
        marginTop: '0.875rem',
        textAlign: 'center',
        fontFamily: "'JetBrains Mono', monospace",
        letterSpacing: "0.04em",
      }}>
        DATA: ALPACA MARKETS · REAL-TIME SNAPSHOTS · US STOCKS $1+ · 50K+ DAILY VOL · CLICK ROW TO ANALYZE
      </p>
    </div>
  );
}
