import { useState, useEffect } from "react";
import { apiRequest } from "@/lib/queryClient";

interface SectorData {
  name: string;
  etf: string;
  change: number;
  marketCap: number; // relative weight
}

const SECTOR_ETFS: { name: string; etf: string; weight: number }[] = [
  { name: "Technology", etf: "XLK", weight: 30 },
  { name: "Healthcare", etf: "XLV", weight: 13 },
  { name: "Financials", etf: "XLF", weight: 13 },
  { name: "Consumer Disc.", etf: "XLY", weight: 10 },
  { name: "Industrials", etf: "XLI", weight: 8 },
  { name: "Communications", etf: "XLC", weight: 9 },
  { name: "Consumer Stap.", etf: "XLP", weight: 6 },
  { name: "Energy", etf: "XLE", weight: 4 },
  { name: "Utilities", etf: "XLU", weight: 3 },
  { name: "Real Estate", etf: "XLRE", weight: 2 },
  { name: "Materials", etf: "XLB", weight: 2 },
];

function getHeatColor(change: number): string {
  if (change >= 3) return "rgba(48, 209, 88, 0.8)";
  if (change >= 2) return "rgba(48, 209, 88, 0.6)";
  if (change >= 1) return "rgba(48, 209, 88, 0.4)";
  if (change >= 0.5) return "rgba(48, 209, 88, 0.25)";
  if (change >= 0) return "rgba(48, 209, 88, 0.1)";
  if (change >= -0.5) return "rgba(255, 69, 58, 0.1)";
  if (change >= -1) return "rgba(255, 69, 58, 0.25)";
  if (change >= -2) return "rgba(255, 69, 58, 0.4)";
  if (change >= -3) return "rgba(255, 69, 58, 0.6)";
  return "rgba(255, 69, 58, 0.8)";
}

function getTextColor(change: number): string {
  if (change >= 0.5) return "#30d158";
  if (change <= -0.5) return "#ff453a";
  return "#7a8ba0";
}

export default function SectorHeatmap() {
  const [sectors, setSectors] = useState<SectorData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchSectors() {
      try {
        const res = await apiRequest("GET", "/api/market/sectors");
        const data = await res.json();
        const results: SectorData[] = [];
        for (const s of SECTOR_ETFS) {
          const snap = data[s.etf];
          if (snap) {
            const bar = snap.dailyBar || {};
            const prev = snap.prevDailyBar || {};
            const c = bar.c || 0;
            const pc = prev.c || c;
            const change = pc > 0 ? ((c - pc) / pc) * 100 : 0;
            results.push({ name: s.name, etf: s.etf, change: Math.round(change * 100) / 100, marketCap: s.weight });
          }
        }
        setSectors(results);
      } catch {}
      setLoading(false);
    }
    fetchSectors();
  }, []);

  if (loading) {
    return (
      <div style={{ padding: "2rem", textAlign: "center", color: "#4a5c70", fontFamily: "'JetBrains Mono', monospace", fontSize: 12 }}>
        LOADING SECTOR DATA...
      </div>
    );
  }

  // Treemap layout: simple row-based packing
  const totalWeight = sectors.reduce((s, d) => s + d.marketCap, 0);

  return (
    <div style={{ width: "100%" }}>
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        marginBottom: "0.75rem",
      }}>
        <h3 style={{
          fontSize: 13, fontWeight: 700, color: "#c8d6e5", margin: 0,
          fontFamily: "'JetBrains Mono', monospace", letterSpacing: "0.1em",
          textTransform: "uppercase",
        }}>
          SECTOR HEATMAP
        </h3>
        <span style={{ fontSize: 10, color: "#4a5c70", fontFamily: "'JetBrains Mono', monospace" }}>
          S&amp;P 500 SECTORS
        </span>
      </div>

      <div style={{
        display: "flex", flexWrap: "wrap", gap: 2,
        borderRadius: 6, overflow: "hidden",
        border: "1px solid rgba(0, 229, 255, 0.08)",
      }}>
        {sectors.map(sector => {
          const widthPct = (sector.marketCap / totalWeight) * 100;
          return (
            <div
              key={sector.etf}
              style={{
                flex: `0 0 calc(${Math.max(widthPct, 8)}% - 2px)`,
                minWidth: "80px",
                height: widthPct > 10 ? "90px" : "70px",
                background: getHeatColor(sector.change),
                border: "1px solid rgba(0, 229, 255, 0.06)",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                padding: "6px 4px",
                cursor: "default",
                transition: "transform 150ms ease, box-shadow 150ms ease",
                position: "relative",
              }}
              onMouseOver={e => {
                (e.currentTarget as HTMLElement).style.transform = "scale(1.02)";
                (e.currentTarget as HTMLElement).style.boxShadow = "0 0 12px rgba(0, 229, 255, 0.15)";
                (e.currentTarget as HTMLElement).style.zIndex = "10";
              }}
              onMouseOut={e => {
                (e.currentTarget as HTMLElement).style.transform = "scale(1)";
                (e.currentTarget as HTMLElement).style.boxShadow = "none";
                (e.currentTarget as HTMLElement).style.zIndex = "1";
              }}
            >
              <div style={{
                fontSize: 10, fontWeight: 600, color: "#c8d6e5",
                fontFamily: "'JetBrains Mono', monospace",
                textAlign: "center", lineHeight: 1.2,
                letterSpacing: "0.05em",
              }}>
                {sector.name}
              </div>
              <div style={{
                fontSize: 9, color: "#7a8ba0",
                fontFamily: "'JetBrains Mono', monospace",
                marginTop: 2,
              }}>
                {sector.etf}
              </div>
              <div style={{
                fontSize: 14, fontWeight: 700,
                color: getTextColor(sector.change),
                fontFamily: "'JetBrains Mono', monospace",
                marginTop: 4,
              }}>
                {sector.change >= 0 ? "+" : ""}{sector.change.toFixed(2)}%
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
