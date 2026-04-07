import { useEffect, useRef, useState, useCallback } from "react";
import { createChart, ColorType, LineStyle, CrosshairMode, IChartApi, ISeriesApi, CandlestickSeries, HistogramSeries } from "lightweight-charts";
import { apiRequest } from "@/lib/queryClient";
import { RefreshCw, TrendingUp, TrendingDown, Clock, Target, Shield } from "lucide-react";

interface Position {
  ticker: string;
  qty: number;
  side: string;
  entryPrice: number;
  currentPrice: number;
  marketValue: number;
  pnl: number;
  pnlPct: number;
  stopPrice: number;
  takeProfitPrice: number | null;
  phase: number;
  rMultiple: number;
  highestPnl: number;
  daysHeld: number;
}

interface Bar {
  time?: number;   // unix timestamp (from existing endpoint)
  t?: string;      // ISO string (fallback)
  open?: number; o?: number;
  high?: number; h?: number;
  low?: number;  l?: number;
  close?: number; c?: number;
  volume?: number; v?: number;
}

type Timeframe = "1Min" | "5Min" | "15Min" | "1Hour" | "1Day";

const TIMEFRAMES: { label: string; value: Timeframe; bars: number }[] = [
  { label: "1m", value: "1Min", bars: 120 },
  { label: "5m", value: "5Min", bars: 100 },
  { label: "15m", value: "15Min", bars: 80 },
  { label: "1h", value: "1Hour", bars: 72 },
  { label: "1D", value: "1Day", bars: 60 },
];

function SingleTradeChart({ position }: { position: Position }) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const [timeframe, setTimeframe] = useState<Timeframe>("5Min");
  const [bars, setBars] = useState<number>(100);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastPrice, setLastPrice] = useState(position.currentPrice);

  const pnlPositive = position.pnlPct >= 0;
  const phaseColors = ["", "#00ffd5", "#fbbf24", "#ff8c00", "#ff4444"];
  const phaseColor = phaseColors[position.phase] || "#00ffd5";

  const loadBars = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const tf = TIMEFRAMES.find(t => t.value === timeframe)!;
      const resp = await apiRequest("GET", `/api/bot/bars/${position.ticker}?timeframe=${timeframe}&limit=${tf.bars}`);
      const data = await resp.json();
      // Existing bars endpoint returns { bars: [...], ticker, timeframe }
      const rawBars: Bar[] = (data && data.bars) ? data.bars : Array.isArray(data) ? data : [];

      if (!chartRef.current || !candleSeriesRef.current) return;

      // Existing bars endpoint pre-converts t to unix timestamp as b.time
      const candleData = rawBars.map((b: any) => ({
        time: (b.time || Math.floor(new Date(b.t).getTime() / 1000)) as any,
        open: b.open ?? b.o, high: b.high ?? b.h, low: b.low ?? b.l, close: b.close ?? b.c,
      })).sort((a, b) => a.time - b.time);

      const volumeData = rawBars.map((b: any) => ({
        time: (b.time || Math.floor(new Date(b.t).getTime() / 1000)) as any,
        value: b.volume ?? b.v,
        color: (b.close ?? b.c) >= (b.open ?? b.o) ? "rgba(0, 255, 213, 0.3)" : "rgba(255, 68, 68, 0.3)",
      })).sort((a, b) => a.time - b.time);

      candleSeriesRef.current.setData(candleData);
      volumeSeriesRef.current?.setData(volumeData);

      if (candleData.length > 0) {
        setLastPrice(candleData[candleData.length - 1].close);
      }

      // Entry line
      candleSeriesRef.current.createPriceLine({
        price: position.entryPrice,
        color: "#60a5fa",
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: `Entry $${position.entryPrice.toFixed(2)}`,
      });

      // Stop loss line (red)
      candleSeriesRef.current.createPriceLine({
        price: position.stopPrice,
        color: "#ff4444",
        lineWidth: 2,
        lineStyle: LineStyle.Dotted,
        axisLabelVisible: true,
        title: `Stop P${position.phase} $${position.stopPrice.toFixed(2)}`,
      });

      // Take profit line (green) — only if not in Phase 3+
      if (position.takeProfitPrice) {
        candleSeriesRef.current.createPriceLine({
          price: position.takeProfitPrice,
          color: "#00ffd5",
          lineWidth: 2,
          lineStyle: LineStyle.Dotted,
          axisLabelVisible: true,
          title: `Target $${position.takeProfitPrice.toFixed(2)}`,
        });
      }

      // Current price line
      candleSeriesRef.current.createPriceLine({
        price: position.currentPrice,
        color: pnlPositive ? "#00ffd5" : "#ff4444",
        lineWidth: 1,
        lineStyle: LineStyle.Solid,
        axisLabelVisible: true,
        title: `Now $${position.currentPrice.toFixed(2)}`,
      });

      chartRef.current.timeScale().fitContent();
    } catch (e: any) {
      setError(e.message || "Failed to load chart data");
    } finally {
      setLoading(false);
    }
  }, [position, timeframe]);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#0d1117" },
        textColor: "#8b949e",
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "#161b22" },
        horzLines: { color: "#161b22" },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: "#00ffd5", width: 1, style: LineStyle.Dotted },
        horzLine: { color: "#00ffd5", width: 1, style: LineStyle.Dotted },
      },
      rightPriceScale: {
        borderColor: "#21262d",
        textColor: "#8b949e",
      },
      timeScale: {
        borderColor: "#21262d",
        textColor: "#8b949e",
        timeVisible: true,
        secondsVisible: false,
      },
      handleScroll: true,
      handleScale: true,
    });

    // v5 API: addSeries(SeriesType, options) replaces addCandlestickSeries/addHistogramSeries
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#00ffd5",
      downColor: "#ff4444",
      borderUpColor: "#00ffd5",
      borderDownColor: "#ff4444",
      wickUpColor: "#00ffd5",
      wickDownColor: "#ff4444",
    });

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
      scaleMargins: { top: 0.8, bottom: 0 },
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;

    const resizeObserver = new ResizeObserver(() => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      }
    });
    resizeObserver.observe(chartContainerRef.current);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    loadBars();
  }, [loadBars]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(loadBars, 30000);
    return () => clearInterval(interval);
  }, [loadBars]);

  const riskReward = position.takeProfitPrice
    ? ((position.takeProfitPrice - position.entryPrice) / (position.entryPrice - position.stopPrice)).toFixed(1)
    : "∞";

  return (
    <div
      style={{
        background: "#0d1117",
        border: `1px solid ${pnlPositive ? "rgba(0,255,213,0.3)" : "rgba(255,68,68,0.3)"}`,
        borderRadius: 8,
        overflow: "hidden",
        fontFamily: "'JetBrains Mono', monospace",
      }}
    >
      {/* Header */}
      <div style={{ padding: "10px 14px", borderBottom: "1px solid #21262d", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ color: "#e6edf3", fontWeight: 700, fontSize: 15 }}>{position.ticker}</span>
          <span style={{
            padding: "2px 8px", borderRadius: 4, fontSize: 11,
            background: pnlPositive ? "rgba(0,255,213,0.15)" : "rgba(255,68,68,0.15)",
            color: pnlPositive ? "#00ffd5" : "#ff4444",
          }}>
            {pnlPositive ? <TrendingUp size={10} style={{ display: "inline", marginRight: 3 }} /> : <TrendingDown size={10} style={{ display: "inline", marginRight: 3 }} />}
            {pnlPositive ? "+" : ""}{position.pnlPct.toFixed(2)}%
          </span>
          <span style={{ color: phaseColor, fontSize: 11, padding: "2px 6px", border: `1px solid ${phaseColor}40`, borderRadius: 4 }}>
            Phase {position.phase} · {position.rMultiple.toFixed(1)}R
          </span>
        </div>

        {/* Stats */}
        <div style={{ display: "flex", gap: 16, fontSize: 11, color: "#8b949e" }}>
          <span><span style={{ color: "#60a5fa" }}>Entry</span> ${position.entryPrice.toFixed(2)}</span>
          <span><Shield size={10} style={{ display: "inline", color: "#ff4444", marginRight: 3 }} /><span style={{ color: "#ff4444" }}>Stop</span> ${position.stopPrice.toFixed(2)}</span>
          {position.takeProfitPrice && (
            <span><Target size={10} style={{ display: "inline", color: "#00ffd5", marginRight: 3 }} /><span style={{ color: "#00ffd5" }}>Target</span> ${position.takeProfitPrice.toFixed(2)} ({riskReward}:1)</span>
          )}
          <span><Clock size={10} style={{ display: "inline", marginRight: 3 }} />{position.daysHeld}d</span>
        </div>

        {/* Timeframe selector */}
        <div style={{ display: "flex", gap: 4 }}>
          {TIMEFRAMES.map(tf => (
            <button
              key={tf.value}
              onClick={() => setTimeframe(tf.value)}
              style={{
                padding: "2px 8px", fontSize: 11, borderRadius: 4, cursor: "pointer",
                background: timeframe === tf.value ? "rgba(0,255,213,0.15)" : "transparent",
                color: timeframe === tf.value ? "#00ffd5" : "#8b949e",
                border: `1px solid ${timeframe === tf.value ? "#00ffd540" : "#21262d"}`,
              }}
            >
              {tf.label}
            </button>
          ))}
          <button onClick={loadBars} style={{ padding: "2px 8px", background: "transparent", border: "1px solid #21262d", borderRadius: 4, color: "#8b949e", cursor: "pointer" }}>
            <RefreshCw size={10} />
          </button>
        </div>
      </div>

      {/* Chart */}
      <div style={{ position: "relative", height: 320 }}>
        {loading && (
          <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "#0d1117", zIndex: 10, color: "#8b949e", fontSize: 12 }}>
            Loading chart...
          </div>
        )}
        {error && (
          <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "#0d1117", zIndex: 10, color: "#ff4444", fontSize: 12 }}>
            {error}
          </div>
        )}
        <div ref={chartContainerRef} style={{ width: "100%", height: "100%" }} />
      </div>

      {/* P&L Bar */}
      <div style={{ padding: "8px 14px", borderTop: "1px solid #21262d", display: "flex", justifyContent: "space-between", fontSize: 11 }}>
        <span style={{ color: "#8b949e" }}>
          {position.qty} shares · ${Math.abs(position.marketValue || position.qty * position.currentPrice).toLocaleString()}
        </span>
        <span style={{ color: pnlPositive ? "#00ffd5" : "#ff4444", fontWeight: 700 }}>
          {pnlPositive ? "+" : ""}${position.pnl.toFixed(2)} P&L
          {position.highestPnl > 0 && ` · Peak ${position.highestPnl.toFixed(1)}%`}
        </span>
      </div>
    </div>
  );
}

export default function TradeCharts({ positions }: { positions: Position[] }) {
  if (!positions || positions.length === 0) {
    return (
      <div style={{
        padding: "40px 20px", textAlign: "center", color: "#8b949e",
        fontFamily: "'JetBrains Mono', monospace", fontSize: 13,
        border: "1px solid #21262d", borderRadius: 8, background: "#0d1117",
      }}>
        No open positions — charts will appear when the bot enters trades
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontFamily: "'JetBrains Mono', monospace", color: "#00ffd5", fontSize: 13, fontWeight: 700 }}>
          LIVE POSITIONS — {positions.length} chart{positions.length !== 1 ? "s" : ""}
        </span>
        <span style={{ fontFamily: "'JetBrains Mono', monospace", color: "#8b949e", fontSize: 11 }}>
          Auto-refreshes every 30s · Entry / Stop / Target lines shown
        </span>
      </div>
      {positions.map(pos => (
        <SingleTradeChart key={pos.ticker} position={pos} />
      ))}
    </div>
  );
}
