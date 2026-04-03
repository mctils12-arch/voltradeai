import { useState, useEffect, useRef, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { createChart, IChartApi, ISeriesApi, CandlestickData, HistogramData, ColorType, CandlestickSeries, HistogramSeries, LineSeries } from "lightweight-charts";
import { Clock, TrendingUp, TrendingDown, Search } from "lucide-react";

// ─── Timeframes ──────────────────────────────────────────────────────────────
const TIMEFRAMES = [
  { label: "1m", value: "1Min" },
  { label: "5m", value: "5Min" },
  { label: "15m", value: "15Min" },
  { label: "1H", value: "1Hour" },
  { label: "1D", value: "1Day" },
  { label: "1W", value: "1Week" },
];

// ─── Market Status Banner ────────────────────────────────────────────────────
function MarketStatusBanner() {
  const { data } = useQuery({
    queryKey: ["/api/bot/market-status"],
    queryFn: async () => { const r = await apiRequest("GET", "/api/bot/market-status"); return r.json(); },
    refetchInterval: 30000,
  });

  const [countdown, setCountdown] = useState("");

  useEffect(() => {
    if (!data) return;
    const interval = setInterval(() => {
      const target = data.isOpen ? data.nextClose : data.nextOpen;
      if (!target) return;
      const diff = new Date(target).getTime() - Date.now();
      if (diff <= 0) { setCountdown("now"); return; }
      const h = Math.floor(diff / 3600000);
      const m = Math.floor((diff % 3600000) / 60000);
      const s = Math.floor((diff % 60000) / 1000);
      setCountdown(`${h}h ${m}m ${s}s`);
    }, 1000);
    return () => clearInterval(interval);
  }, [data]);

  if (!data) return null;

  const isOpen = data.isOpen;
  const now = new Date();
  const hour = now.getUTCHours() - 4; // ET approximation
  const isPreMarket = !isOpen && hour >= 4 && hour < 9.5;
  const isAfterHours = !isOpen && hour >= 16 && hour < 20;

  let status = "Market Closed";
  let color = "#ff453a";
  let bgColor = "rgba(255,69,58,0.1)";
  let borderColor = "rgba(255,69,58,0.2)";

  if (isOpen) {
    status = "Market Open";
    color = "#30d158";
    bgColor = "rgba(48,209,88,0.1)";
    borderColor = "rgba(48,209,88,0.2)";
  } else if (isPreMarket) {
    status = "Pre-Market";
    color = "#ff9f0a";
    bgColor = "rgba(255,159,10,0.1)";
    borderColor = "rgba(255,159,10,0.2)";
  } else if (isAfterHours) {
    status = "After Hours";
    color = "#bf5af2";
    bgColor = "rgba(191,90,242,0.1)";
    borderColor = "rgba(191,90,242,0.2)";
  }

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: "10px", padding: "8px 14px",
      background: bgColor, border: `1px solid ${borderColor}`, borderRadius: "10px",
      fontSize: "12px", marginBottom: "16px",
    }}>
      <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: color, boxShadow: `0 0 8px ${color}` }} />
      <span style={{ color, fontWeight: 600 }}>{status}</span>
      <span style={{ color: "#6e6e73" }}>
        {isOpen ? `Closes in ${countdown}` : `Opens in ${countdown}`}
      </span>
      <span style={{ marginLeft: "auto", color: "#6e6e73", fontSize: "11px" }}>
        Bot is {isOpen ? "monitoring signals" : "updating algorithms"}
      </span>
    </div>
  );
}

// ─── SMA Calculation ─────────────────────────────────────────────────────────
function calcSMA(data: CandlestickData[], period: number) {
  const result: { time: any; value: number }[] = [];
  for (let i = period - 1; i < data.length; i++) {
    let sum = 0;
    for (let j = 0; j < period; j++) sum += (data[i - j] as any).close;
    result.push({ time: data[i].time, value: sum / period });
  }
  return result;
}

// ─── RSI Calculation ─────────────────────────────────────────────────────────
function calcRSI(data: CandlestickData[], period: number = 14) {
  const result: { time: any; value: number }[] = [];
  if (data.length < period + 1) return result;

  let gains = 0, losses = 0;
  for (let i = 1; i <= period; i++) {
    const diff = (data[i] as any).close - (data[i - 1] as any).close;
    if (diff > 0) gains += diff; else losses -= diff;
  }
  let avgGain = gains / period;
  let avgLoss = losses / period;
  const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
  result.push({ time: data[period].time, value: 100 - 100 / (1 + rs) });

  for (let i = period + 1; i < data.length; i++) {
    const diff = (data[i] as any).close - (data[i - 1] as any).close;
    avgGain = (avgGain * (period - 1) + (diff > 0 ? diff : 0)) / period;
    avgLoss = (avgLoss * (period - 1) + (diff < 0 ? -diff : 0)) / period;
    const rs2 = avgLoss === 0 ? 100 : avgGain / avgLoss;
    result.push({ time: data[i].time, value: 100 - 100 / (1 + rs2) });
  }
  return result;
}

// ─── Chart Component ─────────────────────────────────────────────────────────
export default function ChartPage() {
  const [ticker, setTicker] = useState("SPY");
  const [inputTicker, setInputTicker] = useState("SPY");
  const [timeframe, setTimeframe] = useState("1Day");
  const [showSMA20, setShowSMA20] = useState(true);
  const [showSMA50, setShowSMA50] = useState(true);
  const [showVolume, setShowVolume] = useState(true);
  const [showRSI, setShowRSI] = useState(false);

  const chartContainerRef = useRef<HTMLDivElement>(null);
  const rsiContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const rsiChartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const sma20Ref = useRef<ISeriesApi<"Line"> | null>(null);
  const sma50Ref = useRef<ISeriesApi<"Line"> | null>(null);
  const rsiRef = useRef<ISeriesApi<"Line"> | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["/api/bot/bars", ticker, timeframe],
    queryFn: async () => {
      const r = await apiRequest("GET", `/api/bot/bars/${ticker}?timeframe=${timeframe}&limit=300`);
      return r.json();
    },
    staleTime: 30000,
  });

  const bars: CandlestickData[] = data?.bars || [];

  // Create chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Clean up existing chart
    if (chartRef.current) { chartRef.current.remove(); chartRef.current = null; }

    const chart = createChart(chartContainerRef.current, {
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#6e6e73" },
      grid: { vertLines: { color: "rgba(255,255,255,0.04)" }, horzLines: { color: "rgba(255,255,255,0.04)" } },
      crosshair: { mode: 0 },
      rightPriceScale: { borderColor: "rgba(255,255,255,0.1)" },
      timeScale: { borderColor: "rgba(255,255,255,0.1)", timeVisible: timeframe.includes("Min") || timeframe === "1Hour" },
      width: chartContainerRef.current.clientWidth,
      height: 420,
    });

    const candle = chart.addSeries(CandlestickSeries, {
      upColor: "#30d158", downColor: "#ff453a",
      borderUpColor: "#30d158", borderDownColor: "#ff453a",
      wickUpColor: "#30d158", wickDownColor: "#ff453a",
    });
    candleRef.current = candle;

    const vol = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "vol",
    });
    chart.priceScale("vol").applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
    volRef.current = vol;

    const sma20 = chart.addSeries(LineSeries, { color: "#0a84ff", lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
    sma20Ref.current = sma20;

    const sma50 = chart.addSeries(LineSeries, { color: "#ff9f0a", lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
    sma50Ref.current = sma50;

    chartRef.current = chart;

    const handleResize = () => {
      if (chartContainerRef.current) chart.applyOptions({ width: chartContainerRef.current.clientWidth });
    };
    window.addEventListener("resize", handleResize);
    return () => { window.removeEventListener("resize", handleResize); chart.remove(); };
  }, [timeframe]);

  // Create RSI chart
  useEffect(() => {
    if (!showRSI || !rsiContainerRef.current) {
      if (rsiChartRef.current) { rsiChartRef.current.remove(); rsiChartRef.current = null; }
      return;
    }
    if (rsiChartRef.current) { rsiChartRef.current.remove(); rsiChartRef.current = null; }

    const rsiChart = createChart(rsiContainerRef.current, {
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#6e6e73" },
      grid: { vertLines: { color: "rgba(255,255,255,0.04)" }, horzLines: { color: "rgba(255,255,255,0.04)" } },
      rightPriceScale: { borderColor: "rgba(255,255,255,0.1)" },
      timeScale: { visible: false },
      width: rsiContainerRef.current.clientWidth,
      height: 100,
    });

    const rsiLine = rsiChart.addSeries(LineSeries, { color: "#bf5af2", lineWidth: 2, priceLineVisible: false });
    rsiRef.current = rsiLine;
    rsiChartRef.current = rsiChart;

    const handleResize = () => {
      if (rsiContainerRef.current) rsiChart.applyOptions({ width: rsiContainerRef.current.clientWidth });
    };
    window.addEventListener("resize", handleResize);
    return () => { window.removeEventListener("resize", handleResize); rsiChart.remove(); };
  }, [showRSI, timeframe]);

  // Update data
  useEffect(() => {
    if (!bars.length) return;

    if (candleRef.current) candleRef.current.setData(bars);

    if (volRef.current && showVolume) {
      const volData: HistogramData[] = bars.map((b: any) => ({
        time: b.time,
        value: b.volume,
        color: b.close >= b.open ? "rgba(48,209,88,0.3)" : "rgba(255,69,58,0.3)",
      }));
      volRef.current.setData(volData);
    }

    if (sma20Ref.current) {
      sma20Ref.current.setData(showSMA20 ? calcSMA(bars, 20) : []);
    }
    if (sma50Ref.current) {
      sma50Ref.current.setData(showSMA50 ? calcSMA(bars, 50) : []);
    }

    if (rsiRef.current && showRSI) {
      rsiRef.current.setData(calcRSI(bars, 14));
    }

    if (chartRef.current) chartRef.current.timeScale().fitContent();
    if (rsiChartRef.current) rsiChartRef.current.timeScale().fitContent();
  }, [bars, showSMA20, showSMA50, showVolume, showRSI]);

  // Current price info
  const lastBar = bars.length > 0 ? bars[bars.length - 1] : null;
  const prevBar = bars.length > 1 ? bars[bars.length - 2] : null;
  const priceChange = lastBar && prevBar ? (lastBar as any).close - (prevBar as any).close : 0;
  const priceChangePct = prevBar ? (priceChange / (prevBar as any).close) * 100 : 0;
  const isUp = priceChange >= 0;

  function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (inputTicker.trim()) setTicker(inputTicker.trim().toUpperCase());
  }

  const chipStyle = (active: boolean): React.CSSProperties => ({
    padding: "4px 10px", borderRadius: "6px", border: "none", fontSize: "11px", fontWeight: 600,
    cursor: "pointer", transition: "all 0.15s",
    background: active ? "rgba(10,132,255,0.2)" : "rgba(255,255,255,0.06)",
    color: active ? "#0a84ff" : "#6e6e73",
  });

  return (
    <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "20px 16px" }}>

      <MarketStatusBanner />

      {/* Ticker + Price Header */}
      <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", marginBottom: "16px", flexWrap: "wrap", gap: "12px" }}>
        <div>
          <form onSubmit={handleSearch} style={{ display: "flex", gap: "8px", marginBottom: "8px" }}>
            <div style={{ position: "relative" }}>
              <Search size={14} style={{ position: "absolute", left: "10px", top: "50%", transform: "translateY(-50%)", color: "#6e6e73" }} />
              <input
                value={inputTicker}
                onChange={e => setInputTicker(e.target.value.toUpperCase())}
                placeholder="Ticker"
                style={{ padding: "8px 10px 8px 30px", background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "8px", color: "#f5f5f7", fontSize: "14px", width: "120px", outline: "none", fontFamily: "monospace", fontWeight: 700 }}
              />
            </div>
            <button type="submit" style={{ padding: "8px 14px", background: "#0a84ff", color: "white", border: "none", borderRadius: "8px", fontSize: "12px", fontWeight: 600, cursor: "pointer" }}>Go</button>
          </form>
          <div style={{ display: "flex", alignItems: "baseline", gap: "12px" }}>
            <span style={{ fontSize: "24px", fontWeight: 700, color: "#f5f5f7", fontFamily: "monospace" }}>{ticker}</span>
            {lastBar && (
              <>
                <span style={{ fontSize: "24px", fontWeight: 700, color: "#f5f5f7" }}>${(lastBar as any).close.toFixed(2)}</span>
                <span style={{ fontSize: "14px", fontWeight: 600, color: isUp ? "#30d158" : "#ff453a", display: "flex", alignItems: "center", gap: "3px" }}>
                  {isUp ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                  {isUp ? "+" : ""}{priceChange.toFixed(2)} ({isUp ? "+" : ""}{priceChangePct.toFixed(2)}%)
                </span>
              </>
            )}
          </div>
        </div>

        {/* Timeframe selector */}
        <div style={{ display: "flex", gap: "4px" }}>
          {TIMEFRAMES.map(tf => (
            <button key={tf.value} onClick={() => setTimeframe(tf.value)} style={chipStyle(timeframe === tf.value)}>
              {tf.label}
            </button>
          ))}
        </div>
      </div>

      {/* Indicator toggles */}
      <div style={{ display: "flex", gap: "8px", marginBottom: "12px", flexWrap: "wrap" }}>
        <button onClick={() => setShowSMA20(!showSMA20)} style={chipStyle(showSMA20)}>SMA 20</button>
        <button onClick={() => setShowSMA50(!showSMA50)} style={chipStyle(showSMA50)}>SMA 50</button>
        <button onClick={() => setShowVolume(!showVolume)} style={chipStyle(showVolume)}>Volume</button>
        <button onClick={() => setShowRSI(!showRSI)} style={chipStyle(showRSI)}>RSI</button>
        {showSMA20 && <span style={{ fontSize: "10px", color: "#0a84ff", alignSelf: "center" }}>— SMA 20</span>}
        {showSMA50 && <span style={{ fontSize: "10px", color: "#ff9f0a", alignSelf: "center" }}>— SMA 50</span>}
      </div>

      {/* Chart */}
      <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "12px", overflow: "hidden" }}>
        {isLoading && (
          <div style={{ height: "420px", display: "flex", alignItems: "center", justifyContent: "center", color: "#6e6e73" }}>Loading chart data...</div>
        )}
        <div ref={chartContainerRef} style={{ width: "100%", minHeight: "420px" }} />
        {showRSI && (
          <div style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
            <div style={{ padding: "4px 12px", fontSize: "10px", color: "#bf5af2", fontWeight: 600 }}>RSI (14)</div>
            <div ref={rsiContainerRef} style={{ width: "100%", minHeight: "100px" }} />
          </div>
        )}
      </div>

      {/* Quick tickers */}
      <div style={{ display: "flex", gap: "6px", marginTop: "12px", flexWrap: "wrap" }}>
        {["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "AMD", "COIN"].map(t => (
          <button key={t} onClick={() => { setTicker(t); setInputTicker(t); }}
            style={{ padding: "4px 10px", borderRadius: "6px", border: "1px solid rgba(255,255,255,0.08)", background: ticker === t ? "rgba(10,132,255,0.15)" : "transparent", color: ticker === t ? "#0a84ff" : "#6e6e73", fontSize: "11px", fontWeight: 600, cursor: "pointer", fontFamily: "monospace" }}>
            {t}
          </button>
        ))}
      </div>
    </div>
  );
}
