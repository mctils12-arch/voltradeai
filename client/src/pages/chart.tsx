import { useState, useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { Clock } from "lucide-react";

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
  const hour = now.getUTCHours() - 4;
  const isPreMarket = !isOpen && hour >= 4 && hour < 9.5;
  const isAfterHours = !isOpen && hour >= 16 && hour < 20;

  let status = "Market Closed";
  let color = "#ff453a";
  let bgColor = "rgba(255,69,58,0.1)";
  let borderColor = "rgba(255,69,58,0.2)";

  if (isOpen) {
    status = "Market Open"; color = "#30d158";
    bgColor = "rgba(48,209,88,0.1)"; borderColor = "rgba(48,209,88,0.2)";
  } else if (isPreMarket) {
    status = "Pre-Market"; color = "#ff9f0a";
    bgColor = "rgba(255,159,10,0.1)"; borderColor = "rgba(255,159,10,0.2)";
  } else if (isAfterHours) {
    status = "After Hours"; color = "#bf5af2";
    bgColor = "rgba(191,90,242,0.1)"; borderColor = "rgba(191,90,242,0.2)";
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
        <Clock size={10} style={{ display: "inline", verticalAlign: "middle", marginRight: "4px" }} />
        AI engine is {isOpen ? "monitoring signals" : "updating algorithms"}
      </span>
    </div>
  );
}

// ─── TradingView Advanced Chart Widget ───────────────────────────────────────
function TradingViewChart({ ticker }: { ticker: string }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    containerRef.current.innerHTML = "";

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
    script.type = "text/javascript";
    script.async = true;
    script.innerHTML = JSON.stringify({
      autosize: true,
      symbol: ticker,
      interval: "D",
      timezone: "America/New_York",
      theme: "dark",
      style: "1",
      locale: "en",
      backgroundColor: "rgba(0, 0, 0, 0)",
      gridColor: "rgba(255, 255, 255, 0.04)",
      hide_top_toolbar: false,
      hide_legend: false,
      allow_symbol_change: true,
      save_image: true,
      calendar: false,
      studies: ["RSI@tv-basicstudies", "MASimple@tv-basicstudies"],
      support_host: "https://www.tradingview.com",
    });

    const widget = document.createElement("div");
    widget.className = "tradingview-widget-container";
    widget.style.height = "100%";
    widget.style.width = "100%";

    const widgetInner = document.createElement("div");
    widgetInner.className = "tradingview-widget-container__widget";
    widgetInner.style.height = "calc(100% - 32px)";
    widgetInner.style.width = "100%";

    widget.appendChild(widgetInner);
    widget.appendChild(script);
    containerRef.current.appendChild(widget);

    return () => {
      if (containerRef.current) containerRef.current.innerHTML = "";
    };
  }, [ticker]);

  return (
    <div ref={containerRef} style={{ height: "500px", width: "100%", borderRadius: "8px", overflow: "hidden" }} />
  );
}

// ─── Main Export ──────────────────────────────────────────────────────────────
export default function ChartPage() {
  const [ticker, setTicker] = useState("SPY");
  const [inputTicker, setInputTicker] = useState("SPY");

  function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (inputTicker.trim()) setTicker(inputTicker.trim().toUpperCase());
  }

  return (
    <div>
      <MarketStatusBanner />

      {/* Ticker search + quick picks */}
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px", flexWrap: "wrap" }}>
        <form onSubmit={handleSearch} style={{ display: "flex", gap: "6px" }}>
          <input
            value={inputTicker}
            onChange={e => setInputTicker(e.target.value.toUpperCase())}
            placeholder="Ticker"
            style={{ padding: "7px 12px", background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "8px", color: "#f5f5f7", fontSize: "13px", width: "100px", outline: "none", fontFamily: "monospace", fontWeight: 700 }}
          />
          <button type="submit" style={{ padding: "7px 12px", background: "#0a84ff", color: "white", border: "none", borderRadius: "8px", fontSize: "12px", fontWeight: 600, cursor: "pointer" }}>Go</button>
        </form>
        {["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "AMD"].map(t => (
          <button key={t} onClick={() => { setTicker(t); setInputTicker(t); }}
            style={{ padding: "4px 10px", borderRadius: "6px", border: "1px solid rgba(255,255,255,0.08)", background: ticker === t ? "rgba(10,132,255,0.15)" : "transparent", color: ticker === t ? "#0a84ff" : "#6e6e73", fontSize: "11px", fontWeight: 600, cursor: "pointer", fontFamily: "monospace" }}>
            {t}
          </button>
        ))}
      </div>

      {/* TradingView Chart */}
      <TradingViewChart ticker={ticker} />
    </div>
  );
}
