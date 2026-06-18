import { useState } from "react";
import { Target, TrendingUp, ShieldAlert, Crosshair, ArrowRight } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";

interface PlanTarget {
  r: number; price: number; gain_per_share: number; total_gain: number; gain_pct: number;
}
interface TradePlan {
  ticker: string; direction: string; entry: number; stop: number; stop_pct: number;
  per_share_risk: number; risk_amount: number; risk_pct: number; shares: number;
  position_value: number; account_pct: number; reward_risk: number | null;
  daily_vol_pct: number | null; targets: PlanTarget[]; notes: string[]; warnings: string[];
  live_price: number;
}

const usd = (n: number) => n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });
const usd0 = (n: number) => n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const pct = (n: number) => `${n >= 0 ? "" : ""}${(n * 100).toFixed(1)}%`;

export default function PlannerPage() {
  const [ticker, setTicker] = useState("");
  const [account, setAccount] = useState("25000");
  const [riskPct, setRiskPct] = useState("1");
  const [entry, setEntry] = useState("");
  const [stop, setStop] = useState("");
  const [stopMult, setStopMult] = useState("2");
  const [direction, setDirection] = useState("long");
  const [targetPrice, setTargetPrice] = useState("");
  const [taxRate, setTaxRate] = useState("24");
  const [plan, setPlan] = useState<TradePlan | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function buildPlan() {
    if (!ticker.trim()) { setError("Enter a ticker."); return; }
    setBusy(true); setError(null);
    try {
      const payload: any = {
        ticker: ticker.trim().toUpperCase(),
        account_value: parseFloat(account) || 0,
        risk_pct: (parseFloat(riskPct) || 0) / 100,
        direction,
        stop_atr_mult: parseFloat(stopMult) || 2,
      };
      if (entry) payload.entry = parseFloat(entry);
      if (stop) payload.stop = parseFloat(stop);
      if (targetPrice) payload.target_price = parseFloat(targetPrice);
      const res = await apiRequest("POST", "/api/plan", payload);
      const data: TradePlan = await res.json();
      setPlan(data);
      if (!entry && data.entry) setEntry(String(data.entry));
    } catch (e: any) {
      setError(e?.message || "Plan failed.");
    } finally {
      setBusy(false);
    }
  }

  const rate = (parseFloat(taxRate) || 0) / 100;
  const label: React.CSSProperties = { fontSize: 11, color: "#7e8ca0", marginBottom: 4, display: "block" };
  const input: React.CSSProperties = {
    background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: 6, padding: "8px 10px", color: "#eef3fb", fontSize: 13, width: "100%",
    fontFamily: "Geist, sans-serif",
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 22, maxWidth: 1000 }}>
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, color: "#eef3fb", fontSize: 18, fontWeight: 700 }}>
          <Crosshair size={18} /> Trade Planner
        </div>
        <p style={{ color: "#7e8ca0", fontSize: 12.5, margin: "6px 0 0", maxWidth: 740, lineHeight: 1.5 }}>
          Plan a trade before you buy: it suggests a volatility-based stop, sizes the position to your risk,
          and lays out price targets. You place the trade yourself — this never executes anything.
        </p>
      </div>

      {/* Inputs */}
      <section style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 10, padding: 18 }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 14 }}>
          <div>
            <label style={label}>Ticker</label>
            <input style={{ ...input, textTransform: "uppercase" }} placeholder="AAPL" value={ticker}
              onChange={e => setTicker(e.target.value.toUpperCase())}
              onKeyDown={e => e.key === "Enter" && buildPlan()} />
          </div>
          <div>
            <label style={label}>Direction</label>
            <select style={input} value={direction} onChange={e => setDirection(e.target.value)}>
              <option value="long">Long (buy)</option>
              <option value="short">Short (sell)</option>
            </select>
          </div>
          <div>
            <label style={label}>Account value ($)</label>
            <input style={input} type="number" value={account} onChange={e => setAccount(e.target.value)} />
          </div>
          <div>
            <label style={label}>Risk per trade (%)</label>
            <input style={input} type="number" step="0.25" value={riskPct} onChange={e => setRiskPct(e.target.value)} />
          </div>
          <div>
            <label style={label}>Entry ($, blank = live)</label>
            <input style={input} type="number" placeholder="live price" value={entry} onChange={e => setEntry(e.target.value)} />
          </div>
          <div>
            <label style={label}>Stop ($, blank = auto)</label>
            <input style={input} type="number" placeholder="volatility-based" value={stop} onChange={e => setStop(e.target.value)} />
          </div>
          <div>
            <label style={label}>Auto-stop width (× daily move)</label>
            <input style={input} type="number" step="0.5" value={stopMult} onChange={e => setStopMult(e.target.value)} />
          </div>
          <div>
            <label style={label}>Target ($, optional)</label>
            <input style={input} type="number" placeholder="or use 1R/2R/3R" value={targetPrice} onChange={e => setTargetPrice(e.target.value)} />
          </div>
        </div>
        <button onClick={buildPlan} disabled={busy} style={{ marginTop: 16, display: "flex", alignItems: "center", gap: 7, background: "#4d9fff", color: "#050a13", border: "none", borderRadius: 8, padding: "11px 22px", fontSize: 14, fontWeight: 700, cursor: busy ? "default" : "pointer", opacity: busy ? 0.6 : 1 }}>
          <Crosshair size={16} /> {busy ? "Planning…" : "Build trade plan"}
        </button>
      </section>

      {error && <div style={{ color: "#ff6b5a", fontSize: 13 }}>{error}</div>}

      {plan && (
        <section style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Headline stats */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: 16 }}>
            <Stat label="Live price" value={usd(plan.live_price)} accent="#b3c2d8" />
            <Stat label="Entry" value={usd(plan.entry)} accent="#eef3fb" />
            <Stat label="Stop-loss" value={`${usd(plan.stop)} (${pct(plan.stop_pct)})`} accent="#ff6b5a" />
            <Stat label="Shares" value={String(plan.shares)} accent="#4d9fff" big />
            <Stat label="Position" value={`${usd0(plan.position_value)} · ${pct(plan.account_pct)}`} accent="#eef3fb" />
            <Stat label="Risk if stopped" value={usd0(plan.risk_amount)} accent="#f5a623" />
          </div>

          {/* Targets */}
          <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 10, padding: 18 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 12, fontWeight: 700, color: "#b3c2d8", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 14 }}>
              <Target size={14} /> Price targets
            </div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, minWidth: 560 }}>
                <thead>
                  <tr style={{ color: "#7e8ca0", textAlign: "right" }}>
                    <th style={{ textAlign: "left", padding: "4px 8px", fontWeight: 500 }}>Target</th>
                    <th style={{ padding: "4px 8px", fontWeight: 500 }}>Price</th>
                    <th style={{ padding: "4px 8px", fontWeight: 500 }}>Move</th>
                    <th style={{ padding: "4px 8px", fontWeight: 500 }}>Gain</th>
                    <th style={{ padding: "4px 8px", fontWeight: 500 }}>After tax ({taxRate}%)</th>
                  </tr>
                </thead>
                <tbody style={{ fontFamily: "Geist Mono, monospace" }}>
                  {plan.targets.map((t, i) => (
                    <tr key={i} style={{ borderTop: "1px solid rgba(255,255,255,0.05)" }}>
                      <td style={{ textAlign: "left", padding: "8px", color: "#cdd8e8" }}>{t.r}R</td>
                      <td style={{ textAlign: "right", padding: "8px", color: "#eef3fb" }}>{usd(t.price)}</td>
                      <td style={{ textAlign: "right", padding: "8px", color: "#7e8ca0" }}>{pct(t.gain_pct)}</td>
                      <td style={{ textAlign: "right", padding: "8px", color: "#30d158" }}>{usd0(t.total_gain)}</td>
                      <td style={{ textAlign: "right", padding: "8px", color: "#9fe6b8" }}>{usd0(t.total_gain * (1 - rate))}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 12 }}>
              <span style={{ fontSize: 11, color: "#7e8ca0" }}>Est. tax rate on gains (%)</span>
              <input style={{ ...input, width: 80, padding: "5px 8px" }} type="number" value={taxRate} onChange={e => setTaxRate(e.target.value)} />
              <span style={{ fontSize: 11, color: "#5d6b80" }}>— set your real rate on the Taxes tab.</span>
            </div>
            {plan.reward_risk != null && (
              <p style={{ color: "#7e8ca0", fontSize: 12, marginTop: 10 }}>
                Reward:risk at primary target ≈ <b style={{ color: "#4d9fff" }}>{plan.reward_risk}:1</b>
                {plan.daily_vol_pct != null && <> · daily volatility ≈ {(plan.daily_vol_pct * 100).toFixed(1)}%</>}
              </p>
            )}
          </div>

          {/* The trade ticket — planner only */}
          <div style={{ background: "rgba(77,159,255,0.05)", border: "1px solid rgba(77,159,255,0.25)", borderRadius: 10, padding: "16px 20px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 7, color: "#4d9fff", fontWeight: 700, fontSize: 13, marginBottom: 8 }}>
              <TrendingUp size={15} /> Your plan
            </div>
            <p style={{ color: "#eef3fb", fontSize: 14, margin: 0, lineHeight: 1.6, fontFamily: "Geist Mono, monospace" }}>
              {plan.direction === "long" ? "Buy" : "Short"} <b>{plan.shares}</b> {plan.ticker} @ ~{usd(plan.entry)},
              stop <b style={{ color: "#ff6b5a" }}>{usd(plan.stop)}</b>,
              first target <b style={{ color: "#30d158" }}>{usd(plan.targets.find(t => t.r === 2)?.price ?? plan.targets[0].price)}</b>.
            </p>
            <p style={{ color: "#7e8ca0", fontSize: 11.5, margin: "10px 0 0" }}>
              Place this yourself in your broker. AlphaDesk plans and tracks — it never executes trades.
            </p>
          </div>

          {plan.warnings.length > 0 && (
            <div style={{ background: "rgba(255,107,90,0.07)", border: "1px solid rgba(255,107,90,0.35)", borderRadius: 8, padding: "12px 16px" }}>
              {plan.warnings.map((w, i) => (
                <div key={i} style={{ display: "flex", gap: 7, color: "#ff9d90", fontSize: 12.5, lineHeight: 1.5 }}>
                  <ShieldAlert size={14} style={{ flexShrink: 0, marginTop: 2 }} /> {w}
                </div>
              ))}
            </div>
          )}
          {plan.notes.length > 0 && (
            <div style={{ color: "#5d6b80", fontSize: 11.5, lineHeight: 1.5 }}>
              {plan.notes.map((n, i) => <div key={i}>{n}</div>)}
            </div>
          )}
        </section>
      )}
    </div>
  );
}

function Stat({ label, value, accent, big }: { label: string; value: string; accent: string; big?: boolean }) {
  return (
    <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: "12px 18px", minWidth: 130 }}>
      <div style={{ color: "#7e8ca0", fontSize: 11, marginBottom: 5 }}>{label}</div>
      <div style={{ color: accent, fontSize: big ? 22 : 15, fontWeight: 700, fontFamily: "Geist Mono, monospace" }}>{value}</div>
    </div>
  );
}
