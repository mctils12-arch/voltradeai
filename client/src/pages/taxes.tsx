import { useState } from "react";
import { Calculator, Plus, Trash2, Scale } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";

// ── Types ─────────────────────────────────────────────────────────────────
type Account =
  | "taxable" | "roth_ira" | "traditional_ira" | "roth_401k"
  | "traditional_401k" | "hsa";

interface TradeRow {
  symbol: string;
  account: Account;
  buy_date: string;
  sell_date: string;
  proceeds: string;
  cost_basis: string;
}

interface TaxResult {
  tax_year: number;
  filing_status: string;
  ordinary_income: number;
  taxable_ordinary_income: number;
  marginal_rate: number;
  short_term_gain: number;
  long_term_gain: number;
  sheltered_gain: number;
  wash_sale_disallowed: number;
  federal_st_tax: number;
  federal_lt_tax: number;
  niit: number;
  state_tax: number;
  se_tax: number;
  total_tax: number;
  effective_rate_on_gains: number;
  after_tax_gain: number;
  notes: string[];
  flags: string[];
}

const FILING = [
  { id: "single", label: "Single" },
  { id: "mfj", label: "Married filing jointly" },
  { id: "hoh", label: "Head of household" },
  { id: "mfs", label: "Married filing separately" },
];

const ACCOUNTS: { id: Account; label: string; sheltered: boolean }[] = [
  { id: "taxable", label: "Taxable", sheltered: false },
  { id: "roth_ira", label: "Roth IRA", sheltered: true },
  { id: "traditional_ira", label: "Traditional IRA", sheltered: true },
  { id: "roth_401k", label: "Roth 401(k)", sheltered: true },
  { id: "traditional_401k", label: "Traditional 401(k)", sheltered: true },
  { id: "hsa", label: "HSA", sheltered: true },
];

// State individual income tax — representative top marginal rate (2025 est.).
// Most states tax capital gains as ordinary income; user can override. No-income-
// tax states are 0. Estimates — verify for your situation.
const STATES: { code: string; name: string; rate: number }[] = [
  { code: "AL", name: "Alabama", rate: 5.0 }, { code: "AK", name: "Alaska", rate: 0 },
  { code: "AZ", name: "Arizona", rate: 2.5 }, { code: "AR", name: "Arkansas", rate: 3.9 },
  { code: "CA", name: "California", rate: 13.3 }, { code: "CO", name: "Colorado", rate: 4.4 },
  { code: "CT", name: "Connecticut", rate: 6.99 }, { code: "DE", name: "Delaware", rate: 6.6 },
  { code: "DC", name: "District of Columbia", rate: 10.75 }, { code: "FL", name: "Florida", rate: 0 },
  { code: "GA", name: "Georgia", rate: 5.39 }, { code: "HI", name: "Hawaii", rate: 11.0 },
  { code: "ID", name: "Idaho", rate: 5.8 }, { code: "IL", name: "Illinois", rate: 4.95 },
  { code: "IN", name: "Indiana", rate: 3.05 }, { code: "IA", name: "Iowa", rate: 3.8 },
  { code: "KS", name: "Kansas", rate: 5.7 }, { code: "KY", name: "Kentucky", rate: 4.0 },
  { code: "LA", name: "Louisiana", rate: 3.0 }, { code: "ME", name: "Maine", rate: 7.15 },
  { code: "MD", name: "Maryland", rate: 5.75 }, { code: "MA", name: "Massachusetts", rate: 5.0 },
  { code: "MI", name: "Michigan", rate: 4.25 }, { code: "MN", name: "Minnesota", rate: 9.85 },
  { code: "MS", name: "Mississippi", rate: 4.7 }, { code: "MO", name: "Missouri", rate: 4.8 },
  { code: "MT", name: "Montana", rate: 5.9 }, { code: "NE", name: "Nebraska", rate: 5.84 },
  { code: "NV", name: "Nevada", rate: 0 }, { code: "NH", name: "New Hampshire", rate: 0 },
  { code: "NJ", name: "New Jersey", rate: 10.75 }, { code: "NM", name: "New Mexico", rate: 5.9 },
  { code: "NY", name: "New York", rate: 10.9 }, { code: "NC", name: "North Carolina", rate: 4.5 },
  { code: "ND", name: "North Dakota", rate: 2.5 }, { code: "OH", name: "Ohio", rate: 3.5 },
  { code: "OK", name: "Oklahoma", rate: 4.75 }, { code: "OR", name: "Oregon", rate: 9.9 },
  { code: "PA", name: "Pennsylvania", rate: 3.07 }, { code: "RI", name: "Rhode Island", rate: 5.99 },
  { code: "SC", name: "South Carolina", rate: 6.4 }, { code: "SD", name: "South Dakota", rate: 0 },
  { code: "TN", name: "Tennessee", rate: 0 }, { code: "TX", name: "Texas", rate: 0 },
  { code: "UT", name: "Utah", rate: 4.55 }, { code: "VT", name: "Vermont", rate: 8.75 },
  { code: "VA", name: "Virginia", rate: 5.75 }, { code: "WA", name: "Washington", rate: 0 },
  { code: "WV", name: "West Virginia", rate: 4.82 }, { code: "WI", name: "Wisconsin", rate: 7.65 },
  { code: "WY", name: "Wyoming", rate: 0 },
];

const usd = (n: number) =>
  n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

const emptyTrade = (): TradeRow => ({
  symbol: "", account: "taxable", buy_date: "", sell_date: "", proceeds: "", cost_basis: "",
});

// ── Component ───────────────────────────────────────────────────────────────
export default function TaxesPage() {
  const [status, setStatus] = useState("single");
  const [state, setState] = useState("");
  const [stateRate, setStateRate] = useState("0");
  const [stateQuery, setStateQuery] = useState("");
  const [showStates, setShowStates] = useState(false);

  const stateMatches = (() => {
    const q = stateQuery.trim().toLowerCase();
    const list = !q
      ? STATES
      : STATES.filter(s => s.code.toLowerCase().startsWith(q) || s.name.toLowerCase().includes(q));
    return list.slice(0, 8);
  })();
  const pickState = (s: { code: string; name: string; rate: number }) => {
    setState(s.code);
    setStateQuery(s.code);
    setStateRate(String(s.rate));
    setShowStates(false);
  };
  const [w2, setW2] = useState("");
  const [inc1099, setInc1099] = useState("");
  const [other, setOther] = useState("");
  const [trades, setTrades] = useState<TradeRow[]>([emptyTrade()]);
  const [result, setResult] = useState<TaxResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const setTrade = (i: number, patch: Partial<TradeRow>) =>
    setTrades(ts => ts.map((t, idx) => (idx === i ? { ...t, ...patch } : t)));
  const addTrade = () => setTrades(ts => [...ts, emptyTrade()]);
  const removeTrade = (i: number) => setTrades(ts => ts.filter((_, idx) => idx !== i));

  async function calculate() {
    setBusy(true);
    setError(null);
    try {
      const payload = {
        profile: {
          filing_status: status,
          state,
          state_rate: (parseFloat(stateRate) || 0) / 100,
          w2_income: parseFloat(w2) || 0,
          income_1099: parseFloat(inc1099) || 0,
          other_income: parseFloat(other) || 0,
        },
        trades: trades
          .filter(t => t.sell_date && (t.proceeds || t.cost_basis))
          .map(t => ({
            symbol: t.symbol || "—",
            account: t.account,
            buy_date: t.buy_date,
            sell_date: t.sell_date,
            proceeds: parseFloat(t.proceeds) || 0,
            cost_basis: parseFloat(t.cost_basis) || 0,
          })),
      };
      const res = await apiRequest("POST", "/api/tax/estimate", payload);
      setResult(await res.json());
    } catch (e: any) {
      setError(e?.message || "Calculation failed.");
    } finally {
      setBusy(false);
    }
  }

  const label: React.CSSProperties = { fontSize: 11, color: "#7e8ca0", marginBottom: 4, display: "block", letterSpacing: "0.02em" };
  const input: React.CSSProperties = {
    background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: 6, padding: "8px 10px", color: "#eef3fb", fontSize: 13, width: "100%",
    fontFamily: "Geist, sans-serif",
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 22, maxWidth: 1000 }}>
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, color: "#eef3fb", fontSize: 18, fontWeight: 700 }}>
          <Calculator size={18} /> Tax Estimator
        </div>
        <p style={{ color: "#7e8ca0", fontSize: 12.5, margin: "6px 0 0", maxWidth: 720, lineHeight: 1.5 }}>
          Estimate the tax on your realized trades — short vs long-term, NIIT, state, wash sales, and which
          gains are sheltered in a Roth/IRA/HSA. Tax-year 2025 brackets. Estimate for planning only — not tax advice.
        </p>
      </div>

      {/* Profile */}
      <section style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 10, padding: 18 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: "#b3c2d8", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 14 }}>Your profile</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 14 }}>
          <div>
            <label style={label}>Filing status</label>
            <select style={input} value={status} onChange={e => setStatus(e.target.value)}>
              {FILING.map(f => <option key={f.id} value={f.id}>{f.label}</option>)}
            </select>
          </div>
          <div style={{ position: "relative" }}>
            <label style={label}>State</label>
            <input
              style={input}
              placeholder="Type a state…"
              value={stateQuery}
              onChange={e => {
                const v = e.target.value;
                setStateQuery(v);
                setShowStates(true);
                // Exact code/name match auto-fills the rate as you type.
                const hit = STATES.find(s => s.code.toLowerCase() === v.trim().toLowerCase()
                  || s.name.toLowerCase() === v.trim().toLowerCase());
                if (hit) { setState(hit.code); setStateRate(String(hit.rate)); }
              }}
              onFocus={() => setShowStates(true)}
              onBlur={() => setTimeout(() => setShowStates(false), 150)}
            />
            {showStates && stateMatches.length > 0 && (
              <div style={{ position: "absolute", top: "100%", left: 0, right: 0, zIndex: 30, marginTop: 4, background: "#0b1320", border: "1px solid rgba(255,255,255,0.14)", borderRadius: 8, maxHeight: 240, overflowY: "auto", boxShadow: "0 8px 24px rgba(0,0,0,0.5)" }}>
                {stateMatches.map(s => (
                  <div
                    key={s.code}
                    onMouseDown={() => pickState(s)}
                    style={{ display: "flex", justifyContent: "space-between", padding: "8px 11px", fontSize: 12.5, color: "#cdd8e8", cursor: "pointer", borderBottom: "1px solid rgba(255,255,255,0.04)" }}
                    onMouseEnter={e => (e.currentTarget.style.background = "rgba(77,159,255,0.12)")}
                    onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                  >
                    <span>{s.code} — {s.name}</span>
                    <span style={{ color: s.rate === 0 ? "#30d158" : "#7e8ca0", fontFamily: "Geist Mono, monospace" }}>
                      {s.rate === 0 ? "no tax" : `${s.rate}%`}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div>
            <label style={label}>State rate (%)</label>
            <input style={input} type="number" value={stateRate} onChange={e => setStateRate(e.target.value)} />
          </div>
          <div>
            <label style={label}>W-2 income ($)</label>
            <input style={input} type="number" placeholder="0" value={w2} onChange={e => setW2(e.target.value)} />
          </div>
          <div>
            <label style={label}>1099 / self-employed ($)</label>
            <input style={input} type="number" placeholder="0" value={inc1099} onChange={e => setInc1099(e.target.value)} />
          </div>
          <div>
            <label style={label}>Other income ($)</label>
            <input style={input} type="number" placeholder="0" value={other} onChange={e => setOther(e.target.value)} />
          </div>
        </div>
      </section>

      {/* Trades */}
      <section style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 10, padding: 18 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: "#b3c2d8", textTransform: "uppercase", letterSpacing: "0.05em" }}>Realized trades</div>
          <button onClick={addTrade} style={{ display: "flex", alignItems: "center", gap: 5, background: "rgba(77,159,255,0.12)", color: "#4d9fff", border: "1px solid rgba(77,159,255,0.3)", borderRadius: 6, padding: "5px 10px", fontSize: 12, cursor: "pointer" }}>
            <Plus size={13} /> Add trade
          </button>
        </div>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5, minWidth: 720 }}>
            <thead>
              <tr style={{ color: "#7e8ca0", textAlign: "left" }}>
                <th style={{ padding: "4px 6px", fontWeight: 500 }}>Symbol</th>
                <th style={{ padding: "4px 6px", fontWeight: 500 }}>Account</th>
                <th style={{ padding: "4px 6px", fontWeight: 500 }}>Buy date</th>
                <th style={{ padding: "4px 6px", fontWeight: 500 }}>Sell date</th>
                <th style={{ padding: "4px 6px", fontWeight: 500 }}>Proceeds</th>
                <th style={{ padding: "4px 6px", fontWeight: 500 }}>Cost basis</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t, i) => {
                const cell: React.CSSProperties = { ...input, padding: "6px 8px", fontSize: 12.5 };
                return (
                  <tr key={i}>
                    <td style={{ padding: "3px 4px" }}><input style={{ ...cell, width: 80, textTransform: "uppercase" }} value={t.symbol} onChange={e => setTrade(i, { symbol: e.target.value.toUpperCase() })} /></td>
                    <td style={{ padding: "3px 4px" }}>
                      <select style={{ ...cell, width: 150 }} value={t.account} onChange={e => setTrade(i, { account: e.target.value as Account })}>
                        {ACCOUNTS.map(a => <option key={a.id} value={a.id}>{a.label}</option>)}
                      </select>
                    </td>
                    <td style={{ padding: "3px 4px" }}><input style={{ ...cell, width: 130 }} type="date" value={t.buy_date} onChange={e => setTrade(i, { buy_date: e.target.value })} /></td>
                    <td style={{ padding: "3px 4px" }}><input style={{ ...cell, width: 130 }} type="date" value={t.sell_date} onChange={e => setTrade(i, { sell_date: e.target.value })} /></td>
                    <td style={{ padding: "3px 4px" }}><input style={{ ...cell, width: 100 }} type="number" placeholder="0" value={t.proceeds} onChange={e => setTrade(i, { proceeds: e.target.value })} /></td>
                    <td style={{ padding: "3px 4px" }}><input style={{ ...cell, width: 100 }} type="number" placeholder="0" value={t.cost_basis} onChange={e => setTrade(i, { cost_basis: e.target.value })} /></td>
                    <td style={{ padding: "3px 4px" }}>
                      {trades.length > 1 && (
                        <button onClick={() => removeTrade(i)} style={{ background: "none", border: "none", color: "#ff6b5a", cursor: "pointer", padding: 4 }}><Trash2 size={14} /></button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p style={{ color: "#5d6b80", fontSize: 11, marginTop: 10 }}>
          Held &gt; 365 days = long-term (lower rate). Gains in Roth/IRA/HSA are sheltered. Enter a sell date for a trade to be counted.
        </p>
      </section>

      <button onClick={calculate} disabled={busy} style={{ alignSelf: "flex-start", display: "flex", alignItems: "center", gap: 7, background: "#4d9fff", color: "#050a13", border: "none", borderRadius: 8, padding: "11px 22px", fontSize: 14, fontWeight: 700, cursor: busy ? "default" : "pointer", opacity: busy ? 0.6 : 1 }}>
        <Calculator size={16} /> {busy ? "Calculating…" : "Estimate my taxes"}
      </button>

      {error && <div style={{ color: "#ff6b5a", fontSize: 13 }}>{error}</div>}

      {/* Result */}
      {result && (
        <section style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 16 }}>
            <Stat label="Estimated tax on gains" value={usd(result.total_tax)} accent="#ff6b5a" big />
            <Stat label="After-tax gain" value={usd(result.after_tax_gain)} accent="#30d158" big />
            <Stat label="Effective rate on gains" value={pct(result.effective_rate_on_gains)} accent="#f5a623" big />
          </div>

          <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 10, padding: 18 }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#b3c2d8", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 14 }}>Breakdown</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "10px 28px", fontSize: 13 }}>
              <Line label={`Short-term gain (taxed at ${pct(result.marginal_rate)})`} value={usd(result.short_term_gain)} />
              <Line label="Federal tax on short-term" value={usd(result.federal_st_tax)} sub />
              <Line label="Long-term gain (0/15/20%)" value={usd(result.long_term_gain)} />
              <Line label="Federal tax on long-term" value={usd(result.federal_lt_tax)} sub />
              <Line label="NIIT (3.8% surtax)" value={usd(result.niit)} />
              <Line label={`State tax${result.filing_status ? "" : ""}`} value={usd(result.state_tax)} />
              {result.sheltered_gain !== 0 && <Line label="Sheltered in Roth/IRA/HSA (untaxed)" value={usd(result.sheltered_gain)} good />}
              {result.se_tax !== 0 && <Line label="Self-employment tax (1099, est.)" value={usd(result.se_tax)} />}
              {result.wash_sale_disallowed !== 0 && <Line label="Wash-sale losses disallowed" value={usd(result.wash_sale_disallowed)} />}
            </div>
          </div>

          {(result.flags.length > 0) && (
            <div style={{ background: "rgba(245,166,35,0.07)", border: "1px solid rgba(245,166,35,0.35)", borderRadius: 8, padding: "12px 16px" }}>
              {result.flags.map((f, i) => <div key={i} style={{ color: "#f5c97a", fontSize: 12.5, lineHeight: 1.5 }}>⚠ {f}</div>)}
            </div>
          )}
          <div style={{ color: "#5d6b80", fontSize: 11, lineHeight: 1.5 }}>
            {result.notes.map((n, i) => <div key={i}>{n}</div>)}
          </div>
        </section>
      )}
    </div>
  );
}

function Stat({ label, value, accent, big }: { label: string; value: string; accent: string; big?: boolean }) {
  return (
    <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: "14px 20px", minWidth: 180 }}>
      <div style={{ color: "#7e8ca0", fontSize: 11, marginBottom: 6 }}>{label}</div>
      <div style={{ color: accent, fontSize: big ? 24 : 16, fontWeight: 700, fontFamily: "Geist Mono, monospace" }}>{value}</div>
    </div>
  );
}

function Line({ label, value, sub, good }: { label: string; value: string; sub?: boolean; good?: boolean }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 12, paddingLeft: sub ? 14 : 0 }}>
      <span style={{ color: sub ? "#7e8ca0" : "#cdd8e8" }}>{label}</span>
      <span style={{ color: good ? "#30d158" : "#eef3fb", fontFamily: "Geist Mono, monospace" }}>{value}</span>
    </div>
  );
}
