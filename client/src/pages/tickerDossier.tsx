// Ticker dossier (2026-10-29, human-asked) — replaces the AI Engine look-up's
// entry point: the same box (and "/" shortcut) now opens this page, and the
// look-up card itself lives on here in the stock layer, unchanged.
// Weather, not prophecy: the RANGE is the hero (calibrated bands with their
// backtest and live ledger), direction appears only with a measured track
// record, every layer says where it came from, how old it is and whether it
// ran. Top-down: Market → Sector → Stock → Options → Events, then the trade
// plan. Data: GET /api/ticker-dossier/:symbol (ticker_dossier.py, owner only)
// + the look-up (/api/bot/lookup/:symbol) for the bot's own score and gates.
// Layout (2026-10-29, human-asked): the AI Engine's cockpit — a KPI strip
// (price, both ranges, options vs model, regime, bot score), the range hero and
// the trade plan as 2-wide panels, then the layers as panels in one full-width
// grid (pages/cockpit.tsx). Every number, label, provenance line and the
// position-size input are as before; only the layout changed.
import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, FileSearch, RefreshCw, Search } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import {
  type Dossier, type RangeH, type LayerMeta, type Plan, type Trend, type EventItem,
  fmtPx, fmtPct, fmtRate, parseSymbol, symbolFromHash, bandGeometry, ladderGeometry, positionSize, decileFor, decileLine,
  coverageLine, ledgerRangeLine, ledgerPlanLine, directionHeadline, trackRecordLine, layerGaps, ageText, pushRecent, historyLine, insiderCluster, dossierKpis,
  DOSSIER_RECENT_KEY, TIER_SHORT,
} from "@/lib/tickerDossier";
import { LookupCard } from "@/bot/panels";
import type { LookupResult, LedgerReport } from "@/bot/lookup";
import { KpiStrip, KpiTile, Panel } from "./cockpit";

async function getJson<T>(url: string, timeoutMs: number): Promise<T> {
  const r = await apiRequest("GET", url, undefined, { timeoutMs });
  return (await r.json()) as T;
}

function readRecent(): string[] {
  try {
    const v = JSON.parse(window.localStorage.getItem(DOSSIER_RECENT_KEY) || "[]");
    return Array.isArray(v) ? v.filter((s): s is string => typeof s === "string").slice(0, 8) : [];
  } catch (e) {
    console.debug("dossier: recent list unavailable", e);
    return [];
  }
}
function writeRecent(list: string[]): void {
  try { window.localStorage.setItem(DOSSIER_RECENT_KEY, JSON.stringify(list)); } catch (e) { console.debug("dossier: recent list not saved", e); }
}

function Status({ l }: { l?: LayerMeta | null }) {
  const s = l?.status ?? "unavailable";
  return <span className="vt-ds-status" data-s={s}>{s}</span>;
}
function Prov({ l, now }: { l?: LayerMeta | null; now: number }) {
  if (!l) return null;
  return <p className="vt-ds-prov">{l.source}{l.as_of ? ` · as of ${l.as_of.length === 10 ? l.as_of : ageText(l.as_of, now)}` : ""}</p>;
}
function Kv({ k, v, tone }: { k: string; v: string; tone?: "g" | "r" | "o" }) {
  return <div className="vt-ds-kv"><span>{k}</span><b data-tone={tone}>{v}</b></div>;
}
const signTone = (v: number | null | undefined): "g" | "r" | undefined => (typeof v === "number" ? (v > 0 ? "g" : v < 0 ? "r" : undefined) : undefined);

function TrendRow({ t }: { t?: Trend }) {
  if (!t) return null;
  return (
    <div className="vt-ds-kvs">
      <Kv k="trend" v={t.label} tone={t.label === "up" ? "g" : t.label === "down" ? "r" : undefined} />
      <Kv k="5 sessions" v={fmtPct(t.ret_5d_pct, 1, true)} tone={signTone(t.ret_5d_pct)} />
      <Kv k="20 sessions" v={fmtPct(t.ret_20d_pct, 1, true)} tone={signTone(t.ret_20d_pct)} />
      <Kv k="60 sessions" v={fmtPct(t.ret_60d_pct, 1, true)} tone={signTone(t.ret_60d_pct)} />
      <Kv k="vs 50-day avg" v={fmtPct(t.vs_sma50_pct, 1, true)} tone={signTone(t.vs_sma50_pct)} />
      <Kv k="vs 200-day avg" v={fmtPct(t.vs_sma200_pct, 1, true)} tone={signTone(t.vs_sma200_pct)} />
    </div>
  );
}

/** The hero: one horizon's 68% / 90% band as a picture, with the price on it. */
function BandCard({ title, r, base, price, ledger }: { title: string; r?: RangeH; base?: number; price?: number; ledger: { all: string; mine: string } }) {
  if (!r || !r.bands || typeof base !== "number") {
    return <div className="vt-ds-band" data-empty="true"><h3>{title}</h3><p className="vt-ds-note">No range — not enough price history to fit the volatility model.</p></div>;
  }
  const b = r.bands;
  const g = bandGeometry(b, base, price);
  return (
    <div className="vt-ds-band">
      <div className="vt-ds-bandhead">
        <h3>{title}</h3>
        <span className="vt-ds-big vt-ds-m">±{b.move68_pct.toFixed(1)}%</span>
        <span className="vt-ds-meta">68% range · 90%: ±{b.move90_pct.toFixed(1)}%</span>
      </div>
      <div className="vt-ds-bandviz" role="img" aria-label={`68% range ${fmtPx(b.lo68)} to ${fmtPx(b.hi68)}, 90% range ${fmtPx(b.lo90)} to ${fmtPx(b.hi90)}`}>
        <div className="vt-ds-b90" style={{ left: `${g.lo90}%`, width: `${g.hi90 - g.lo90}%` }} />
        <div className="vt-ds-b68" style={{ left: `${g.lo68}%`, width: `${g.hi68 - g.lo68}%` }} />
        <div className="vt-ds-bbase" style={{ left: `${g.base}%` }} title={`last close ${fmtPx(base)}`} />
        {g.price !== null ? <div className="vt-ds-bpx" data-out={g.priceOutside ?? undefined} style={{ left: `${g.price}%` }} title={`now ${fmtPx(price)}`}><span>now</span></div> : null}
      </div>
      <div className="vt-ds-bandlbl vt-ds-m">
        <span style={{ left: `${g.lo90}%` }}>{fmtPx(b.lo90)}</span>
        <span style={{ left: `${g.lo68}%` }}>{fmtPx(b.lo68)}</span>
        <span style={{ left: `${g.hi68}%` }}>{fmtPx(b.hi68)}</span>
        <span style={{ left: `${g.hi90}%` }}>{fmtPx(b.hi90)}</span>
      </div>
      <p className="vt-ds-note">{coverageLine(r.backtest)}</p>
      <p className="vt-ds-note"><b>live record</b> {ledger.all}</p>
      <p className="vt-ds-note">this ticker: {ledger.mine}</p>
      <p className="vt-ds-meta">model {r.model === "har" ? "HAR-RV" : r.model === "ewma" ? "EWMA" : "—"}{r.fallback_reason ? ` (${r.fallback_reason})` : ""}</p>
    </div>
  );
}

function PlanCard({ p, ledgerLine, hasDirection }: { p: Plan; ledgerLine: string; hasDirection: boolean }) {
  const [risk, setRisk] = useState("");
  const g = ladderGeometry(p);
  const size = positionSize(Number(risk), p.entry_ref, p.stop);
  const t = p.touch, t2 = p.touch_stretch;
  return (
    <div className="vt-ds-plan">
      <div className="vt-ds-ladder" role="img" aria-label={`stop ${fmtPx(p.stop)}, entry ${fmtPx(p.entry_ref)}, target ${fmtPx(p.target)}, stretch ${fmtPx(p.stretch_target)}`}>
        <div className="vt-ds-lrisk" style={{ left: `${g.stop}%`, width: `${g.entry - g.stop}%` }} />
        <div className="vt-ds-lreward" style={{ left: `${g.entry}%`, width: `${g.stretch - g.entry}%` }} />
        {([["stop", g.stop, p.stop], ["entry", g.entry, p.entry_ref], ["target", g.target, p.target], ["stretch", g.stretch, p.stretch_target]] as const).map(([k, x, v]) => (
          <div key={k} className="vt-ds-lline" data-k={k} style={{ left: `${x}%` }}><span>{k}<br /><b className="vt-ds-m">{fmtPx(v)}</b></span></div>
        ))}
      </div>
      <div className="vt-ds-plangrid">
        <div className="vt-ds-tile"><span>Stop</span><b className="vt-ds-m" data-tone="r">{fmtPx(p.stop)}</b><em>{fmtPct(p.stop_pct, 1, true)} · below the 90% 5-session band</em></div>
        <div className="vt-ds-tile"><span>Target</span><b className="vt-ds-m" data-tone="g">{fmtPx(p.target)}</b><em>{fmtPct(p.target_pct, 1, true)} · top of the 68% band</em></div>
        <div className="vt-ds-tile"><span>Stretch</span><b className="vt-ds-m">{fmtPx(p.stretch_target)}</b><em>{fmtPct(p.stretch_pct, 1, true)} · top of the 90% band</em></div>
        <div className="vt-ds-tile"><span>Reward : risk</span><b className="vt-ds-m">{p.reward_risk?.toFixed(2) ?? "—"}</b><em>stretch {p.stretch_reward_risk?.toFixed(2) ?? "—"}</em></div>
      </div>
      <div className="vt-ds-touch">
        <div><span>Target before stop</span><b className="vt-ds-m">{fmtRate(t.p_target_first)}</b></div>
        <div><span>Stop before target</span><b className="vt-ds-m">{fmtRate(t.p_stop_first)}</b></div>
        <div><span>Neither in 5 sessions</span><b className="vt-ds-m">{fmtRate(t.p_neither)}</b></div>
        <div><span>Stretch before stop</span><b className="vt-ds-m">{fmtRate(t2.p_target_first)}</b></div>
      </div>
      <p className="vt-ds-note">
        {hasDirection ? "" : "No validated drift: "}these odds come from the volatility model alone ({p.rules.probabilities}). With no edge, a near target is hit often and a far stop rarely, but the expected value is about zero before costs — the levels size the risk; they do not predict the move.
      </p>
      <p className="vt-ds-note"><b>record</b> {ledgerLine}</p>
      <form className="vt-ds-size" onSubmit={(e) => e.preventDefault()}>
        <label htmlFor="vt-ds-risk">Dollars you are willing to lose</label>
        <input id="vt-ds-risk" inputMode="decimal" placeholder="e.g. 250" value={risk} onChange={(e) => setRisk(e.target.value.replace(/[^0-9.]/g, ""))} />
        <span className="vt-ds-m">{size ? (size.shares ? `${size.shares} shares · ${fmtPx(size.notional)} position · ${fmtPx(size.lossAtStop)} lost at the stop` : "less than one share at this stop distance") : `shares = dollars ÷ (entry ${fmtPx(p.entry_ref)} − stop ${fmtPx(p.stop)})`}</span>
      </form>
      <details className="vt-ds-details">
        <summary>How the levels are set</summary>
        <ul>{Object.entries(p.rules).map(([k, v]) => <li key={k}><b>{k}</b> — {v}</li>)}</ul>
        <p className="vt-ds-note">Entry reference = the latest price when the page was built. The plan is scored on the 5 full sessions after today (daily high/low; a session that touches both is counted as "same-day both").</p>
      </details>
    </div>
  );
}

function EventRow({ e }: { e: EventItem }) {
  const hist = e.history ? historyLine(e.history, { window: e.history.study_window, path: e.history.study_path, date: e.history.study_date }) : null;
  return (
    <li className="vt-ds-ev" data-kind={e.kind}>
      <div className="vt-ds-evtop">
        <span className="vt-ds-tier" data-tier={e.tier} title={`credibility tier ${e.tier}`}>T{e.tier} · {TIER_SHORT[e.tier] ?? "unverified"}</span>
        <span className="vt-ds-meta vt-ds-m">{e.date}{e.form ? ` · ${e.form}` : ""} · {e.source}</span>
      </div>
      {e.url ? <a href={e.url} target="_blank" rel="noopener noreferrer" className="vt-ds-evt">{e.title || "(untitled)"}</a> : <span className="vt-ds-evt">{e.title || "(untitled)"}</span>}
      {hist ? <span className="vt-ds-react" data-history="true"><i>history</i> {hist}</span> : null}
    </li>
  );
}

function Skeleton() {
  return (
    <div className="vt-ck-grid" aria-busy="true" aria-label="Loading the dossier">
      {[180, 120, 220, 160, 160].map((h, i) => <div key={i} className={`vt-ds-skel${i === 0 ? " vt-ck-span2" : ""}`} style={{ height: h }} />)}
    </div>
  );
}

export default function TickerDossierPage() {
  const [sym, setSym] = useState<string | null>(() => symbolFromHash(window.location.hash));
  const [input, setInput] = useState(sym ?? "");
  const [recent, setRecent] = useState<string[]>(readRecent);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const t = window.setInterval(() => setNow(Date.now()), 30_000); return () => window.clearInterval(t); }, []);
  useEffect(() => {
    const onHash = () => { const s = symbolFromHash(window.location.hash); if (s) { setSym(s); setInput(s); } };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => { if (!sym) inputRef.current?.focus(); }, [sym]);

  const q = useQuery({ queryKey: ["/api/ticker-dossier", sym], enabled: !!sym, queryFn: () => getJson<Dossier>(`/api/ticker-dossier/${sym}`, 125_000), staleTime: 4 * 60_000, retry: 1 });
  const lk = useQuery({ queryKey: ["/api/bot/lookup", sym], enabled: !!sym, queryFn: () => getJson<LookupResult>(`/api/bot/lookup/${sym}`, 120_000), staleTime: 60_000, retry: 0 });
  const lg = useQuery({ queryKey: ["/api/bot/lookup-ledger"], enabled: !!sym, queryFn: () => getJson<LedgerReport>("/api/bot/lookup-ledger", 20_000), staleTime: 5 * 60_000, retry: 0 });

  useEffect(() => {
    if (q.data?.stock?.status === "live" && sym) setRecent((r) => { const n = pushRecent(r, sym); writeRecent(n); return n; });
  }, [q.data, sym]);

  const go = (raw: string) => {
    const s = parseSymbol(raw);
    if (!s) return;
    setInput(s); setSym(s);
    window.location.hash = `#/dossier/${s}`;
  };

  const d = q.data;
  const st = d?.stock;
  const botScore = typeof lk.data?.deep?.score === "number" ? lk.data.deep.score : null;
  const dec = useMemo(() => decileFor(d?.bot_track, botScore), [d?.bot_track, botScore]);
  const dir = directionHeadline(d?.direction);
  const gaps = layerGaps(d);
  const opt = d?.options;
  const kpis = useMemo(() => (d ? dossierKpis(d, { score: botScore, minScore: lk.data?.regime?.min_score, verdict: lk.data?.decision?.verdict, busy: lk.isLoading }) : []), [d, botScore, lk.data, lk.isLoading]);

  return (
    <div className="vt-ck vt-ds" data-vt-dossier>
      <header className="vt-ck-head">
        <h1><FileSearch size={16} aria-hidden="true" /> {sym ? <>Dossier <span className="vt-ds-m">{sym}</span></> : "Ticker dossier"}</h1>
        <span className="vt-ds-chip" title="Research output — not a recommendation to buy or sell">Research, not advice</span>
        <form className="vt-ds-search" role="search" onSubmit={(e) => { e.preventDefault(); go(input); }}>
          <Search size={14} aria-hidden="true" />
          <input ref={inputRef} value={input} onChange={(e) => setInput(e.target.value)} placeholder="Symbol, e.g. RKLB" aria-label="Look up a stock or option" spellCheck={false} autoComplete="off" maxLength={21} />
          <button type="submit" className="vt-ds-btn" disabled={!parseSymbol(input) || q.isFetching}>{q.isFetching ? "building…" : "Open"}</button>
        </form>
        {recent.length ? <div className="vt-ds-recent">{recent.filter((s) => s !== sym).map((s) => <button key={s} type="button" className="vt-ds-btn" data-small="true" onClick={() => go(s)}>{s}</button>)}</div> : null}
        {d && !q.isLoading ? (
          <div className="vt-ck-fresh">
            <span>built <b>{ageText(d.generated_at, now)}</b>{d.cached ? " (cached)" : ""}</span>
            {st?.base_date ? <span>last close <b>{st.base_date}</b></span> : null}
            {st?.price_day ? <span>price <b>{st.price_is_partial_session ? `${st.price_day} (session in progress)` : st.price_day}</b></span> : null}
            <button type="button" className="vt-ds-btn" onClick={() => { void q.refetch(); void lk.refetch(); }} disabled={q.isFetching} aria-label="Rebuild the dossier">
              <RefreshCw size={12} aria-hidden="true" className={q.isFetching ? "vt-ds-spin" : undefined} /> {q.isFetching ? "building" : "rebuild"}
            </button>
          </div>
        ) : null}
        <a className="vt-ds-back" href="#/bot"><ArrowLeft size={13} aria-hidden="true" /> AI Engine</a>
      </header>

      {!sym ? (
        <div className="vt-ds-state"><b>Look up any stock.</b> The dossier reads it like a weather forecast — the market, then the sector, then the stock, its options and its news — and leads with what is genuinely forecastable: how far the price is likely to move. Direction is shown only where a signal has a measured record.</div>
      ) : q.isLoading ? <Skeleton /> : q.isError && !d ? (
        <div className="vt-ds-state" data-tone="red" role="alert"><b>The dossier could not be built.</b> {String(q.error instanceof Error ? q.error.message : q.error).slice(0, 180)}</div>
      ) : d && d.errors?.symbol ? (
        <div className="vt-ds-state"><b>{d.symbol || input}</b> is not a symbol the dossier can read: {d.errors.symbol}</div>
      ) : d && st?.status !== "live" ? (
        <div className="vt-ds-state"><b>No price history for {d.symbol}.</b> {st?.reason ?? d.errors?.stock ?? d.errors?.dossier ?? "The symbol may be delisted, an OTC name, or not on Alpaca's feed."}</div>
      ) : d && st ? (
        <>
          <KpiStrip n={6} label="Dossier summary">
            {kpis.map((k) => <KpiTile key={k.label} label={k.label} value={k.value} sub={k.sub} subTone={k.subTone} line={k.line} tone={k.tone} />)}
          </KpiStrip>

          <div className="vt-ck-grid">
            {/* HERO — the range */}
            <Panel id="vt-ds-range" title="The range — what is forecastable" span={2} className="vt-ds-hero"
              meta={<span className="vt-ds-px vt-ds-m">{fmtPx(st.price)} <em data-tone={signTone(st.trend?.ret_5d_pct)}>{fmtPct(st.trend?.ret_5d_pct, 1, true)} 5d</em></span>}>
              <div className="vt-ds-bands">
                <BandCard title="Next session" r={st.ranges?.["1"]} base={st.base_close} price={st.price} ledger={{ all: ledgerRangeLine(d.ledger, "1"), mine: ledgerRangeLine(d.ledger, "1", "ticker") }} />
                <BandCard title="Next 5 sessions" r={st.ranges?.["5"]} base={st.base_close} price={st.price} ledger={{ all: ledgerRangeLine(d.ledger, "5"), mine: ledgerRangeLine(d.ledger, "5", "ticker") }} />
              </div>
              {st.expected_move ? (
                <p className="vt-ds-note">Today's expected daily move ±{st.expected_move.sigma_1d_pct.toFixed(1)}% vs this stock's median day over the past year ±{st.expected_move.median_abs_move_pct.toFixed(1)}%{typeof st.expected_move.sigma_pctile_1y === "number" ? ` — calmer than ${Math.round((1 - st.expected_move.sigma_pctile_1y) * 100)}% of its last year` : ""}.</p>
              ) : null}
              <p className="vt-ds-meta">{st.range_method}</p>
            </Panel>

            {/* TRADE PLAN */}
            <Panel id="vt-ds-plan" title="Trade plan · 5 sessions" span={2} meta="levels from the range model">
              {d.plan ? <PlanCard p={d.plan} ledgerLine={ledgerPlanLine(d.ledger)} hasDirection={d.direction?.status === "call"} /> : <div className="vt-ds-state">No plan — the volatility model could not be fitted.</div>}
            </Panel>

            {/* DIRECTION — honest; with the bot's read it is the column beside the hero from 3 grid columns up (index.css .vt-ds-side) */}
            <div className="vt-ds-side">
            <Panel id="vt-ds-dir" title="Direction" meta={<span className="vt-ds-dirword" data-tone={dir.tone}>{dir.word}</span>}>
              <p className="vt-ds-lead">{d.direction?.status === "call" && d.direction.call
                ? <>The weekly screen calls <b>{d.direction.call.direction}</b> through {d.direction.call.horizon_end} with P(up) {fmtRate(d.direction.call.p_up)}.</>
                : <>{d.direction?.statement ?? "No validated direction signal for this ticker — near a coin flip."}</>}</p>
              <p className="vt-ds-note">{trackRecordLine(d.direction)}{d.direction?.calibration_bin && d.direction.calibration_bin.n ? ` · calls near ${fmtRate(d.direction.calibration_bin.mean_p)} rose ${fmtRate(d.direction.calibration_bin.freq)} of the time (n=${d.direction.calibration_bin.n})` : ""}</p>
            </Panel>

            <Panel id="vt-ds-botread" title="The bot's own read" meta={lk.isLoading ? "scoring…" : botScore !== null ? `deep score ${botScore.toFixed(1)}` : "no score"}>
              <p className="vt-ds-note">{lk.isLoading ? "the bot is scoring this name through its scan path…" : botScore !== null ? <>deep score <b className="vt-ds-m">{botScore.toFixed(1)}</b>{lk.data?.regime ? ` vs MIN_SCORE ${lk.data.regime.min_score}` : ""} · {lk.data?.decision?.verdict ?? "undecided"}</> : "no bot score (the look-up could not score this name)"}</p>
              <p className="vt-ds-note"><b>past candidates in this score band</b> {decileLine(dec, d.bot_track?.base)}</p>
              {dec && dec.d20.n ? <p className="vt-ds-note">{decileLine(dec, d.bot_track?.base, "d20")}</p> : null}
              {d.bot_guards ? (
                <p className="vt-ds-note"><b>regime guards</b> {d.bot_guards.would_block.length ? <span data-tone="r">would block: {d.bot_guards.would_block.map((g) => d.bot_guards?.reasons[g] ?? g).join("; ")}</span> : `nothing blocks a new long in ${d.bot_guards.composite ?? "this regime"}`}</p>
              ) : null}
              {d.bot_track?.source ? <p className="vt-ds-meta">{d.bot_track.source}{d.bot_track.n ? ` · ${d.bot_track.n.toLocaleString()} settled candidates` : ""}</p> : <p className="vt-ds-meta">no shadow-candidate outcomes available</p>}
            </Panel>
            </div>

            {/* LAYERS, top-down */}
            <Panel id="vt-ds-mkt" title="1 · Market" meta={<Status l={d.market} />} status={d.market?.status}>
              {d.market?.status === "live" ? (
                <>
                  <div className="vt-ds-kvs">
                    <Kv k="regime" v={d.market.composite ?? "—"} />
                    <Kv k="exposure" v={typeof d.market.exposure === "number" ? `${Math.round(d.market.exposure * 100)}%` : "—"} />
                    <Kv k="trend" v={d.market.trend ?? "—"} /><Kv k="volatility" v={d.market.vol ?? "—"} /><Kv k="breadth" v={d.market.breadth ?? "—"} />
                    <Kv k="VIX" v={typeof d.market.stress?.vix === "number" ? (d.market.stress.vix as number).toFixed(1) : "—"} />
                    <Kv k="VIX/VIX3M" v={typeof d.market.stress?.vix_vix3m === "number" ? (d.market.stress.vix_vix3m as number).toFixed(2) : "—"} tone={d.market.stress?.vix_vix3m_inverted ? "r" : undefined} />
                    <Kv k="days in regime" v={String(d.market.days_in_regime ?? "—")} />
                  </div>
                  {d.market.spy ? <><h3 className="vt-ds-h3">SPY</h3><TrendRow t={d.market.spy.trend} /></> : null}
                </>
              ) : <p className="vt-ds-note">{d.market?.reason ?? "unavailable"}</p>}
              <Prov l={d.market} now={now} />
            </Panel>

            <Panel id="vt-ds-sector" title={`2 · Sector${d.sector?.label ? ` — ${d.sector.label}` : ""}`} meta={<Status l={d.sector} />} status={d.sector?.status}>
              {d.sector?.status === "live" ? (
                <>
                  <div className="vt-ds-kvs">
                    <Kv k="sector ETF" v={d.sector.etf ?? "—"} />
                    <Kv k="ETF vs SPY 20d" v={fmtPct(d.sector.rs_vs_spy_20d, 1, true)} tone={signTone(d.sector.rs_vs_spy_20d)} />
                    <Kv k="ETF vs SPY 60d" v={fmtPct(d.sector.rs_vs_spy_60d, 1, true)} tone={signTone(d.sector.rs_vs_spy_60d)} />
                    <Kv k={`${d.symbol} vs ETF 20d`} v={fmtPct(d.sector.stock_vs_sector_20d, 1, true)} tone={signTone(d.sector.stock_vs_sector_20d)} />
                    <Kv k={`${d.symbol} vs ETF 60d`} v={fmtPct(d.sector.stock_vs_sector_60d, 1, true)} tone={signTone(d.sector.stock_vs_sector_60d)} />
                  </div>
                  <TrendRow t={d.sector.trend} />
                </>
              ) : (
                <>
                  <p className="vt-ds-note">{d.sector?.reason ?? "unavailable"}</p>
                  <div className="vt-ds-kvs"><Kv k={`${d.symbol} vs SPY 20d`} v={fmtPct(d.sector?.stock_vs_spy_20d, 1, true)} tone={signTone(d.sector?.stock_vs_spy_20d)} /><Kv k={`${d.symbol} vs SPY 60d`} v={fmtPct(d.sector?.stock_vs_spy_60d, 1, true)} tone={signTone(d.sector?.stock_vs_spy_60d)} /></div>
                </>
              )}
              <Prov l={d.sector} now={now} />
            </Panel>

            <Panel id="vt-ds-stock" title="3 · Stock" meta={<Status l={st} />} status={st.status} className="vt-ds-stock">
              <div className="vt-ds-kvs">
                <Kv k="price" v={fmtPx(st.price)} /><Kv k="last close" v={fmtPx(st.base_close)} />
                <Kv k="52-week range" v={`${fmtPx(st.low_52w)} – ${fmtPx(st.high_52w)}`} />
                <Kv k="realized vol 20d" v={fmtPct(st.realized_vol_20d_pct, 0)} /><Kv k="realized vol 60d" v={fmtPct(st.realized_vol_60d_pct, 0)} />
                <Kv k="vs SPY 20d" v={fmtPct(st.vs_spy_20d, 1, true)} tone={signTone(st.vs_spy_20d)} /><Kv k="vs SPY 60d" v={fmtPct(st.vs_spy_60d, 1, true)} tone={signTone(st.vs_spy_60d)} />
              </div>
              <TrendRow t={st.trend} />
              <h3 className="vt-ds-h3">The bot's look-up</h3>
              <LookupCard r={lk.data ?? (lk.isError ? { symbol: sym ?? "", at: new Date().toISOString(), errors: { lookup: String(lk.error instanceof Error ? lk.error.message : lk.error).slice(0, 160) } } : null)} busy={lk.isLoading} history={[]} ledger={lg.data ?? null} />
              <Prov l={st} now={now} />
            </Panel>

            <Panel id="vt-ds-opt" title="4 · Options" meta={<Status l={opt} />} status={opt?.status}>
              {opt?.status !== "unavailable" && typeof opt?.atm_iv === "number" ? (
                <>
                  <p className="vt-ds-lead">Options price a <b className="vt-ds-m">±{fmtPct(opt.implied_move_5d_pct)}</b> week; our model <b className="vt-ds-m">±{fmtPct(opt.model_move_5d_pct)}</b>{typeof opt.implied_vs_model_5d === "number" ? ` (implied is ${opt.implied_vs_model_5d.toFixed(2)}× the model)` : ""}.</p>
                  <div className="vt-ds-kvs">
                    <Kv k="ATM IV (front)" v={fmtPct(opt.atm_iv, 0)} /><Kv k="ATM IV ~30d" v={fmtPct(opt.atm_iv_30d, 0)} />
                    <Kv k="implied 1-day move" v={`±${fmtPct(opt.implied_move_1d_pct)}`} /><Kv k="model 1-day move" v={`±${fmtPct(opt.model_move_1d_pct)}`} />
                    <Kv k="put − call skew" v={typeof opt.skew_put_minus_call_pts === "number" ? `${opt.skew_put_minus_call_pts > 0 ? "+" : ""}${opt.skew_put_minus_call_pts.toFixed(1)} pts` : "—"} />
                    <Kv k="front expiry" v={opt.front ? `${opt.front.exp_date} (${opt.front.days_out}d)` : "—"} />
                  </div>
                  <p className="vt-ds-meta">{opt.skew_strikes}. An earnings date inside the front expiry inflates its IV.</p>
                </>
              ) : <p className="vt-ds-note">{opt?.reason ?? "unavailable"}</p>}
              <p className="vt-ds-note"><b>next earnings</b> {opt?.earnings?.status === "live" ? `${opt.earnings.date}${opt.earnings.hour ? ` (${opt.earnings.hour})` : ""} · ${opt.earnings.source}` : opt?.earnings?.reason ?? "unknown"}</p>
              <Prov l={opt} now={now} />
            </Panel>

            <Panel id="vt-ds-ev" title={`5 · Events · last ${d.events?.window_days ?? 60} days`} meta={<Status l={d.events} />} status={d.events?.status} span={2} className="vt-ds-events">
              {d.events?.items?.length ? <ul className="vt-ds-evs">{d.events.items.map((e, i) => <EventRow key={`${e.time}-${i}`} e={e} />)}</ul> : <p className="vt-ds-note">{d.events?.status === "unavailable" ? d.events.reason : "No filings or news in the window."}</p>}
              {d.events?.form4_count ? <p className="vt-ds-note">{d.events.form4_count} Form 4 insider filing{d.events.form4_count === 1 ? "" : "s"} in the window{d.events.form4_count > 5 ? " (newest 5 listed)" : ""}.</p> : null}
              {d.insiders && d.insiders.filings ? (
                <p className="vt-ds-note"><b>insider trades (archive, {d.insiders.windowDays}d)</b> {d.insiders.buys} open-market buy{d.insiders.buys === 1 ? "" : "s"} ({d.insiders.buyShares.toLocaleString()} sh) · {d.insiders.sells} sale{d.insiders.sells === 1 ? "" : "s"} ({d.insiders.sellShares.toLocaleString()} sh) — {d.insiders.note}</p>
              ) : null}
              {d.insiders && insiderCluster(d.insiders.rows) && d.events?.event_history?.insider_cluster ? (
                <p className="vt-ds-note vt-ds-react" data-history="true"><i>history</i> {historyLine(d.events.event_history.insider_cluster, d.events.event_study)}</p>
              ) : null}
              <p className="vt-ds-meta">{d.events?.history_note}</p>
              <details className="vt-ds-details">
                <summary>Source credibility tiers</summary>
                <ul>{Object.entries(d.credibility_tiers ?? {}).map(([k, v]) => <li key={k}><b>T{k}</b> — {v}</li>)}</ul>
                <p className="vt-ds-note">A documented static table: a tier says what kind of source it is, not whether a story is right.</p>
              </details>
              <Prov l={d.events} now={now} />
            </Panel>
          </div>

          {gaps.length ? <div className="vt-ds-gaps" role="note"><b>Could not run:</b> {gaps.join(" · ")}</div> : null}
          <footer className="vt-ck-foot"><p>Research, not advice. Ranges are model estimates with their measured hit rates shown; nothing on this page places, sizes or gates a trade.{d.timings?.total_s ? ` Built in ${d.timings.total_s}s${d.via ? ` via ${d.via}` : ""}.` : ""}</p></footer>
        </>
      ) : null}
    </div>
  );
}
