import { type Express } from "express";
import { type Server } from "http";
import { exec } from "child_process";
import { promisify } from "util";
import path from "path";
import fs from "fs";
import https from "https";
import { WebSocket as WSClient } from "ws";
import cookieParser from "cookie-parser";
// datacore JSON is imported statically so esbuild bakes it into dist/index.cjs
// — the frozen Dockerfile's runtime stage copies selective paths and datacore/
// never reaches the image, which made runtime fs reads return {} in prod.
import datacoreLayers from "../datacore/layers.json";
import datacoreSites from "../datacore/sites/strategic_sites.json";
import datacorePowerplants from "../datacore/powerplants/us_power_plants.json";
import {
  archiveAircraft, archiveVessels, archiveTrains, compressOldHours, rollupOldDays,
  recentTrack, archiveStats,
} from "./datacoreArchive";
import { registerAuthRoutes, db } from "./auth";
import { registerBotRoutes } from "./bot";
import { vesselStreamEnabled, bootVesselStream } from "./vesselStream";
import { complianceAuditTick, setComplianceAuditWriter } from "./providerCompliance";
import { mapDigitraffic, mapEntur, ENTUR_VEHICLES_QUERY } from "./trainsFeed";
import { computeShadowStats } from "./shadowFleet";
import { computePortDwell, portsFromSites } from "./portDwell";
import shadowZones from "../datacore/shadow_zones.json";
import { bootForm4Poll, latestForm4Filings, readFilingHistory } from "./edgarForm4";

const execAsync = promisify(exec);

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface ScanResult {
  ticker: string;
  scan_score?: number;
  price?: number;
  change_pct?: number;
  volume?: number;
  iv_rank?: number;
  iv_percentile?: number;
  put_call_ratio?: number;
  unusual_activity?: boolean;
  signal?: string;
  sentiment_score?: number;
  sentiment_signal?: string;
  rec_action?: string;
  rec_signal?: string;
  freshness?: "fresh" | "recent" | "stale";
  scanned_at?: number; // epoch ms
  error?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tier 1 — fast list (~150 tickers, refreshes every 5 min)
// ─────────────────────────────────────────────────────────────────────────────

const TIER1_BASE: string[] = [
  // ── Original ~60 from routes_current ──
  "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC", "QCOM",
  "AVGO", "MU", "AMAT", "LRCX", "KLAC", "TXN", "ADI", "MRVL", "ARM", "SMCI",
  "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "XLU",
  "COIN", "MSTR", "RIOT", "MARA", "HUT", "CLSK", "BTBT", "WULF", "IREN", "CIFR",
  "PTON", "NFLX", "DIS", "ROKU", "SPOT", "TTD", "TRADE", "APPS", "MGNI", "PUBM",
  "GME", "AMC", "BBBY", "KOSS", "EXPR", "BB", "NOK", "CLOV", "WKHS", "GOEV",
];

const EXTRA_TIER1: string[] = [
  "UBER", "LYFT", "SNAP", "PINS", "RBLX", "HOOD", "SOFI", "LCID", "RIVN", "NKLA",
  "PLTR", "SOUN", "AI", "BBAI", "IONQ", "RGTI", "QBTS", "ARRY", "CHPT", "BLNK",
  "ENPH", "SEDG", "FSLR", "NEE", "PLUG", "BE", "RUN", "SPWR", "NOVA", "STEM",
  "RKLB", "ASTS", "LUNR", "RDW", "MNTS", "ASTR", "SPCE", "VORB", "LMT", "RTX",
  "GD", "NOC", "BA", "HII", "TDG", "HEI", "AXON", "TASER", "MRNA", "BNTX",
  "NVAX", "VRTX", "REGN", "BIIB", "ILMN", "PACB", "CRSP", "BEAM", "EDIT", "NTLA",
  "SGEN", "ALNY", "BMRN", "SRPT", "RARE", "FOLD", "ACAD", "SAGE", "NBIX", "INCY",
  "WFC", "BAC", "JPM", "GS", "MS", "C", "USB", "PNC", "TFC", "KEY",
  "SQ", "PYPL", "AFRM", "UPST", "LC", "OPEN", "OFLD", "TREE", "NRDS", "DAVE",
  "W", "ETSY", "CHWY", "OSTK", "PRTS", "REAL", "POSH", "GENI", "MAPS", "VERV",
  "ZM", "DOCU", "DOCN", "NET", "FSLY", "ESTC", "MDB", "DDOG", "SNOW", "S",
  "U", "MTTR", "OUST", "LIDR", "MVIS", "LAZR", "VLDR", "INVZ", "AEVA",
];

// Deduplicate
const TIER1_TICKERS: string[] = Array.from(new Set([...TIER1_BASE, ...EXTRA_TIER1]));

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

const FULL_UNIVERSE_CACHE_MAX = 500; // Cap to prevent unbounded memory growth
let fullUniverseCache: Map<string, ScanResult> = new Map();
let tier1Cache: ScanResult[] = [];
let tier1LastUpdate = 0;
let fullScanProgress = {
  current: 0,
  total: 0,
  running: false,
  lastFullCycle: 0,
};

// ─────────────────────────────────────────────────────────────────────────────
// Freshness helper
// ─────────────────────────────────────────────────────────────────────────────

function upsertCache(r: ScanResult) {
  fullUniverseCache.set(r.ticker, r);
  // Evict oldest entries if cache exceeds limit
  if (fullUniverseCache.size > FULL_UNIVERSE_CACHE_MAX) {
    let oldestKey = "";
    let oldestTime = Infinity;
    fullUniverseCache.forEach((val, key) => {
      const t = val.scanned_at ?? 0;
      if (t < oldestTime) { oldestTime = t; oldestKey = key; }
    });
    if (oldestKey) fullUniverseCache.delete(oldestKey);
  }
}

function getFreshness(scannedAt: number | undefined): "fresh" | "recent" | "stale" {
  if (!scannedAt) return "stale";
  const ageMs = Date.now() - scannedAt;
  if (ageMs < 5 * 60 * 1000) return "fresh";
  if (ageMs < 20 * 60 * 1000) return "recent";
  return "stale";
}

function applyFreshness(results: ScanResult[]): ScanResult[] {
  return results.map((r) => ({ ...r, freshness: getFreshness(r.scanned_at) }));
}

function sortByScore(results: ScanResult[]): ScanResult[] {
  return [...results].sort((a, b) => (b.scan_score ?? 0) - (a.scan_score ?? 0));
}

// ─────────────────────────────────────────────────────────────────────────────
// Scan a single ticker via Python
// ─────────────────────────────────────────────────────────────────────────────

async function scanSingleTicker(ticker: string): Promise<ScanResult | null> {
  const scriptPath = path.resolve(process.cwd(), "analyze.py");
  try {
    const { stdout } = await execAsync(
      `python3 "${scriptPath}" "${ticker.toUpperCase()}" --mode=scan`,
      { timeout: 15000, maxBuffer: 1024 * 1024 * 2 }
    );
    const output = stdout.trim();
    if (!output) return null;

    const raw = JSON.parse(output);

    // Normalise into ScanResult shape
    const result: ScanResult = {
      ticker: ticker.toUpperCase(),
      scan_score: raw.scan_score ?? raw.score ?? 0,
      price: raw.price,
      change_pct: raw.change_pct ?? raw.change_percent,
      volume: raw.volume,
      iv_rank: raw.iv_rank,
      iv_percentile: raw.iv_percentile,
      put_call_ratio: raw.put_call_ratio,
      unusual_activity: raw.unusual_activity,
      signal: raw.signal,
      sentiment_score: raw.sentiment_score,
      sentiment_signal: raw.sentiment_signal,
      rec_action: raw.rec_action,
      rec_signal: raw.rec_signal,
      scanned_at: Date.now(),
    };
    return result;
  } catch {
    return null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scan a batch of tickers concurrently
// ─────────────────────────────────────────────────────────────────────────────

async function scanBatch(tickers: string[]): Promise<ScanResult[]> {
  const results = await Promise.allSettled(tickers.map(scanSingleTicker));
  const valid: ScanResult[] = [];
  for (const r of results) {
    if (r.status === "fulfilled" && r.value !== null) {
      valid.push(r.value);
    }
  }
  return valid;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tier 1 refresh (called by interval + on-demand if cache expired)
// ─────────────────────────────────────────────────────────────────────────────

async function refreshTier1(): Promise<void> {
  try {
    const BATCH = 20;
    const fresh: ScanResult[] = [];
    for (let i = 0; i < TIER1_TICKERS.length; i += BATCH) {
      const batch = TIER1_TICKERS.slice(i, i + BATCH);
      const results = await scanBatch(batch);
      fresh.push(...results);
      // Also update fullUniverseCache
      for (const r of results) {
        upsertCache(r);
      }
    }
    tier1Cache = sortByScore(applyFreshness(fresh));
    tier1LastUpdate = Date.now();
  } catch (err) {
    console.error("[scanner] Tier1 refresh error:", err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CBOE universe fetch
// ─────────────────────────────────────────────────────────────────────────────

const CBOE_CSV_URL =
  "https://www.cboe.com/us/options/symboldir/equity_index_options/?download=csv";
const CBOE_CACHE_PATH = "/tmp/cboe_universe.json";
const CBOE_CACHE_TTL_MS = 24 * 60 * 60 * 1000; // 1 day

// Hardcoded 500-ticker fallback (S&P 500 + Nasdaq 100 representative set)
const FALLBACK_UNIVERSE: string[] = [
  ...TIER1_TICKERS,
  "MMM","AOS","ABT","ABBV","ACN","ADBE","AES","AFL","A","APD","ABNB","AKAM","ALB",
  "ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMGN","AMP","AMT","AWK",
  "ATO","T","ADSK","ADP","AZO","AVB","AVY","BKR","BALL","BDX","BRK.B","BBY",
  "BIO","TECH","BIIB","BLK","BX","BA","BKNG","BWA","BXP","BSX","BMY","AVGO",
  "BR","CHRW","CDNS","CPB","COF","CAH","KMX","CCL","CARR","CTLT","CAT","CBOE",
  "CBRE","CDW","CE","CNC","CNP","CDAY","CF","CRL","SCHW","CHTR","CVX","CMG","CB",
  "CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH",
  "CL","CMCSA","CMA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CTVA",
  "CSGP","COST","CTRA","CCI","CSX","CMI","CVS","DHI","DHR","DRI","DVA","DE",
  "DAL","XRAY","DVN","DXCM","FANG","DLR","DFS","DIS","DG","DLTR","D","DPZ",
  "DOV","DOW","DTE","DUK","DRE","DD","EMN","ETN","EBAY","ECL","EIX","EW","EA",
  "ELV","LLY","EMR","ENPH","ETR","EOG","EPAM","EFX","EQIX","EQR","ESS","EL",
  "ETSY","EVRG","ES","EXC","EXPD","EXPE","EXR","XOM","FFIV","FDS","FICO","FAST",
  "FRT","FDX","FITB","FRC","FE","FIS","FISV","FLT","FMC","F","FTNT","FTV",
  "FOXA","FOX","BEN","FCX","GRMN","IT","GEHC","GEN","GNRC","GD","GE","GIS",
  "GM","GPC","GILD","GL","GPN","HAL","HIG","HAS","HCA","PEAK","HSIC","HES",
  "HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUM","HII","IBM",
  "IEX","IDXX","ITW","ILMN","INCY","IR","PODD","INTC","ICE","IP","IPG","IFF",
  "INTU","ISRG","IVZ","INVH","IQV","IRM","JBHT","JKHY","J","JNJ","JCI","JPM",
  "JNPR","K","KDP","KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX",
  "LH","LRCX","LW","LVS","LDOS","LEN","LNC","LIN","LYV","LKQ","LMT","L",
  "LOW","LYB","MTB","MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH",
  "MKC","MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP","MU","MSFT",
  "MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI",
  "MSCI","NDAQ","NTAP","NFLX","NWL","NEM","NWSA","NWS","NEE","NKE","NI","NDSN",
  "NSC","NTRS","NOC","NLOK","NCLH","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY",
  "ODFL","OMC","ON","OKE","ORCL","OGN","OTIS","PCAR","PKG","PANW","PARA","PH",
  "PAYX","PAYC","PYPL","PNR","PEP","PKI","PFE","PCG","PM","PSX","PNW","PXD",
  "PNC","POOL","PPG","PPL","PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA",
  "PHM","QRVO","PWR","QCOM","RL","RJF","RTX","O","REG","REGN","RF","RSG",
  "RMD","RHI","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC","SLB","STX",
  "SEE","SRE","NOW","SHW","SBNY","SPG","SWKS","SJM","SNA","SEDG","SO","LUV",
  "SWK","SBUX","STT","STE","SYK","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR",
  "TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN","TXT","COO","HAS","TMO",
  "TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN","USB","UDR","ULTA",
  "UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VRSN","VRSK","VZ","VRTX",
  "VFC","VTRS","V","VNO","VMC","WAB","WBA","WMT","WBD","WM","WAT","WEC","WFC",
  "WELL","WST","WDC","WRK","WY","WHR","WMB","WTW","WLTW","GWW","XEL","XYL",
  "YUM","ZBRA","ZBH","ZION","ZTS",
];

function fetchUrl(url: string, timeoutMs = 10000): Promise<string> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("CBOE fetch timeout")), timeoutMs);
    https
      .get(url, (res) => {
        if (res.statusCode !== 200) {
          clearTimeout(timer);
          reject(new Error(`HTTP ${res.statusCode}`));
          return;
        }
        let data = "";
        res.on("data", (chunk) => (data += chunk));
        res.on("end", () => {
          clearTimeout(timer);
          resolve(data);
        });
        res.on("error", (e) => {
          clearTimeout(timer);
          reject(e);
        });
      })
      .on("error", (e) => {
        clearTimeout(timer);
        reject(e);
      });
  });
}

async function fetchCBOEUniverse(): Promise<string[]> {
  // Check local cache first
  try {
    if (fs.existsSync(CBOE_CACHE_PATH)) {
      const stat = fs.statSync(CBOE_CACHE_PATH);
      if (Date.now() - stat.mtimeMs < CBOE_CACHE_TTL_MS) {
        const cached = JSON.parse(fs.readFileSync(CBOE_CACHE_PATH, "utf8"));
        if (Array.isArray(cached) && cached.length > 0) {
          console.log(`[scanner] CBOE universe loaded from cache (${cached.length} tickers)`);
          return cached;
        }
      }
    }
  } catch {
    // ignore cache read errors
  }

  // Try fetching live from CBOE
  try {
    const csv = await fetchUrl(CBOE_CSV_URL, 15000);
    const lines = csv.split("\n");
    const symbols: string[] = [];

    // Parse CSV — find the column that holds the symbol
    // CBOE CSV typically has: "Symbol","Company","Exchange","..."
    let symbolCol = 0;
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;

      if (i === 0) {
        // header row — find symbol column index
        const headers = line.split(",").map((h) => h.replace(/"/g, "").trim().toLowerCase());
        const idx = headers.findIndex((h) => h === "symbol" || h === "ticker");
        symbolCol = idx >= 0 ? idx : 0;
        continue;
      }

      const cols = line.split(",");
      if (cols.length <= symbolCol) continue;
      const sym = cols[symbolCol].replace(/"/g, "").trim().toUpperCase();

      // Filter: 1-6 uppercase letters, no special chars except dots (for classes like BRK.B)
      if (/^[A-Z]{1,6}(\.[A-Z])?$/.test(sym)) {
        symbols.push(sym);
      }
    }

    if (symbols.length > 50) {
      // Persist cache
      try {
        fs.writeFileSync(CBOE_CACHE_PATH, JSON.stringify(symbols));
      } catch {
        // ignore write errors
      }
      console.log(`[scanner] CBOE universe fetched live (${symbols.length} tickers)`);
      return symbols;
    }
  } catch (err) {
    console.warn("[scanner] CBOE fetch failed, using fallback universe:", err);
  }

  // Fallback: deduplicated static list
  const fallback = Array.from(new Set(FALLBACK_UNIVERSE));
  console.log(`[scanner] Using fallback universe (${fallback.length} tickers)`);
  return fallback;
}

// ─────────────────────────────────────────────────────────────────────────────
// Background rolling scanner
// ─────────────────────────────────────────────────────────────────────────────

const SCAN_BATCH_SIZE = 3; // reduced to prevent CPU overload

async function runBackgroundScanner(): Promise<void> {
  console.log("[scanner] Background scanner starting…");

  // Use Tier 1 only — keeps CPU usage low on local machines
  let universe: string[] = Array.from(new Set(TIER1_TICKERS));
  console.log(`[scanner] Universe size: ${universe.length} tickers (Tier 1 only)`);

  // Perpetual loop
  while (true) {
    fullScanProgress.running = true;
    fullScanProgress.total = universe.length;
    fullScanProgress.current = 0;

    for (let i = 0; i < universe.length; i += SCAN_BATCH_SIZE) {
      const batch = universe.slice(i, i + SCAN_BATCH_SIZE);
      try {
        const results = await scanBatch(batch);
        for (const r of results) {
          upsertCache(r);
        }
        fullScanProgress.current = Math.min(i + SCAN_BATCH_SIZE, universe.length);
      } catch (err) {
        console.error("[scanner] Batch error:", err);
      }

      // Yield — longer pause between batches to keep CPU usage low
      await new Promise((resolve) => setTimeout(resolve, 4000));
    }

    fullScanProgress.running = false;
    fullScanProgress.lastFullCycle = Date.now();
    fullScanProgress.current = universe.length;

    console.log(
      `[scanner] Full cycle complete — ${fullUniverseCache.size} tickers cached. Next cycle in 30s.`
    );

    // Short pause between full cycles before starting again
    await new Promise((resolve) => setTimeout(resolve, 5 * 60_000)); // 5 min between full cycles

    // Refresh CBOE universe daily (cache handles this)
    // Cap to TIER1 only to keep memory bounded — deep scanning happens in Python
    try {
      universe = Array.from(new Set(TIER1_TICKERS));
    } catch {
      // keep existing universe
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tier 1 auto-refresh every 5 minutes
// ─────────────────────────────────────────────────────────────────────────────

// Tier1 auto-refresh disabled — saves CPU on local machines
// setInterval(async () => { await refreshTier1(); }, 5 * 60 * 1000);

// ─────────────────────────────────────────────────────────────────────────────
// Route registration
// ─────────────────────────────────────────────────────────────────────────────

export async function registerRoutes(httpServer: Server, app: Express): Promise<Server> {

  // ── Auth & Bot ────────────────────────────────────────────────────────────
  app.use(cookieParser());

  // ── AUTH MIDDLEWARE (2026-04-20 security fix Bug #22) ──────────────────
  // Centralizes the 3-step session lookup previously copy-pasted in 3 places.
  // Returns null if not authenticated, or the user object if OK.
  function _checkSession(req: any): { email: string; id: number } | null {
    const session = req?.cookies?.session;
    if (!session) return null;
    try {
      const sessionRow = db.prepare("SELECT user_id FROM sessions WHERE token = ?").get(session) as any;
      if (!sessionRow) return null;
      const user = db.prepare("SELECT id, email FROM users WHERE id = ?").get(sessionRow.user_id) as any;
      return user || null;
    } catch { return null; }
  }

  // Required-auth wrapper — returns 401 if not authenticated
  function requireAuth(handler: (req: any, res: any) => any) {
    return async (req: any, res: any) => {
      const user = _checkSession(req);
      if (!user) return res.status(401).json({ error: "Authentication required" });
      (req as any).user = user;
      return handler(req, res);
    };
  }

  // ── ALPACA_BASE from env (2026-04-20 fix Bug #23) ──────────────────────
  // Previously several endpoints had hardcoded paper URLs, meaning switching
  // to live trading would show stale paper data on the dashboard forever.
  const ALPACA_BASE_URL = process.env.ALPACA_BASE_URL || "https://paper-api.alpaca.markets";

  registerAuthRoutes(app);
  registerBotRoutes(app);

  // ── Watchlist persistence ─────────────────────────────────────────────────
  try {
    db.prepare("CREATE TABLE IF NOT EXISTS watchlists (user_email TEXT, ticker TEXT, added_at TEXT, PRIMARY KEY (user_email, ticker))").run();
  } catch {}

  app.get("/api/watchlist", (req, res) => {
    const session = (req as any).cookies?.session;
    if (!session) return res.json({ tickers: [] });

    const sessionRow = db.prepare("SELECT user_id FROM sessions WHERE token = ?").get(session) as any;
    if (!sessionRow) return res.json({ tickers: [] });

    const user = db.prepare("SELECT email FROM users WHERE id = ?").get(sessionRow.user_id) as any;
    if (!user) return res.json({ tickers: [] });

    const rows = db.prepare("SELECT ticker FROM watchlists WHERE user_email = ? ORDER BY added_at DESC").all(user.email) as any[];
    res.json({ tickers: rows.map((r: any) => r.ticker) });
  });

  app.post("/api/watchlist/add", (req, res) => {
    const session = (req as any).cookies?.session;
    if (!session) return res.status(401).json({ error: "Not authenticated" });

    const sessionRow = db.prepare("SELECT user_id FROM sessions WHERE token = ?").get(session) as any;
    if (!sessionRow) return res.status(401).json({ error: "Not authenticated" });

    const user = db.prepare("SELECT email FROM users WHERE id = ?").get(sessionRow.user_id) as any;
    if (!user) return res.status(401).json({ error: "Not authenticated" });

    const { ticker } = req.body || {};
    if (!ticker || typeof ticker !== "string") return res.status(400).json({ error: "Ticker required" });
    if (!/^[A-Za-z.]{1,10}$/.test(ticker)) return res.status(400).json({ error: "Invalid ticker format" });

    try {
      db.prepare("INSERT OR IGNORE INTO watchlists (user_email, ticker, added_at) VALUES (?, ?, ?)").run(
        user.email, ticker.toUpperCase(), new Date().toISOString()
      );
    } catch (e) { console.error("[watchlist-add]", e); }

    res.json({ ok: true });
  });

  app.post("/api/watchlist/remove", (req, res) => {
    const session = (req as any).cookies?.session;
    if (!session) return res.status(401).json({ error: "Not authenticated" });

    const sessionRow = db.prepare("SELECT user_id FROM sessions WHERE token = ?").get(session) as any;
    if (!sessionRow) return res.status(401).json({ error: "Not authenticated" });

    const user = db.prepare("SELECT email FROM users WHERE id = ?").get(sessionRow.user_id) as any;
    if (!user) return res.status(401).json({ error: "Not authenticated" });

    const { ticker } = req.body || {};
    if (ticker && typeof ticker === "string") {
      db.prepare("DELETE FROM watchlists WHERE user_email = ? AND ticker = ?").run(user.email, ticker.toUpperCase());
    }

    res.json({ ok: true });
  });

  // ── Single-ticker analysis ────────────────────────────────────────────────
  app.get("/api/analyze/:ticker", async (req, res) => {
    const { ticker } = req.params;

    if (!ticker || !/^[A-Za-z.]{1,10}$/.test(ticker)) {
      return res.status(400).json({ error: "Invalid ticker symbol. Please use letters only (e.g. AAPL, SPY, TSLA)." });
    }

    const scriptPath = path.resolve(process.cwd(), "analyze.py");

    try {
      const { stdout } = await execAsync(
        `python3 "${scriptPath}" "${ticker.toUpperCase()}"`,
        { timeout: 120000, maxBuffer: 1024 * 1024 * 2 }
      );

      const output = stdout.trim();
      if (!output) {
        return res.status(500).json({ error: "No output from analysis engine. Try again." });
      }

      const data = JSON.parse(output);
      return res.json(data);
    } catch (err: any) {
      if (err.stdout) {
        try {
          const data = JSON.parse(err.stdout.trim());
          return res.status(400).json(data);
        } catch {}
      }
      return res.status(500).json({ error: "Analysis failed. Please check the ticker and try again." });
    }
  });

  // ── Single-ticker INSIGHTS view (insider + institutional + float + S/R + options) ──
  // ALPHA AUDIT 2026-05-03 batch 3: smart-money / structure data
  // surfaced from the bot's existing data layer in a single payload.
  // Backend: insights.py. Has its own timeout because it touches more
  // sources (Finnhub, SEC EDGAR, yfinance) than /api/analyze.
  app.get("/api/insights/:ticker", async (req, res) => {
    const { ticker } = req.params;

    if (!ticker || !/^[A-Za-z.]{1,10}$/.test(ticker)) {
      return res.status(400).json({ error: "Invalid ticker symbol. Please use letters only (e.g. AAPL, SPY, TSLA)." });
    }

    const scriptPath = path.resolve(process.cwd(), "insights.py");

    try {
      const { stdout } = await execAsync(
        `python3 "${scriptPath}" "${ticker.toUpperCase()}"`,
        { timeout: 90000, maxBuffer: 1024 * 1024 * 2 }
      );

      const output = stdout.trim();
      if (!output) {
        return res.status(500).json({ error: "No output from insights engine. Try again." });
      }

      const data = JSON.parse(output);
      return res.json(data);
    } catch (err: any) {
      if (err.stdout) {
        try {
          const data = JSON.parse(err.stdout.trim());
          return res.status(400).json(data);
        } catch {}
      }
      return res.status(500).json({ error: "Insights fetch failed. Please check the ticker and try again." });
    }
  });

  // ── AlphaDesk EQUITY RESEARCH ─────────────────────────────────────────────
  // Explainable buy/sell verdict from five weighted pillars (fundamentals,
  // valuation, supply/demand, market context, filings) + an after-tax horizon
  // comparison. Backed by the `alphadesk` Python package (clean-room engine in
  // ./alphadesk). Runs offline on sample data; fills in live vendor numbers when
  // the same Alpaca/Polygon/Finnhub keys used elsewhere are present in env.
  // Optional tax query params (?bracket=0.37&state=0.093&ltcg=0.20&niit=0)
  // tune the after-tax math. Research/education only — never places orders.
  app.get("/api/research/:ticker", async (req, res) => {
    const { ticker } = req.params;

    if (!ticker || !/^[A-Za-z.]{1,10}$/.test(ticker)) {
      return res.status(400).json({ error: "Invalid ticker symbol. Please use letters only (e.g. AAPL, MSFT, NVDA)." });
    }

    // Build the optional tax flags from query params, validating each as a
    // plausible 0–1 rate so nothing untrusted reaches the shell.
    const taxFlags: string[] = [];
    const rate = (v: unknown) => {
      const n = typeof v === "string" ? parseFloat(v) : NaN;
      return Number.isFinite(n) && n >= 0 && n <= 1 ? n : null;
    };
    const bracket = rate(req.query.bracket);
    const state = rate(req.query.state);
    const ltcg = rate(req.query.ltcg);
    if (bracket !== null) taxFlags.push(`--bracket ${bracket}`);
    if (state !== null) taxFlags.push(`--state ${state}`);
    if (ltcg !== null) taxFlags.push(`--ltcg ${ltcg}`);
    if (req.query.niit === "0" || req.query.niit === "false") taxFlags.push("--no-niit");

    // AlphaDesk runs as a module (`python -m alphadesk`) from its own folder.
    const cwd = path.resolve(process.cwd(), "alphadesk");

    try {
      const { stdout } = await execAsync(
        `python3 -m alphadesk "${ticker.toUpperCase()}" --json ${taxFlags.join(" ")}`.trim(),
        { timeout: 120000, maxBuffer: 1024 * 1024 * 2, cwd }
      );

      const output = stdout.trim();
      if (!output) {
        return res.status(500).json({ error: "No output from research engine. Try again." });
      }

      const data = JSON.parse(output);
      return res.json(data);
    } catch (err: any) {
      if (err.stdout) {
        try {
          const data = JSON.parse(err.stdout.trim());
          return res.status(400).json(data);
        } catch {}
      }
      console.error("[research] AlphaDesk error:", err?.message || err);
      return res.status(500).json({ error: "Research failed. Please check the ticker and try again." });
    }
  });

  // ── DATACORE BOUNDARY (/api/data/*) ──────────────────────────────────────
  // The spinout-ready data layer's API boundary (CLAUDE.md: SPINOUT-READY
  // DATA LAYER). All /data map overlay data is served here — the frontend
  // never calls external data sources directly. Layers registry is static
  // metadata from datacore/layers.json; overlay routes land one per slice.
  app.get("/api/data/layers", (_req, res) => {
    const layers = ((datacoreLayers as any).layers || []).map((l: any) =>
      // vessels goes live automatically the moment AISSTREAM_KEY exists
      l.id === "vessels"
        ? { ...l, status: vesselStreamEnabled() ? "live" : "awaiting_key" }
        : l
    );
    res.json({ layers });
  });

  // Live aircraft overlay (RAW) — community ADS-B chain, THREE deep
  // (human directive 2026-07-03; self-hosted receivers declined):
  // adsb.lol primary (ODbL 1.0, the only provider lawful under
  // monetization) -> airplanes.live -> adsb.fi (both non-commercial
  // licenses — fine for the current no-revenue POC, must be dropped or
  // upgraded before billing goes live; see wishlist MONETIZATION
  // TRIPWIRE). All three are global community networks sharing the
  // readsb point+radius JSON shape (adsb.fi differs only in URL pattern
  // and array key). OpenSky removed 2026-07-03 (human decision):
  // Railway egress rejects it even with OAuth creds, and its operational-use
  // clause requires a written agreement — requested by the human; reinstate
  // and re-verify Railway connectivity if granted. One shared upstream
  // request per bbox (in-flight dedup + 30s cache) protects rate limits no
  // matter how many visitors watch. Exponential backoff per provider;
  // stale-over-error; every fresh snapshot feeds the permanent position
  // archive (datacoreArchive). Frontend never calls upstreams.
  const ARCHIVE_SITES = ((datacoreSites as any).sites || []).map((s: any) => ({ lat: s.lat, lon: s.lon }));
  // Monetization tripwire boot check (throttled inside; also ticked per
  // aircraft request) — loud COMPLIANCE-WARNING if billing activates while
  // a non-commercial-licensed provider is still in the chain. Writer wired
  // to the persistent audit_log (same table bot.ts persistAudit uses; the
  // compliance module itself stays db-free for testability).
  setComplianceAuditWriter((type, message) => {
    try {
      db.prepare("INSERT INTO audit_log (time, type, message) VALUES (?, ?, ?)").run(
        new Date().toISOString(), type, message.slice(0, 500)
      );
    } catch {}
  });
  complianceAuditTick();
  const aircraftCache: Map<string, { at: number; data: any }> = new Map();
  const aircraftInflight: Map<string, Promise<any>> = new Map();
  const feedBackoff: Record<string, { failures: number; until: number }> = {};

  const backoffActive = (p: string) => (feedBackoff[p]?.until || 0) > Date.now();
  const backoffBump = (p: string) => {
    const b = (feedBackoff[p] ||= { failures: 0, until: 0 });
    b.failures++;
    b.until = Date.now() + Math.min(15 * 60_000, 30_000 * 2 ** (b.failures - 1));
  };
  const backoffClear = (p: string) => { feedBackoff[p] = { failures: 0, until: 0 }; };

  async function fetchAircraft(lamin: number, lamax: number, lomin: number, lomax: number) {
    const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" };
    let aircraft: any[] = [];
    let source = "";
    let coverage = "full";
    let coverage_note = "";
    const time = Math.floor(Date.now() / 1000);

    // Both providers share the point+radius API shape (hard max 250nm) —
    // wide viewports are served best-effort around view center. Two
    // independent networks so one flaking doesn't kill the layer — the
    // 2026-07-03 prod incident was a single provider's egress flake
    // exponentially backing off = zero aircraft for fresh bboxes.
    const clat = (lamin + lamax) / 2;
    const clon = (lomin + lomax) / 2;
    const latSpanNm = Math.abs(lamax - lamin) * 60;
    const lonSpanNm = Math.abs(lomax - lomin) * 60 * Math.max(0.1, Math.cos((clat * Math.PI) / 180));
    const neededNm = Math.ceil(Math.sqrt(latSpanNm ** 2 + lonSpanNm ** 2) / 2);
    const radiusNm = Math.min(250, Math.max(50, neededNm));
    if (neededNm > 250) {
      coverage = "partial";
      coverage_note = `feed covers ~250nm around view center (viewport needs ~${neededNm}nm) — zoom in for full coverage`;
    }
    const PROVIDERS = [
      { key: "adsblol", url: `https://api.adsb.lol/v2/point/${clat.toFixed(3)}/${clon.toFixed(3)}/${radiusNm}`, label: "adsb.lol (ADS-B, community)", arr: "ac" },
      { key: "airplaneslive", url: `https://api.airplanes.live/v2/point/${clat.toFixed(3)}/${clon.toFixed(3)}/${radiusNm}`, label: "airplanes.live (ADS-B, community)", arr: "ac" },
      { key: "adsbfi", url: `https://opendata.adsb.fi/api/v2/lat/${clat.toFixed(3)}/lon/${clon.toFixed(3)}/dist/${radiusNm}`, label: "adsb.fi (ADS-B, community)", arr: "aircraft" },
    ];
    const errs: string[] = [];
    for (const fb of PROVIDERS) {
      if (source) break;
      if (backoffActive(fb.key)) { errs.push(`${fb.key} in backoff`); continue; }
      try {
        const r2 = await fetch(fb.url, { headers: UA, signal: AbortSignal.timeout(12000) });
        if (!r2.ok) throw new Error(`${fb.key} ${r2.status}`);
        const raw2: any = await r2.json();
        aircraft = (raw2[fb.arr] || []).slice(0, 5000).map((a: any) => ({
          icao24: a.hex,
          callsign: String(a.flight || "").trim(),
          origin_country: a.r || "",
          lon: a.lon,
          lat: a.lat,
          altitude_m: a.alt_baro === "ground" || a.alt_baro == null ? null : Math.round(a.alt_baro * 0.3048),
          on_ground: a.alt_baro === "ground",
          velocity_ms: a.gs == null ? null : Math.round(a.gs * 0.5144),
          heading: a.track ?? null,
          type: a.t || null,                    // ICAO type designator (e.g. B738, C172)
          category: a.category || null,          // ADS-B emitter category (A1..A7)
        })).filter((a: any) => a.lat != null && a.lon != null);
        source = fb.label;
        backoffClear(fb.key);
      } catch (e: any) {
        backoffBump(fb.key);
        // capture the underlying cause — bare "fetch failed" is undiagnosable
        const cause = e?.cause?.code || e?.cause?.message || "";
        errs.push(`${fb.key}: ${e?.message}${cause ? ` (${cause})` : ""}`);
      }
    }
    if (!source) throw new Error(errs.join(" | "));

    // Feed the permanent archive (adaptive thinning inside; fire-and-forget).
    try { archiveAircraft(aircraft, ARCHIVE_SITES); } catch {}

    return { source, kind: "raw", time, coverage, coverage_note: coverage_note || undefined, count: aircraft.length, aircraft };
  }

  app.get("/api/data/aircraft", async (req, res) => {
    complianceAuditTick();
    const num = (v: any, lo: number, hi: number, dflt: number) => {
      const n = parseFloat(String(v));
      return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : dflt;
    };
    // GLOBAL defaults; rounded to 1dp so panning re-hits the shared cache.
    const lamin = Math.round(num(req.query.lamin, -85, 85, -85) * 10) / 10;
    const lamax = Math.round(num(req.query.lamax, -85, 85, 85) * 10) / 10;
    const lomin = Math.round(num(req.query.lomin, -180, 180, -180) * 10) / 10;
    const lomax = Math.round(num(req.query.lomax, -180, 180, 180) * 10) / 10;
    const key = `${lamin},${lamax},${lomin},${lomax}`;
    const hit = aircraftCache.get(key);
    if (hit && Date.now() - hit.at < 30_000) {
      // Delta support: if the client already holds this snapshot, don't
      // re-send the payload.
      if (String(req.query.since || "") === String(hit.data.time)) {
        return res.json({ unchanged: true, time: hit.data.time, count: hit.data.count });
      }
      return res.json({ ...hit.data, cached: true });
    }
    try {
      // In-flight dedup: concurrent visitors on the same bbox share ONE
      // upstream request (rate-limit protection is server-wide, not per-tab).
      let p = aircraftInflight.get(key);
      if (!p) {
        p = fetchAircraft(lamin, lamax, lomin, lomax).finally(() => aircraftInflight.delete(key));
        aircraftInflight.set(key, p);
      }
      const data = await p;
      aircraftCache.set(key, { at: Date.now(), data });
      if (aircraftCache.size > 20) {
        const oldest = Array.from(aircraftCache.entries()).sort((a, b) => a[1].at - b[1].at)[0];
        if (oldest) aircraftCache.delete(oldest[0]);
      }
      if (String(req.query.since || "") === String(data.time)) {
        return res.json({ unchanged: true, time: data.time, count: data.count });
      }
      res.json(data);
    } catch (e: any) {
      // Stale-beats-spinner (DESIGN.md performance budget): serve the last
      // snapshot with its timestamp rather than an empty error.
      if (hit) return res.json({ ...hit.data, cached: true, stale: true, stale_at: hit.at });
      res.status(502).json({ error: `aircraft feed unavailable: ${e?.message || e}`, aircraft: [] });
    }
  });

  // Live vessels (RAW) — aisstream.io AIS websocket, key-gated. Without
  // AISSTREAM_KEY the route reports enabled:false (the layer panel shows
  // "awaiting API key" — signup flagged in research/wishlist.md). With the
  // key, a lazy singleton subscriber keeps an in-memory latest-positions map
  // (US coastal boxes, PositionReports only), pruned for staleness and
  // capped in size. The frontend polls this route only (boundary rule).
  const vesselPositions: Map<string, { lat: number; lon: number; sog: number | null; cog: number | null; name: string; at: number }> = new Map();
  // Static data (ship type, destination) arrives in separate AIS messages —
  // kept in a side map and merged into position reads.
  const vesselStatics: Map<string, { shiptype: number | null; destination: string | null; name: string | null }> = new Map();
  let vesselSocket: WSClient | null = null;
  let vesselSocketUp = 0;

  function ensureVesselStream(): boolean {
    const key = process.env.AISSTREAM_KEY || "";
    if (!key) return false;
    if (vesselSocket && vesselSocket.readyState === 1) return true;   // OPEN
    if (vesselSocket && vesselSocket.readyState === 0) return true;   // CONNECTING
    try { vesselSocket?.terminate(); } catch {}
    try {
      const ws = new WSClient("wss://stream.aisstream.io/v0/stream");
      vesselSocket = ws;
      ws.on("open", () => {
        vesselSocketUp = Date.now();
        ws.send(JSON.stringify({
          APIKey: key,
          // GLOBAL coverage. Honest limit (stated in the layer panel):
          // aisstream aggregates terrestrial receivers, so mid-ocean gaps
          // are physics, not a bug — satellite AIS is a priced product
          // (BUILD-FIRST rule: raw material inaccessible free).
          BoundingBoxes: [[[-90, -180], [90, 180]]],
          FilterMessageTypes: ["PositionReport", "ShipStaticData"],
        }));
      });
      ws.on("message", (buf: any) => {
        try {
          const m = JSON.parse(buf.toString());
          const meta = m.MetaData || {};
          const mmsi = String(meta.MMSI || "");
          if (!mmsi) return;
          if (m.MessageType === "ShipStaticData") {
            const s = m.Message?.ShipStaticData || {};
            vesselStatics.set(mmsi, {
              shiptype: s.Type ?? null,
              destination: (s.Destination || "").trim() || null,
              name: (s.Name || meta.ShipName || "").trim() || null,
            });
            if (vesselStatics.size > 30_000) vesselStatics.clear();
            return;
          }
          if (m.MessageType !== "PositionReport") return;
          const pos = m.Message?.PositionReport || {};
          const lat = pos.Latitude ?? meta.latitude;
          const lon = pos.Longitude ?? meta.longitude;
          if (lat == null || lon == null) return;
          vesselPositions.set(mmsi, {
            lat, lon,
            sog: pos.Sog ?? null, cog: pos.Cog ?? null,
            name: String(meta.ShipName || "").trim() || mmsi,
            at: Date.now(),
          });
          // bound memory: prune stale (>20min) when large, then cap hard
          if (vesselPositions.size > 20_000) {
            const cutoff = Date.now() - 20 * 60_000;
            vesselPositions.forEach((v, k) => {
              if (v.at < cutoff) vesselPositions.delete(k);
            });
            while (vesselPositions.size > 20_000) {
              const first = vesselPositions.keys().next().value;
              if (first === undefined) break;
              vesselPositions.delete(first);
            }
          }
        } catch {}
      });
      ws.on("error", (e: any) => console.error("[datacore] aisstream:", e?.message || e));
      ws.on("close", () => { if (vesselSocket === ws) vesselSocket = null; });
      return true;
    } catch (e: any) {
      console.error("[datacore] aisstream connect:", e?.message || e);
      vesselSocket = null;
      return false;
    }
  }

  // KNOWN BROKEN #9 fix: connect at boot instead of waiting for the first
  // /api/data/vessels request, so every deploy doesn't leave the vessels
  // layer (and its archive recording) cold until someone opens the map.
  bootVesselStream(process.env, ensureVesselStream);

  app.get("/api/data/vessels", (req, res) => {
    if (!vesselStreamEnabled()) {
      return res.json({
        enabled: false,
        reason: "AISSTREAM_KEY not set — free signup at aisstream.io (see research/wishlist.md)",
        vessels: [],
      });
    }
    ensureVesselStream();
    const num = (v: any, dflt: number) => {
      const n = parseFloat(String(v));
      return Number.isFinite(n) ? n : dflt;
    };
    const lamin = num(req.query.lamin, -85), lamax = num(req.query.lamax, 85);
    const lomin = num(req.query.lomin, -180), lomax = num(req.query.lomax, 180);
    const cutoff = Date.now() - 20 * 60_000;
    const vessels: any[] = [];
    vesselPositions.forEach((v, mmsi) => {
      if (vessels.length >= 5000) return;
      if (v.at < cutoff) return;
      if (v.lat < lamin || v.lat > lamax || v.lon < lomin || v.lon > lomax) return;
      const st = vesselStatics.get(mmsi);
      vessels.push({
        mmsi, name: st?.name || v.name, lat: v.lat, lon: v.lon,
        sog: v.sog, cog: v.cog,
        shiptype: st?.shiptype ?? null,
        destination: st?.destination ?? null,
      });
    });
    res.json({
      enabled: true,
      source: "aisstream.io (AIS, terrestrial receivers — mid-ocean coverage gaps are inherent)",
      kind: "raw",
      warming_up: vessels.length === 0 && Date.now() - vesselSocketUp < 30_000,
      count: vessels.length,
      vessels,
    });
  });

  // ── PERMANENT POSITION ARCHIVE (ARCHIVE EVERYTHING directive) ────────────
  // Vessels: snapshot the in-memory position map into the archive every 60s
  // (adaptive thinning happens inside archiveVessels). Aircraft archive on
  // every fresh upstream fetch (see fetchAircraft). Maintenance: gzip old
  // hours every 30min; roll raw days older than the retention window into
  // per-entity track summaries every 6h. All on the Railway volume.
  setInterval(() => {
    try {
      if (vesselPositions.size === 0) return;
      const pts: any[] = [];
      const cutoff = Date.now() - 5 * 60_000;
      vesselPositions.forEach((v, mmsi) => {
        if (v.at < cutoff) return;
        const st = vesselStatics.get(mmsi);
        pts.push({ mmsi, name: st?.name || v.name, lat: v.lat, lon: v.lon,
                   sog: v.sog, cog: v.cog, shiptype: st?.shiptype ?? null,
                   destination: st?.destination ?? null });
      });
      archiveVessels(pts, ARCHIVE_SITES);
    } catch (e: any) { console.error("[archive] vessel tick:", e?.message || e); }
  }, 60_000).unref?.();
  setInterval(() => { try { compressOldHours(); } catch {} }, 30 * 60_000).unref?.();
  setInterval(() => { try { rollupOldDays(); } catch {} }, 6 * 3600_000).unref?.();

  // Recent trail for one entity (serves the client's track-on-click).
  app.get("/api/data/track/:kind/:id", (req, res) => {
    const kind = req.params.kind === "vessels" ? "vessels"
               : req.params.kind === "trains" ? "trains" : "aircraft";
    const id = String(req.params.id || "").slice(0, 24);
    if (!id) return res.status(400).json({ error: "id required" });
    try {
      const points = recentTrack(kind, id);
      res.json({ kind, id, points, count: points.length,
                 note: points.length === 0 ? "no archived positions yet for this id (archive began 2026-07-03)" : undefined });
    } catch (e: any) {
      res.status(500).json({ error: e?.message || "track read failed" });
    }
  });

  // Archive growth observability (volume watch — see wishlist).
  app.get("/api/data/archive/stats", (_req, res) => {
    try { res.json(archiveStats()); } catch (e: any) { res.status(500).json({ error: e?.message }); }
  });

  // Strategic sites (RAW) — static reference data from datacore/sites.
  app.get("/api/data/sites", (_req, res) => {
    const d = datacoreSites as any;
    res.json({ kind: "raw", categories: d.categories || {}, sites: d.sites || [] });
  });

  // US power plants (RAW) — static reference data compiled from the WRI
  // Global Power Plant Database (CC BY 4.0) by scripts/build_powerplants.py.
  // Whole-file response, day-cached: ~760KB raw / ~200KB gzipped, one fetch
  // per visitor-day; clustering/decluttering is client-side (DESIGN.md:
  // heavy geo work never on the Railway box).
  app.get("/api/data/powerplants", (_req, res) => {
    res.set("Cache-Control", "public, max-age=86400");
    res.json({ kind: "raw", ...(datacorePowerplants as any) });
  });

  // Live trains overlay (RAW) — Finland Digitraffic (CC BY 4.0) + Norway
  // Entur (NLOD); mapping in server/trainsFeed.ts (pure, unit-tested).
  // Shared 30s cache + in-flight dedup + per-source backoff; the response
  // carries per-source status so the panel labels coverage HONESTLY
  // (launch coverage is FI+NO only). US freight rail positions are
  // proprietary — no free source exists (open_questions; do not chase).
  // Every fresh snapshot feeds the permanent position archive.
  let trainsCache: { at: number; data: any } | null = null;
  let trainsInflight: Promise<any> | null = null;
  async function fetchTrains() {
    const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" };
    const sources: any[] = [];
    const trains: any[] = [];
    await Promise.all([
      (async () => {
        if (backoffActive("digitraffic")) { sources.push({ key: "digitraffic", country: "FI", status: "backoff", count: 0 }); return; }
        try {
          const r = await fetch("https://rata.digitraffic.fi/api/v1/train-locations/latest",
            { headers: { ...UA, "Digitraffic-User": "voltradeai-datacore" }, signal: AbortSignal.timeout(12000) });
          if (!r.ok) throw new Error(`digitraffic ${r.status}`);
          const mapped = mapDigitraffic(await r.json());
          trains.push(...mapped);
          backoffClear("digitraffic");
          sources.push({ key: "digitraffic", country: "FI", status: "ok", count: mapped.length });
        } catch (e: any) {
          backoffBump("digitraffic");
          sources.push({ key: "digitraffic", country: "FI", status: "error", count: 0, error: e?.message });
        }
      })(),
      (async () => {
        if (backoffActive("entur")) { sources.push({ key: "entur", country: "NO", status: "backoff", count: 0 }); return; }
        try {
          const r = await fetch("https://api.entur.io/realtime/v1/vehicles/graphql", {
            method: "POST",
            headers: { ...UA, "Content-Type": "application/json", "ET-Client-Name": "voltradeai-datacore" },
            body: JSON.stringify({ query: ENTUR_VEHICLES_QUERY }),
            signal: AbortSignal.timeout(12000),
          });
          if (!r.ok) throw new Error(`entur ${r.status}`);
          const mapped = mapEntur(await r.json());
          trains.push(...mapped);
          backoffClear("entur");
          sources.push({ key: "entur", country: "NO", status: "ok", count: mapped.length });
        } catch (e: any) {
          backoffBump("entur");
          sources.push({ key: "entur", country: "NO", status: "error", count: 0, error: e?.message });
        }
      })(),
    ]);
    try { archiveTrains(trains); } catch {}
    return {
      source: "Digitraffic Finland (CC BY 4.0) + Entur Norway (NLOD)",
      kind: "raw",
      time: Math.floor(Date.now() / 1000),
      coverage: "FI + NO (launch); US freight positions are proprietary — no free source",
      sources,
      count: trains.length,
      trains,
    };
  }
  app.get("/api/data/trains", async (_req, res) => {
    if (trainsCache && Date.now() - trainsCache.at < 30_000) return res.json(trainsCache.data);
    if (!trainsInflight) trainsInflight = fetchTrains().finally(() => { trainsInflight = null; });
    try {
      const data = await trainsInflight;
      trainsCache = { at: Date.now(), data };
      res.json(data);
    } catch (e: any) {
      // fetchTrains never throws by design (per-source status instead);
      // this is a last-resort stale-over-error path.
      if (trainsCache) return res.json({ ...trainsCache.data, stale: true });
      res.status(502).json({ error: e?.message || "trains fetch failed" });
    }
  });

  // SEC EDGAR Form 4 (insider transactions) — RAW as-filed display (EDGE
  // DOCTRINE #1 "build data, don't buy it"; ROOT VALIDATION LADDER gate 1
  // passed, see server/edgarForm4.test.ts). No API key needed. Polls the
  // public "getcurrent" feed on a background timer started at boot (same
  // eager-boot pattern as vessels, KNOWN BROKEN #9's lesson: don't wait for
  // the first request to start collecting). This is a display of what was
  // filed, not an interpreted claim — the "does clustering predict returns"
  // question is gate 2, unattempted, tracked in research/open_questions.md.
  bootForm4Poll();
  app.get("/api/data/insider", (_req, res) => {
    const hit = latestForm4Filings();
    if (!hit) {
      return res.json({ kind: "raw", source: "SEC EDGAR (Form 4)", warming_up: true, count: 0, filings: [] });
    }
    res.json({
      kind: "raw",
      source: "SEC EDGAR (Form 4) — sec.gov/cgi-bin/browse-edgar",
      time: hit.at,
      count: hit.filings.length,
      filings: hit.filings,
    });
  });

  // Accumulated Form 4 history from the filings archive (COLLECT-EVERYTHING:
  // the poll loop appends every filing to disk; this serves the /data/filings
  // full view). Merges the live cache on top so the newest poll shows even
  // before its day file is re-read.
  app.get("/api/data/insider/history", (req, res) => {
    const days = Math.min(90, Math.max(1, parseInt(String(req.query.days || "30"), 10) || 30));
    try {
      const archived = readFilingHistory(days);
      const live = latestForm4Filings()?.filings || [];
      const seen = new Set(archived.map((f: any) => f.accession));
      const merged = [...live.filter((f: any) => !seen.has(f.accession)), ...archived];
      res.json({
        kind: "raw",
        source: "SEC EDGAR (Form 4) — accumulated archive (began 2026-07-04)",
        days,
        count: merged.length,
        filings: merged,
      });
    } catch (e: any) {
      res.status(500).json({ error: e?.message || "history read failed" });
    }
  });

  // Dark-ship RAW statistics — derived from OUR OWN AIS archive (shadow-fleet
  // directive 2026-07-04). Counts only: per-vessel claims are SIGNAL-class
  // and ladder-gated (validation plan in open_questions). 10-min cache — the
  // computation reads up to 72h of archive JSONL.
  let shadowCache: { at: number; data: any } | null = null;
  app.get("/api/data/shadowstats", (_req, res) => {
    if (shadowCache && Date.now() - shadowCache.at < 10 * 60_000) return res.json(shadowCache.data);
    try {
      const zones = (shadowZones as any).zones || [];
      const data = {
        kind: "raw",
        source: "Derived from our own AIS position archive (terrestrial coverage; began 2026-07-03)",
        zones: zones.map((z: any) => ({ id: z.id, name: z.name })),
        ...computeShadowStats(zones),
      };
      shadowCache = { at: Date.now(), data };
      res.json(data);
    } catch (e: any) {
      res.status(500).json({ error: e?.message || "shadowstats failed" });
    }
  });

  // Port dwell analytics (RAW) — arrival/departure detection + dwell
  // distributions per imagery-verified port geofence, from OUR OWN AIS
  // archive (fusion directive 2026-07-04). Anomaly FLAGS are 3x-median,
  // suppressed on thin history; the dwell-anomaly SIGNAL stays ladder-gated.
  // 10-min cache — the computation reads up to 7d of archive JSONL.
  let dwellCache: { at: number; data: any } | null = null;
  app.get("/api/data/portdwell", (_req, res) => {
    if (dwellCache && Date.now() - dwellCache.at < 10 * 60_000) return res.json(dwellCache.data);
    try {
      const ports = portsFromSites((datacoreSites as any).sites || []);
      const data = {
        kind: "raw",
        source: "Derived from our own AIS position archive (terrestrial coverage; began 2026-07-03)",
        ...computePortDwell(ports),
      };
      dwellCache = { at: Date.now(), data };
      res.json(data);
    } catch (e: any) {
      res.status(500).json({ error: e?.message || "portdwell failed" });
    }
  });

  // ── TAX ESTIMATOR ─────────────────────────────────────────────────────────
  // Account-aware capital-gains + income tax estimate. Takes a profile (filing
  // status, state rate, W-2/1099 income) + a list of realized trades and returns
  // a full breakdown (ST/LT, NIIT, state, wash sales, sheltered gains). Backed by
  // alphadesk's pure `tax_engine`. Payload is base64'd so the trade list passes
  // safely through argv. Estimate/education only — not tax advice.
  app.post("/api/tax/estimate", async (req, res) => {
    const body = req.body && typeof req.body === "object" ? req.body : {};
    const trades = Array.isArray(body.trades) ? body.trades : [];
    if (trades.length > 2000) {
      return res.status(400).json({ error: "Too many trades (max 2000)." });
    }
    const payload = Buffer.from(JSON.stringify(body)).toString("base64");
    const cwd = path.resolve(process.cwd(), "alphadesk");
    try {
      const { stdout } = await execAsync(
        `python3 -m alphadesk taxes --payload "${payload}"`,
        { timeout: 30000, maxBuffer: 1024 * 1024 * 2, cwd }
      );
      const out = stdout.trim();
      if (!out) return res.status(500).json({ error: "No output from tax engine." });
      return res.json(JSON.parse(out));
    } catch (err: any) {
      console.error("[tax] estimate error:", err?.message || err);
      return res.status(500).json({ error: "Tax estimate failed." });
    }
  });

  // ── PRE-TRADE PLANNER ─────────────────────────────────────────────────────
  // Given a ticker + account value + risk-per-trade, fetches the live price and
  // realized vol and returns a sized trade plan: volatility-based stop, share
  // count, position value, and R-multiple targets. Backed by alphadesk's pure
  // `planner`. Plans only — it never places an order. Education, not advice.
  app.post("/api/plan", async (req, res) => {
    const body = req.body && typeof req.body === "object" ? req.body : {};
    if (!body.ticker || !/^[A-Za-z.]{1,10}$/.test(String(body.ticker))) {
      return res.status(400).json({ error: "Invalid ticker symbol." });
    }
    const payload = Buffer.from(JSON.stringify(body)).toString("base64");
    const cwd = path.resolve(process.cwd(), "alphadesk");
    try {
      const { stdout } = await execAsync(
        `python3 -m alphadesk plan --payload "${payload}"`,
        { timeout: 120000, maxBuffer: 1024 * 1024 * 2, cwd }
      );
      const out = stdout.trim();
      if (!out) return res.status(500).json({ error: "No output from planner." });
      return res.json(JSON.parse(out));
    } catch (err: any) {
      console.error("[plan] error:", err?.message || err);
      return res.status(500).json({ error: "Trade plan failed." });
    }
  });

  // ── ETF BUILDER / ANALYZER ────────────────────────────────────────────────
  // Pulls ETF metadata (expense ratio, holdings, sector breakdown, active vs
  // passive heuristic, typical rebalance schedule) plus rich per-holding data
  // for the ETF Builder view. Backed by etf_analyzer.py which threads the
  // per-holding yfinance calls. Heavier than /api/analyze — give it 3min.
  app.get("/api/etf/:ticker", async (req, res) => {
    const { ticker } = req.params;

    if (!ticker || !/^[A-Za-z.]{1,10}$/.test(ticker)) {
      return res.status(400).json({ error: "Invalid ticker symbol. Use letters only (e.g. SPY, QQQ, VTI)." });
    }

    const scriptPath = path.resolve(process.cwd(), "etf_analyzer.py");

    try {
      const { stdout } = await execAsync(
        `python3 "${scriptPath}" "${ticker.toUpperCase()}"`,
        { timeout: 180000, maxBuffer: 1024 * 1024 * 4 }
      );

      const output = stdout.trim();
      if (!output) {
        return res.status(500).json({ error: "No output from ETF analyzer. Try again." });
      }

      const data = JSON.parse(output);
      return res.json(data);
    } catch (err: any) {
      if (err.stdout) {
        try {
          const data = JSON.parse(err.stdout.trim());
          return res.status(400).json(data);
        } catch {}
      }
      return res.status(500).json({ error: "ETF analysis failed. Please check the ticker (must be an ETF, not a stock) and try again." });
    }
  });

  // ── DEBUG: raw Finnhub insider response ──────────────────────────────────
  // ALPHA AUDIT 2026-05-03 batch 5: diagnostic endpoint to figure out why
  // some tickers (e.g. NVDA) show empty insider data despite Finnhub
  // having Form 4 filings. Returns BOTH the raw API response and the
  // parsed result our production pipeline produces, so we can see whether
  // the issue is in the API (premium-gated, empty response) or in our
  // parser. Gated by DEBUG_ENABLED env var — should be 'true' on Railway
  // only while diagnosing, then unset.
  app.get("/api/debug/finnhub-insider/:ticker", async (req, res) => {
    if (process.env.DEBUG_ENABLED !== "true") {
      return res.status(403).json({
        error: "Debug endpoint disabled. Set DEBUG_ENABLED=true to enable.",
      });
    }

    const { ticker } = req.params;
    if (!ticker || !/^[A-Za-z.]{1,10}$/.test(ticker)) {
      return res.status(400).json({ error: "Invalid ticker symbol." });
    }

    const scriptPath = path.resolve(process.cwd(), "finnhub_data.py");

    try {
      const { stdout } = await execAsync(
        `python3 "${scriptPath}" "${ticker.toUpperCase()}" --raw insider`,
        { timeout: 30000, maxBuffer: 1024 * 1024 }
      );
      const output = stdout.trim();
      if (!output) {
        return res.status(500).json({ error: "No output from finnhub_data.py" });
      }
      const data = JSON.parse(output);
      return res.json(data);
    } catch (err: any) {
      if (err.stdout) {
        try {
          return res.status(400).json(JSON.parse(err.stdout.trim()));
        } catch {}
      }
      return res.status(500).json({
        error: "Debug fetch failed.",
        detail: String(err?.message || err).slice(0, 200),
      });
    }
  });

  // ── DEBUG: comprehensive ML pipeline diagnostics ──────────────────────────
  // ALPHA AUDIT 2026-05-04 batch 8: one-shot inspector for the full ML
  // process state. Returns environment (RSS, env vars, container memory),
  // filesystem state (model/feedback files), library versions, last
  // retrain status, and a static smoke test that walks through what a
  // retrain would do without actually retraining.
  //
  // Use case: ML retrain has been failing for 90+ hours with various
  // errors (NoneType, SIGKILL). Hitting this endpoint once gives us
  // every piece of state we need to diagnose without round-trips.
  //
  // Gated by DEBUG_ENABLED=true env var. Runs in a child process with
  // a 90-second timeout — the smoke test includes a tiny lightgbm fit
  // and a small bars fetch which can take 5-30s combined.
  app.get("/api/debug/ml-diagnostics", async (_req, res) => {
    if (process.env.DEBUG_ENABLED !== "true") {
      return res.status(403).json({
        error: "Debug endpoint disabled. Set DEBUG_ENABLED=true to enable.",
      });
    }

    const scriptPath = path.resolve(process.cwd(), "ml_diagnostics.py");

    try {
      const { stdout } = await execAsync(
        `python3 "${scriptPath}"`,
        { timeout: 90000, maxBuffer: 1024 * 1024 * 4 }
      );
      const output = stdout.trim();
      if (!output) {
        return res.status(500).json({ error: "No output from ml_diagnostics.py" });
      }
      const data = JSON.parse(output);
      return res.json(data);
    } catch (err: any) {
      // ml_diagnostics.py is designed to never raise — but if execAsync
      // itself fails (timeout, signal, etc.), surface the details so we
      // know what killed it. Mirror the fingerprint pattern from the
      // TIER3-ML-ERROR classifier.
      const _stderr = String(err?.stderr || "").slice(-400);
      const _stdout = String(err?.stdout || "").slice(-400);
      const _signal = err?.signal || "none";
      const _code = err?.code === undefined ? "?" : err.code;
      return res.status(500).json({
        error: "Diagnostics script failed at the process level.",
        process_state: {
          exit_code: _code,
          kill_signal: _signal,
          stderr_tail: _stderr,
          stdout_tail: _stdout,
        },
        hint:
          _signal === "SIGKILL"
            ? "Process was SIGKILL'd. With 8GB available this is likely the exec timeout (90s) firing — the diagnostics smoke test (lightgbm fit + bars fetch) ran longer than expected."
            : "See exit_code and stderr_tail for the cause.",
      });
    }
  });

  // ── System self-test (batch 11) ─────────────────────────────────────────────
  // ALPHA AUDIT 2026-05-06 batch 11: when you push new code, this endpoint
  // tells you whether each subsystem is wired up correctly. Verifies:
  //   - size-tier module produces expected values across equity tiers
  //   - get_adaptive_params overlays size-tier on top of regime correctly
  //   - size_portfolio uses tier max_positions (not hardcoded 8)
  //   - check_sector_correlation accepts the override kwarg
  //   - bot_engine top_10 includes the new factor breakdown / rank fields
  //   - tie-breaking refinement block is present in source
  //   - scan timings cache is reachable
  //
  // Returns 'overall: ok' if everything passes, 'degraded' if some warns,
  // 'broken' if any fails. Hit this after a deploy to confirm the new code
  // is actually running (vs the old version still cached).
  //
  // Unlike ml-diagnostics, this endpoint is NOT gated by DEBUG_ENABLED
  // because it exposes no secrets — just describes which code is running.
  app.get("/api/bot/system-status", async (_req, res) => {
    const scriptPath = path.resolve(process.cwd(), "system_status.py");
    try {
      const { stdout } = await execAsync(
        `python3 "${scriptPath}"`,
        { timeout: 15000, maxBuffer: 1024 * 1024 * 2 }
      );
      const output = stdout.trim();
      if (!output) {
        return res.status(500).json({ error: "No output from system_status.py" });
      }
      return res.json(JSON.parse(output));
    } catch (err: any) {
      return res.status(500).json({
        error: "system-status script failed",
        detail: String(err?.message || err).slice(0, 200),
        stderr_tail: String(err?.stderr || "").slice(-300),
      });
    }
  });

  // ── Market scanner ────────────────────────────────────────────────────────
  app.get("/api/scan", async (req, res) => {
    const now = Date.now();
    const tier1Age = now - tier1LastUpdate;
    const tier1Stale = tier1LastUpdate === 0 || tier1Age > 5 * 60 * 1000;

    // If tier1 cache is stale and we have nothing yet, do a synchronous seed
    if (tier1Cache.length === 0 || tier1Stale) {
      // Fire off refresh in background — don't await so the endpoint stays fast
      // If nothing is cached yet, we do a quick seed of first 20 tickers synchronously
      if (tier1Cache.length === 0) {
        try {
          const seedBatch = TIER1_TICKERS.slice(0, 20);
          const seedResults = await scanBatch(seedBatch);
          if (tier1Cache.length === 0) {
            tier1Cache = sortByScore(applyFreshness(seedResults));
            tier1LastUpdate = Date.now();
            for (const r of seedResults) upsertCache(r);
          }
        } catch {
          // Seed failed — return empty with progress
        }
      } else {
        // Refresh in background
        refreshTier1().catch(console.error);
      }
    }

    const fullResults = sortByScore(
      applyFreshness(Array.from(fullUniverseCache.values()))
    );

    const cached = !tier1Stale;
    const ageSeconds = tier1LastUpdate > 0 ? Math.round(tier1Age / 1000) : 0;

    return res.json({
      results: applyFreshness(tier1Cache),
      full_results: fullResults,
      cached,
      age_seconds: ageSeconds,
      progress: { ...fullScanProgress },
    });
  });

  // ── Scan progress ─────────────────────────────────────────────────────────
  app.get("/api/scan/progress", (_req, res) => {
    const total = fullScanProgress.total || 1;
    const pct = Math.round((fullScanProgress.current / total) * 100);
    return res.json({
      progress: pct,
      cached_count: fullUniverseCache.size,
    });
  });

  // ── Market snapshot (Polygon grouped daily) ───────────────────────────────
  app.get("/api/market-snapshot", async (req, res) => {
    const POLYGON_KEY = process.env.POLYGON_API_KEY || "";
    try {
      const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];
      const url = `https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/${yesterday}?adjusted=true&apiKey=${POLYGON_KEY}`;
      const response = await fetch(url);
      const data: any = await response.json();

      if (!data.results) return res.json({ results: [] });

      const results = data.results
        .filter((r: any) => r.v > 50000 && r.c > 1 && r.T && !r.T.includes('.'))
        .map((r: any) => ({
          ticker: r.T,
          close: r.c,
          open: r.o,
          high: r.h,
          low: r.l,
          volume: r.v,
          change_pct: Number(((r.c - r.o) / r.o * 100).toFixed(2)),
          vwap: r.vw,
        }))
        .sort((a: any, b: any) => b.volume - a.volume)
        .slice(0, 500); // Cap to top 500 by volume to limit response size
      res.json({ results, date: yesterday, total: results.length });
    } catch (err) {
      console.error("[market-snapshot] Error:", err);
      res.status(500).json({ error: "Market snapshot failed" });
    }
  });

  // ── Polygon news ─────────────────────────────────────────────────────────────
  app.get("/api/news", async (req, res) => {
    const POLYGON_KEY = process.env.POLYGON_API_KEY || "";
    const ticker = req.query.ticker as string || "";
    // Bug 32: validate ticker param
    if (ticker && !/^[A-Za-z.]{1,10}$/.test(ticker)) return res.status(400).json({ error: "Invalid ticker" });
    try {
      const url = ticker
        ? `https://api.polygon.io/v2/reference/news?ticker=${encodeURIComponent(ticker)}&limit=20&apiKey=${POLYGON_KEY}`
        : `https://api.polygon.io/v2/reference/news?limit=20&apiKey=${POLYGON_KEY}`;
      const response = await fetch(url);
      const data = await response.json();
      res.json(data);
    } catch (err) {
      console.error("[news] Error:", err);
      res.status(500).json({ error: "News fetch failed" });
    }
  });

  // ── Alpaca Data Proxy (keeps API keys server-side) ──────────────────────────
  const ALPACA_KEY = process.env.ALPACA_KEY || "";
  const ALPACA_SECRET = process.env.ALPACA_SECRET || "";
  const alpacaHeaders = { "APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET };

  // Market scanner data
  app.get("/api/market/scanner", async (_req, res) => {
    try {
      const [activeRes, moversRes] = await Promise.all([
        fetch("https://data.alpaca.markets/v1beta1/screener/stocks/most-actives?by=volume&top=100", { headers: alpacaHeaders }),
        fetch("https://data.alpaca.markets/v1beta1/screener/stocks/movers?top=50", { headers: alpacaHeaders }),
      ]);
      const active = await activeRes.json();
      const movers = await moversRes.json();

      const tickerSet = new Set<string>();
      (active.most_actives || []).forEach((s: any) => tickerSet.add(s.symbol));
      (movers.gainers || []).forEach((s: any) => tickerSet.add(s.symbol));
      (movers.losers || []).forEach((s: any) => tickerSet.add(s.symbol));
      ["AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","AMD","NFLX","SPY","QQQ","DIS","BA","JPM","GS","V","MA","COIN","PLTR","SOFI"].forEach(t => tickerSet.add(t));

      const allTickers = Array.from(tickerSet).slice(0, 150);
      const stocks: any[] = [];

      for (let i = 0; i < allTickers.length; i += 50) {
        const batch = allTickers.slice(i, i + 50).join(",");
        try {
          const snapRes = await fetch(`https://data.alpaca.markets/v2/stocks/snapshots?symbols=${batch}&feed=sip`, { headers: alpacaHeaders });
          const snapData = await snapRes.json();
          for (const [ticker, snap] of Object.entries(snapData) as any) {
            const bar = snap.dailyBar || {};
            const prev = snap.prevDailyBar || {};
            const c = bar.c || 0;
            const pc = prev.c || c;
            const change = pc > 0 ? ((c - pc) / pc) * 100 : 0;
            if (c > 1 && bar.v > 50000) {
              stocks.push({ ticker, close: c, open: bar.o || c, high: bar.h || c, low: bar.l || c, volume: bar.v || 0, vwap: bar.vw || c, change_pct: Math.round(change * 100) / 100 });
            }
          }
        } catch {}
      }
      stocks.sort((a: any, b: any) => b.volume - a.volume);
      res.json({ results: stocks, date: new Date().toISOString().split("T")[0] });
    } catch (e: any) {
      res.status(500).json({ error: e.message });
    }
  });

  // Sector heatmap data
  app.get("/api/market/sectors", async (_req, res) => {
    try {
      const etfs = "XLK,XLF,XLE,XLV,XLI,XLC,XLY,XLP,XLU,XLRE,XLB";
      const snapRes = await fetch(`https://data.alpaca.markets/v2/stocks/snapshots?symbols=${etfs}&feed=sip`, { headers: alpacaHeaders });
      res.json(await snapRes.json());
    } catch (e: any) {
      res.status(500).json({ error: e.message });
    }
  });

  // Stock snapshots (for watchlist prices)
  app.get("/api/market/snapshots", async (req, res) => {
    try {
      const symbols = req.query.symbols as string || "";
      if (!symbols) return res.json({});
      // Bug 31: validate symbols — comma-separated tickers only
      if (!/^[A-Za-z.,]{1,500}$/.test(symbols)) return res.status(400).json({ error: "Invalid symbols" });
      const snapRes = await fetch(`https://data.alpaca.markets/v2/stocks/snapshots?symbols=${encodeURIComponent(symbols)}&feed=sip`, { headers: alpacaHeaders });
      res.json(await snapRes.json());
    } catch (e: any) {
      res.status(500).json({ error: e.message });
    }
  });

  // News proxy
  app.get("/api/market/news", async (req, res) => {
    try {
      const ticker = req.query.ticker as string || "";
      const url = ticker
        ? `https://data.alpaca.markets/v1beta1/news?limit=20&sort=desc&symbols=${ticker}`
        : `https://data.alpaca.markets/v1beta1/news?limit=20&sort=desc`;
      const newsRes = await fetch(url, { headers: alpacaHeaders });
      const json = await newsRes.json();
      // Transform to standard format
      const results = (json.news || []).map((n: any) => ({
        id: n.id || "",
        title: n.headline || "",
        description: n.summary || "",
        published_utc: n.created_at || "",
        article_url: n.url || "",
        tickers: n.symbols || [],
        keywords: [],
        publisher: { name: n.source || "Unknown", favicon_url: "" },
      }));
      res.json({ results });
    } catch (e: any) {
      res.status(500).json({ error: e.message, results: [] });
    }
  });

  // ── Trading Activity Dashboard (landing page) ───────────────────────────

  /** Compute the current trading-day start: 4:00 AM ET today, or yesterday if before 4 AM ET */
  function getTradingDayStart(): string {
    const nowET = new Date(new Date().toLocaleString("en-US", { timeZone: "America/New_York" }));
    // If before 4 AM ET, use yesterday's 4 AM
    if (nowET.getHours() < 4) {
      nowET.setDate(nowET.getDate() - 1);
    }
    nowET.setHours(4, 0, 0, 0);
    // Convert back to UTC: build an ISO string in ET then let the offset handle it
    // We need the UTC equivalent of this ET time
    const etYear = nowET.getFullYear();
    const etMonth = String(nowET.getMonth() + 1).padStart(2, "0");
    const etDay = String(nowET.getDate()).padStart(2, "0");
    // Determine ET offset (EDT = -4, EST = -5)
    const jan = new Date(etYear, 0, 1);
    const jul = new Date(etYear, 6, 1);
    const stdOffset = Math.max(jan.getTimezoneOffset(), jul.getTimezoneOffset());
    const isDST = nowET.getTimezoneOffset() < stdOffset;
    // For server-side: compute UTC hour directly
    // 4 AM ET = 4 + offset hours UTC (EDT=+4, EST=+5)
    const utcHour = isDST ? 8 : 9; // 4AM EDT = 08:00 UTC, 4AM EST = 09:00 UTC
    return `${etYear}-${etMonth}-${etDay}T${String(utcHour).padStart(2, "0")}:00:00Z`;
  }

  // Today's filled orders with P/L enrichment
  app.get("/api/trades/today", async (req, res) => {
    // AUTH (2026-04-20 fix Bug #22)
    const _user = _checkSession(req);
    if (!_user) return res.status(401).json({ error: "Authentication required" });
    try {
      const tradingDayStart = getTradingDayStart();

      // Fetch filled orders and current positions in parallel
      const [ordersResponse, positionsResponse] = await Promise.all([
        fetch(
          `${ALPACA_BASE_URL}/v2/orders?status=closed&after=${encodeURIComponent(tradingDayStart)}&limit=200&direction=desc`,
          { headers: alpacaHeaders }
        ),
        fetch(
          `${ALPACA_BASE_URL}/v2/positions`,
          { headers: alpacaHeaders }
        ),
      ]);

      if (!ordersResponse.ok) {
        const errText = await ordersResponse.text();
        return res.status(ordersResponse.status).json({ error: errText, trades: [] });
      }

      const orders: any[] = await ordersResponse.json();
      const filled = orders.filter((o: any) => o.status === "filled");

      // Build positions lookup: symbol -> { avg_entry_price, current_price, qty }
      const posMap: Record<string, { avgEntry: number; currentPrice: number; qty: number }> = {};
      if (positionsResponse.ok) {
        const positions: any[] = await positionsResponse.json();
        for (const p of positions) {
          posMap[p.symbol] = {
            avgEntry: parseFloat(p.avg_entry_price) || 0,
            currentPrice: parseFloat(p.current_price) || 0,
            qty: parseFloat(p.qty) || 0,
          };
        }
      }

      // Build a map of today's buy fills by symbol for entry price lookups
      const buysBySymbol: Record<string, number[]> = {};
      for (const o of [...filled].reverse()) {
        const side = (o.side || "").toLowerCase();
        if (side === "buy") {
          const sym = o.symbol || "";
          if (!buysBySymbol[sym]) buysBySymbol[sym] = [];
          buysBySymbol[sym].push(parseFloat(o.filled_avg_price) || 0);
        }
      }

      // Enrich each trade with entry/exit/P&L
      const enriched = filled.map((o: any) => {
        const sym = o.symbol || "";
        const side = (o.side || "").toLowerCase();
        const fillPrice = parseFloat(o.filled_avg_price) || 0;
        const qty = parseFloat(o.filled_qty || o.qty) || 0;
        const pos = posMap[sym];

        let entry_price: number | null = null;
        let exit_price: number | null = null;
        let pnl: number | null = null;
        let pnl_pct: number | null = null;

        if (side === "sell" || side === "sell_short") {
          // Sell: entry = position avg_entry or earlier buy price, exit = fill price
          exit_price = fillPrice;
          if (pos) {
            entry_price = pos.avgEntry;
          } else if (buysBySymbol[sym] && buysBySymbol[sym].length > 0) {
            entry_price = buysBySymbol[sym][0]; // earliest buy
          }
          if (entry_price && entry_price > 0) {
            pnl = (exit_price - entry_price) * qty;
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100;
          }
        } else {
          // Buy: entry = fill price, exit = current price (unrealized) or null
          entry_price = fillPrice;
          if (pos && pos.currentPrice > 0) {
            exit_price = pos.currentPrice;
            pnl = (exit_price - entry_price) * qty;
            pnl_pct = entry_price > 0 ? ((exit_price - entry_price) / entry_price) * 100 : null;
          }
        }

        return {
          ...o,
          entry_price,
          exit_price,
          pnl: pnl !== null ? Math.round(pnl * 100) / 100 : null,
          pnl_pct: pnl_pct !== null ? Math.round(pnl_pct * 100) / 100 : null,
        };
      });

      res.json({ trades: enriched });
    } catch (e: any) {
      res.status(500).json({ error: e.message, trades: [] });
    }
  });

  // Open orders
  app.get("/api/orders/open", async (req, res) => {
    // AUTH (2026-04-20 fix Bug #22)
    const _user = _checkSession(req);
    if (!_user) return res.status(401).json({ error: "Authentication required" });
    try {
      const response = await fetch(
        `${ALPACA_BASE_URL}/v2/orders?status=open&limit=200`,
        { headers: alpacaHeaders }
      );
      if (!response.ok) {
        const errText = await response.text();
        return res.status(response.status).json({ error: errText, orders: [] });
      }
      const orders: any[] = await response.json();
      res.json({ orders });
    } catch (e: any) {
      res.status(500).json({ error: e.message, orders: [] });
    }
  });

  // Open positions
  app.get("/api/positions", async (req, res) => {
    // AUTH (2026-04-20 fix Bug #22)
    const _user = _checkSession(req);
    if (!_user) return res.status(401).json({ error: "Authentication required" });
    try {
      const response = await fetch(
        `${ALPACA_BASE_URL}/v2/positions`,
        { headers: alpacaHeaders }
      );
      if (!response.ok) {
        const errText = await response.text();
        return res.status(response.status).json({ error: errText, positions: [] });
      }
      const positions: any[] = await response.json();
      res.json({ positions });
    } catch (e: any) {
      res.status(500).json({ error: e.message, positions: [] });
    }
  });

  // ── Intraday Shorts Dashboard API (v1.0.27) ─────────────────────────────
  app.get("/api/shorts/dashboard", async (req, res) => {
    // AUTH (2026-04-20 fix Bug #22)
    const _user = _checkSession(req);
    if (!_user) return res.status(401).json({ error: "Authentication required" });
    try {
      const { stdout } = await execAsync(
        `python3 -c "import sys; sys.path.insert(0,'.'); from intraday_shorts import get_dashboard_data; import json; print(json.dumps(get_dashboard_data()))"`,
        { timeout: 10000 }
      );
      const jsonStart = stdout.indexOf("{");
      if (jsonStart === -1) throw new Error("No JSON");
      res.json(JSON.parse(stdout.slice(jsonStart)));
    } catch (e: any) {
      res.json({
        enabled: true, total_trades: 0, open_trades: 0, win_rate: 0,
        avg_pnl_pct: 0, total_pnl_pct: 0, total_pnl_dollar: 0,
        recent_trades: [], strategy_status: "waiting_for_signals",
        error: e.message,
      });
    }
  });

  // ── Trade History (filled orders from Alpaca) ──────────────────────────
  app.get("/api/trades/history", async (req, res) => {
    // AUTH (2026-04-20 fix Bug #22)
    const _user = _checkSession(req);
    if (!_user) return res.status(401).json({ error: "Authentication required" });
    try {
      const response = await fetch(
        `${ALPACA_BASE_URL}/v2/orders?status=filled&limit=50&direction=desc`,
        { headers: alpacaHeaders }
      );
      if (!response.ok) {
        const errText = await response.text();
        return res.status(response.status).json({ error: errText, trades: [] });
      }
      const orders: any[] = await response.json();

      // Also fetch market clock to know if we're in extended hours
      let marketOpen = false;
      try {
        const clockRes = await fetch(`${ALPACA_BASE_URL}/v2/clock`, { headers: alpacaHeaders });
        const clock = await clockRes.json();
        marketOpen = clock.is_open === true;
      } catch { /* ignore */ }

      // Group fills by symbol to pair buys with sells for round-trip P&L
      // Each order: symbol, side (buy/sell), filled_qty, filled_avg_price, filled_at
      interface TradeRecord {
        symbol: string;
        side: string;
        shares: number;
        entryPrice: number;
        exitPrice: number | null;
        pnl: number | null;
        pnlPct: number | null;
        filledAt: string;
      }

      const trades: TradeRecord[] = [];

      // Separate buys and sells per symbol
      const buyQueue: Map<string, Array<{ qty: number; price: number; filledAt: string }>> = new Map();

      // Process orders oldest first to match buys to sells
      const sorted = [...orders].reverse(); // oldest first
      for (const o of sorted) {
        const sym = o.symbol;
        const qty = parseFloat(o.filled_qty ?? "0");
        const price = parseFloat(o.filled_avg_price ?? "0");
        const filledAt = o.filled_at ?? o.updated_at ?? "";
        if (!qty || !price) continue;

        if (o.side === "buy") {
          if (!buyQueue.has(sym)) buyQueue.set(sym, []);
          buyQueue.get(sym)!.push({ qty, price, filledAt });
        } else if (o.side === "sell") {
          // Match against oldest buy
          const buys = buyQueue.get(sym) || [];
          const matchBuy = buys.shift();
          if (matchBuy) {
            const pnl = (price - matchBuy.price) * qty;
            const pnlPct = matchBuy.price > 0 ? ((price - matchBuy.price) / matchBuy.price) * 100 : 0;
            trades.push({
              symbol: sym,
              side: "SELL",
              shares: qty,
              entryPrice: matchBuy.price,
              exitPrice: price,
              pnl,
              pnlPct,
              filledAt,
            });
          } else {
            // No matching buy — record as standalone sell
            trades.push({
              symbol: sym,
              side: "SELL",
              shares: qty,
              entryPrice: price,
              exitPrice: price,
              pnl: 0,
              pnlPct: 0,
              filledAt,
            });
          }
        }
      }

      // Also include unmatched buys (still open or partial)
      // — skip those, they're open positions

      // Sort most-recent first
      trades.sort((a, b) => new Date(b.filledAt).getTime() - new Date(a.filledAt).getTime());

      res.json({ trades, marketOpen });
    } catch (e: any) {
      res.status(500).json({ error: e.message, trades: [] });
    }
  });

  // NOTE: /api/bot/performance is registered in bot.ts — removed duplicate here

  // Pre-warm: run a quick scan of top 10 tickers on startup so scanner has data
  setTimeout(() => {
    console.log("[scanner] Pre-warming with top 10 tickers...");
    const warmBatch = TIER1_TICKERS.slice(0, 10);
    scanBatch(warmBatch).then(results => {
      tier1Cache = sortByScore(applyFreshness(results));
      tier1LastUpdate = Date.now();
      for (const r of results) upsertCache(r);
      console.log(`[scanner] Pre-warm complete — ${results.length} tickers cached`);
    }).catch(err => console.error("[scanner] Pre-warm failed:", err));
  }, 3000);



// ── ML Model Status & Toggle ──────────────────────────────────────────────
  app.get("/api/ml/status", async (_req, res) => {
    try {
      const { exec } = await import("child_process");
      const { promisify } = await import("util");
      const execAsync = promisify(exec);
      const { stdout } = await execAsync("python3 ml_status.py", { timeout: 10000 });
      res.json(JSON.parse(stdout.trim()));
    } catch (err: any) {
      res.json({
        model_exists: false,
        enabled: false,
        error: err?.message?.slice(0, 200),
        contributes_to_cagr: false,
        note: "ML status check failed"
      });
    }
  });

  app.post("/api/ml/toggle", async (req, res) => {
    // AUTH (2026-04-20 fix Bug #22)
    const _user = _checkSession(req);
    if (!_user) return res.status(401).json({ error: "Authentication required" });
    try {
      const enabled = !!(req.body?.enabled);
      const action = enabled ? "enable" : "disable";
      const { exec } = await import("child_process");
      const { promisify } = await import("util");
      const execAsync = promisify(exec);
      const { stdout } = await execAsync(`python3 ml_toggle.py ${action}`, { timeout: 5000 });
      res.json(JSON.parse(stdout.trim()));
    } catch (err: any) {
      res.status(500).json({ error: err?.message?.slice(0, 200) });
    }
  });

  app.post("/api/ml/retrain", async (req, res) => {
    // AUTH (2026-04-20 fix Bug #22)
    const _user = _checkSession(req);
    if (!_user) return res.status(401).json({ error: "Authentication required" });
    try {
      res.json({ status: "started", message: "ML retrain started in background" });
      const { exec } = await import("child_process");
      const { promisify } = await import("util");
      const execAsync = promisify(exec);
      const { stdout } = await execAsync("python3 ml_retrain_safe.py", { timeout: 300000 });
      // Save result to persistent DATA_DIR so /api/ml/status can read it
      const result = JSON.parse(stdout.trim());
      const fs = await import("fs");
      const path = await import("path");
      const dataDir = process.env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp");
      fs.mkdirSync(dataDir, { recursive: true });
      fs.writeFileSync(path.join(dataDir, "ml_status.json"), JSON.stringify(result));
      // Also enable toggle automatically after successful retrain
      if (result.status === "ok" || result.status === "success" || result.steps?.includes("training_done")) {
        fs.writeFileSync(path.join(dataDir, "ml_toggle.json"), JSON.stringify({ enabled: true }));
      }
    } catch (err: any) {
      // Background — save error status
      try {
        const fs = await import("fs");
        const path = await import("path");
        const dataDir = process.env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp");
        fs.mkdirSync(dataDir, { recursive: true });
        const errMsg = (err?.message || String(err)).slice(0, 500);
        fs.writeFileSync(path.join(dataDir, "ml_status.json"), JSON.stringify({ status: "error", error: errMsg }));
      } catch {} 
    }
  });

  


  return httpServer;
}

