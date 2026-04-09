import { type Express } from "express";
import { type Server } from "http";
import { exec } from "child_process";
import { promisify } from "util";
import path from "path";
import fs from "fs";
import https from "https";
import cookieParser from "cookie-parser";
import { registerAuthRoutes, db } from "./auth";
import { registerBotRoutes } from "./bot";

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
        fullUniverseCache.set(r.ticker, r);
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
          fullUniverseCache.set(r.ticker, r);
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
    try {
      universe = await fetchCBOEUniverse();
      const us = new Set(universe);
      for (const t of TIER1_TICKERS) us.add(t);
      universe = Array.from(us);
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

    const { ticker } = req.body;
    if (!ticker) return res.status(400).json({ error: "Ticker required" });

    try {
      db.prepare("INSERT OR IGNORE INTO watchlists (user_email, ticker, added_at) VALUES (?, ?, ?)").run(
        user.email, ticker.toUpperCase(), new Date().toISOString()
      );
    } catch {}

    res.json({ ok: true });
  });

  app.post("/api/watchlist/remove", (req, res) => {
    const session = (req as any).cookies?.session;
    if (!session) return res.status(401).json({ error: "Not authenticated" });

    const sessionRow = db.prepare("SELECT user_id FROM sessions WHERE token = ?").get(session) as any;
    if (!sessionRow) return res.status(401).json({ error: "Not authenticated" });

    const user = db.prepare("SELECT email FROM users WHERE id = ?").get(sessionRow.user_id) as any;
    if (!user) return res.status(401).json({ error: "Not authenticated" });

    const { ticker } = req.body;
    if (ticker) {
      db.prepare("DELETE FROM watchlists WHERE user_email = ? AND ticker = ?").run(user.email, ticker.toUpperCase());
    }

    res.json({ ok: true });
  });

  // ── Single-ticker analysis ────────────────────────────────────────────────
  app.get("/api/analyze/:ticker", async (req, res) => {
    const { ticker } = req.params;

    if (!ticker || !/^[A-Za-z.^-]{1,10}$/.test(ticker)) {
      return res.status(400).json({ error: "Invalid ticker symbol. Please use letters only (e.g. AAPL, SPY, TSLA)." });
    }

    const scriptPath = path.resolve(process.cwd(), "analyze.py");

    try {
      const { stdout } = await execAsync(
        `python3 "${scriptPath}" "${ticker.toUpperCase()}"`,
        { timeout: 120000, maxBuffer: 1024 * 1024 * 10 }
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
            for (const r of seedResults) fullUniverseCache.set(r.ticker, r);
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
    const POLYGON_KEY = process.env.POLYGON_API_KEY || "UNwTHo3kvZMBckeIaHQbBLuaaURmFUQP";
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
        .sort((a: any, b: any) => b.volume - a.volume);
      // No slice — return all tradeable stocks
      res.json({ results, date: yesterday, total: results.length });
    } catch (err) {
      console.error("[market-snapshot] Error:", err);
      res.status(500).json({ error: "Market snapshot failed" });
    }
  });

  // ── Polygon news ─────────────────────────────────────────────────────────────
  app.get("/api/news", async (req, res) => {
    const POLYGON_KEY = process.env.POLYGON_API_KEY || "UNwTHo3kvZMBckeIaHQbBLuaaURmFUQP";
    const ticker = req.query.ticker as string || "";
    try {
      const url = ticker
        ? `https://api.polygon.io/v2/reference/news?ticker=${ticker}&limit=20&apiKey=${POLYGON_KEY}`
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
  const ALPACA_KEY = process.env.ALPACA_KEY || "PKMDHJOVQEVIB4UHZXUYVTIDBU";
  const ALPACA_SECRET = process.env.ALPACA_SECRET || "9jnjnhts7fsNjefFZ6U3g7sUvuA5yCvcx2qJ7mZb78Et";
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
      const snapRes = await fetch(`https://data.alpaca.markets/v2/stocks/snapshots?symbols=${symbols}&feed=sip`, { headers: alpacaHeaders });
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

  // ── Intraday Shorts Dashboard API (v1.0.27) ─────────────────────────────
  app.get("/api/shorts/dashboard", async (_req, res) => {
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
  app.get("/api/trades/history", async (_req, res) => {
    try {
      const response = await fetch(
        "https://paper-api.alpaca.markets/v2/orders?status=filled&limit=50&direction=desc",
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
        const clockRes = await fetch("https://paper-api.alpaca.markets/v2/clock", { headers: alpacaHeaders });
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

  // Pre-warm: run a quick scan of top 10 tickers on startup so scanner has data
  setTimeout(() => {
    console.log("[scanner] Pre-warming with top 10 tickers...");
    const warmBatch = TIER1_TICKERS.slice(0, 10);
    scanBatch(warmBatch).then(results => {
      tier1Cache = sortByScore(applyFreshness(results));
      tier1LastUpdate = Date.now();
      for (const r of results) fullUniverseCache.set(r.ticker, r);
      console.log(`[scanner] Pre-warm complete — ${results.length} tickers cached`);
    }).catch(err => console.error("[scanner] Pre-warm failed:", err));
  }, 3000);

  return httpServer;

// ── ML Model Status & Toggle ──────────────────────────────────────────────
  app.get("/api/ml/status", async (_req, res) => {
    try {
      const { exec } = await import("child_process");
      const { promisify } = await import("util");
      const execAsync = promisify(exec);
      const { stdout } = await execAsync(
        `python3 -c "
import json, os, time
model_path = os.environ.get('ML_MODEL_PATH', 'voltrade_ml_v2.pkl')
data_dir = os.environ.get('DATA_DIR', '/tmp')
status_path = os.path.join(data_dir, 'ml_status.json')
toggle_path = os.path.join(data_dir, 'ml_toggle.json')

# Check model file
model_exists = os.path.exists(model_path) or os.path.exists(os.path.join(data_dir, model_path))
model_age_hours = None
for p in [model_path, os.path.join(data_dir, model_path), os.path.join(data_dir, 'voltrade_ml_v2.pkl')]:
    if os.path.exists(p):
        model_age_hours = round((time.time() - os.path.getmtime(p)) / 3600, 1)
        model_exists = True
        break

# Check toggle state
enabled = False
try:
    with open(toggle_path) as f:
        enabled = json.load(f).get('enabled', False)
except: pass

# Check last training result
last_train = {}
try:
    with open(status_path) as f:
        last_train = json.load(f)
except: pass

print(json.dumps({
    'model_exists': model_exists,
    'model_age_hours': model_age_hours,
    'enabled': enabled,
    'last_accuracy': last_train.get('accuracy'),
    'last_samples': last_train.get('samples'),
    'last_features': last_train.get('feature_count'),
    'last_status': last_train.get('status', 'unknown'),
    'last_train_time': last_train.get('timestamp'),
    'contributes_to_cagr': False,
    'note': 'ML is disabled by default. The 20.3% CAGR runs without ML. Enable to test if ML adds edge.'
}))
"`,
        { timeout: 10000 }
      );
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
    try {
      const { enabled } = req.body;
      const { exec } = await import("child_process");
      const { promisify } = await import("util");
      const execAsync = promisify(exec);
      await execAsync(
        `python3 -c "
import json, os
data_dir = os.environ.get('DATA_DIR', '/tmp')
toggle_path = os.path.join(data_dir, 'ml_toggle.json')
with open(toggle_path, 'w') as f:
    json.dump({'enabled': ${enabled ? 'True' : 'False'}}, f)
print(json.dumps({'enabled': ${enabled ? 'true' : 'false'}, 'status': 'ok'}))
"`,
        { timeout: 5000 }
      );
      res.json({ enabled: !!enabled, status: "ok" });
    } catch (err: any) {
      res.status(500).json({ error: err?.message?.slice(0, 200) });
    }
  });

  app.post("/api/ml/retrain", async (_req, res) => {
    try {
      res.json({ status: "started", message: "ML retrain started in background" });
      const { exec } = await import("child_process");
      const { promisify } = await import("util");
      const execAsync = promisify(exec);
      const { stdout } = await execAsync("python3 ml_retrain_safe.py", { timeout: 300000 });
      // Save result
      const result = JSON.parse(stdout.trim());
      await execAsync(
        `python3 -c "import json; f=open('/tmp/ml_status.json','w'); json.dump(${JSON.stringify(JSON.stringify(result))}, f)"`,
        { timeout: 5000 }
      );
    } catch (err: any) {
      // Background — no response to send
    }
  });

  


}

