import { type Express } from "express";
import { type Server } from "http";
import { exec } from "child_process";
import { promisify } from "util";
import path from "path";
import fs from "fs";
import https from "https";

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
  const universe: string[] = Array.from(new Set(TIER1_TICKERS));
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
}
