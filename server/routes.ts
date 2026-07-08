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
import datacoreBoundaries from "../datacore/boundaries/ne_110m_admin0.json";
import { version as pkgVersion } from "../package.json";
import {
  archiveAircraft, archiveVessels, archiveTrains, compressOldHours, rollupOldDays,
  recentTrack, archiveStats,
} from "./datacoreArchive";
import { applyViewport } from "./viewport";
import { budgetStatus as tiles3dBudgetStatus, loadLedger as loadTiles3dLedger, authorizeRoot as tiles3dAuthorizeRoot, recordRoot as tiles3dRecordRoot } from "./tiles3dBudget";
import { is3dTilesUrl, withKey as tiles3dWithKey, ROOT_URL as TILES3D_ROOT_URL } from "./tiles3dProxy";
import { registerAuthRoutes, db } from "./auth";
import { registerBotRoutes } from "./bot";
import { vesselStreamEnabled, bootVesselStream } from "./vesselStream";
import { complianceAuditTick, setComplianceAuditWriter } from "./providerCompliance";
import { mapDigitraffic, mapEntur, ENTUR_VEHICLES_QUERY } from "./trainsFeed";
import { computeShadowStatsAsync } from "./shadowFleet";
import { computePortDwellAsync, portsFromSites } from "./portDwell";
import { buildGraph, neighborhood, resolveEntityId } from "./entityGraph";
import {
  validateWxTile, owmTileUrl, classifyOwmStatus, owmStatusNote, makeTileCache,
  amplifyWeatherTile, TILE_TTL_MS, NEGATIVE_TTL_MS, WxLayer,
} from "./owmTiles";
import {
  parseApiKeys, makeRateLimiter, meterUsage, apiMeta, LICENSE_MARKS, ApiTier,
} from "./apiProduct";
import { addToWaitlist } from "./waitlist";
import {
  snapBbox, bboxKey, gridPoints, spacingKm, owmPointUrl, parseOwmPoint,
  GRID_TTL_MS, UPSTREAM_PER_MIN_GUARD, WeatherSample,
} from "./weatherGrid";
import shadowZones from "../datacore/shadow_zones.json";
import { bootForm4Poll, latestForm4Filings, readFilingHistory } from "./edgarForm4";
import { archivedAircraftCached, entityForHex, loadEntitySpine } from "./aircraftEntities";
import { bootAlertsPoll, latestAlerts } from "./nwsAlerts";
import { bootTreasuryPoll, latestAuctions } from "./treasuryAuctions";
import { bootDroughtPoll, latestDrought } from "./droughtMonitor";
import { bootCensusPoll, latestImports, censusEnabled } from "./censusImports";
import { bootShortVolPoll, latestShortVol, readSummaryHistory, lookupSymbolHistory } from "./finraShortVolume";
import { bootCotPoll, latestCot } from "./cftcCot";
import { bootTffPoll, latestTff } from "./cftcTff";
import { bootDtsPoll, latestDts } from "./treasuryDts";
import { bootFailuresPoll, latestFailures } from "./fdicBanks";
import { bootComplaintsPoll, latestComplaintStats } from "./nhtsaComplaints";
import { bootGridDemandPoll, latestDemand, gridDemandEnabled } from "./gridDemand";
import { bootGridStressPoll, latestGridStress, gridStressEnabled } from "./gridStress";
import { bootEuLoadPoll, latestLoad, euLoadEnabled } from "./euLoad";
import { bootAirQualityPoll, latestAirQuality, airQualityEnabled } from "./airQuality";
import { bootSatellitesPoll, satellitesResponse } from "./satellites";
import { bootCropConditionsPoll, latestConditions, cropConditionsEnabled } from "./cropConditions";
import { bootOccPoll, latestOcc } from "./occVolume";
import { bootAttentionPoll, latestAttention, lastAttentionCycle, ARTICLES as WIKI_ARTICLES } from "./wikiAttention";
import { bootFaaPoll, latestFaaStatus } from "./faaStatus";
import { bootBorderWaitPoll, latestBorderWaits } from "./cbpBorderWait";
import { fleetSeriesCached } from "./fleetUtilization";
import { siteTimelineCached, type SiteRef } from "./siteTimeline";
import { queryWindowCached } from "./queryEngine";
import { analystResponse } from "./analyst";
import { firmsEnabled, bootFirmsPoll, latestFirms, buildFiresResponse } from "./nasaFirms";
import { firesNearFacilities } from "./firesFacilities";
import { gaugePoints, plantsNearGauges, type PlantTuple } from "./riverPlants";
import { plantsUnderAlerts } from "./plantsUnderAlerts";
import { bootChainArchive } from "./optionsChainArchive";
import { platformStats } from "./platformStats";
import { bootEarnings8kPoll, latestEarnings8Ks, readEarnings8kHistory } from "./sec8kEarnings";
import { boot13FPoll, latest13FFilings, read13FHistory, trimHoldings, FOCUSED_MAX_HOLDINGS } from "./edgar13f";
import { bootFredPoll, latestFredSeries, buildMacroPayload, fredEnabled } from "./fredMacro";
import { raceDeadline, slotExpired, makeSlot, ROUTE_DEADLINE_MS, type InflightSlot } from "./routeGuards";
import { bootContractsPoll, latestContracts } from "./usaSpending";
import { bootFdaPoll, latestFdaEvents } from "./fdaEvents";
import { bootUsgsPoll, latestGauges } from "./usgsWater";
import { bootGdeltPoll, latestGdeltEvents } from "./gdeltEvents";
import { bootStreamsInventoryPoll, getStreamsInventoryCached } from "./streamsInventory";
import { bootFinraQueryPoll, latestFinraSi, latestFinraAts } from "./finraQuery";
import { bootFtdPoll, latestFtd } from "./secFtd";
import { bootEuMacroPoll, latestEuMacro } from "./euMacro";
import { bootQuakesPoll, latestQuakes } from "./usgsQuakes";
import { bootBuoysPoll, latestBuoys } from "./ndbcBuoys";

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
      // vessels/fires go live automatically the moment their key exists
      l.id === "vessels"
        ? { ...l, status: vesselStreamEnabled() ? "live" : "awaiting_key" }
      : l.id === "fires"
        ? { ...l, status: firmsEnabled() ? "live" : "awaiting_key" }
      // [REPAIR 2026-07-05, audit defect #2] trains health override: the
      // registry statically claimed "live" through the entire trains
      // outage — a dead feed advertising live is the exact "empty feed
      // that claims live status" failure mode. The eager archive tick
      // refreshes trainsCache every 10 min, so a cache >45 min old (or
      // both sources erroring) means the feed is genuinely down.
      : l.id === "trains"
        ? (() => {
            const fresh = trainsCache && Date.now() - trainsCache.at < 45 * 60_000;
            const anyOk = (trainsCache?.data?.sources || []).some((s: any) => s.status === "ok");
            return fresh && anyOk ? l : { ...l, status: "down", status_note: "feed outage — sources unreachable; auto-recovers" };
          })()
        : l
    );
    // server_version lets the client detect an OPEN-TAB VERSION SKEW: a
    // long-lived tab that remounts the /data page re-fetches this registry
    // (new layer rows) while still running an old bundle (no effects for
    // them) — pill flips, label stays "off", nothing renders (the
    // 2026-07-04 production desync). The client compares against its
    // baked-in version and tells the user to reload.
    res.json({ layers, server_version: pkgVersion });
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
  const aircraftInflight: Map<string, InflightSlot<any>> = new Map();
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
        aircraft = (raw2[fb.arr] || []).slice(0, 10000).map((a: any) => ({
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
      // SCALE S1(a): optional ?bbox= viewport filter (scale_program.md).
      // Absent/invalid bbox = payload unchanged; helper lives in viewport.ts
      // so this handler body (and the raceDeadline guard window) is untouched.
      return res.json(applyViewport(({ ...hit.data, cached: true }), req.query.bbox, "aircraft", (a: any) => [a.lon, a.lat]));
    }
    try {
      // In-flight dedup: concurrent visitors on the same bbox share ONE
      // upstream request (rate-limit protection is server-wide, not per-tab).
      // Same stuck-slot defense as trains ([REPAIR 2026-07-05]): a slot
      // older than the expiry is abandoned, no request waits past the route
      // deadline (falls to stale-beats-spinner below), and a late-settling
      // orphan can't clobber its replacement.
      let slot = aircraftInflight.get(key);
      if (slotExpired(slot, Date.now())) {
        slot = makeSlot(() => fetchAircraft(lamin, lamax, lomin, lomax), Date.now(), (self) => {
          if (aircraftInflight.get(key) === self) aircraftInflight.delete(key);
        });
        aircraftInflight.set(key, slot);
      }
      const data = await raceDeadline(slot!.p, ROUTE_DEADLINE_MS, "aircraft");
      aircraftCache.set(key, { at: Date.now(), data });
      if (aircraftCache.size > 20) {
        const oldest = Array.from(aircraftCache.entries()).sort((a, b) => a[1].at - b[1].at)[0];
        if (oldest) aircraftCache.delete(oldest[0]);
      }
      if (String(req.query.since || "") === String(data.time)) {
        return res.json({ unchanged: true, time: data.time, count: data.count });
      }
      res.json(applyViewport(data, req.query.bbox, "aircraft", (a: any) => [a.lon, a.lat]));
    } catch (e: any) {
      // Stale-beats-spinner (DESIGN.md performance budget): serve the last
      // snapshot with its timestamp rather than an empty error.
      if (hit) return res.json(applyViewport(({ ...hit.data, cached: true, stale: true, stale_at: hit.at }), req.query.bbox, "aircraft", (a: any) => [a.lon, a.lat]));
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
    // Raised 5000→15000: the WebGL symbol layer handles 10k+ (DESIGN.md budget),
    // so at wide zoom in dense AIS regions the layer no longer SILENTLY drops
    // ships. total_in_view counts every fresh in-bbox vessel so, when the cap
    // still bites, we disclose it honestly (coverage_note, surfaced by the
    // client's existing partial-coverage note path) instead of hiding the drop.
    const VESSEL_CAP = 15000;
    const vessels: any[] = [];
    let totalInView = 0;
    vesselPositions.forEach((v, mmsi) => {
      if (v.at < cutoff) return;
      if (v.lat < lamin || v.lat > lamax || v.lon < lomin || v.lon > lomax) return;
      totalInView += 1;
      if (vessels.length >= VESSEL_CAP) return;
      const st = vesselStatics.get(mmsi);
      vessels.push({
        mmsi, name: st?.name || v.name, lat: v.lat, lon: v.lon,
        sog: v.sog, cog: v.cog,
        shiptype: st?.shiptype ?? null,
        destination: st?.destination ?? null,
      });
    });
    const capped = totalInView > vessels.length;
    res.json({
      enabled: true,
      source: "aisstream.io (AIS, terrestrial receivers — mid-ocean coverage gaps are inherent)",
      kind: "raw",
      warming_up: vessels.length === 0 && Date.now() - vesselSocketUp < 30_000,
      count: vessels.length,
      total_in_view: totalInView,
      ...(capped ? {
        coverage: "partial",
        coverage_note: `showing ${vessels.length.toLocaleString()} of ${totalInView.toLocaleString()} vessels in view — zoom in to see the rest`,
      } : {}),
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

  // Fires × facilities cross-tie (worldview-globe Pillar 6, backend inference).
  // Joins live NASA active-fire detections to our strategic-facility archive:
  // which assets have fires nearby right now. RAW cross-tie (gate-0 observation),
  // NOT a validated signal — the fires-near-assets→insurer/utility-returns
  // hypothesis is filed in open_questions.md and gated. Cache-only read (fires
  // poller already runs); default radius 25 km, capped 1–200. Sites-only for now
  // (16 assets = cheap exact join); powerplants need a spatial index (SCALE S3).
  app.get("/api/data/fires-near-facilities", (req, res) => {
    try {
      if (!firmsEnabled()) return res.json({ enabled: false, awaiting_key: true, note: "NASA_FIRMS_MAP_KEY not set — fires feed inactive", facilities: [] });
      const latest = latestFirms();
      if (!latest) return res.json({ warming_up: true, facilities: [], note: "fires feed warming up" });
      const r = Math.min(200, Math.max(1, parseFloat(String(req.query.radius_km)) || 25));
      const sites: any[] = (datacoreSites as any).sites || (datacoreSites as any) || [];
      const hits = firesNearFacilities(sites, latest.detections || [], r);
      res.json({
        enabled: true, kind: "raw", as_of: latest.at, radius_km: r,
        fires_checked: (latest.detections || []).length,
        facilities_with_fires: hits.length,
        facilities: hits,
        note: "RAW cross-tie: strategic assets with active VIIRS 375m fire detections nearby (NRT ~3h). No predictive claim — the fires-near-assets → insurer/utility hypothesis is validation-gated. Not for safety-of-life use.",
        source: "NASA FIRMS/LANCE (VIIRS_SNPP_NRT) × datacore/sites strategic facilities",
      });
    } catch (e: any) { res.status(500).json({ error: e?.message }); }
  });

  // Plants × river gauges cross-tie (worldview-globe Pillar 6, backend inference).
  // Joins the WRI power-plant archive to the live barge-corridor USGS gauges: which
  // generating capacity sits near each gauge, so a LOW-WATER reading immediately
  // names the plants exposed to cooling-water limits (river-cooled thermal) and to
  // barge coal-delivery draft restrictions. RAW cross-tie (gate-0 observation), NOT
  // a validated signal — the low-water→operator/utility-returns hypothesis is filed
  // in open_questions.md and gated. Cache-only read (USGS poller already runs);
  // default radius 50 km, capped 1–200. Keyless: USGS public domain × WRI CC BY 4.0.
  app.get("/api/data/plants-near-rivergauges", (req, res) => {
    try {
      const hit = latestGauges();
      if (!hit) return res.json({ warming_up: true, gauges: [], note: "river-gauge feed warming up" });
      const r = Math.min(200, Math.max(1, parseFloat(String(req.query.radius_km)) || 50));
      const pts = gaugePoints(hit.gauges || []);
      const plants = (datacorePowerplants as unknown as PlantTuple[]) || [];
      const gauges = plantsNearGauges(pts, plants, r);
      res.json({
        kind: "raw", as_of: hit.at, radius_km: r,
        gauges_checked: pts.length, plants_checked: plants.length,
        gauges_with_plants: gauges.length, gauges,
        note: "RAW cross-tie: generating capacity near each barge-corridor river gauge, so a low-water reading names the plants exposed to cooling-water limits and barge coal-delivery draft restrictions. No predictive claim — the low-water → operator/utility hypothesis is validation-gated. Static plant locations (WRI 2021 vintage) × live USGS gauge positions.",
        source: "USGS NWIS barge-corridor gauges (public domain) × WRI Global Power Plant DB (CC BY 4.0)",
      });
    } catch (e: any) { res.status(500).json({ error: e?.message }); }
  });

  // Plants × severe-weather warnings cross-tie (worldview-globe Pillar 6, backend
  // inference). Which generating capacity sits INSIDE an active NWS warning
  // polygon right now — a live storm-risk-to-generation read, distinct from the
  // water-risk river cross-tie above. RAW cross-tie (gate-0 observation), NOT a
  // validated signal — the warning-over-generation→operator/grid hypothesis is
  // filed in open_questions.md and gated. Cache-only read (NWS poller already
  // runs, polygon-carrying alerts only; zone-only alerts have no geometry to test
  // and are honestly excluded). Keyless: NWS public domain × WRI CC BY 4.0.
  app.get("/api/data/plants-under-alerts", (_req, res) => {
    try {
      const hit = latestAlerts();
      if (!hit) return res.json({ warming_up: true, alerts: [], note: "NWS alerts feed warming up" });
      const plants = (datacorePowerplants as unknown as PlantTuple[]) || [];
      const alerts = plantsUnderAlerts((hit.display || []) as any, plants);
      res.json({
        kind: "raw", as_of: hit.at,
        alerts_checked: (hit.display || []).length, zone_only_excluded: hit.zoneOnly ?? 0,
        plants_checked: plants.length,
        alerts_with_plants: alerts.length, alerts,
        note: "RAW cross-tie: generating capacity inside active NWS warning polygons — the plants a severe-weather event currently covers. No predictive claim — the warning-over-generation → operator/grid-disruption hypothesis is validation-gated. Zone-only alerts (no polygon) are excluded, not silently counted as covering nothing. Static plant locations (WRI 2021 vintage) × live NWS warnings.",
        source: "NWS api.weather.gov active warnings (public domain) × WRI Global Power Plant DB (CC BY 4.0)",
      });
    } catch (e: any) { res.status(500).json({ error: e?.message }); }
  });

  // Google Photorealistic 3D Tiles cost-guard status (worldview-globe G3). Safe,
  // NO-SPEND: reports the daily/monthly root-request budget so the client and a
  // health check can see headroom before 3D mode is ever activated. The actual
  // key-holding root proxy (which spends a billable root request) ships WITH the
  // deck.gl deep-zoom client, gated by authorizeRoot() — same discipline as the
  // RunPod budget shipping before the launcher.
  app.get("/api/data/3dtiles/status", (_req, res) => {
    try {
      const keyed = !!(process.env.GOOGLE_MAPS_API_KEY || "").trim();
      res.json({
        enabled: keyed,
        awaiting_key: !keyed,
        note: keyed
          ? "3D Tiles available; activates only at deep zoom, root requests capped in the free tier"
          : "GOOGLE_MAPS_API_KEY not set — 3D Tiles inactive",
        budget: tiles3dBudgetStatus(loadTiles3dLedger(), Date.now()),
      });
    } catch (e: any) { res.status(500).json({ error: e?.message }); }
  });

  // 3D Tiles ROOT — the one BILLABLE request (starts a session). Key-gated,
  // guard-gated (authorizeRoot refuses past the daily/monthly free-tier cap), and
  // ledgered on success. Returns Google's tileset JSON; the client's loader then
  // fetches children through /proxy (below), so the key never leaves the server.
  app.get("/api/data/3dtiles/root", async (_req, res) => {
    const key = (process.env.GOOGLE_MAPS_API_KEY || "").trim();
    if (!key) return res.status(503).json({ enabled: false, awaiting_key: true, error: "GOOGLE_MAPS_API_KEY not set" });
    const decision = tiles3dAuthorizeRoot(loadTiles3dLedger(), Date.now());
    if (!decision.authorized) return res.status(429).json({ error: decision.detail, reason: decision.reason, budget: decision });
    try {
      const r = await fetch(tiles3dWithKey(TILES3D_ROOT_URL, key), { signal: AbortSignal.timeout(15000) });
      if (!r.ok) return res.status(502).json({ error: `google 3dtiles root ${r.status}` });
      const json = await r.json();
      tiles3dRecordRoot(Date.now(), "root"); // ledger only on a real, successful spend
      res.json(json);
    } catch (e: any) { res.status(502).json({ error: e?.message || "root fetch failed" }); }
  });

  // 3D Tiles CHILD PROXY — free within the session. Streams a Google 3D-tiles
  // child (json/glTF/glb) with the key appended server-side. Tight allowlist
  // (is3dTilesUrl) so this can never become an open proxy / SSRF vector.
  app.get("/api/data/3dtiles/proxy", async (req, res) => {
    const key = (process.env.GOOGLE_MAPS_API_KEY || "").trim();
    if (!key) return res.status(503).end();
    const u = String(req.query.u || "");
    if (!is3dTilesUrl(u)) return res.status(400).json({ error: "url not allowed" });
    try {
      const r = await fetch(tiles3dWithKey(u, key), { signal: AbortSignal.timeout(20000) });
      if (!r.ok) return res.status(502).end();
      res.setHeader("content-type", r.headers.get("content-type") || "application/octet-stream");
      res.setHeader("cache-control", "public, max-age=3600");
      res.end(Buffer.from(await r.arrayBuffer()));
    } catch { res.status(502).end(); }
  });

  // Streams inventory (DATACORE MAXIMUS Phase 4, 2026-07-06): one call =
  // every manifested stream with source/hypothesis/recording-since/count/
  // freshness/health/latest-peek. Cache-only request path (event-loop
  // rule); the scan runs on a 5-min timer + eager boot. Coverage is
  // mechanical — the aggregator enumerates datacore/manifests/ at runtime
  // (pinned by the inventory-coverage ratchet in streamsInventory.test.ts).
  bootStreamsInventoryPoll();
  app.get("/api/data/streams", (_req, res) => {
    try {
      const inv = getStreamsInventoryCached();
      if (!inv) { res.json({ warming_up: true, count: 0, streams: [] }); return; }
      res.json(inv);
    } catch (e: any) { res.status(500).json({ error: e?.message }); }
  });

  // Aircraft entity spine v1 (GIP Part 3b; BUILD ORDER 2 #1, 2026-07-05).
  // /hexes = distinct archived icao24s (the join-key list the session-side
  // FAA build script consumes; 6h-cached scan, stale served during refresh).
  // /entity/:hex = identity from the committed spine artifact, evidence
  // envelope included; degrades to entity:null until the artifact ships.
  app.get("/api/data/aircraft/hexes", async (_req, res) => {
    try {
      res.set("Cache-Control", "public, max-age=3600");
      const { hexes, as_of } = await archivedAircraftCached();
      res.json({ kind: "raw", count: hexes.length, as_of, hexes });
    } catch (e: any) { res.status(500).json({ error: e?.message }); }
  });
  app.get("/api/data/aircraft/entity/:hex", (req, res) => {
    try {
      const hex = String(req.params.hex || "").toLowerCase();
      if (!/^[0-9a-f]{6}$/.test(hex)) { res.status(400).json({ error: "hex must be 6 hex chars" }); return; }
      res.set("Cache-Control", "public, max-age=3600");
      res.json({ kind: "raw", hex, entity: entityForHex(hex), spine_built: loadEntitySpine() !== null });
    } catch (e: any) { res.status(500).json({ error: e?.message }); }
  });

  // Honest self-updating platform counters for the landing hero
  // (hero-refinements directive 2026-07-05): layers from the registry,
  // streams from registry+archive, observations = REAL line counts
  // (10-min cached). Never hardcoded — grows as the system grows.
  app.get("/api/data/platform/stats", async (_req, res) => {
    try {
      res.set("Cache-Control", "public, max-age=300");
      res.json(await platformStats((datacoreLayers as any).layers || []));
    } catch (e: any) { res.status(500).json({ error: e?.message }); }
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

  // Admin boundaries (RAW; Natural Earth 1:110m admin-0, PUBLIC DOMAIN —
  // atlas-parity layer 3). Self-hosted datacore compile (254KB slim):
  // no external dependency, no license constraint on resale. Day-cached,
  // fetched only when the layer is enabled (zero-cost-when-off).
  app.get("/api/data/boundaries", (_req, res) => {
    res.set("Cache-Control", "public, max-age=86400");
    res.json({ kind: "raw", source: "Natural Earth 1:110m admin-0 (public domain)", ...(datacoreBoundaries as any) });
  });

  // Live trains overlay (RAW) — Finland Digitraffic (CC BY 4.0) + Norway
  // Entur (NLOD); mapping in server/trainsFeed.ts (pure, unit-tested).
  // Shared 30s cache + in-flight dedup + per-source backoff; the response
  // carries per-source status so the panel labels coverage HONESTLY
  // (launch coverage is FI+NO only). US freight rail positions are
  // proprietary — no free source exists (open_questions; do not chase).
  // Every fresh snapshot feeds the permanent position archive.
  let trainsCache: { at: number; data: any } | null = null;
  let trainsInflight: InflightSlot<any> | null = null;
  async function fetchTrains() {
    const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" };
    const sources: any[] = [];
    const trains: any[] = [];
    await Promise.all([
      (async () => {
        if (backoffActive("digitraffic")) { sources.push({ key: "digitraffic", country: "FI", status: "backoff", count: 0 }); return; }
        try {
          // Digitraffic REQUIRES gzip since 2026-07 (406 without it) —
          // explicit even though undici usually sends it, so a bundler or
          // runtime that drops the default can't silently kill the feed.
          const r = await fetch("https://rata.digitraffic.fi/api/v1/train-locations/latest",
            { headers: { ...UA, "Digitraffic-User": "voltradeai-datacore", "Accept-Encoding": "gzip" }, signal: AbortSignal.timeout(12000) });
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
    // [REPAIR 2026-07-05] Production outage: one fetchTrains got stuck past
    // its per-source timeouts; `.finally()` only clears on SETTLE, so every
    // request forever after awaited the same dead promise (endpoint hung
    // >90s with zero bytes until redeploy). Defense in depth via
    // routeGuards: a stale slot is abandoned (fresh fetch starts), no
    // request waits past the route deadline (falls to stale-beats-spinner),
    // and an orphan settling late can't clobber the replacement slot.
    if (slotExpired(trainsInflight, Date.now())) {
      trainsInflight = makeSlot(fetchTrains, Date.now(), (self) => {
        if (trainsInflight === self) trainsInflight = null;
      });
    }
    const slot = trainsInflight!; // non-null: the expired branch just assigned it
    try {
      const data = await raceDeadline(slot.p, ROUTE_DEADLINE_MS, "trains");
      trainsCache = { at: Date.now(), data };
      res.json(data);
    } catch (e: any) {
      // stale-over-error: fetchTrains itself never throws (per-source
      // status instead) — this path is the route deadline firing.
      if (trainsCache) return res.json({ ...trainsCache.data, stale: true });
      res.status(503).json({ error: e?.message || "trains fetch failed", trains: [], count: 0 });
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

  // NASA FIRMS active fires (RAW) — Tier-1(c) geospatial root, key-gated
  // exactly like vessels (research/wishlist.md: free MAP_KEY, commercial-
  // lawful, VIIRS 375m NRT ~3h latency, LANCE attribution + "not for
  // safety-of-life" disclaimer required). No free history exists upstream,
  // so every fetch archives from day one (see server/nasaFirms.ts). Boots
  // eagerly (KNOWN BROKEN #9 lesson) but only actually polls once a key is
  // present — bootFirmsPoll no-ops without NASA_FIRMS_MAP_KEY.
  bootFirmsPoll();

  // Daily options-chain snapshot archive (approved options-data decision
  // 2026-07-04: forward-only history that can never be back-bought; feed=
  // indicative honesty in every record + manifest). No-ops without Alpaca
  // creds; runs once per trading day after the close.
  bootChainArchive();

  // [REPAIR 2026-07-05, audit defect #1] EAGER ARCHIVE TICK. Aircraft and
  // trains archiving was REQUEST-driven (fetch functions archive as a side
  // effect) — no visitors meant no archive, and 11 hourly gaps each were
  // already permanent in the archive's first 36h (gaps never refill).
  // Vessels got the eager-boot fix (KNOWN BROKEN #9); these two didn't.
  // Every 10 minutes: one aircraft snapshot rotating across four
  // strategic-site regions (point-radius API — a fixed region per tick,
  // each region ~every 40 min) + one trains snapshot (single global
  // fetch, its archive is internal). One extra upstream call per 10 min
  // per feed — negligible vs the visitor-driven load it replaces, and the
  // archive stops being visitor-biased.
  const ARCHIVE_TICK_REGIONS = [
    { lamin: 33, lamax: 39, lomin: -100, lomax: -93 },  // Cushing / south-central US
    { lamin: 37, lamax: 42, lomin: -78, lomax: -71 },   // US northeast corridor
    { lamin: 32, lamax: 36, lomin: -120, lomax: -114 }, // LA / Long Beach
    { lamin: 49, lamax: 54, lomin: 2, lomax: 9 },       // Rotterdam / ARA range
  ];
  let archiveTickIdx = 0;
  const archiveTick = () => {
    const r = ARCHIVE_TICK_REGIONS[archiveTickIdx++ % ARCHIVE_TICK_REGIONS.length];
    fetchAircraft(r.lamin, r.lamax, r.lomin, r.lomax).catch(() => {});
    fetchTrains().then((d) => { trainsCache = { at: Date.now(), data: d }; }).catch(() => {});
  };
  archiveTick();
  setInterval(archiveTick, 10 * 60_000).unref?.();

  app.get("/api/data/fires", (req, res) => {
    if (!firmsEnabled()) {
      return res.json({
        enabled: false,
        kind: "raw",
        reason: "FIRMS key not set (NASA_FIRMS_MAP_KEY or FIRMS_MAP_KEY) — free registration at firms.modaps.eosdis.nasa.gov (see research/wishlist.md)",
        fires: [],
      });
    }
    const hit = latestFirms();
    if (!hit) {
      return res.json({ enabled: true, kind: "raw", warming_up: true, count: 0, fires: [] });
    }
    res.json(buildFiresResponse(hit.detections, req.query.bbox, hit.at));
  });

  // USGS real-time earthquakes (RAW, M2.5+, global, rolling 24h) — free-data
  // pipeline build (EDGE DOCTRINE #1: build data, don't buy it). US
  // government public domain, keyless, no rate limit documented — boots
  // eagerly, no key gate needed (see server/usgsQuakes.ts). No map layer
  // yet — pipeline+API-first sequencing, same as sec8kEarnings/finraQuery
  // part 1 (client picks it up once archive history accumulates).
  bootQuakesPoll();
  app.get("/api/data/earthquakes", (_req, res) => {
    const hit = latestQuakes();
    if (!hit) {
      return res.json({ kind: "raw", source: "USGS Earthquake Hazards Program", warming_up: true, count: 0, quakes: [] });
    }
    res.set("Cache-Control", "public, max-age=120");
    res.json({
      kind: "raw",
      source: "USGS Earthquake Hazards Program (M2.5+, global, rolling 24h)",
      attribution: "U.S. Geological Survey",
      time: hit.at,
      count: hit.events.length,
      quakes: hit.events,
    });
  });

  // NOAA NDBC buoy/C-MAN latest observations (RAW, ~889 stations worldwide,
  // one row per station) — free-data pipeline build (EDGE DOCTRINE #1). US
  // government public domain, keyless, single global file — boots eagerly,
  // no key gate needed (see server/ndbcBuoys.ts). No map layer yet —
  // pipeline+API-first sequencing, same as usgsQuakes/sec8kEarnings (client
  // picks it up once archive history accumulates).
  bootBuoysPoll();
  app.get("/api/data/buoys", (_req, res) => {
    const hit = latestBuoys();
    if (!hit) {
      return res.json({ kind: "raw", source: "NOAA National Data Buoy Center", warming_up: true, count: 0, buoys: [] });
    }
    res.set("Cache-Control", "public, max-age=300");
    res.json({
      kind: "raw",
      source: "NOAA National Data Buoy Center (latest observations, all reporting stations)",
      attribution: "NOAA National Data Buoy Center",
      time: hit.at,
      count: hit.obs.length,
      buoys: hit.obs,
    });
  });

  // SEC 8-K Item 2.02 earnings-language (RAW as-filed exhibit display —
  // NEW DATA ROOTS #1 in research/open_questions.md: EDGAR is
  // public-record/free, no key required, same fair-access terms
  // edgarForm4.ts already relies on. ROOT VALIDATION
  // LADDER gate 1 (DATA) passed for extraction — see
  // server/sec8kEarnings.test.ts. Gate 2 (does guidance/results language
  // predict forward returns) is NOT attempted; this is a display of the
  // issuer's own as-filed press release, no predictive claim. HONEST GAP:
  // Q&A sessions are almost never filed as an exhibit — prepared remarks/
  // guidance language only. Boots eagerly (KNOWN BROKEN #9 lesson).
  bootEarnings8kPoll();
  app.get("/api/data/earnings-language", (_req, res) => {
    const hit = latestEarnings8Ks();
    if (!hit) {
      return res.json({ kind: "raw", source: "SEC EDGAR (8-K Item 2.02 / Exhibit 99)", warming_up: true, count: 0, filings: [] });
    }
    res.json({
      kind: "raw",
      source: "SEC EDGAR (8-K Item 2.02 / Exhibit 99) — sec.gov/cgi-bin/browse-edgar",
      time: hit.at,
      count: hit.filings.length,
      filings: hit.filings,
    });
  });

  // Accumulated earnings-language history from the archive (COLLECT-
  // EVERYTHING: the poll loop appends every extracted filing to disk; free
  // substitute for a paid transcript-history product per BUILD-FIRST rule
  // #2 — every day not archived is permanently lost).
  app.get("/api/data/earnings-language/history", (req, res) => {
    const days = Math.min(90, Math.max(1, parseInt(String(req.query.days || "30"), 10) || 30));
    try {
      const archived = readEarnings8kHistory(days);
      const live = latestEarnings8Ks()?.filings || [];
      const seen = new Set(archived.map((f) => f.accession));
      const merged = [...live.filter((f) => !seen.has(f.accession)), ...archived];
      res.json({
        kind: "raw",
        source: "SEC EDGAR (8-K Item 2.02 / Exhibit 99) — accumulated archive (began 2026-07-04)",
        days,
        count: merged.length,
        filings: merged,
      });
    } catch (e: any) {
      res.status(500).json({ error: e?.message || "history read failed" });
    }
  });

  // SEC 13F-HR institutional holdings (RAW as-filed display — stream #2 of
  // the DATA STREAM EXPANSION build order, directive 2026-07-05; hypothesis
  // + ladder path filed in research/open_questions.md BEFORE the first
  // pull). ROOT VALIDATION LADDER gate 1 (DATA) passed for the parser —
  // see server/edgar13f.test.ts. Gate 2 (new-position clustering vs forward
  // returns, 45-day lag modeled) NOT attempted; no predictive claim here.
  // Same EDGAR fair-access terms as Form 4. Boots eagerly (KNOWN BROKEN #9
  // lesson). Holdings in API responses are trimmed to top rows by value —
  // the archive keeps the full as-filed table.
  boot13FPoll();
  app.get("/api/data/filings13f", (_req, res) => {
    const hit = latest13FFilings();
    if (!hit) {
      return res.json({ kind: "raw", source: "SEC EDGAR (13F-HR)", warming_up: true, count: 0, focused_cap: FOCUSED_MAX_HOLDINGS, filings: [] });
    }
    res.json({
      kind: "raw",
      source: "SEC EDGAR (13F-HR) — sec.gov/cgi-bin/browse-edgar",
      time: hit.at,
      count: hit.filings.length,
      focused_cap: FOCUSED_MAX_HOLDINGS,
      filings: hit.filings.map((f) => trimHoldings(f)),
    });
  });
  app.get("/api/data/filings13f/history", (req, res) => {
    const days = Math.min(120, Math.max(1, parseInt(String(req.query.days || "30"), 10) || 30));
    try {
      const archived = read13FHistory(days);
      const live = (latest13FFilings()?.filings || []).map((f) => trimHoldings(f));
      const seen = new Set(archived.map((f) => f.accession));
      const merged = [...live.filter((f) => !seen.has(f.accession)), ...archived];
      res.json({
        kind: "raw",
        source: "SEC EDGAR (13F-HR) — accumulated archive (began 2026-07-05)",
        days,
        count: merged.length,
        focused_cap: FOCUSED_MAX_HOLDINGS,
        filings: merged,
      });
    } catch (e: any) {
      res.status(500).json({ error: e?.message || "history read failed" });
    }
  });

  // GDELT facility events (RAW display — stream #7 of the DATA STREAM
  // EXPANSION order): world-media unrest/strike event mentions inside
  // ~0.5° boxes around OUR strategic sites — the alert-trigger raw
  // material for burst -> own-sensor confirmation (gate-2). Geo is
  // city/ADM-approximate (stated); CAMEO can't see industrial accidents
  // (FIRMS is the fire sensor). Attribution required by license.
  bootGdeltPoll();
  app.get("/api/data/facility-events", (_req, res) => {
    const hit = latestGdeltEvents();
    if (!hit) {
      return res.json({ kind: "raw", source: "The GDELT Project", warming_up: true, count: 0, events: [] });
    }
    res.set("Cache-Control", "public, max-age=300");
    res.json({
      kind: "raw",
      source: "The GDELT Project (https://www.gdeltproject.org/) — unrest/strike mentions near tracked facilities",
      attribution: "The GDELT Project",
      time: hit.at,
      count: hit.events.length,
      note: "media event MENTIONS, geo approximate — verification prompts for our own sensors, not confirmations",
      events: hit.events,
    });
  });

  // USGS river gauges (RAW display — stream #6 of the DATA STREAM
  // EXPANSION order): 14 live-verified Mississippi/Ohio corridor gauges,
  // keyless, public domain. Raw stage/discharge readings only — the
  // low-water barge-stress SIGNAL is gate-2-locked. The /data map layer
  // (registry + icons + legend) ships as its own [PRODUCT] PR per the
  // legend same-PR rule. Boots eagerly.
  bootUsgsPoll();
  app.get("/api/data/rivergauges", (_req, res) => {
    const hit = latestGauges();
    if (!hit) {
      return res.json({ kind: "raw", source: "USGS NWIS", warming_up: true, count: 0, gauges: [] });
    }
    res.set("Cache-Control", "public, max-age=300");
    res.json({
      kind: "raw",
      source: "USGS Water Services (instantaneous values) — public domain",
      attribution: "Data courtesy U.S. Geological Survey",
      time: hit.at,
      count: hit.gauges.length,
      note: "raw stage/discharge readings; provisional values revise — the archive keeps every vintage",
      gauges: hit.gauges,
    });
  });

  // NWS active severe-weather alerts (RAW overlay — BUILD ORDER 2 #3):
  // official warnings displayed as-is with attribution, no predictive
  // claim. Zone-only alerts (no polygon) are COUNTED in the payload —
  // a visible cap, never silent. Boots eagerly.
  bootAlertsPoll();
  app.get("/api/data/alerts", (_req, res) => {
    const hit = latestAlerts();
    if (!hit) {
      return res.json({ kind: "raw", source: "NWS api.weather.gov", warming_up: true, count: 0, zone_only: 0, alerts: [] });
    }
    res.set("Cache-Control", "public, max-age=120");
    res.json({
      kind: "raw",
      source: "National Weather Service (api.weather.gov) — US government work, public domain",
      attribution: "National Weather Service",
      time: hit.at,
      count: hit.display.length,
      zone_only: hit.zoneOnly,
      note: "polygon-carrying alerts only; zone-coded alerts without geometry are counted in zone_only (resolution is a filed follow-up)",
      alerts: hit.display,
    });
  });

  // Treasury auction results (RAW — BUILD ORDER 2 #4): bid-to-cover +
  // bidder-class splits per auction, keyless TreasuryDirect TA_WS.
  // The tail-vs-WI stress metric is paid data — never faked here.
  bootTreasuryPoll();
  app.get("/api/data/treasury-auctions", (_req, res) => {
    const hit = latestAuctions();
    if (!hit) {
      return res.json({ kind: "raw", source: "TreasuryDirect TA_WS", warming_up: true, count: 0, auctions: [] });
    }
    res.set("Cache-Control", "public, max-age=1800");
    res.json({
      kind: "raw",
      source: "TreasuryDirect TA_WS (securities/auctioned) — US government work, public domain",
      attribution: "U.S. Department of the Treasury (TreasuryDirect)",
      time: hit.at,
      count: hit.auctions.length,
      note: "results-complete auctions from the last 30 days; dealer_take is DERIVED (pd accepted / competitive accepted); no tail metric — WI quotes are paid data",
      auctions: hit.auctions,
    });
  });

  // Corporate-fleet utilization (DERIVED, from our own archives — BUILD
  // ORDER 3 #1; gate 1 join accuracy PASSED 20/20, experiments.md
  // v1.0.120). Owners are FAA REGISTRANTS — trustee/leasing shells hide
  // beneficial owners; hours are lower bounds under adaptive sampling.
  app.get("/api/data/fleet-utilization", async (req, res) => {
    try {
      res.set("Cache-Control", "public, max-age=1800");
      const { series, as_of } = await fleetSeriesCached();
      const top = Math.min(parseInt(String(req.query.top || "50"), 10) || 50, 200);
      const kept = series.filter((s) => s.n_airframes >= 2).slice(0, top);
      res.json({
        kind: "derived",
        source: "voltradeai aircraft position archive x FAA entity spine",
        attribution: "FAA Aircraft Registry (identity); own archive (positions)",
        as_of,
        owners_total: series.length,
        count: kept.length,
        note: "owners are FAA registrants (trustee/leasing entities hide beneficial owners); airborne hours are LOWER BOUNDS under adaptive archive sampling; weeks without coverage are absent, not zero; single-airframe registrants excluded from this payload (top= caps at 200)",
        owners: kept,
      });
    } catch (e: any) { res.status(500).json({ error: e?.message }); }
  });

  // Everything Graph R1 (BUILD ORDER 3 #5, [PRODUCT]): per-site event
  // timeline composed from archives we already record (alerts, fires,
  // gauges, own traffic density). Raw overlay composition — each element
  // labeled at its source stream; no ladder gate (standing rule).
  app.get("/api/data/site-timeline/:siteId", async (req, res) => {
    try {
      const sites: SiteRef[] = ((datacoreSites as any).sites || [])
        .filter((s: any) => s.lat != null && s.lon != null)
        .map((s: any) => ({ id: s.id, name: s.name, lat: s.lat, lon: s.lon }));
      const wanted = sites.find((s) => s.id === req.params.siteId);
      if (!wanted) { res.status(404).json({ error: "unknown site" }); return; }
      res.set("Cache-Control", "public, max-age=1800");
      const { timelines, as_of } = await siteTimelineCached(sites);
      const t = timelines.get(wanted.id);
      res.json({
        kind: "raw",
        source: "own archives: NWS alerts + FIRMS fires + USGS gauges + position density",
        as_of,
        site: wanted.id,
        window_days: 7,
        note: "events capped at 12 (newest first); zone-only alerts have no geometry and are excluded; density = archived points within 50 km under adaptive sampling (near-site is full-resolution); absent days are absent, not zero",
        events: t?.events ?? [],
        density: t?.density ?? {},
      });
    } catch (e: any) { res.status(500).json({ error: e?.message }); }
  });

  // ANALYST CONSOLE W4: cross-layer geo-temporal query — everything the
  // archives know about a point+radius+window, per-layer provenance and
  // freshness. Archive reads only; no network at query time. RAW overlay
  // composition — each element labeled at its source stream, no ladder gate.
  app.get("/api/data/query", async (req, res) => {
    try {
      const lat = parseFloat(String(req.query.lat));
      const lon = parseFloat(String(req.query.lon));
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
        res.status(400).json({ error: "lat and lon are required numbers" }); return;
      }
      const radiusKm = req.query.radiusKm != null ? parseFloat(String(req.query.radiusKm)) : undefined;
      const days = req.query.days != null ? parseInt(String(req.query.days), 10) : undefined;
      const layers = req.query.layers
        ? String(req.query.layers).split(",").map((s) => s.trim()).filter(Boolean)
        : undefined;
      res.set("Cache-Control", "public, max-age=300");
      res.json({ kind: "raw", ...(await queryWindowCached({ lat, lon, radiusKm, days, layers })) });
    } catch (e: any) {
      res.status(String(e?.message).startsWith("invalid") ? 400 : 500).json({ error: e?.message });
    }
  });

  // ANALYST CONSOLE W6: LLM analyst tool-loop (key-gated on ANTHROPIC_API_KEY;
  // awaiting_key honesty like euLoad/censusImports — activates on key detect).
  // Session-required: LLM calls spend a real daily token budget, so anonymous
  // visitors can't burn it (_checkSession, same gate as the account routes).
  // v1 is single-question: { question: string }; multi-turn is a later PR.
  app.post("/api/analyst", async (req, res) => {
    const user = _checkSession(req);
    if (!user) return res.status(401).json({ error: "Authentication required" });
    const { status, body } = await analystResponse(req.body?.question);
    res.status(status).json(body);
  });

  // Census monthly port imports (RAW — BUILD ORDER 3 #4, key-gated like
  // fredMacro: awaiting_key honesty until CENSUS_API_KEY is set; activates
  // automatically on deploy with the key).
  bootCensusPoll();
  app.get("/api/data/imports", (_req, res) => {
    if (!censusEnabled()) {
      return res.json({ kind: "raw", enabled: false, reason: "CENSUS_API_KEY not set (free signup — see wishlist)", count: 0, imports: [] });
    }
    const hit = latestImports();
    if (!hit) {
      return res.json({ kind: "raw", source: "US Census Bureau", warming_up: true, count: 0, imports: [] });
    }
    res.set("Cache-Control", "public, max-age=3600");
    res.json({
      kind: "raw",
      source: "US Census Bureau intltrade (imports/porths, FT920 monthly) — public domain",
      attribution: "U.S. Census Bureau (USA Trade Online / FT920)",
      time: hit.at,
      count: hit.imports.length,
      note: "monthly port-level import values incl. containerized (~45-day lag); revisions append as new vintages in the archive",
      imports: hit.imports,
    });
  });

  // FINRA daily short-sale volume summary (RAW — BUILD ORDER 5 #1,
  // keyless). Serves the poller's cached day summary only — the full
  // ~12K-row day files live in the archive, never on the request path.
  // This is short-marked EXECUTION volume (a flow proxy), NOT short
  // interest — the label matters (manifest confidence_model).
  bootShortVolPoll();
  app.get("/api/data/short-volume", (_req, res) => {
    const hit = latestShortVol();
    if (!hit) {
      return res.json({ kind: "raw", source: "FINRA Reg SHO daily short sale volume", warming_up: true });
    }
    res.set("Cache-Control", "public, max-age=1800");
    res.json({
      kind: "raw",
      source: "FINRA Reg SHO daily short sale volume (CNMS consolidated file) — free with attribution",
      attribution: "FINRA Reg SHO daily short sale volume",
      time: hit.at,
      note: "daily short-marked execution volume (flow proxy — NOT short interest); top list floors total volume at " +
            `${hit.summary.floor_total_vol.toLocaleString()} shares and caps at ${hit.summary.top_cap} symbols (stated, not hidden)`,
      summary: hit.summary,
    });
  });

  // FINRA short-volume history (#/data/short-volume full view). Two modes,
  // both bounded reads (never the request-path-materializes-the-whole-
  // archive mistake — see finraShortVolume.ts's summary-history comment):
  // (1) no symbol — the tiny persisted market-wide trend log (one float/day,
  // separate from the 12K-row day archives); (2) ?symbol=X — reads up to
  // `days` (<=90, same cap convention as insider/earnings history) archived
  // day-files directly, one day materialized and discarded at a time, to
  // serve that ticker's ratio series from the 2024-06-17+ deep archive.
  app.get("/api/data/short-volume/history", (req, res) => {
    const days = Math.min(90, Math.max(1, parseInt(String(req.query.days || "30"), 10) || 30));
    const symbol = typeof req.query.symbol === "string" ? req.query.symbol.trim() : "";
    try {
      if (symbol) {
        const series = lookupSymbolHistory(symbol, days, undefined);
        return res.json({
          kind: "raw",
          source: "FINRA Reg SHO daily short sale volume (CNMS consolidated file) — free with attribution",
          symbol: symbol.toUpperCase(),
          days,
          count: series.length,
          note: series.length ? undefined : "no rows for this symbol in the archived window (delisted, not yet listed, or a typo)",
          series,
        });
      }
      res.json({
        kind: "raw",
        source: "FINRA Reg SHO daily short sale volume (CNMS consolidated file) — free with attribution",
        days,
        today: latestShortVol()?.summary ?? null,
        trend: readSummaryHistory(days, undefined),
        note: "market-wide agg_short_ratio trend; this log started accumulating 2026-07-06 — pass ?symbol=TICKER for that ticker's multi-year ratio series from the existing day-archive instead",
      });
    } catch (e: any) {
      res.status(500).json({ error: e?.message || "history read failed" });
    }
  });

  // FINRA Query API — settlement-stress datasets (DATACORE MAXIMUS
  // census build #4 part 1, contract live-verified 2026-07-07):
  // consolidatedShortInterest (semi-monthly per-symbol SHORT INTEREST —
  // positions, unlike the short-VOLUME flow above) + thresholdList
  // (daily Reg SHO threshold names). Cache-only request path.
  bootFinraQueryPoll();
  app.get("/api/data/short-interest", (_req, res) => {
    const hit = latestFinraSi();
    if (!hit) {
      return res.json({ kind: "raw", source: "FINRA Query API (consolidatedShortInterest + thresholdList)", warming_up: true });
    }
    res.set("Cache-Control", "public, max-age=3600");
    res.json({
      kind: "raw",
      source: "FINRA Query API — consolidated short interest (semi-monthly) + Reg SHO threshold list (daily); free with attribution",
      attribution: "FINRA Query API",
      time: hit.at,
      settlement_date: hit.si?.settlement_date ?? null,
      si_records: hit.si?.records ?? 0,
      note: "SHORT INTEREST (positions, ~T+9 semi-monthly publish) — distinct from /api/data/short-volume (daily flow). " +
            "Leaderboards floor ADV/position/previous-position (stated in payload) to keep near-zero-base artifacts out; " +
            "threshold list here is FINRA's OTC side only.",
      si: hit.si,
      threshold: hit.threshold,
    });
  });

  // FINRA Query API — ATS venue summaries (census build #4 part 2,
  // contract live-verified 2026-07-08): weeklySummary + monthlySummary
  // (per-symbol ATS/OTC volume leaderboards, *_SMBL rows only — see
  // finraQuery.ts header for the granularity-mixing rationale) +
  // blocksSummary (per-venue block-trading ranks). RAW, no predictive
  // claim; the settlement-stress composite hypothesis stays a queued
  // [RESEARCH] item until ladder gate 2. bootFinraQueryPoll() above
  // already polls this alongside part 1 — same 6h cadence, same cache
  // shape, cache-only request path.
  app.get("/api/data/ats-summary", (_req, res) => {
    const hit = latestFinraAts();
    if (!hit) {
      return res.json({ kind: "raw", source: "FINRA Query API (weeklySummary + monthlySummary + blocksSummary)", warming_up: true });
    }
    res.set("Cache-Control", "public, max-age=3600");
    res.json({
      kind: "raw",
      source: "FINRA Query API — ATS/OTC weekly + monthly per-symbol volume, ATS venue block-trading ranks; free with attribution",
      attribution: "FINRA Query API",
      time: hit.at,
      note: "Weekly/monthly leaderboards use only the per-symbol cross-firm rows (summaryTypeCode *_SMBL) — " +
            "composition[] on each summary states every granularity mixed in the source partition, so nothing " +
            "ranked here double-counts firm-level rows. tiers_covered states exactly which tiers fed the reading.",
      weekly: hit.weekly,
      monthly: hit.monthly,
      blocks: hit.blocks,
    });
  });

  // European macro cluster (census build #7): ECB + Eurostat +
  // Bundesbank regime features, keyless, per-series attribution
  // (all three licenses verified — commercial reuse w/ attribution).
  // Vintage-honest archive; cache-only request path.
  bootEuMacroPoll();
  app.get("/api/data/eu-macro", (_req, res) => {
    const hit = latestEuMacro();
    if (!hit) {
      return res.json({ kind: "raw", source: "ECB + Eurostat + Bundesbank (European macro cluster)", warming_up: true });
    }
    res.set("Cache-Control", "public, max-age=3600");
    res.json({
      kind: "raw",
      source: "European macro cluster — ECB Data Portal + Eurostat + Deutsche Bundesbank (keyless, free with attribution)",
      attribution: "per-series (each series carries its required attribution string)",
      time: hit.at,
      note: "REGIME INPUT feed (never a direct signal). Values as currently published; sources revise in place — " +
            "our archive keeps every vintage as-seen. Weekly ECB balance-sheet dates are the statement's Friday reference date.",
      series: hit.series,
    });
  });

  // SEC CNS fails-to-deliver (census build #6, public domain — the
  // resale-safe source). Half-month files, trailer-checksummed;
  // QUANTITY is a fail BALANCE (level), not a flow. Cache-only path.
  bootFtdPoll();
  app.get("/api/data/ftd", (_req, res) => {
    const hit = latestFtd();
    if (!hit) {
      return res.json({ kind: "raw", source: "SEC CNS fails-to-deliver (half-month files)", warming_up: true });
    }
    res.set("Cache-Control", "public, max-age=3600");
    res.json({
      kind: "raw",
      source: "SEC CNS fails-to-deliver, half-month files (public domain)",
      attribution: "U.S. Securities and Exchange Commission (CNS fails-to-deliver)",
      time: hit.at,
      note: "aggregate net fail BALANCES per settlement date (a level, not a daily flow); published on a 2.5-4.5 week lag; " +
            "top list covers the newest settlement date, floors quantity at " +
            `${hit.summary.qty_floor.toLocaleString()} shares (stated); raw FTD spikes alone are a crowded signal — ` +
            "this stream exists for the settlement-stress composite (see manifest)",
      summary: hit.summary,
    });
  });

  // CFTC Commitments of Traders, disaggregated futures-only (RAW —
  // BUILD ORDER 5 #2, keyless Socrata named-fields source). Weekly
  // report; serves the poller's cached week only (event-loop rule).
  bootCotPoll();
  app.get("/api/data/cot", (_req, res) => {
    const hit = latestCot();
    if (!hit) {
      return res.json({ kind: "raw", source: "CFTC Commitments of Traders (disaggregated)", warming_up: true });
    }
    res.set("Cache-Control", "public, max-age=3600");
    res.json({
      kind: "raw",
      source: "CFTC Commitments of Traders, disaggregated futures-only (public domain)",
      attribution: "CFTC Commitments of Traders (disaggregated)",
      time: hit.at,
      report_date: hit.report_date,
      count: hit.rows.length,
      note: "weekly positioning by trader category (Tuesday as-of, Friday publish); futures-ONLY report; positioning-extreme signals need trailing history the archive is only beginning to accumulate",
      markets: hit.rows,
    });
  });

  // CFTC Traders in Financial Futures, futures-only (RAW — BUILD
  // ORDER 6 #1, keyless Socrata sibling of /api/data/cot). Weekly
  // report; serves the poller's cached week only (event-loop rule).
  bootTffPoll();
  app.get("/api/data/tff", (_req, res) => {
    const hit = latestTff();
    if (!hit) {
      return res.json({ kind: "raw", source: "CFTC Traders in Financial Futures", warming_up: true });
    }
    res.set("Cache-Control", "public, max-age=3600");
    res.json({
      kind: "raw",
      source: "CFTC Traders in Financial Futures, futures-only (public domain)",
      attribution: "CFTC Traders in Financial Futures",
      time: hit.at,
      report_date: hit.report_date,
      count: hit.rows.length,
      note: "weekly financial-futures positioning by trader category (Tuesday as-of, Friday publish); futures-ONLY report; positioning-extreme signals need trailing history the archive is only beginning to accumulate",
      markets: hit.rows,
    });
  });

  // Treasury Daily Statement, deposits & withdrawals (RAW — BUILD
  // ORDER 6 #2, keyless FiscalData). Daily statement; serves the
  // poller's cached day only (event-loop rule).
  bootDtsPoll();
  app.get("/api/data/dts", (_req, res) => {
    const hit = latestDts();
    if (!hit) {
      return res.json({ kind: "raw", source: "U.S. Treasury Fiscal Data — Daily Treasury Statement", warming_up: true });
    }
    res.set("Cache-Control", "public, max-age=3600");
    res.json({
      kind: "raw",
      source: "U.S. Treasury Fiscal Data — Daily Treasury Statement, Table II (public domain)",
      attribution: "U.S. Treasury Fiscal Data — Daily Treasury Statement",
      time: hit.at,
      record_date: hit.record_date,
      count: hit.rows.length,
      note: "daily TGA deposits/withdrawals by category, $ millions as published (~1-2 business-day lag); the withheld-tax payroll-nowcast hypothesis is gate-locked until ladder validation",
      lines: hit.rows,
    });
  });

  // FDIC bank failures (RAW — BUILD ORDER 6 #3 v1, keyless event
  // stream). Serves the poller's cached window only (event-loop rule).
  bootFailuresPoll();
  app.get("/api/data/bank-failures", (_req, res) => {
    const hit = latestFailures();
    if (!hit) {
      return res.json({ kind: "raw", source: "FDIC Bank Data API", warming_up: true });
    }
    res.set("Cache-Control", "public, max-age=3600");
    res.json({
      kind: "raw",
      source: "FDIC Bank Data API — bank failures (US government data)",
      attribution: "FDIC Bank Data API",
      time: hit.at,
      count: hit.failures.length,
      note: "most recent US bank failures/assistance events, amounts in $ thousands as published; cost is the estimated DIF loss and is null until estimated; regional-bank-stress signals stay gate-locked until ladder validation",
      failures: hit.failures,
    });
  });

  // ENTSO-E actual total load by EU bidding zone (RAW — wishlist 9c,
  // key-gated on ENTSOE_API_KEY/ENTSOE_TOKEN, activated 2026-07-07).
  // Serves the poller's cached per-zone stats only (event-loop rule).
  bootEuLoadPoll();
  app.get("/api/data/eu-load", (_req, res) => {
    if (!euLoadEnabled()) {
      return res.json({ kind: "raw", enabled: false, reason: "ENTSOE_API_KEY not set (free token — see wishlist 9c)", count: 0, zones: [] });
    }
    const hit = latestLoad();
    if (!hit) {
      return res.json({ kind: "raw", source: "ENTSO-E Transparency Platform", warming_up: true, count: 0, zones: [] });
    }
    res.set("Cache-Control", "public, max-age=1800");
    res.json({
      kind: "raw",
      source: "ENTSO-E Transparency Platform (actual total load, A65/A16)",
      attribution: "ENTSO-E Transparency Platform",
      time: hit.at,
      count: hit.stats.length,
      note: "realised total load in MW per bidding zone, ~1-2h publication lag; stored at zone-native resolution, never resampled; zones absent from a cycle are absent, never zero-filled — their last sweep outcome is in `issues`; window min/max/mean expose series shape (some TSOs under-report or publish partial leading edges)",
      zones: hit.stats,
      issues: hit.issues,
    });
  });

  // Google Air Quality (RAW — strategic-site AQI/PM2.5/NO2 archive; key-gated
  // on GOOGLE_MAPS_API_KEY, activates on API-enable). Serves the poller's
  // cache only (event-loop rule); never fetches on request. Free-tier budget
  // guard in the module keeps polling under 150 calls/day.
  bootAirQualityPoll();
  app.get("/api/data/air-quality", (_req, res) => {
    if (!airQualityEnabled()) {
      return res.json({ kind: "raw", enabled: false, reason: "GOOGLE_MAPS_API_KEY not set", awaiting_key: true, count: 0, sites: [] });
    }
    const hit = latestAirQuality();
    if (!hit) {
      return res.json({ kind: "raw", source: "Google Air Quality API", attribution: "Air quality data \u00a9 Google", warming_up: true, count: 0, sites: [] });
    }
    res.set("Cache-Control", "public, max-age=1800");
    res.json({
      kind: "raw",
      source: "Google Maps Platform Air Quality API (current conditions)",
      attribution: "Air quality data \u00a9 Google",
      time: hit.at,
      awaiting_enable: hit.awaiting_enable,
      note: hit.awaiting_enable
        ? "Air Quality API is not yet enabled on the GCP project — enable it in the Google Cloud console; readings populate on the next poll"
        : "latest current-conditions reading per strategic site; universal + US EPA AQI, PM2.5/NO2; archived over time (Google exposes only 30d history)",
      budget: hit.budget,
      count: hit.readings.length,
      sites: hit.readings,
      issues: hit.issues,
    });
  });

  // CelesTrak GP satellite element sets (RAW overlay — ANALYST CONSOLE W2,
  // keyless; licensing verified 2026-07-07: freely available, courtesy
  // limits enforced in the module — 6h/group vs the 2h update cycle).
  // Serves the poller's cache only (event-loop rule); never fetches here.
  bootSatellitesPoll();
  app.get("/api/data/satellites", (req, res) => {
    const { status, body } = satellitesResponse(req.query.group);
    if (status === 200) res.set("Cache-Control", "public, max-age=1800");
    res.status(status).json(body);
  });

  // EIA-930 hourly grid demand (RAW — DATACORE MAXIMUS Phase 0,
  // key-gated on EIA_API_KEY). Serves the poller's cached
  // per-respondent stats only (event-loop rule).
  bootGridDemandPoll();
  app.get("/api/data/grid-demand", (_req, res) => {
    if (!gridDemandEnabled()) {
      return res.json({ kind: "raw", enabled: false, reason: "EIA_API_KEY not set (free signup — see wishlist)", count: 0, respondents: [] });
    }
    const hit = latestDemand();
    if (!hit) {
      return res.json({ kind: "raw", source: "EIA-930 Hourly Electric Grid Monitor", warming_up: true, count: 0, respondents: [] });
    }
    res.set("Cache-Control", "public, max-age=1800");
    res.json({
      kind: "raw",
      source: "EIA-930 Hourly Electric Grid Monitor (public domain)",
      attribution: "EIA-930 Hourly Electric Grid Monitor",
      time: hit.at,
      count: hit.stats.length,
      note: "hourly demand (MWh) for US48 + major balancing authorities, ~1-2h publication lag; latest_forecast_mwh is EIA's day-ahead forecast FOR THE SAME HOUR as latest_mwh (null when not yet published); the US48 forecast aggregate back-fills progressively as BAs report, so it can understate near the leading edge; industrial-activity nowcast signals stay gate-locked until ladder validation",
      respondents: hit.stats,
    });
  });

  // TX/ERCOT grid-stress reading — DESCRIPTIVE ONLY (GRID VISION A1 gate-2
  // FAIL path, research/grid_vision_products.md + experiments.md
  // 2026-07-07 v1.0.190). The v0 index was gate-2 tested against two
  // pre-stated outcome definitions and VOIDED on both (0/10 and insufficient
  // spot-validation hits respectively — see gate2_result.json). Per the
  // design's own locked FAIL path this ships as a labeled-non-predictive
  // descriptive surface, not a signal: no field here may be read as
  // predictive, tradable, or sellable. `predictive: false` on every
  // response, deliberately never omitted.
  bootGridStressPoll();
  app.get("/api/data/grid-stress", (_req, res) => {
    if (!gridStressEnabled()) {
      return res.json({ kind: "descriptive", enabled: false, predictive: false,
                         reason: "EIA_API_KEY not set (free signup — see wishlist)", reading: null });
    }
    const hit = latestGridStress();
    if (!hit) {
      return res.json({ kind: "descriptive", source: "VoltradeAI derived composite (EIA-930 ERCOT + NOAA CPC)",
                         predictive: false, warming_up: true, reading: null });
    }
    res.set("Cache-Control", "public, max-age=1800");
    res.json({
      kind: "descriptive",
      source: "VoltradeAI derived composite (EIA-930 ERCOT demand/forecast + NOAA CPC TX degree days)",
      attribution: "EIA-930 Hourly Electric Grid Monitor; NOAA Climate Prediction Center",
      predictive: false,
      validation_status: "GATE 2 NOT PASSED (2026-07-07) — two outcome designs (forecast-exceedance, "
        + "pooled-peak) were voided on their own pre-stated spot-validation rules before any PASS/FAIL "
        + "lift comparison was trusted; see research/experiments.md and datacore/gridvision/gate2_result.json",
      note: "Descriptive-only TX/ERCOT reading: three raw ingredients (same-month demand percentile, "
        + "day-ahead forecast strain, NOAA weather extremity) plus a plain EQUAL-WEIGHTED composite. "
        + "This is deliberately NOT the gate-2 script's fitted weights — those were fit against an "
        + "outcome variable that failed validation, so reusing them here would smuggle a voided claim "
        + "back in under a new name. Percentiles are withheld (null) rather than guessed when fewer than "
        + "5 same-calendar-month peer days exist in the archive yet.",
      time: hit.at,
      history_depth_days: hit.history_depth_days,
      reading: hit.reading,
    });
  });

  // OCC daily options volume by trade origin (RAW — DATACORE MAXIMUS
  // census #1, keyless, ARCHIVE-NOW: rolling 2-year source window).
  // Serves the poller's cached top-underlyings only (event-loop rule).
  bootOccPoll();
  app.get("/api/data/occ-volume", (_req, res) => {
    const hit = latestOcc();
    if (!hit) {
      return res.json({ kind: "raw", source: "OCC daily volume", warming_up: true, count: 0, top: [] });
    }
    res.set("Cache-Control", "public, max-age=3600");
    res.json({
      kind: "raw",
      source: "The Options Clearing Corporation (OCC) daily cleared volume",
      attribution: "The Options Clearing Corporation (OCC) daily volume",
      time: hit.at,
      report_date: hit.report_date,
      underlyings: hit.underlyings,
      count: hit.top.length,
      note: "top underlyings by cleared volume with customer/market-maker put-call splits (qty counts each clearing side; totals halved); source keeps only a rolling 2-year window — this archive is the durable copy; origin-split signals stay gate-locked until ladder validation",
      top: hit.top,
    });
  });

  // USDA NASS crop conditions (RAW — DATACORE MAXIMUS Phase 0b,
  // key-gated on NASS_API_KEY). Serves the poller's cached newest
  // week only (event-loop rule).
  bootCropConditionsPoll();
  app.get("/api/data/crop-conditions", (_req, res) => {
    if (!cropConditionsEnabled()) {
      return res.json({ kind: "raw", enabled: false, reason: "NASS_API_KEY not set (free signup — see wishlist)", count: 0, rows: [] });
    }
    const hit = latestConditions();
    if (!hit) {
      return res.json({ kind: "raw", source: "USDA NASS QuickStats", warming_up: true, count: 0, rows: [] });
    }
    res.set("Cache-Control", "public, max-age=3600");
    res.json({
      kind: "raw",
      source: "USDA NASS QuickStats — weekly crop condition ratings (US government data)",
      attribution: "USDA National Agricultural Statistics Service (QuickStats)",
      time: hit.at,
      latest_week: hit.latest_week,
      count: hit.rows.length,
      note: "national weekly CONDITION ratings for corn + soybeans (5 classes via short_desc, Monday releases in season); condition-delta signals stay gate-locked until ladder validation",
      rows: hit.rows,
    });
  });

  // NHTSA complaints watchlist (RAW — BUILD ORDER 6 #4, keyless,
  // curated-seed pattern). Serves the sweep's cached per-vehicle
  // stats only (event-loop rule).
  bootComplaintsPoll();
  app.get("/api/data/vehicle-complaints", (_req, res) => {
    const hit = latestComplaintStats();
    if (!hit) {
      return res.json({ kind: "raw", source: "NHTSA Office of Defects Investigation", warming_up: true });
    }
    res.set("Cache-Control", "public, max-age=3600");
    res.json({
      kind: "raw",
      source: "NHTSA ODI complaints API (US government data)",
      attribution: "NHTSA Office of Defects Investigation",
      time: hit.at,
      count: hit.stats.length,
      note: "per-vehicle complaint stats over a CURATED ticker-mapped watchlist (not the full vehicle universe); complaint-velocity signals stay gate-locked until ladder validation",
      vehicles: hit.stats,
    });
  });

  // Wikipedia pageviews attention proxy (RAW — BUILD ORDER 5 #3, the
  // pytrends replacement). Curated 23-ticker seed, raw daily views
  // only — no spike/z-score claims until gate 1 + trailing history.
  bootAttentionPoll();
  app.get("/api/data/attention", (_req, res) => {
    const hit = latestAttention();
    if (!hit) {
      // last_cycle makes a never-filling cache diagnosable from prod
      // without log access (R7: warming persisted across clean deploys)
      return res.json({ kind: "raw", source: "Wikimedia pageviews API", warming_up: true,
                        last_cycle: lastAttentionCycle() });
    }
    res.set("Cache-Control", "public, max-age=3600");
    res.json({
      kind: "raw",
      source: "Wikimedia pageviews API (en.wikipedia, all-access, agent=user)",
      attribution: "Wikimedia pageviews API",
      time: hit.at,
      date: hit.day.date,
      seed_size: Object.keys(WIKI_ARTICLES).length,
      count: hit.day.tickers.length,
      note: "RAW daily user pageviews for a curated ticker->article seed (stated size above); an attention PROXY, not a signal — no spike claims until the archive holds trailing history and gate 1 passes",
      tickers: hit.day.tickers,
    });
  });

  // FAA national airspace status (RAW — BUILD ORDER 5 #4, keyless).
  // Ground stops / GDPs / delays / closures; an empty list is a real
  // state (clear NAS), not an error.
  bootFaaPoll();
  app.get("/api/data/airport-status", (_req, res) => {
    const hit = latestFaaStatus();
    if (!hit) {
      return res.json({ kind: "raw", source: "FAA National Airspace System Status", warming_up: true });
    }
    res.set("Cache-Control", "public, max-age=300");
    res.json({
      kind: "raw",
      source: "FAA National Airspace System Status (public domain)",
      attribution: "FAA National Airspace System Status",
      time: hit.at,
      update_time: hit.update_time,
      count: hit.events.length,
      note: "rolling snapshot (15-min poll); durations as published (human-readable); an empty events list means a clear NAS, not missing data",
      events: hit.events,
    });
  });

  // CBP land-border wait times (RAW — BUILD ORDER 5 #5, keyless).
  // Status strings are localized by CBP's serving region and passed
  // through verbatim; numeric fields are locale-independent.
  bootBorderWaitPoll();
  app.get("/api/data/border-waits", (_req, res) => {
    const hit = latestBorderWaits();
    if (!hit) {
      return res.json({ kind: "raw", source: "CBP Border Wait Times", warming_up: true });
    }
    res.set("Cache-Control", "public, max-age=900");
    res.json({
      kind: "raw",
      source: "CBP Border Wait Times (public domain)",
      attribution: "CBP Border Wait Times",
      time: hit.at,
      count: hit.obs.length,
      note: "hourly snapshot flattened per lane class (commercial standard/FAST + passenger standard); status strings as published (localized by CBP's serving region) — join on port_number; null delay = not published, never zero",
      waits: hit.obs,
    });
  });

  // US Drought Monitor weekly severity (RAW — BUILD ORDER 2 #5):
  // CONUS + ag-belt states, cumulative D0-D4 + published DSCI.
  // Attribution required by the source and carried in every payload.
  bootDroughtPoll();
  app.get("/api/data/drought", (_req, res) => {
    const hit = latestDrought();
    if (!hit) {
      return res.json({ kind: "raw", source: "U.S. Drought Monitor", warming_up: true, count: 0, drought: [] });
    }
    res.set("Cache-Control", "public, max-age=3600");
    res.json({
      kind: "raw",
      source: "U.S. Drought Monitor data services — free with credit",
      attribution: "U.S. Drought Monitor (NDMC/USDA/NOAA)",
      time: hit.at,
      count: hit.drought.length,
      note: "weekly cumulative D0-D4 area %; dsci is the USDM's own published index (DERIVED here from the percentages)",
      drought: hit.drought,
    });
  });

  // FDA binary events (RAW display — stream #5 of the DATA STREAM
  // EXPANSION order): openFDA approvals (backward-looking confirmations)
  // + Federal Register advisory-committee meeting notices (forward-
  // looking catalysts). PDUFA dates are legally unavailable free — the
  // manifest carries the full honesty note; adcom meeting dates carry a
  // parse-confidence label and are never guessed. Boots eagerly.
  bootFdaPoll();
  app.get("/api/data/fda-events", (_req, res) => {
    const hit = latestFdaEvents();
    if (!hit) {
      return res.json({ kind: "raw", source: "openFDA + Federal Register", warming_up: true, count: 0, events: [] });
    }
    res.set("Cache-Control", "public, max-age=300");
    res.json({
      kind: "raw",
      source: "openFDA (drugsfda approvals) + Federal Register (FDA advisory-committee notices)",
      attribution: "openFDA (U.S. FDA); Federal Register (U.S. National Archives)",
      time: hit.at,
      count: hit.events.length,
      note: "No PDUFA target dates — legally unavailable free (21 CFR 314.430); adcom dates carry parse confidence",
      events: hit.events,
    });
  });

  // USAspending federal contract awards (RAW display — stream #4 of the
  // DATA STREAM EXPANSION order; hypothesis + verified research filed in
  // open_questions BEFORE the build). Keyless, US-gov data; UEI only
  // (DUNS is D&B-proprietary and never stored). Ticker matches are
  // precision-first (cache -> exact name -> award-detail parent), and
  // unmatched rows carry tkr:null — consumers skip them, never guess.
  // HONESTY: rt is the event date; DoD publishes ~90 days late (the
  // manifest carries the full note). Boots eagerly.
  bootContractsPoll();
  app.get("/api/data/contracts", (_req, res) => {
    const hit = latestContracts();
    if (!hit) {
      return res.json({ kind: "raw", source: "USAspending.gov", warming_up: true, count: 0, contracts: [] });
    }
    res.set("Cache-Control", "public, max-age=300");
    res.json({
      kind: "raw",
      source: "USAspending.gov, U.S. Department of the Treasury (contracts A-D, |amt| >= $25k)",
      attribution: "Source: USAspending.gov, U.S. Department of the Treasury",
      time: hit.at,
      count: hit.txns.length,
      note: "action dates are signature dates — DoD/USACE publish ~90 days late; rt is the as-seen date",
      contracts: hit.txns.slice(0, 500),
    });
  });

  // FRED macro series (RAW display — stream #3 of the DATA STREAM
  // EXPANSION build order; hypothesis filed BEFORE the build: this is the
  // REGIME INPUT feed, never a direct signal, never traded alone).
  // Key-gated: no-ops without FRED_API_KEY (set in Railway 2026-07-05).
  // LICENSING: the payload builder excludes license:"restricted" series
  // (CBOE VIX, ICE BofA, UMich) — third-party copyrighted data is archived
  // for internal regime use only and never reaches a product surface.
  // Point-in-time vintages live in the archive; this route shows current
  // published values with FRED attribution.
  bootFredPoll();
  app.get("/api/data/macro", (_req, res) => {
    if (!fredEnabled()) {
      return res.json({ kind: "raw", enabled: false, reason: "FRED_API_KEY not set", series: [] });
    }
    const payload = buildMacroPayload(latestFredSeries());
    if (!payload) {
      return res.json({ kind: "raw", source: "FRED API", warming_up: true, series: [] });
    }
    res.set("Cache-Control", "public, max-age=300");
    res.json(payload);
  });

  // Dark-ship RAW statistics — derived from OUR OWN AIS archive (shadow-fleet
  // directive 2026-07-04). Counts only: per-vessel claims are SIGNAL-class
  // and ladder-gated (validation plan in open_questions). 10-min cache — the
  // computation reads up to 72h of archive JSONL.
  // [REPAIR 2026-07-05] The scan used to run SYNCHRONOUSLY on the request
  // path — at current archive size it blocked the whole event loop 26-90s
  // (prod: 000/502 on cold hit; every other route starved meanwhile). Now:
  // async streaming scan on an eager 10-min poller; the route serves only
  // the cache and answers instantly, warming_up before the first pass.
  let shadowCache: { at: number; data: any } | null = null;
  let shadowRunning = false;
  const refreshShadowStats = async () => {
    if (shadowRunning) return;
    shadowRunning = true;
    try {
      const zones = (shadowZones as any).zones || [];
      const data = {
        kind: "raw",
        source: "Derived from our own AIS position archive (terrestrial coverage; began 2026-07-03)",
        zones: zones.map((z: any) => ({ id: z.id, name: z.name })),
        ...(await computeShadowStatsAsync(zones)),
      };
      shadowCache = { at: Date.now(), data };
    } catch (e: any) {
      console.error("[datacore] shadowstats refresh:", e?.message || e);
    } finally {
      shadowRunning = false;
    }
  };
  refreshShadowStats();
  setInterval(() => { refreshShadowStats(); }, 10 * 60_000).unref?.();
  app.get("/api/data/shadowstats", (_req, res) => {
    if (!shadowCache) {
      return res.json({ kind: "raw", warming_up: true, note: "first archive scan in progress — retry shortly" });
    }
    res.set("Cache-Control", "public, max-age=300");
    res.json(shadowCache.data);
  });

  // ── /api/v1 — the DATA PRODUCT surface (throughput/API directive
  // 2026-07-04). Pre-revenue scaffolding: env-seeded keys only, per-tier
  // rate limits, metering from day one, license marks on every response.
  // No billing, no pricing, no public issuance — the monetization
  // readiness checklist (wishlist.md) gates the last mile.
  const apiKeys = parseApiKeys();
  const apiLimiter = makeRateLimiter();
  const requireApiKey = (req: any, res: any): { key: string; tier: ApiTier } | null => {
    const key = String(req.headers["x-api-key"] || req.query.api_key || "");
    const info = apiKeys.get(key);
    if (!info) {
      res.status(401).json({ error: "invalid or missing API key", docs: "/developers", waitlist: true });
      return null;
    }
    const gate = apiLimiter.allow(key, info.tier);
    if (!gate.ok) {
      res.setHeader("retry-after", String(gate.retryAfterSec));
      res.status(429).json({ error: "rate limit", tier: info.tier, retry_after_sec: gate.retryAfterSec });
      try { meterUsage({ key, endpoint: req.path, status: 429, tier: info.tier }); } catch {}
      return null;
    }
    return { key, tier: info.tier };
  };
  const v1Envelope = (mark: keyof typeof LICENSE_MARKS, data: any) => ({
    api_version: "v1",
    ...LICENSE_MARKS[mark],
    disclaimer: "data as-is; not for safety-of-life use",
    data,
  });

  app.get("/api/v1/meta", (_req, res) => res.json(apiMeta())); // reference is public — docs, not data

  // Waitlist capture (/developers) — email only, no billing, "coming soon".
  // The monetization tripwire stays untripped: no pricing enablement here.
  app.post("/api/waitlist", (req, res) => {
    const result = addToWaitlist(String(req.body?.email || ""), String(req.body?.source || "developers"));
    if (result === "invalid") return res.status(400).json({ ok: false, error: "valid email required" });
    res.json({ ok: true, status: result });
  });

  app.get("/api/v1/tracks/:kind/:id", (req, res) => {
    const auth = requireApiKey(req, res);
    if (!auth) return;
    const kind = String(req.params.kind);
    if (!["aircraft", "vessels", "trains"].includes(kind)) {
      return res.status(400).json({ error: "kind must be aircraft|vessels|trains" });
    }
    try {
      const track = recentTrack(kind as any, String(req.params.id));
      const mark = (`tracks/${kind}`) as keyof typeof LICENSE_MARKS;
      res.json(v1Envelope(mark, { id: req.params.id, kind, points: track }));
      meterUsage({ key: auth.key, endpoint: `/api/v1/tracks/${kind}`, status: 200, tier: auth.tier });
    } catch (e: any) {
      res.status(500).json({ error: e?.message || "track read failed" });
      meterUsage({ key: auth.key, endpoint: `/api/v1/tracks/${kind}`, status: 500, tier: auth.tier });
    }
  });

  app.get("/api/v1/stats/portdwell", (req, res) => {
    const auth = requireApiKey(req, res);
    if (!auth) return;
    try {
      // [REPAIR 2026-07-05] was a per-request sync 168h scan (the
      // shadowstats event-loop defect) — serves the shared poller cache.
      if (!dwellCache) {
        res.status(503).set("Retry-After", "60").json({ error: "warming up — first archive scan in progress" });
        meterUsage({ key: auth.key, endpoint: "/api/v1/stats/portdwell", status: 503, tier: auth.tier });
        return;
      }
      res.json(v1Envelope("stats/portdwell", dwellCache.data));
      meterUsage({ key: auth.key, endpoint: "/api/v1/stats/portdwell", status: 200, tier: auth.tier });
    } catch (e: any) {
      res.status(500).json({ error: e?.message });
      meterUsage({ key: auth.key, endpoint: "/api/v1/stats/portdwell", status: 500, tier: auth.tier });
    }
  });

  app.get("/api/v1/stats/shadow", (req, res) => {
    const auth = requireApiKey(req, res);
    if (!auth) return;
    try {
      // [REPAIR 2026-07-05] same event-loop defect as /api/data/shadowstats:
      // this ran the synchronous 72h scan per request. Serves the shared
      // poller cache now; 503 + Retry-After while the first scan warms.
      if (!shadowCache) {
        res.status(503).set("Retry-After", "60").json({ error: "warming up — first archive scan in progress" });
        meterUsage({ key: auth.key, endpoint: "/api/v1/stats/shadow", status: 503, tier: auth.tier });
        return;
      }
      res.json(v1Envelope("stats/shadow", shadowCache.data));
      meterUsage({ key: auth.key, endpoint: "/api/v1/stats/shadow", status: 200, tier: auth.tier });
    } catch (e: any) {
      res.status(500).json({ error: e?.message });
      meterUsage({ key: auth.key, endpoint: "/api/v1/stats/shadow", status: 500, tier: auth.tier });
    }
  });

  app.get("/api/v1/stats/archive", (req, res) => {
    const auth = requireApiKey(req, res);
    if (!auth) return;
    try {
      res.json(v1Envelope("stats/archive", archiveStats()));
      meterUsage({ key: auth.key, endpoint: "/api/v1/stats/archive", status: 200, tier: auth.tier });
    } catch (e: any) {
      res.status(500).json({ error: e?.message });
      meterUsage({ key: auth.key, endpoint: "/api/v1/stats/archive", status: 500, tier: auth.tier });
    }
  });

  // Port dwell analytics (RAW) — arrival/departure detection + dwell
  // distributions per imagery-verified port geofence, from OUR OWN AIS
  // archive (fusion directive 2026-07-04). Anomaly FLAGS are 3x-median,
  // suppressed on thin history; the dwell-anomaly SIGNAL stays ladder-gated.
  // 10-min cache — the computation reads up to 7d of archive JSONL.
  // [REPAIR 2026-07-05] same event-loop defect as shadowstats (fourth
  // site, heavier 168h window): sync scan per cache miss. Eager 10-min
  // poller now; the route serves only the cache.
  let dwellCache: { at: number; data: any } | null = null;
  let dwellRunning = false;
  const refreshPortDwell = async () => {
    if (dwellRunning) return;
    dwellRunning = true;
    try {
      const ports = portsFromSites((datacoreSites as any).sites || []);
      const data = {
        kind: "raw",
        source: "Derived from our own AIS position archive (terrestrial coverage; began 2026-07-03)",
        ...(await computePortDwellAsync(ports)),
      };
      dwellCache = { at: Date.now(), data };
    } catch (e: any) {
      console.error("[datacore] portdwell refresh:", e?.message || e);
    } finally {
      dwellRunning = false;
    }
  };
  refreshPortDwell();
  setInterval(() => { refreshPortDwell(); }, 10 * 60_000).unref?.();
  app.get("/api/data/portdwell", (_req, res) => {
    if (!dwellCache) {
      return res.json({ kind: "raw", warming_up: true, note: "first archive scan in progress — retry shortly" });
    }
    res.set("Cache-Control", "public, max-age=300");
    res.json(dwellCache.data);
  });

  // Everything Graph v1 (datacore/EVERYTHING_GRAPH.md, build step 2) — joins
  // Form 4 insiders, the entity_map operator->ticker table, and our own
  // port-dwell/AIS archive into one node/edge graph. RAW (asserts filed
  // relationships with provenance; no predictive claim). Same eager-poller
  // shape as portdwell/shadowstats above — the graph rebuild reads a 168h
  // AIS fold, so it must never run synchronously per-request (the exact
  // event-loop/OOM class those two routes were fixed for).
  let graphCache: { at: number; data: any } | null = null;
  let graphRunning = false;
  const refreshGraph = async () => {
    if (graphRunning) return;
    graphRunning = true;
    try {
      graphCache = { at: Date.now(), data: await buildGraph() };
    } catch (e: any) {
      console.error("[datacore] entityGraph refresh:", e?.message || e);
    } finally {
      graphRunning = false;
    }
  };
  refreshGraph();
  setInterval(() => { refreshGraph(); }, 15 * 60_000).unref?.();
  app.get("/api/data/graph", (req, res) => {
    if (!graphCache) {
      return res.json({ kind: "raw", warming_up: true, note: "first graph build in progress — retry shortly" });
    }
    const graph = graphCache.data;
    res.set("Cache-Control", "public, max-age=300");
    const entity = typeof req.query.entity === "string" ? req.query.entity : null;
    if (!entity) {
      // No entity requested: counts only (the full graph is tens of
      // thousands of facility nodes — never dump it whole by default).
      return res.json({ kind: "raw", built_at: graph.built_at, counts: graph.counts, caveat: graph.caveat,
        note: "pass ?entity=<ticker|MMSI|CIK|facility id>&hops=1 for a neighborhood query" });
    }
    const id = resolveEntityId(graph, entity);
    if (!id) return res.status(404).json({ kind: "raw", error: `entity not found: ${entity}` });
    const hops = Math.max(0, Math.min(3, parseInt(String(req.query.hops ?? "1"), 10) || 1));
    res.json({ kind: "raw", built_at: graph.built_at, entity: id, hops, caveat: graph.caveat, ...neighborhood(graph, id, hops) });
  });

  // ── OpenWeatherMap global weather fields (Tier-1(b) global half) ─────────
  // Key stays server-side; tiles proxied + cached (free tier is 60 calls/min
  // — the shared cache bounds upstream to unique-tiles-per-TTL across all
  // visitors). Fresh-key honesty: OWM 401s until a new key activates (~2h),
  // classified "activating" with a retry note, never a hard error.
  const wxTileCache = makeTileCache<Buffer>(2000);
  let wxNegativeUntil = 0; // don't hammer OWM while the key activates
  let wxStatusCache: { at: number; data: any } | null = null;

  app.get("/api/data/weather/global/status", async (_req, res) => {
    if (wxStatusCache && Date.now() - wxStatusCache.at < 5 * 60_000) return res.json(wxStatusCache.data);
    const key = process.env.OPENWEATHERMAP_KEY || "";
    let http: number | null = null;
    if (key) {
      try {
        const r = await fetch(owmTileUrl("temp_new", 0, 0, 0, key));
        http = r.status;
      } catch { http = null; }
    }
    const status = classifyOwmStatus(http, !!key);
    const data = { status, note: owmStatusNote(status), attribution: "Weather data © OpenWeatherMap" };
    wxStatusCache = { at: Date.now(), data };
    res.json(data);
  });

  // Sampled weather grid (wind vectors + temperature labels). One shared
  // upstream burst per snapped-viewport bucket per 10-min TTL; minute guard
  // keeps the OWM free budget safe; stale beats spinner on guard trips.
  const wxGridCache = new Map<string, { at: number; data: any }>();
  let wxGridMinute = 0, wxGridMinuteCalls = 0;
  app.get("/api/data/weather/grid", async (req, res) => {
    const key = process.env.OPENWEATHERMAP_KEY || "";
    if (!key) return res.status(503).json({ status: "awaiting_key", note: owmStatusNote("awaiting_key") });
    const parts = String(req.query.bbox || "").split(",").map(Number);
    if (parts.length !== 4 || parts.some((v) => !Number.isFinite(v))) {
      return res.status(400).json({ error: "bbox=s,w,n,e required" });
    }
    const b = snapBbox(parts[0], parts[1], parts[2], parts[3]);
    const ck = bboxKey(b);
    const hit = wxGridCache.get(ck);
    if (hit && Date.now() - hit.at < GRID_TTL_MS) return res.json(hit.data);
    const minute = Math.floor(Date.now() / 60_000);
    if (minute !== wxGridMinute) { wxGridMinute = minute; wxGridMinuteCalls = 0; }
    const pts = gridPoints(b);
    if (wxGridMinuteCalls + pts.length > UPSTREAM_PER_MIN_GUARD) {
      if (hit) return res.json({ ...hit.data, stale: true }); // stale beats spinner
      return res.status(503).json({ status: "budget", note: "shared upstream budget exhausted this minute — retry shortly" });
    }
    wxGridMinuteCalls += pts.length;
    try {
      const samples: WeatherSample[] = [];
      const CONC = 8;
      for (let i = 0; i < pts.length; i += CONC) {
        const chunk = await Promise.all(pts.slice(i, i + CONC).map(async (p) => {
          try {
            const r = await fetch(owmPointUrl(p.la, p.lo, key));
            if (!r.ok) return null;
            return parseOwmPoint(p.la, p.lo, await r.json());
          } catch { return null; }
        }));
        for (const s of chunk) if (s) samples.push(s);
      }
      const data = {
        kind: "raw",
        source: "OpenWeatherMap current-weather point samples (Weather data © OpenWeatherMap)",
        note: `sampled grid — one observation per ~${spacingKm(b, pts.length)} km; arrows/labels never denser than the data`,
        spacing_km: spacingKm(b, pts.length),
        sampled: samples.length,
        points: samples,
      };
      if (wxGridCache.size > 200) wxGridCache.delete(wxGridCache.keys().next().value as string);
      wxGridCache.set(ck, { at: Date.now(), data });
      res.json(data);
    } catch (e: any) {
      res.status(502).json({ status: "error", note: e?.message || "grid fetch failed" });
    }
  });

  app.get("/api/data/wxtile/:layer/:z/:x/:y", async (req, res) => {
    const key = process.env.OPENWEATHERMAP_KEY || "";
    if (!key) return res.status(503).json({ status: "awaiting_key", note: owmStatusNote("awaiting_key") });
    const z = Number(req.params.z), x = Number(req.params.x), y = Number(req.params.y);
    const layer = req.params.layer;
    if (!validateWxTile(layer, z, x, y)) return res.status(400).json({ error: "bad tile request" });
    const ck = `${layer}/${z}/${x}/${y}`;
    const hit = wxTileCache.get(ck);
    if (hit) {
      res.setHeader("content-type", "image/png");
      res.setHeader("cache-control", "public, max-age=600");
      return res.end(hit);
    }
    if (Date.now() < wxNegativeUntil) {
      return res.status(503).json({ status: "activating", note: owmStatusNote("activating") });
    }
    try {
      const r = await fetch(owmTileUrl(layer as WxLayer, z, x, y, key));
      if (r.status === 401 || r.status === 403) {
        wxNegativeUntil = Date.now() + NEGATIVE_TTL_MS;
        wxStatusCache = null; // let the next status probe re-check
        return res.status(503).json({ status: "activating", note: owmStatusNote("activating") });
      }
      if (!r.ok) return res.status(502).json({ status: "error", note: owmStatusNote("error") });
      let buf = Buffer.from(await r.arrayBuffer());
      // v1.0.69: amplify the near-invisible 1.0-palette tiles once, here,
      // then cache the display-ready result (root cause in owmTiles.ts).
      try { buf = amplifyWeatherTile(buf, layer as WxLayer); } catch {}
      wxTileCache.set(ck, buf, TILE_TTL_MS);
      res.setHeader("content-type", "image/png");
      res.setHeader("cache-control", "public, max-age=600");
      res.end(buf);
    } catch {
      res.status(502).json({ status: "error", note: owmStatusNote("error") });
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

