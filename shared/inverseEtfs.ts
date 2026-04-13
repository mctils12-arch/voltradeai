/**
 * Inverse ETFs: held "long" in Alpaca but represent bearish/short bets.
 * When the bot buys these, the position direction should display as "short"
 * because the user is effectively betting against the underlying index.
 *
 * Includes 1x, 2x, and 3x inverse ETFs across major indices and sectors.
 */
export const INVERSE_ETFS = new Set([
  // ── Broad market inverse ──────────────────────────────────────
  "SH",     // ProShares Short S&P 500 (1x)
  "SDS",    // ProShares UltraShort S&P 500 (2x)
  "SPXU",   // ProShares UltraPro Short S&P 500 (3x)
  "SPXS",   // Direxion Daily S&P 500 Bear 3x
  "SPDN",   // Direxion Daily S&P 500 Bear 1x

  // ── Nasdaq inverse ────────────────────────────────────────────
  "PSQ",    // ProShares Short QQQ (1x)
  "QID",    // ProShares UltraShort QQQ (2x)
  "SQQQ",   // ProShares UltraPro Short QQQ (3x)

  // ── Dow Jones inverse ─────────────────────────────────────────
  "DOG",    // ProShares Short Dow 30 (1x)
  "DXD",    // ProShares UltraShort Dow 30 (2x)
  "SDOW",   // ProShares UltraPro Short Dow 30 (3x)

  // ── Russell 2000 inverse ──────────────────────────────────────
  "RWM",    // ProShares Short Russell 2000 (1x)
  "TWM",    // ProShares UltraShort Russell 2000 (2x)
  "SRTY",   // ProShares UltraPro Short Russell 2000 (3x)
  "TZA",    // Direxion Daily Small Cap Bear 3x

  // ── Sector inverse ────────────────────────────────────────────
  "SOXS",   // Direxion Daily Semiconductor Bear 3x
  "TECS",   // Direxion Daily Technology Bear 3x
  "FAZ",    // Direxion Daily Financial Bear 3x
  "SKF",    // ProShares UltraShort Financials (2x)
  "LABD",   // Direxion Daily S&P Biotech Bear 3x
  "ERY",    // Direxion Daily Energy Bear 2x
  "DRIP",   // Direxion Daily S&P Oil & Gas Bear 2x
  "DUST",   // Direxion Daily Gold Miners Bear 2x
  "JDST",   // Direxion Daily Junior Gold Miners Bear 2x
  "YANG",   // Direxion Daily FTSE China Bear 3x
  "EDZ",    // Direxion Daily Emerging Markets Bear 3x
  "WEBS",   // Direxion Daily Dow Jones Internet Bear 3x

  // ── Volatility inverse (short vol = bearish on volatility) ────
  "SVXY",   // ProShares Short VIX Short-Term Futures
]);

/**
 * Returns the display side for a position. If the ticker is an inverse ETF
 * and the raw side is "long", the display side is "short" because the user
 * is effectively betting against the underlying index.
 */
export function getDisplaySide(ticker: string, rawSide: string): string {
  if (INVERSE_ETFS.has(ticker) && rawSide === "long") {
    return "short";
  }
  return rawSide;
}
