// The boot RULES banner's state-dependent lines ([REPAIR] 2026-11-06).
//
// Every boot writes a RULES banner to the audit log, which sessions, the
// watchdog and the improve routines all read. Two of its lines were fixed
// strings that had stopped being true:
// - "Kill switch: OFF (manual emergency stop available)" was written on every
//   boot, including all 39 boots of 2026-11-06 while the drawdown-kill latch
//   restored from disk was ON (the bot booted halted each time).
// - the satellite-exposure line said the cap excludes the "QQQ floor"; the
//   floor has been the volatility-managed SPY/SSO floor since 2026-11-05.
// The banner now reads the kill switch's live state, the configured
// drawdown threshold and the live score threshold. Nothing here decides
// anything; it only describes.

export function killSwitchBannerLine(on: boolean, reason: string | null | undefined, maxDrawdownPct: number): string {
  if (on) {
    const why = reason && reason.trim() ? `: ${reason.trim().slice(0, 160)}` : "";
    return `Kill switch: ON (latched${why}) — no new entries; exits and the floor's reduce-only pass still run; the paper auto-resume check or the owner clears it`;
  }
  return `Kill switch: OFF (armed: trips at ${maxDrawdownPct}% from the equity peak; manual stop available)`;
}

export function satelliteExposureBannerLine(maxTotalExposure: number): string {
  return `Max active satellite exposure: ${Math.round(maxTotalExposure * 100)}% (the floor + VRP excluded) | Total portfolio cap: 95% (regime-adaptive)`;
}

export function minScoreBannerLine(minScoreThreshold: number): string {
  return `Minimum score to trade: ${minScoreThreshold}/100 combined edge score`;
}
