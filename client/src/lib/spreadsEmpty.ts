// Empty state for the Analyze page's "Top Ranked Spreads" section.
//
// analyze.py only builds spreads from contracts with a LIVE two-sided quote
// (bid > 0 and ask > 0). Outside market hours, and for thinly-traded or
// option-less stocks, that leaves nothing to rank — and the section used to
// silently disappear, which reads as "the feature is broken". Instead the
// page says what happened and why, and never ranks stale/one-sided prices.

/** hasLiveQuotes: the response carried a vol surface (at least one expiry
 *  had an at-the-money contract with a live bid and ask). */
export function spreadsEmptyMessage(ticker: string, hasLiveQuotes: boolean): { title: string; body: string } {
  const t = (ticker || "this stock").toUpperCase();
  if (!hasLiveQuotes) {
    return {
      title: "No spreads to rank right now",
      body: `No ${t} option contracts have a live bid and ask at the moment. Options only quote both sides `
        + `during market hours (9:30 a.m.–4:00 p.m. ET, weekdays), and stocks with thin or no listed options `
        + `may not have them at all. Spreads are ranked from live quotes only — never from stale or one-sided `
        + `prices — so try again during market hours.`,
    };
  }
  return {
    title: "No spreads to rank right now",
    body: `${t} has live option quotes, but none of the scanned expirations had strikes that form a spread `
      + `the ranker can price (it needs a live bid and ask on every leg). Try again later in the session, `
      + `when more strikes are quoted.`,
  };
}
