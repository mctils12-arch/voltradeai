// Terms of Service — W6 hardening sprint (2026-07-18).
//
// Canonical ToS content lives HERE as structured data (one source of truth).
// Served two ways, both mounted in index.ts BEFORE routes/static:
//   • GET /api/legal/terms  → JSON (for a future React /terms page to consume)
//   • GET /terms            → a self-contained, theme-aware HTML page
//                             (works standalone so the client footer can link
//                              straight to /terms today; the footer wiring is
//                              the client session's job)
//
// Plain, readable language. Prohibits scraping / automated harvesting / bulk
// export / redistribution of platform data; data provided "as is" for
// informational use; credits upstream open-data sources; reserves the right
// to suspend violators.

import type { Request, Response, Express } from "express";

export const TERMS_EFFECTIVE_DATE = "2026-07-18";

export interface TermsSection {
  heading: string;
  body: string[];
}

export const TERMS_SECTIONS: TermsSection[] = [
  {
    heading: "1. Acceptance",
    body: [
      "By accessing VolTradeAI (the \"Platform\"), including its website, data maps, and APIs, you agree to these Terms of Service. If you do not agree, do not use the Platform.",
    ],
  },
  {
    heading: "2. Informational use only — not financial advice",
    body: [
      "The Platform aggregates public and derived data about the physical and financial economy and presents analysis, signals, and visualizations. All content is provided for general informational purposes only. Nothing on the Platform is investment, legal, tax, or other professional advice, and nothing is an offer or solicitation to buy or sell any security or instrument.",
      "Trading activity shown on the Platform is executed against a paper (simulated) brokerage account. Past or simulated performance does not predict future results.",
    ],
  },
  {
    heading: "3. Data provided \"as is\"",
    body: [
      "All data, signals, estimates, and derived readings are provided \"as is\" and \"as available\", without warranties of any kind, express or implied, including accuracy, completeness, timeliness, fitness for a particular purpose, or non-infringement. Estimated or predicted values are labeled as such and are not ground truth. You are solely responsible for any decisions you make using the Platform.",
    ],
  },
  {
    heading: "4. Prohibited use — no scraping, harvesting, or bulk export",
    body: [
      "You may view and interact with the Platform through its normal user interface. You may NOT, without our prior written permission:",
      "• scrape, crawl, spider, or use any automated system or bot to access, index, or harvest data from the Platform or its APIs;",
      "• bulk-download, extract, or export the Platform's data, tiles, or catalogues, whether manually or by automated means;",
      "• redistribute, resell, sublicense, or publicly republish Platform data or derived signals, in whole or in part;",
      "• circumvent, disable, or interfere with rate limits, access controls, honeypots, or other technical protection measures;",
      "• probe, scan, or test the vulnerability of the Platform, or breach its security or authentication.",
      "Programmatic access is available only through officially issued API keys, subject to their own plan limits and terms.",
    ],
  },
  {
    heading: "5. Upstream data sources & attribution",
    body: [
      "The Platform incorporates data from open and public sources, which remain subject to their own licenses and terms. These include, among others: OpenStreetMap (© OpenStreetMap contributors, ODbL); Natural Earth (public domain); U.S. NOAA / National Weather Service; U.S. Geological Survey (USGS); NASA and Copernicus / Sentinel Earth-observation programs; GEBCO bathymetry; CelesTrak orbital data; adsb.lol aircraft data; and various U.S. government open-data endpoints (SEC, FINRA, CFTC, FEMA, and others).",
      "Attribution to these upstream sources is preserved on the Platform. Your use of any upstream data must comply with that source's own license. We claim no ownership over upstream open data; we do assert rights in our compilation, processing, derived signals, and presentation.",
    ],
  },
  {
    heading: "6. Accounts",
    body: [
      "You are responsible for safeguarding your account credentials and for all activity under your account. Notify us promptly of any unauthorized use. Repeated failed authentication attempts may be rate-limited or temporarily locked for security.",
    ],
  },
  {
    heading: "7. Suspension & enforcement",
    body: [
      "We may throttle, suspend, or permanently block access — with or without notice — for any account, IP address, or client that we reasonably believe is scraping, harvesting, redistributing data, abusing the service, or otherwise violating these Terms. We may also pursue any other remedies available at law.",
    ],
  },
  {
    heading: "8. Limitation of liability",
    body: [
      "To the maximum extent permitted by law, VolTradeAI and its operators are not liable for any indirect, incidental, special, consequential, or punitive damages, or for any loss of profits, data, or trading losses, arising from your use of or inability to use the Platform.",
    ],
  },
  {
    heading: "9. Changes",
    body: [
      "We may update these Terms from time to time. Continued use of the Platform after changes take effect constitutes acceptance of the revised Terms. The effective date below reflects the current version.",
    ],
  },
];

export function termsAsJson() {
  return {
    title: "Terms of Service",
    effectiveDate: TERMS_EFFECTIVE_DATE,
    sections: TERMS_SECTIONS,
  };
}

export function termsAsText(): string {
  const out: string[] = ["VolTradeAI — Terms of Service", `Effective: ${TERMS_EFFECTIVE_DATE}`, ""];
  for (const s of TERMS_SECTIONS) {
    out.push(s.heading);
    for (const b of s.body) out.push(b);
    out.push("");
  }
  return out.join("\n");
}

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export function termsAsHtml(): string {
  const sections = TERMS_SECTIONS.map(
    (s) =>
      `<section><h2>${esc(s.heading)}</h2>${s.body.map((b) => `<p>${esc(b)}</p>`).join("")}</section>`,
  ).join("\n");
  return `<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Terms of Service — VolTradeAI</title>
<style>
  :root { color-scheme: light dark; }
  body { margin: 0; background: #0b0e14; color: #e6e9ef; font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  @media (prefers-color-scheme: light) { body { background: #f7f8fa; color: #1a1d24; } }
  main { max-width: 760px; margin: 0 auto; padding: 48px 20px 96px; }
  h1 { font-size: 28px; letter-spacing: -0.01em; margin: 0 0 4px; }
  .eff { opacity: 0.65; font-size: 14px; margin: 0 0 32px; }
  h2 { font-size: 18px; margin: 32px 0 8px; }
  p { margin: 0 0 12px; }
  a { color: #5b9dff; }
  footer { margin-top: 48px; opacity: 0.6; font-size: 13px; }
</style>
</head><body>
<main>
  <h1>Terms of Service</h1>
  <p class="eff">Effective ${esc(TERMS_EFFECTIVE_DATE)}</p>
  ${sections}
  <footer>VolTradeAI · Data provided for informational use only · <a href="/">Back to site</a></footer>
</main>
</body></html>`;
}

export function registerTerms(app: Express) {
  app.get("/api/legal/terms", (_req: Request, res: Response) => {
    res.json(termsAsJson());
  });
  app.get("/terms", (_req: Request, res: Response) => {
    res.type("html").send(termsAsHtml());
  });
}
