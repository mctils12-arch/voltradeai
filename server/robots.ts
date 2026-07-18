// robots.txt — W6 hardening sprint (2026-07-18).
//
// Served from the Node layer (mounted in index.ts BEFORE routes / static so
// nothing shadows it). Disallows the entire /api/ surface and every honeypot
// trap. Well-behaved crawlers stay out of the data API; scrapers that ignore
// this and hit the Disallowed honeypots self-identify to antiScraping.ts.

import type { Request, Response, Express } from "express";
import { HONEYPOT_PATHS } from "./antiScraping";

export function buildRobotsTxt(): string {
  const lines = [
    "User-agent: *",
    "Disallow: /api/",
    "Disallow: /tiles/",
  ];
  for (const h of HONEYPOT_PATHS) lines.push(`Disallow: ${h}`);
  lines.push("");
  lines.push("# The public site is crawlable; the data API and internal");
  lines.push("# endpoints are not. Automated harvesting of platform data is");
  lines.push("# prohibited by the Terms of Service (/terms).");
  lines.push("");
  return lines.join("\n");
}

export function registerRobots(app: Express) {
  app.get("/robots.txt", (_req: Request, res: Response) => {
    res.type("text/plain").send(buildRobotsTxt());
  });
}
