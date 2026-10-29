// Ticker dossier — the Node half: symbol validation, the subprocess command,
// the Form 4 archive summary, the cache, and the bot.ts / daemon wiring.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseDossierSymbol, dossierSubprocessCmd, insiderSummary, DossierCache } from "./tickerDossier";
import type { Form4Filing } from "./edgarForm4";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const read = (rel: string) => fs.readFileSync(path.join(HERE, "..", rel), "utf8");

test("symbols: letters only reach a command line; an OCC code resolves to its underlying", () => {
  assert.equal(parseDossierSymbol("rklb"), "RKLB");
  assert.equal(parseDossierSymbol(" RKLB261120C00030000 "), "RKLB");
  for (const bad of ["", "RK-LB", "RKLB;rm", "TOOLONGX", "BRK.B", "12", null, undefined]) assert.equal(parseDossierSymbol(bad), null, String(bad));
  assert.match(dossierSubprocessCmd("RKLB"), /build\(sys\.argv\[1\]\).* RKLB$/);
  assert.throws(() => dossierSubprocessCmd("RKLB; rm -rf /"));
});

function f4(sym: string, filedAt: string, code: "P" | "S" | "A", shares: number): Form4Filing {
  return {
    issuerCik: "1", issuerName: "X", issuerTradingSymbol: sym, periodOfReport: filedAt.slice(0, 10), accession: `${sym}-${filedAt}-${code}`, filedAt, cik: "1", indexUrl: "https://www.sec.gov/x",
    owners: [{ cik: "2", name: "Jane Doe", isDirector: false, isOfficer: true, isTenPercentOwner: false, officerTitle: "CFO" }],
    transactions: [{ table: "nonDerivative", securityTitle: "Common", transactionDate: filedAt.slice(0, 10), transactionCode: code, kind: code === "P" ? "open_market_buy" : code === "S" ? "open_market_sale" : "award_grant", shares, pricePerShare: 25, acquiredDisposedCode: code === "S" ? "D" : "A", sharesOwnedAfter: null }],
  };
}

test("insider summary: open-market buys and sales for this issuer inside the window, raw", () => {
  const now = Date.parse("2026-10-29T12:00:00Z");
  const s = insiderSummary([f4("RKLB", "2026-10-20T10:00:00Z", "P", 1000), f4("RKLB", "2026-10-22T10:00:00Z", "S", 500), f4("RKLB", "2026-10-23T10:00:00Z", "A", 9999),
    f4("ASTS", "2026-10-22T10:00:00Z", "P", 7), f4("RKLB", "2026-08-01T10:00:00Z", "P", 3)], "RKLB", now);
  assert.equal(s.filings, 3);
  assert.deepEqual([s.buys, s.sells, s.buyShares, s.sellShares], [1, 1, 1000, 500]);
  assert.equal(s.rows[0].kind, "sell");
  assert.equal(s.rows[0].role, "CFO");
  assert.match(s.note, /not a signal/);
});

test("cache: one load per symbol in flight, successes cached for the TTL, failures not cached", async () => {
  let t = 0, loads = 0;
  const c = new DossierCache(1000, () => t);
  const load = async () => { loads += 1; return { ok: true, body: { n: loads } }; };
  const [a, b] = await Promise.all([c.get("X", load), c.get("X", load)]);
  assert.equal(loads, 1); assert.deepEqual(a, b);
  assert.equal((await c.get("X", load)).cached, true);
  t = 2000;
  await c.get("X", load); assert.equal(loads, 2);
  let fails = 0;
  const bad = async () => { fails += 1; return { ok: false, body: { err: true } }; };
  await c.get("Y", bad); await c.get("Y", bad); assert.equal(fails, 2);
});

test("wiring: owner-only route, daemon method registered, nightly scorer beats", () => {
  const bot = read("server/bot.ts");
  assert.match(bot, /app\.get\("\/api\/ticker-dossier\/:symbol", requireOwner,/);
  assert.match(bot, /pythonCall\("ticker_dossier", \{ symbol \}, dossierSubprocessCmd\(symbol\)/);
  assert.match(bot, /from ticker_dossier import score_ledger/);
  assert.ok(bot.includes(`beat("dossier_ledger_score", "start")`) && bot.includes(`beat("dossier_ledger_score", "finish", "ok")`));
  const daemon = read("voltrade_daemon.py");
  assert.match(daemon, /"ticker_dossier": \("ticker_dossier", "build"\)/);
  assert.match(daemon, /"ticker_dossier_score": \("ticker_dossier", "score_ledger"\)/);
});
