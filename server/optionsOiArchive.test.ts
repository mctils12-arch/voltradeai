// Open-interest sidecar — pure helpers, pagination, idempotence, honesty of nulls.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "path";
import {
  compactOi, contractsUrl, fetchOiForUnderlying, captureOpenInterest, oiDayFile, isOiCaptured, OI_PAGE_LIMIT,
} from "./optionsOiArchive";
import { CHAIN_ALWAYS } from "./optionsChainArchive";

const NOW = Date.UTC(2026, 9, 29, 20, 45); // Thu 2026-10-29 16:45 ET
const env = { ALPACA_KEY: "k", ALPACA_SECRET: "s" } as any;
const row = (sym: string, oi: any, d: any = "2026-10-28") => ({ symbol: sym, open_interest: oi, open_interest_date: d });

test("compactOi: string OI parsed, absent OI stays null (never 0), bad date dropped", () => {
  assert.deepEqual(compactOi(row("SPY261030C00600000", "12345"), "SPY", NOW), { s: "SPY261030C00600000", u: "SPY", t: NOW, oi: 12345, oid: "2026-10-28" });
  assert.equal(compactOi(row("SPY261030C00600000", undefined), "SPY", NOW)!.oi, null);
  assert.equal(compactOi(row("SPY261030C00600000", ""), "SPY", NOW)!.oi, null);
  assert.equal(compactOi(row("SPY261030C00600000", "0"), "SPY", NOW)!.oi, 0, "a reported zero is a real zero");
  assert.equal(compactOi(row("X", "5", "10/28/2026"), "SPY", NOW)!.oid, null);
  assert.equal(compactOi({ open_interest: "5" }, "SPY", NOW), null);
});

test("contractsUrl: bounded window, max page size, token appended", () => {
  const u = contractsUrl("SPY", "2026-10-29", "2026-12-28", "tok/1");
  assert.match(u, /underlying_symbols=SPY&expiration_date_gte=2026-10-29&expiration_date_lte=2026-12-28&limit=10000&page_token=tok%2F1$/);
  assert.equal(OI_PAGE_LIMIT, 10000);
});

test("fetchOiForUnderlying follows next_page_token and flags a repeated token", async () => {
  const pages = [{ option_contracts: [row("A", "1")], next_page_token: "p2" }, { option_contracts: [row("B", "2")] }];
  let i = 0;
  const f: any = async () => ({ ok: true, status: 200, json: async () => pages[i++] });
  const got = await fetchOiForUnderlying("SPY", env, f, NOW);
  assert.equal(got.records.length, 2); assert.equal(got.pages, 2); assert.equal(got.truncated, false);
  const loop: any = async () => ({ ok: true, status: 200, json: async () => ({ option_contracts: [row("A", "1")], next_page_token: "same" }) });
  assert.equal((await fetchOiForUnderlying("SPY", env, loop, NOW)).truncated, true);
});

test("captureOpenInterest: publishes once per ET day, idempotent, one failure leaves no file", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "oi-"));
  const ok: any = async () => ({ ok: true, status: 200, json: async () => ({ option_contracts: [row("A", "10"), row("B", null)] }) });
  const r = await captureOpenInterest(env, ok, NOW, base, 0);
  assert.equal(r.status, "published"); assert.equal(r.rows, 2 * CHAIN_ALWAYS.length); assert.equal(r.withOi, CHAIN_ALWAYS.length);
  assert.equal(isOiCaptured("2026-10-29", base), true);
  assert.ok(fs.existsSync(oiDayFile("2026-10-29", base)));
  assert.equal((await captureOpenInterest(env, ok, NOW, base, 0)).status, "skipped");
  const base2 = fs.mkdtempSync(path.join(os.tmpdir(), "oi-"));
  let n = 0;
  const flaky: any = async () => (++n === 2 ? { ok: false, status: 500, json: async () => ({}) } : { ok: true, status: 200, json: async () => ({ option_contracts: [row("A", "1")] }) });
  const bad = await captureOpenInterest(env, flaky, NOW, base2, 0);
  assert.equal(bad.status, "failed");
  assert.equal(isOiCaptured("2026-10-29", base2), false, "a partial day must not be claimed (would bias own-GEX low)");
});
