/**
 * Pins the site's email identity (human-directed 2026-10-13): every Resend
 * send uses EMAIL_FROM on the verified voltradeai.com domain, and the public
 * pages offer one contact address, hello@voltradeai.com.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { EMAIL_FROM } from "./emailSender";

const here = path.dirname(fileURLToPath(import.meta.url));
const read = (p: string) => fs.readFileSync(path.join(here, p), "utf8");

test("EMAIL_FROM is the send-only address on the verified domain", () => {
  assert.equal(EMAIL_FROM, "VolTradeAI <noreply@voltradeai.com>");
});

test("every Resend send uses EMAIL_FROM; Resend's test sender (owner-only delivery) is gone", () => {
  const senders = fs.readdirSync(here).filter((f) => f.endsWith(".ts") && !f.endsWith(".test.ts"))
    .filter((f) => read(f).includes("api.resend.com/emails"));
  assert.deepEqual(senders.sort(), ["auth.ts", "bot.ts"], "a new Resend caller must be added here deliberately");
  for (const f of senders) {
    const src = read(f);
    assert.ok(!src.includes("onboarding@resend.dev"), `${f} still sends from Resend's test address`);
    const sends = src.split("api.resend.com/emails").length - 1;
    const uses = (src.match(/from: EMAIL_FROM,/g) || []).length;
    assert.equal(uses, sends, `${f}: every send must use EMAIL_FROM`);
  }
});

test("public pages offer hello@voltradeai.com and none of the retired addresses", () => {
  for (const page of ["contact.tsx", "careers.tsx"]) {
    const src = read(path.join("..", "client", "src", "pages", page));
    assert.ok(src.includes("hello@voltradeai.com"), `${page} must show hello@voltradeai.com`);
    for (const gone of ["support@voltradeai.com", "press@voltradeai.com", "careers@voltradeai.com", "noreply@voltradeai.com"]) {
      assert.ok(!src.includes(gone), `${page} must not show ${gone}`);
    }
  }
});
