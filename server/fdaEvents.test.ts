import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import {
  parseApprovals,
  parseAdcomNotices,
  parseMeetingDate,
  fetchFdaEvents,
  archiveFdaEvents,
} from "./fdaEvents";

const here = path.dirname(fileURLToPath(import.meta.url));

// Documented openFDA drugsfda shape (fields verified live 2026-07-05:
// application_number, sponsor_name, submissions[].submission_status/AP +
// submission_status_date YYYYMMDD, products[].brand_name).
const APPROVALS_JSON = {
  results: [
    {
      application_number: "NDA219853", sponsor_name: "EXAMPLE THERAPEUTICS",
      products: [{ brand_name: "EXAMPLDRUG" }],
      submissions: [
        { submission_status: "AP", submission_status_date: "20260630", submission_number: "1" },
        { submission_status: "TA", submission_status_date: "20260401", submission_number: "2" },
      ],
    },
    { application_number: "ANDA211111", sponsor_name: "GENERICS CO", products: [], submissions: [
      { submission_status: "AP", submission_status_date: "20260615", submission_number: "3" },
    ] },
  ],
};

// Real-shape Federal Register notice (title pattern verified live: committee
// + sponsor + BLA + drug in the title; meeting date in the abstract).
const ADCOM_JSON = {
  results: [
    {
      document_number: "2026-12345",
      title: "Cellular, Tissue, and Gene Therapies Advisory Committee; Notice of Meeting; Establishment of a Public Docket; Request for Comments — BLA 125842 From Capricor, Inc. for Deramiocel",
      publication_date: "2026-06-20",
      html_url: "https://www.federalregister.gov/documents/2026/06/20/2026-12345/x",
      abstract: "The meeting will be held on August 21, 2026, from 8:30 a.m. to 5 p.m. ...",
    },
    {
      document_number: "2026-99999",
      title: "Oncologic Drugs Advisory Committee; Notice of Meeting",
      publication_date: "2026-06-25",
      html_url: "https://www.federalregister.gov/documents/x",
      abstract: "Details to follow.",
    },
  ],
};

test("parseApprovals: AP submissions only, dates normalized, brand joined", () => {
  const ev = parseApprovals(APPROVALS_JSON, "2026-07-05");
  assert.equal(ev.length, 2, "non-AP submissions skipped");
  // window filter: openFDA returns whole applications with full history —
  // out-of-window AP submissions must be excluded (live E2E catch)
  assert.equal(parseApprovals(APPROVALS_JSON, "2026-07-05", "2026-06-20").length, 1,
    "AP submission before the window must be dropped");
  assert.equal(ev[0].t, "approval");
  assert.equal(ev[0].appl, "NDA219853");
  assert.equal(ev[0].date, "2026-06-30");
  assert.equal(ev[0].brand, "EXAMPLDRUG");
  assert.equal(ev[0].key, "NDA219853|1|2026-06-30");
});

test("parseMeetingDate extracts Month D, YYYY; misses stay null", () => {
  assert.equal(parseMeetingDate("held on August 21, 2026, from 8:30"), "2026-08-21");
  assert.equal(parseMeetingDate("held on March 3, 2027."), "2027-03-03");
  assert.equal(parseMeetingDate("Details to follow."), null);
});

test("parseAdcomNotices: committee/sponsor/BLA extracted; unparsed dates labeled, never guessed", () => {
  const ev = parseAdcomNotices(ADCOM_JSON, "2026-07-05");
  assert.equal(ev.length, 2);
  assert.equal(ev[0].t, "adcom");
  assert.equal(ev[0].key, "2026-12345");
  assert.equal(ev[0].date, "2026-08-21");
  assert.equal(ev[0].dateConf, "parsed");
  assert.ok(ev[0].committee!.includes("Gene Therapies Advisory Committee"));
  assert.equal(ev[0].appl, "BLA125842");
  assert.equal(ev[0].sponsor, "Capricor");
  assert.equal(ev[0].pub, "2026-06-20", "publication date = when the public could know");
  assert.equal(ev[1].date, null);
  assert.equal(ev[1].dateConf, "unparsed", "missed parse must be labeled, not guessed");
});

test("fetchFdaEvents: one-source failure never kills the batch", async () => {
  const impl = async (url: string) => {
    if (url.includes("api.fda.gov")) return { ok: false, status: 500, text: async () => "" };
    return { ok: true, status: 200, text: async () => JSON.stringify(ADCOM_JSON) };
  };
  const ev = await fetchFdaEvents(impl as any, Date.parse("2026-07-05T12:00:00Z"));
  assert.equal(ev.length, 2, "adcom events survive an openFDA outage");
  assert.ok(ev.every((e) => e.t === "adcom"));
});

test("archive round-trip with dedup by key", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtfda-"));
  const t0 = Date.parse("2026-07-05T12:00:00Z");
  const ev = parseApprovals(APPROVALS_JSON, "2026-07-05");
  assert.equal(archiveFdaEvents(ev, dir, t0), 2);
  assert.equal(archiveFdaEvents(ev, dir, t0), 0, "same keys never archive twice");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("routes.ts boots the FDA poll and registers /api/data/fda-events; manifest states the PDUFA honesty", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("bootFdaPoll()"), "FDA poll must boot eagerly");
  assert.ok(routes.includes('"/api/data/fda-events"'), "route registered");
  const manifest = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "manifests", "fda.json"), "utf8"));
  assert.equal(manifest.stream, "fda");
  assert.ok(String(manifest.license).includes("PDUFA"), "PDUFA unavailability must be stated");
  assert.ok(String(manifest.field_map.dateConf).includes("never guessed"), "parse-confidence honesty stated");
});
