import { test } from "node:test";
import assert from "node:assert/strict";

import { catalogFetchPlan } from "./catalogMirror";

test("token present: documented private-repo contents API with raw media type", () => {
  const p = catalogFetchPlan("gp_active.json", "github_pat_TEST");
  assert.equal(p.source, "private-api");
  assert.equal(
    p.url,
    "https://api.github.com/repos/mctils12-arch/voltradeai/contents/gp_active.json?ref=celestial-catalog-data",
  );
  assert.equal(p.headers.Authorization, "Bearer github_pat_TEST");
  // raw media type is REQUIRED for files between 1MB and 100MB (the GP
  // catalog is ~13MB) — without it the API returns base64 JSON or errors.
  assert.equal(p.headers.Accept, "application/vnd.github.raw+json");
  // api.github.com rejects UA-less requests outright.
  assert.ok(p.headers["User-Agent"]);
});

test("token is trimmed; whitespace-only token means NO token (public fallback, no auth header)", () => {
  const padded = catalogFetchPlan("meta.json", "  github_pat_X  ");
  assert.equal(padded.headers.Authorization, "Bearer github_pat_X");

  for (const empty of ["", "   ", undefined as unknown as string]) {
    const p = catalogFetchPlan("meta.json", empty);
    assert.equal(p.source, "public-raw");
    assert.equal(p.url, "https://raw.githubusercontent.com/mctils12-arch/voltradeai-catalog/data/meta.json");
    assert.equal("Authorization" in p.headers, false);
    assert.ok(p.headers["User-Agent"]);
  }
});

test("file name interpolates into both plan shapes", () => {
  assert.match(catalogFetchPlan("satcat.csv", "t").url, /contents\/satcat\.csv\?ref=/);
  assert.match(catalogFetchPlan("satcat.csv", "").url, /\/data\/satcat\.csv$/);
});
