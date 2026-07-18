// Celestial catalog mirror — fetch planning (pure; unit-tested).
//
// DESIGN HISTORY (all 2026-07-18, each step human-approved):
//   v1 mirrored into THIS repo's celestial-catalog-data branch — FAILED in
//      production: the repo is private and raw.githubusercontent.com 404s
//      private content without credentials Railway didn't have.
//   v2 retargeted at a public voltradeai-catalog repo — the human chose
//      Railway credentials instead ("use railway") before creating it.
//   v3 (current): a fine-grained PAT (Contents: read-only, voltradeai only)
//      lives in Railway as GITHUB_CATALOG_TOKEN. With it, the relay fetches
//      the mirror branch through the documented contents API with the raw
//      media type — the supported path for private-repo files up to 100MB
//      (gp_active.json is ~13MB). Without the token, the public-repo raw
//      URL remains as a zero-secret fallback so the relay still recovers if
//      that repo is ever created instead.
//
// api.github.com REJECTS requests without a User-Agent header — every plan
// carries one. Rate limits are irrelevant at our volume (≤ a few fetches
// per hour against 5000/h authenticated).

export interface CatalogFetchPlan {
  url: string;
  headers: Record<string, string>;
  source: "private-api" | "public-raw";
}

const UA = "voltradeai-catalog-relay/1.0";

export function catalogFetchPlan(file: string, token: string): CatalogFetchPlan {
  const t = (token || "").trim();
  if (t) {
    return {
      url: `https://api.github.com/repos/mctils12-arch/voltradeai/contents/${file}?ref=celestial-catalog-data`,
      headers: {
        Authorization: `Bearer ${t}`,
        Accept: "application/vnd.github.raw+json",
        "User-Agent": UA,
        "X-GitHub-Api-Version": "2022-11-28",
      },
      source: "private-api",
    };
  }
  return {
    url: `https://raw.githubusercontent.com/mctils12-arch/voltradeai-catalog/data/${file}`,
    headers: { "User-Agent": UA },
    source: "public-raw",
  };
}
