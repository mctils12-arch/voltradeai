# EVERYTHING GRAPH — design doc (v1 spec)

Filed 2026-07-04 per the charter directive; flagship PRODUCT roadmap item
(open_questions.md MAP V2 ROADMAP R5). Status: DESIGN — no build yet.
VISION.md pillar 6.

## Why a graph

Every dataset we collect is about the same underlying economy, keyed by
different identifiers: Form 4 speaks CIK, the market speaks ticker, the
sites registry speaks facility, the AIS archive speaks MMSI. Fusion
hypotheses (insider × facility, generation × operator, ships × retail)
are JOINs across these keys. The graph is that join, materialized once,
so every future hypothesis starts from linked entities instead of
re-deriving the mapping. It is also the product's shape: "show me
everything the platform knows about STLD" is a graph query.

## Ground rules (from the constitution)

1. datacore boundary: the graph builder imports archives and registries,
   never trading logic; the bot consumes graph queries through the API
   like any customer.
2. RAW vs SIGNAL: the graph itself is RAW (it asserts relationships from
   filings and registries, with provenance). Any INTERPRETATION on top
   (anomaly scores, "insider cluster near facility drawdown") is
   SIGNAL-class and ladder-gated.
3. Archive ingredients, recompute derivations: v1 is a MATERIALIZED VIEW
   rebuilt from existing archives + git-versioned registries — no new
   mutable store whose loss would lose data. Everything the graph knows
   is recomputable from what we already persist.

## Entity types (v1 — only what we already collect)

| type | key | source of truth | notes |
|---|---|---|---|
| company | ticker (+ CIK, name aliases) | Form 4 issuer records; entity_map | CIK↔ticker from filings is authoritative-as-filed |
| person | CIK | Form 4 owner records | SEC assigns stable person CIKs — free entity resolution |
| facility | site id / plant id | datacore/sites, datacore/powerplants | lat/lon intrinsic; imagery-verified flag carried |
| vessel | MMSI | AIS archive | name as last-broadcast (mutable — shadow-fleet lesson) |
| aircraft_operator | — | PLANNED | blocked on tail→operator mapping (gate-1 item, open_questions ARCHIVE-ENABLED hypotheses); not in v1 |

## Relationship types (v1)

| edge | from → to | derived from | attributes |
|---|---|---|---|
| insider_of | person → company | Form 4 filings archive | roles (director/officer/10%/title), filing count, first/last seen, last transaction kind |
| operates | company → facility | sites `operator` + plants `owner` via entity_map | confidence, provenance (registry string), verified flag |
| located_at | facility → geo | registries | lat/lon, imagery-verified flag |
| calls_at | vessel → facility(port) | portDwell visits (v1.0.60) | visit count, last call, median dwell — the graph's first MOVING edges |

Deliberately excluded from v1 (need sources we don't have yet):
`supplies` (company→company supply chains — no free authoritative
source; NOT blocked-by-access, derivable later from 10-K/8-K text at
low confidence, but that is an extraction project, not a join),
`banks_with`, `audited_by`, ownership trees (13F/13D — queued EDGAR
expansion).

## The hard part: entity resolution (operator strings → tickers)

`datacore/entity_map.json` — a git-versioned mapping table
{registry string → ticker, confidence, provenance, verified_against}.
Built once, verified against 10-K subsidiary lists (fusion hypothesis
(b) gate-1 already requires exactly this table — the graph formalizes
it as shared infrastructure). Rows carry confidence: exact-name match
(high) / alias match (medium) / manual research (with citation).
Unmapped operators stay as string-keyed pseudo-nodes — honest gaps,
never guessed tickers. REFERENCE DATA ACCURACY applies: the map is
reference data; spot-verification before shipping.

## Storage decision

**v1: pure builder module + cache, no database.**
`server/entityGraph.ts` (pure, baseDir-injectable like
shadowFleet/portDwell) reads: filings archive (edgarForm4
readFilingHistory), sites/plants registries, entity_map, portDwell
visits → emits `{nodes: [...], edges: [...]}`. Served at
`/api/data/graph` (neighborhood queries: `?entity=STLD&hops=1`),
15-min cache. Estimated v1 size: ~10k company nodes worst case from
filings (30d window: hundreds), 9.8k facility nodes, edges in the low
tens of thousands — trivially in-memory.

**Evolution trigger (stated now, decided later): when rebuild exceeds
~2s or edges exceed ~250k**, materialize into a sqlite file
(better-sqlite3 already a dependency; own db file on the volume, NEVER
auth.ts's db — standing rule: db by injection, and the graph db is
rebuildable state, droppable without data loss). Schema then: nodes
(id, type, label, attrs JSON), edges (src, dst, type, attrs JSON,
first_seen, last_seen, source, confidence) with (src,type) and
(dst,type) indexes.

Every edge carries `{source, confidence, first_seen, last_seen}` —
aligned with the UNIVERSAL SCHEMA envelope proposal (wishlist, pending
approval) so archived data and graph edges speak the same metadata
dialect.

## Product surface (when v1 lands — not before)

/data gains a graph panel: entity search → neighborhood card (the
company's insiders + recent buys, operated facilities on the map,
vessels calling at its ports). Map integration: selecting a company
HIGHLIGHTS its facilities/ports — the graph drives the map rather than
being a separate visualization. Graph queries are the /data feature;
no separate site.

## Build plan (each its own PR)

1. `datacore/entity_map.json` for the ~30 operators in sites +
   top-100 plants, verified; builder test pins coverage honesty
   (unmapped stays unmapped). [also unblocks fusion (b) gate 1]
2. `server/entityGraph.ts` + `/api/data/graph` + tests (fixture
   archives → expected nodes/edges; provenance/confidence pins).
3. /data graph panel per DESIGN.md (self-see, three widths, theme
   tokens) + company→facility map highlighting.
4. (later, gated) SIGNAL layers on top — each through the ladder.

## What the graph does for trading (the customer)

Fusion gate-1 efforts stop being bespoke joins: insider×facility (a)
reads insider_of ⨝ operates; generation×operator (b) reads operates
on plants; ships×retail (c) reads calls_at aggregates. Each hypothesis
still validates through its own ladder — the graph only removes the
join labor, it grants no evidential shortcut.
