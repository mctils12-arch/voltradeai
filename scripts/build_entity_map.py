#!/usr/bin/env python3
"""build_entity_map.py — Everything Graph R5 build-plan step 1
(datacore/EVERYTHING_GRAPH.md: "datacore/entity_map.json ... a
git-versioned mapping table {registry string -> ticker, confidence,
provenance, verified_against}").

Scope (per the spec, "~30 operators in sites + top-100 plants"): every
`operator` string in datacore/sites/strategic_sites.json, and every
`owner` string among datacore/powerplants/us_power_plants.json's
top-100-by-capacity_mw plants. Strings are used EXACTLY as they appear
in the source registries (including their irregular EIA-860 whitespace)
because they are the join key server/entityGraph.ts will look up.

Research method (2026-07-05, REFERENCE DATA ACCURACY rule): each row
below is hand-verified against a live WebSearch this session (SEC
filings, company investor-relations pages, or company/Wikipedia pages
citing primary sources) — not recalled from training data alone,
because regulated-utility subsidiary structures are stable but
merchant-generator ownership (private equity, joint ventures,
post-bankruptcy spinoffs) changes and several entries below turned out
to be stale (Homer City decommissioned, Louisiana Generating sold to
Cleco in 2019). confidence="unmapped" is a first-class, honest outcome
— per the spec, unmapped operators stay as string-keyed pseudo-nodes,
never guessed tickers.

Re-run: python3 scripts/build_entity_map.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT_PATH = os.path.join(ROOT, "datacore", "entity_map.json")

# {exact registry string: (ticker_or_None, confidence, parent_or_None, note)}
# confidence in {"high", "medium", "unmapped"}.
ENTITIES = {
    # --- datacore/sites/strategic_sites.json operators (13) ---
    "Enbridge": ("ENB", "high", "Enbridge Inc.",
        "Publicly traded pipeline/midstream operator (NYSE/TSX: ENB); named on-site at the Cushing hub."),
    "Plains All American": ("PAA", "high", "Plains All American Pipeline, L.P.",
        "Publicly traded (NASDAQ: PAA); named on-site at the Cushing hub."),
    "Steel Dynamics (STLD)": ("STLD", "high", "Steel Dynamics, Inc.",
        "Ticker is literally in the registry string; NASDAQ: STLD."),
    "Multiple (Enbridge, Plains, ONEOK, Energy Transfer)": (None, "unmapped", None,
        "Cushing hub is genuinely multi-operator; forcing one of the four constituent "
        "tickers (ENB/PAA/OKE/ET, each individually public) would misattribute the site "
        "to a single company. Left as an honest multi-operator gap, not guessed."),
    "City of LA (POLA)": (None, "unmapped", None, "Municipal port authority (Port of LA), no public equity."),
    "Georgia Ports Authority": (None, "unmapped", None, "State port authority, no public equity."),
    "NWSA": (None, "unmapped", None, "Northwest Seaport Alliance, a Seattle/Tacoma joint public authority, no public equity."),
    "PANYNJ": (None, "unmapped", None, "Port Authority of NY & NJ, bi-state public authority, no public equity."),
    "POLB": (None, "unmapped", None, "Port of Long Beach, municipal department, no public equity."),
    "Port Houston": (None, "unmapped", None, "Port of Houston Authority, county-level public authority, no public equity."),
    "Port of Oakland": (None, "unmapped", None, "Municipal department of the City of Oakland, no public equity."),
    "SC Ports": (None, "unmapped", None, "South Carolina Ports Authority, state authority, no public equity."),
    "Virginia Port Authority": (None, "unmapped", None, "Virginia state authority, no public equity."),

    # --- datacore/powerplants/us_power_plants.json owners, top-100-by-MW (56) ---
    "Alabama Power Co": ("SO", "high", "The Southern Company", "Wholly owned regulated operating subsidiary."),
    "Georgia Power Co": ("SO", "high", "The Southern Company", "Wholly owned regulated operating subsidiary."),
    "Mississippi Power Co": ("SO", "high", "The Southern Company", "Wholly owned regulated operating subsidiary."),
    "Southern Power Co": ("SO", "high", "The Southern Company", "Wholly owned wholesale-generation subsidiary."),
    "Appalachian Power Co": ("AEP", "high", "American Electric Power Co., Inc.", "Wholly owned regulated operating subsidiary."),
    "Indiana Michigan Power Co": ("AEP", "high", "American Electric Power Co., Inc.", "Wholly owned regulated operating subsidiary."),
    "Arizona Public Service Co": ("PNW", "high", "Pinnacle West Capital Corp.", "Wholly owned regulated operating subsidiary."),
    "Consumers Energy Co": ("CMS", "high", "CMS Energy Corp.", "Wholly owned regulated operating subsidiary."),
    "DTE Electric Company": ("DTE", "high", "DTE Energy Co.", "Wholly owned regulated operating subsidiary."),
    "Dominion Energy Nuclear Conn Inc": ("D", "high", "Dominion Energy, Inc.", "Wholly owned subsidiary; operates Millstone."),
    "Virginia Electric & Power Co": ("D", "high", "Dominion Energy, Inc.", "d/b/a Dominion Energy Virginia; wholly owned regulated operating subsidiary."),
    "Duke Energy Carolinas  LLC": ("DUK", "high", "Duke Energy Corp.", "Wholly owned regulated operating subsidiary."),
    "Duke Energy Florida  LLC": ("DUK", "high", "Duke Energy Corp.", "Wholly owned regulated operating subsidiary."),
    "Duke Energy Indiana  LLC": ("DUK", "high", "Duke Energy Corp.", "Wholly owned regulated operating subsidiary."),
    "Duke Energy Progress - (NC)": ("DUK", "high", "Duke Energy Corp.", "Wholly owned regulated operating subsidiary."),
    "Entergy Arkansas LLC": ("ETR", "high", "Entergy Corp.", "Wholly owned regulated operating subsidiary."),
    "Entergy Louisiana LLC": ("ETR", "high", "Entergy Corp.", "Wholly owned regulated operating subsidiary."),
    "Entergy Texas Inc.": ("ETR", "high", "Entergy Corp.", "Wholly owned regulated operating subsidiary."),
    "Evergy Kansas Central  Inc": ("EVRG", "high", "Evergy, Inc.",
        "Verified via WebSearch 2026-07-05 (Evergy Inc. investor pages): subsidiary of NASDAQ: EVRG."),
    "FirstEnergy Generation Corp": ("FE", "high", "FirstEnergy Corp.", "Wholly owned subsidiary."),
    "FirstEnergy Nuclear Operating Company": ("FE", "high", "FirstEnergy Corp.", "Wholly owned subsidiary."),
    "Monongahela Power Co": ("FE", "high", "FirstEnergy Corp.", "Wholly owned regulated operating subsidiary (WV/OH)."),
    "Florida Power & Light Co": ("NEE", "high", "NextEra Energy, Inc.", "Wholly owned regulated operating subsidiary."),
    "Indianapolis Power & Light Co": ("AES", "high", "The AES Corporation",
        "Renamed AES Indiana in 2021; wholly owned subsidiary of NYSE: AES since 2001. "
        "Verified via WebSearch 2026-07-05. Registry uses the pre-2021 legal name."),
    "Kentucky Utilities Co": ("PPL", "high", "PPL Corp.", "Wholly owned regulated operating subsidiary."),
    "Louisville Gas & Electric Co": ("PPL", "high", "PPL Corp.", "Wholly owned regulated operating subsidiary."),
    "Luminant Generation Company LLC": ("VST", "high", "Vistra Corp.",
        "Verified via WebSearch 2026-07-05 (Vistra Corp. investor pages / Comanche Peak coverage)."),
    "Dynegy Midwest Generation Inc": ("VST", "high", "Vistra Corp.",
        "Dynegy merged into Vistra Energy Corp. April 2018. Verified via WebSearch 2026-07-05."),
    "NRG Oswego Harbor Power Operations Inc": ("NRG", "high", "NRG Energy, Inc.", "Wholly owned operating subsidiary."),
    "NRG Texas Power LLC": ("NRG", "high", "NRG Energy, Inc.", "Wholly owned operating subsidiary."),
    "Northern Indiana Pub Serv Co": ("NI", "high", "NiSource Inc.",
        "NIPSCO; NiSource retains 80.1% after a 2023 minority (19.9%) sale to Blackstone "
        "Infrastructure Partners — majority public-parent attribution still holds. "
        "Verified via WebSearch 2026-07-05."),
    "Northern States Power Co - Minnesota": ("XEL", "high", "Xcel Energy Inc.", "Wholly owned regulated operating subsidiary (NSP-Minnesota)."),
    "PSEG Nuclear LLC": ("PEG", "high", "Public Service Enterprise Group Inc.", "Wholly owned subsidiary."),
    "PacifiCorp": ("BRK.B", "medium", "Berkshire Hathaway Inc.",
        "Wholly owned but two levels removed: PacifiCorp -> Berkshire Hathaway Energy -> "
        "Berkshire Hathaway Inc. Confidence held at medium — BHE's other assets dwarf any "
        "single plant's materiality to BRK's share price; discount heavily before use in "
        "any trading inference. Verified via WebSearch 2026-07-05."),
    "Pacific Gas & Electric Co.": ("PCG", "high", "PG&E Corp.", "Wholly owned regulated operating subsidiary."),
    "Tampa Electric Co": ("EMA", "high", "Emera Inc.",
        "Wholly owned subsidiary of Emera Inc., dual-listed NYSE/TSX: EMA since May 2025. "
        "Verified via WebSearch 2026-07-05."),
    "Talen Montana LLC": ("TLN", "high", "Talen Energy Corp.",
        "Talen re-listed on Nasdaq (TLN) after its 2022 Chapter 11 emergence. Verified via WebSearch 2026-07-05."),
    "TalenEnergy Susquehanna LLC": ("TLN", "high", "Talen Energy Corp.", "Verified via WebSearch 2026-07-05."),
    "Exelon Nuclear": ("CEG", "high", "Constellation Energy Corp.",
        "The nuclear fleet moved to Constellation Energy Corp. (NASDAQ: CEG) in the Feb-2022 "
        "Exelon spinoff. Verified via WebSearch 2026-07-05 (Peach Bottom now Constellation-operated). "
        "NOTE: some plants under this label are jointly owned (e.g. Peach Bottom 50/50 with PSEG) — "
        "confidence reflects OPERATOR attribution, not exclusive economic ownership."),
    "Constellation Mystic Power LLC": ("CEG", "high", "Constellation Energy Corp.", "Verified via WebSearch 2026-07-05."),
    "Union Electric Co - (MO)": ("AEE", "high", "Ameren Corp.", "d/b/a Ameren Missouri; wholly owned regulated operating subsidiary."),

    "Tennessee Valley Authority": (None, "unmapped", None, "US federal government corporation; issues bonds, has no publicly traded equity."),
    "U S Bureau of Reclamation": (None, "unmapped", None, "US federal agency, no public equity."),
    "USACE Northwestern Division": (None, "unmapped", None, "US Army Corps of Engineers, federal agency, no public equity."),
    "New York Power Authority": (None, "unmapped", None, "NY State public-benefit corporation, no public equity."),
    "South Carolina Public Service Authority": (None, "unmapped", None, "Santee Cooper; SC state-owned utility, no public equity."),
    "STP Nuclear Operating Co": (None, "unmapped", None,
        "South Texas Project is jointly owned (Constellation Energy 42%, CPS Energy 42% "
        "municipal, Austin Energy 16% municipal per the 2024 ownership transaction) — no "
        "single controlling public ticker. Verified via WebSearch 2026-07-05."),
    "KeyCon Operating LLC": (None, "unmapped", None,
        "Keystone/Conemaugh joint operating entity; ownership is fragmented across multiple "
        "private-equity holders plus a minority (~12-16%) Talen Energy stake — no single "
        "controlling public parent. Verified via WebSearch 2026-07-05."),
    "Cardinal Operating Company": (None, "unmapped", None,
        "Joint-venture operator (AEP-affiliate + Buckeye Power municipal/cooperative interests); "
        "no single controlling public parent."),
    "Midland Cogeneration Venture": (None, "unmapped", None, "Complex private joint venture, no single controlling public parent."),
    "Gavin Power  LLC": (None, "unmapped", None,
        "Privately held merchant generator (Lightstone Generation lineage; press reports as of "
        "2026 describe a further private-equity-to-private-equity sale in progress). Not SEC-registered. "
        "Verified via WebSearch 2026-07-05."),
    "GenOn Chalk Point  LLC": (None, "unmapped", None,
        "GenOn Energy emerged from its 2018 Chapter 11 bankruptcy as a privately held company "
        "(ArcLight Capital-affiliated); WebSearch 2026-07-05 found no evidence of a public listing."),
    "Helix Ravenswood  LLC": (None, "unmapped", None, "Privately held merchant generator (GenOn/Helix Generation lineage), no public ticker."),
    "LaFrontera Holdings LLC": (None, "unmapped", None, "Privately held merchant generator (LS Power-affiliated lineage), no public ticker."),
    "Louisiana Generating LLC": (None, "unmapped", None,
        "Sold by NRG Energy to Cleco in Feb 2019; Cleco Corporate Holdings has been privately "
        "held since a 2016 investor buyout — registry attribution to NRG is STALE (a live "
        "example of the WRI GPPD ownership-vintage lag; see manifest note). "
        "Verified via WebSearch 2026-07-05."),
    "NRG Homer City Services LLC": (None, "unmapped", None,
        "STALE registry entry: Homer City Generating Station retired June 2023 and was "
        "physically demolished in 2025 (site now a gas-fired/data-center redevelopment); NRG "
        "was the contracted OPERATOR, never the owner (ownership sat with hedge funds via the "
        "plant's bankruptcy-era ownership vehicle). No going-concern operator to map. "
        "Verified via WebSearch 2026-07-05."),
}

SITE_KEYS = {
    "Enbridge", "Plains All American", "Steel Dynamics (STLD)",
    "Multiple (Enbridge, Plains, ONEOK, Energy Transfer)", "City of LA (POLA)",
    "Georgia Ports Authority", "NWSA", "PANYNJ", "POLB", "Port Houston",
    "Port of Oakland", "SC Ports", "Virginia Port Authority",
}


def main() -> None:
    sites = json.load(open(os.path.join(ROOT, "datacore", "sites", "strategic_sites.json")))
    site_ops = sorted({s["operator"] for s in sites["sites"] if s.get("operator")})
    assert set(site_ops) == SITE_KEYS, f"strategic_sites.json operators changed: {set(site_ops) ^ SITE_KEYS}"

    plants = json.load(open(os.path.join(ROOT, "datacore", "powerplants", "us_power_plants.json")))["plants"]
    top100 = sorted(plants, key=lambda p: p[1] or 0, reverse=True)[:100]
    plant_owners = sorted({p[3] for p in top100 if p[3]})

    all_keys = list(site_ops) + plant_owners
    missing = [k for k in all_keys if k not in ENTITIES]
    if missing:
        raise SystemExit(f"ENTITIES table is missing {len(missing)} operator string(s) — research and add before building: {missing}")

    entities = []
    for key in all_keys:
        ticker, confidence, parent, note = ENTITIES[key]
        entry = {"operator": key, "ticker": ticker, "confidence": confidence, "note": note}
        if parent:
            entry["parent"] = parent
        entities.append(entry)

    mapped = sum(1 for e in entities if e["confidence"] != "unmapped")
    doc = {
        "_doc": (
            "Everything Graph R5 build-plan step 1 (datacore/EVERYTHING_GRAPH.md). "
            "Operator/owner strings from datacore/sites/strategic_sites.json (all sites) and "
            "datacore/powerplants/us_power_plants.json (top-100-by-capacity_mw plants) mapped to "
            "public tickers, hand-verified against SEC filings / company investor-relations pages "
            "via live search 2026-07-05. RAW reference data (per the graph's own RAW-vs-SIGNAL rule): "
            "asserts a relationship with provenance, no predictive claim. Unmapped entries "
            "(government/municipal/federal authorities, privately held merchant generators, "
            "fragmented joint ventures) are honest gaps, never guessed tickers — corporate "
            "structures change (subsidiaries are sold, spun off, or taken private; this build "
            "already caught two registry entries stale for that exact reason), so re-verify "
            "before trading on any inference built atop this table."
        ),
        "schema_version": 1,
        "built": "2026-07-05",
        "sources": [
            "datacore/sites/strategic_sites.json (operator field, all 16 sites)",
            "datacore/powerplants/us_power_plants.json (owner field, top-100-by-capacity_mw plants)",
        ],
        "coverage": {"total": len(entities), "mapped": mapped, "unmapped": len(entities) - mapped},
        "entities": entities,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print(f"wrote {OUT_PATH}: {len(entities)} entities, {mapped} mapped, {len(entities) - mapped} unmapped")


if __name__ == "__main__":
    main()
