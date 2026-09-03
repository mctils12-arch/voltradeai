/**
 * fleetOperatorTickers.ts — hand-verified FAA-registrant -> public-ticker
 * join for server/fleetUtilization.ts's owner strings (CLAUDE.md EDGE
 * DOCTRINE ACTIVE ANGLE-HUNTING #1's own named example: "corporate-fleet
 * aircraft utilization x earnings timing"). Data lives in
 * datacore/fleet_operator_tickers.json — see that file's own _doc for the
 * verification discipline (never a guessed ticker) and why it is a sibling
 * table to datacore/entity_map.json rather than a merge into it (disjoint
 * source scope: FAA aircraft registrants vs. power-plant/site operators).
 */
import fleetTickersJson from "../datacore/fleet_operator_tickers.json";

export type FleetOperatorThesis = "operational_proxy" | "control_comparison";

export interface FleetOperatorTicker {
  operator: string;
  ticker: string;
  confidence: "high" | "medium";
  thesis: FleetOperatorThesis;
  note: string;
}

const ENTRIES = fleetTickersJson.entries as FleetOperatorTicker[];

const BY_OPERATOR = new Map<string, FleetOperatorTicker>(
  ENTRIES.map((e) => [e.operator, e]),
);

/** Looks up a hand-verified ticker for a fleet-utilization owner/group
 *  string (exact match against the FAA registrant name as it appears in
 *  server/fleetUtilization.ts's payload). Returns null for anything not
 *  yet verified — never guesses. */
export function tickerForFleetOwner(owner: string): FleetOperatorTicker | null {
  return BY_OPERATOR.get(owner) ?? null;
}

export function allFleetOperatorTickers(): readonly FleetOperatorTicker[] {
  return ENTRIES;
}
