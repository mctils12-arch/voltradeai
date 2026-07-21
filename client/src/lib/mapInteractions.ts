// mapInteractions — attach click + hover-cursor handlers to one or more map
// layers and return a detach() that map.off()s every one.
//
// Fixes BUG 4: six hand-rolled /data layers (sites, powerplants, trains, fires,
// rivergauges, alerts) attached click/mouseenter/mouseleave ANONYMOUSLY and
// never removed them, so each toggle off→on stacked another set — after N
// cycles a single click fired N detail cards / N trail fetches. wireLivePoints
// (aircraft/vessels) already did this right with NAMED handlers + map.off in
// teardown; this generalizes that discipline. Callers MUST invoke the returned
// detach() in their effect cleanup.

// Structural type so this is unit-testable with a fake map (no maplibre needed).
export interface InteractiveMap {
  on(type: string, layerId: string, handler: (e: any) => void): void;
  off(type: string, layerId: string, handler: (e: any) => void): void;
  getCanvas(): { style: { cursor: string } };
}

export function attachLayerInteractions(
  map: InteractiveMap,
  layerIds: string | string[],
  onClick: (e: any) => void,
): () => void {
  const ids = Array.isArray(layerIds) ? layerIds : [layerIds];
  // AIRCRAFT CLICK CLAIM (2026-07-21): the 3D plane picker stamps
  // __vtAirClaim on the shared originalEvent when a plane is genuinely
  // under the cursor — a plane parked next to a mapped site must win the
  // click over the site marker. Ground handlers on the same click event
  // stand down; unclaimed clicks behave exactly as before.
  const guardedClick = (e: any) => {
    if (e?.originalEvent?.__vtAirClaim) return;
    onClick(e);
  };
  const onEnter = () => { map.getCanvas().style.cursor = "pointer"; };
  const onLeave = () => { map.getCanvas().style.cursor = ""; };
  for (const id of ids) {
    map.on("click", id, guardedClick);
    map.on("mouseenter", id, onEnter);
    map.on("mouseleave", id, onLeave);
  }
  return () => {
    for (const id of ids) {
      try {
        map.off("click", id, guardedClick);
        map.off("mouseenter", id, onEnter);
        map.off("mouseleave", id, onLeave);
      } catch { /* layer already gone */ }
    }
  };
}
