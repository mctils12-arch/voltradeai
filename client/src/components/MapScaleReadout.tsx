// MAP SCALE + ZOOM READOUT — the scale bar with its distance, and the zoom
// level, in the /data map's combined bottom status bar. pages/datamap.tsx
// renders the bar itself and the imagery capture date beside this.
//
// It is its own component so that a camera move re-renders these three spans
// and nothing else. Until 2026-11-12 the camera state behind the readout was
// page state: every frame of every pan or zoom called setState on
// DataMapPage and re-rendered the whole page to change one short line of
// text (SCALE program stability queue item 1, research/scale_program.md).
// React.memo keeps the page's own re-renders (position ticks, hover, panels)
// out of it, the boundary the Legend got in v1.0.373.
//
// The map's "move" event only asks for an animation frame; the frame reads
// the camera. So the camera is read at most once per frame however many move
// events arrive, and nothing is computed inside the event handler.

import { memo, useEffect, useState, useSyncExternalStore } from "react";
import { scaleReading, zoomLabel } from "@/lib/mapScale";
import { getUnits, subscribeUnits } from "@/lib/units";

/** The camera reading the scale depends on. */
export interface ScaleView {
  zoom: number;
  lat: number;
}

/** The map's own start camera (datamap.tsx: zoom 3.6 at latitude 37.5),
 *  shown until there is a map to read. */
export const SCALE_VIEW_AT_START: ScaleView = { zoom: 3.6, lat: 37.5 };

/** The part of a MapLibre map the readout uses. */
export interface ScaleReadoutMap {
  getZoom(): number;
  getCenter(): { lat: number };
  on(type: "move", listener: () => void): unknown;
  off(type: "move", listener: () => void): unknown;
}

/** Animation-frame scheduling: the browser's by default, a fake in tests. */
export interface FrameScheduler {
  request(callback: () => void): number;
  cancel(handle: number): void;
}

const browserFrames: FrameScheduler = {
  request: (callback) => requestAnimationFrame(callback),
  cancel: (handle) => cancelAnimationFrame(handle),
};

/** The camera as a ScaleView, or null when the map cannot be read. */
export function readScaleView(map: ScaleReadoutMap): ScaleView | null {
  let view: ScaleView | null = null;
  try { view = { zoom: map.getZoom(), lat: map.getCenter().lat }; } catch {}
  return view;
}

/**
 * Follow a map's camera: report the view once now, then at most once per
 * animation frame while the map moves. Returns the function that stops it
 * (listener removed, waiting frame cancelled). No React and no DOM, so the
 * tests drive it with a fake map and a fake frame clock.
 */
export function trackScaleView(
  map: ScaleReadoutMap,
  onView: (view: ScaleView) => void,
  frames: FrameScheduler = browserFrames,
): () => void {
  let pending: number | null = null;
  const sync = () => {
    pending = null;
    const view = readScaleView(map);
    if (view) onView(view);
  };
  const onMove = () => {
    if (pending == null) pending = frames.request(sync);
  };
  sync();
  map.on("move", onMove);
  return () => {
    if (pending != null) frames.cancel(pending);
    // on leaving the page the map is removed before this runs
    try { map.off("move", onMove); } catch {}
  };
}

export interface MapScaleReadoutProps {
  /** The map once it is ready, null before. It is the same object for the
   *  page's whole life, so memo's shallow comparison holds. */
  map: ScaleReadoutMap | null;
}

export const MapScaleReadout = memo(function MapScaleReadout({ map }: MapScaleReadoutProps) {
  // Read the camera on the first render when there is a map: the status bar
  // remounts when the space frame hands the viewport back, and must not show
  // the start view for a frame first.
  const [view, setView] = useState<ScaleView>(() => (map && readScaleView(map)) || SCALE_VIEW_AT_START);
  // the site-wide unit setting: a switch re-renders the readout in the new units
  const units = useSyncExternalStore(subscribeUnits, getUnits, getUnits);
  useEffect(() => {
    if (!map) return;
    return trackScaleView(map, setView);
  }, [map]);
  const sc = scaleReading(view.zoom, view.lat, units);
  return (
    <>
      <span className="vt-statusbar-scale" aria-label={`scale ${sc.label}`}>
        <span className="vt-statusbar-bar" style={{ width: `${Math.round(sc.widthPx)}px` }} />
        {sc.label}
      </span>
      <span className="vt-statusbar-sep">·</span>
      <span className="vt-statusbar-zoom">{zoomLabel(view.zoom)}</span>
    </>
  );
});
