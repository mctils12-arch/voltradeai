// RAIL FLYOUT — where a menu opened from the /data map's top-left icon rail
// goes. One placement rule for every rail menu (components/RailMenu.tsx is
// the only caller).
//
// Why this exists (owner report 2026-11-19: "when you click on the mountain,
// it covers the other icon and moves down. Why does it move? It's where I
// have my mouse"): the base-map-style menu was laid out IN FLOW beside its
// button inside a centring flex row, so the taller menu pushed the 44px
// button 23px down onto the button below it. The repair is structural: a rail
// button's slot is a fixed box and its menu is positioned OUT of flow by this
// function, so opening a menu cannot move or cover anything.
//
// Standing UI Law (CLAUDE.md, RENDERING & MOTION LAW): no popup covers another
// element; never off-page; screen-size aware. So the placement:
//   1. opens to the RIGHT of the rail, top-aligned with its button;
//   2. is pulled back inside the container when it would run past an edge;
//   3. when another control already occupies that spot, first NARROWS (its
//      rows wrap) to fit the free width beside its button, and only then
//      steps aside — up/down beside the rail, or right past what is in the
//      way;
//   4. never grows past the container: `maxWidth`/`maxHeight` cap it and the
//      menu wraps / scrolls inside those.
//
// Pure: plain numbers in, plain numbers out, in CONTAINER coordinates (the
// map page root, same convention as lib/panelLayout). The menu's size depends
// on how wide it is allowed to be, so the caller passes a `measure(maxWidth)`
// probe instead of one size. Unit-tested with
// `npx tsx --test client/src/lib/railFlyout.test.ts`.

export interface Box { left: number; top: number; width: number; height: number }

/** space between the rail button and its menu, and between the menu and
 *  anything it steps around (the rail's own 6px rhythm: buttons sit 6px apart) */
export const RAIL_FLYOUT_GAP = 6;
/** the menu keeps this far from the container's edges */
export const RAIL_FLYOUT_MARGIN = 8;
/** never offer the menu less width than this (its widest single option) */
export const RAIL_FLYOUT_MIN_WIDTH = 160;

/** boxes covering more than this share of the page are backdrops (the map
 *  canvas, a full-page view), not controls a menu could step around. High on
 *  purpose: a phone bottom sheet is a control — the time machine's is 54% of
 *  a 390x732 page, and at 0.5 the menu opened over it. */
export const RAIL_BACKDROP_SHARE = 0.9;

/** is this box a backdrop of the container rather than a control on it? */
export function isBackdrop(box: { width: number; height: number }, container: { width: number; height: number }): boolean {
  return box.width * box.height > RAIL_BACKDROP_SHARE * container.width * container.height;
}

export interface RailFlyoutInput {
  /** the rail button's box */
  anchor: Box;
  /** the positioning container (map page) */
  container: { width: number; height: number };
  /** the menu's own size when it may be at most `maxWidth` wide */
  measure: (maxWidth: number) => { width: number; height: number };
  /** boxes of every other control on the page (other rail buttons included) */
  obstacles?: readonly Box[];
}

export interface RailFlyoutPlacement {
  left: number;
  top: number;
  maxWidth: number;
  maxHeight: number;
  /** "beside" = right next to its button; "shifted" = it stepped aside for
   *  another control; "overlap" = no free spot existed (it sits beside its
   *  button, on top of what is there and covering as few controls as it can,
   *  so it is at least usable) */
  mode: "beside" | "shifted" | "overlap";
}

const right = (b: Box) => b.left + b.width;
const bottom = (b: Box) => b.top + b.height;

/** overlapping area of two boxes in px² (0 when they only touch) */
export function overlapArea(a: Box, b: Box): number {
  const w = Math.min(right(a), right(b)) - Math.max(a.left, b.left);
  const h = Math.min(bottom(a), bottom(b)) - Math.max(a.top, b.top);
  return w > 0 && h > 0 ? w * h : 0;
}

/** the widest a menu opening beside `anchor` may be in a container this wide */
export function railFlyoutMaxWidth(anchor: Box, containerWidth: number): number {
  const room = containerWidth - (right(anchor) + RAIL_FLYOUT_GAP) - RAIL_FLYOUT_MARGIN;
  return Math.max(RAIL_FLYOUT_MIN_WIDTH, Math.floor(room));
}

export function placeRailFlyout(input: RailFlyoutInput): RailFlyoutPlacement {
  const { anchor, container } = input;
  const obstacles = input.obstacles ?? [];
  const room = railFlyoutMaxWidth(anchor, container.width);
  const maxHeight = Math.max(0, Math.floor(container.height - 2 * RAIL_FLYOUT_MARGIN));
  const minTop = RAIL_FLYOUT_MARGIN;
  const besideLeft = right(anchor) + RAIL_FLYOUT_GAP;

  // one attempt per width the menu may take: the full room first, then each
  // width that would stop it short of a control standing to its right
  const natural = input.measure(room);
  const caps = [room];
  for (const o of obstacles) {
    const cap = Math.floor(o.left - RAIL_FLYOUT_GAP - besideLeft);
    if (cap >= RAIL_FLYOUT_MIN_WIDTH && cap < natural.width && !caps.includes(cap)) caps.push(cap);
  }
  caps.sort((a, b) => b - a);

  type Try = { cap: number; width: number; height: number; maxTop: number; maxLeft: number; baseLeft: number; baseTop: number };
  const tries: Try[] = caps.map((cap) => {
    const m = cap === room ? natural : input.measure(cap);
    const width = Math.min(m.width, cap);
    const height = Math.min(m.height, maxHeight);
    const maxTop = Math.max(minTop, container.height - RAIL_FLYOUT_MARGIN - height);
    const maxLeft = container.width - RAIL_FLYOUT_MARGIN - width;
    return {
      cap, width, height, maxTop, maxLeft,
      // beside the button, pulled back inside the container (only a container
      // narrower than the rail plus the minimum width ever pulls it back)
      baseLeft: Math.max(RAIL_FLYOUT_MARGIN, Math.min(besideLeft, maxLeft)),
      baseTop: Math.min(maxTop, Math.max(minTop, anchor.top)),
    };
  });
  const free = (t: Try, left: number, top: number) => {
    const b: Box = { left, top, width: t.width, height: t.height };
    return obstacles.every((o) => overlapArea(b, o) === 0);
  };
  const done = (t: Try, left: number, top: number, mode: RailFlyoutPlacement["mode"]): RailFlyoutPlacement =>
    ({ left: Math.round(left), top: Math.round(top), maxWidth: t.cap, maxHeight, mode });

  // 1. right beside its button, at the widest width that is free there
  for (const t of tries) {
    if (free(t, t.baseLeft, t.baseTop)) return done(t, t.baseLeft, t.baseTop, "beside");
  }
  // 2. step aside, widest first: slide up/down beside the rail to clear a
  // control in this column, or stay on the button's row and move right past a
  // control on it. Of the free spots at one width, the smallest move wins
  // (ties: stay beside the rail).
  for (const t of tries) {
    const cands: { left: number; top: number; cost: number }[] = [];
    for (const o of obstacles) {
      if (o.left < t.baseLeft + t.width && right(o) > t.baseLeft) {
        for (const top of [o.top - RAIL_FLYOUT_GAP - t.height, bottom(o) + RAIL_FLYOUT_GAP]) {
          if (top >= minTop && top <= t.maxTop) cands.push({ left: t.baseLeft, top, cost: Math.abs(top - t.baseTop) });
        }
      }
      if (o.top < t.baseTop + t.height && bottom(o) > t.baseTop) {
        const left = right(o) + RAIL_FLYOUT_GAP;
        if (left > t.baseLeft && left <= t.maxLeft) cands.push({ left, top: t.baseTop, cost: left - t.baseLeft + 0.5 });
      }
    }
    cands.sort((a, b) => a.cost - b.cost);
    for (const c of cands) if (free(t, c.left, c.top)) return done(t, c.left, c.top, "shifted");
  }
  // 3. nowhere is free (a narrow page with every panel open): beside the
  // button, on top, at the width that covers the fewest controls, then the
  // least of them (ties: the widest)
  const covered = (t: Try): [number, number] => {
    const b: Box = { left: t.baseLeft, top: t.baseTop, width: t.width, height: t.height };
    const areas = obstacles.map((o) => overlapArea(b, o)).filter((a) => a > 0);
    return [areas.length, areas.reduce((sum, a) => sum + a, 0)];
  };
  let best = tries[0];
  let bestCovered = covered(best);
  for (const t of tries.slice(1)) {
    const c = covered(t);
    if (c[0] < bestCovered[0] || (c[0] === bestCovered[0] && c[1] < bestCovered[1])) { best = t; bestCovered = c; }
  }
  return done(best, best.baseLeft, best.baseTop, "overlap");
}
