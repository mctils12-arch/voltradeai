// Display decodes for the Google Air Quality current-conditions payload
// (/api/data/air-quality, server/airQuality.ts). Kept here rather than inline
// in the page because both decodes below can silently MISREPRESENT a reading,
// which is the one failure mode a data product may not have.
//
// WHY we render the provider's own color instead of our own severity ramp:
// this feed carries two indexes that run in OPPOSITE directions. Universal
// AQI is 0-100 where HIGHER is cleaner; US EPA AQI is 0-500 where HIGHER is
// dirtier. Verified empirically against the 16 live site readings on
// 2026-08-20 — port_oakland reads uaqi 83 "Excellent air quality" alongside
// epa_aqi 32 "Good", while port_savannah reads uaqi 46 "Moderate" alongside
// epa_aqi 62 "Moderate": within each index the category moves monotonically
// with the number, in opposite senses. One shared ramp across both columns
// would paint the cleanest site as the worst. Google ships a per-index color
// with each reading, so we render THAT and never infer a direction ourselves.

export interface AqIndexColor {
  red?: number;
  green?: number;
  blue?: number;
  alpha?: number;
}

const chan = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : null;

/** Google's color is protobuf JSON: a zero-valued channel is OMITTED, not sent
 *  as 0 (`{"green":0.894}` is rgb(0,228,0), the standard EPA green). Returns
 *  null when no channel is present at all — an empty color object carries no
 *  information and must not be rendered as black. Alpha is ignored: these are
 *  opaque category chips. */
export function indexColorCss(color?: AqIndexColor | null): string | null {
  if (!color || typeof color !== "object") return null;
  const r = chan(color.red);
  const g = chan(color.green);
  const b = chan(color.blue);
  if (r === null && g === null && b === null) return null;
  const to255 = (v: number | null) => Math.round((v ?? 0) * 255);
  return `rgb(${to255(r)}, ${to255(g)}, ${to255(b)})`;
}

const srgbToLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));

/** WCAG relative luminance of a provider color, 0 (black) to 1 (white).
 *  Same omitted-channel rule as indexColorCss. */
export function indexLuminance(color?: AqIndexColor | null): number {
  const r = srgbToLinear(chan(color?.red) ?? 0);
  const g = srgbToLinear(chan(color?.green) ?? 0);
  const b = srgbToLinear(chan(color?.blue) ?? 0);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** Text color for a chip painted with `color`. Google's AQI palette spans
 *  near-white greens to dark maroons, so neither black nor white is legible
 *  across the whole range. */
export function indexTextColor(color?: AqIndexColor | null): string {
  return indexLuminance(color) > 0.35 ? "#0b0f14" : "#ffffff";
}

/** The API's unit enums → their conventional symbols. Per CLAUDE.md's UNITS
 *  PREFERENCE rule these are DOMAIN CONVENTIONS (like hPa or MW), not
 *  imperial/metric-switchable quantities, so they are never converted. An
 *  unrecognized enum falls through to the raw string — never guessed. */
export function unitSymbol(units?: string | null): string {
  if (!units) return "";
  switch (units) {
    case "MICROGRAMS_PER_CUBIC_METER":
      return "µg/m³";
    case "PARTS_PER_BILLION":
      return "ppb";
    case "PARTS_PER_MILLION":
      return "ppm";
    default:
      return units;
  }
}

/** Law V (RENDERING & MOTION LAW): a layer that cannot say how old it is may
 *  not claim to be live. `dateTime` is the API's own hour-rounded observation
 *  time, so the freshest possible reading is already up to an hour old.
 *  Returns null for a missing or unparseable timestamp rather than "0m". */
export function readingAge(dateTime: string | null | undefined, nowMs: number): string | null {
  if (!dateTime) return null;
  const t = Date.parse(dateTime);
  if (!Number.isFinite(t)) return null;
  const mins = Math.floor((nowMs - t) / 60_000);
  if (mins < 0) return "just now";
  if (mins < 60) return `${mins}m old`;
  const hours = Math.floor(mins / 60);
  if (hours < 48) return `${hours}h old`;
  return `${Math.floor(hours / 24)}d old`;
}
