// ENTRY SHADOW MODE on the Bot page (2026-10-28 human directive: new stock
// entries and every options open are LOGGED instead of ordered; the floor
// basket and all exits stay live). /api/bot/status carries the switch
// (`entry_shadow`, server/entryShadow.ts entryShadowStatus) and today's (ET)
// would-be entries (`shadow_entries_today`, the in-memory ShadowEntryBook).
// Pure words for the page — so a viewer reads "shadowed", never "broken".
import type { EntryShadow, ShadowEntryRow } from "./useBotData";
import { fmtClockET, fmtOcc } from "./format";

export type ShadowSleeve = "stocks" | "options";

/** The sleeves whose new entries are shadowed; [] when off or not loaded. */
export function shadowSleeves(es: EntryShadow | null | undefined): ShadowSleeve[] {
  if (!es) return [];
  const out: ShadowSleeve[] = [];
  if (es.stocks === true) out.push("stocks");
  if (es.options === true) out.push("options");
  return out;
}

/** One calm sentence: what is shadowed, that it is logged not traded, what stays live. */
export function shadowHeadline(es: EntryShadow | null | undefined): string | null {
  const s = shadowSleeves(es);
  if (!s.length) return null;
  const live = s.length === 2 ? "Floor + exits live." : s[0] === "stocks" ? "Options, floor + exits live." : "Stocks, floor + exits live.";
  return `New entries: SHADOW — ${s.join(" · ")} (logged, not traded). ${live}`;
}

/** Where the switch was set: the env var overrides the system_config default. */
export function shadowSourceNote(es: EntryShadow | null | undefined): string | null {
  if (!shadowSleeves(es).length) return null;
  return es?.source === "env" ? "set by VOLTRADE_SHADOW_ENTRIES (env override)" : "system_config default";
}

export function shadowCountWord(count: number | null | undefined): string {
  const n = typeof count === "number" && Number.isFinite(count) ? count : 0;
  if (n <= 0) return "no would-be entries yet today";
  return `${n} would-be entr${n === 1 ? "y" : "ies"} today`;
}

/** What would have been ordered, beside the ticker the row already names:
 *  "BUY NVDL · ~$4,211", "SELL AMD 155P 20-Nov · ~$15,500", "ENTRY · size not reported"
 *  — never a fake $0 for a size the bot did not report. */
export function shadowRowWhat(r: ShadowEntryRow): string {
  const side = r.side ? r.side.toUpperCase() : "ENTRY";
  const inst = r.instrument && r.instrument !== r.ticker ? ` ${fmtOcc(r.instrument)}` : "";
  const size = typeof r.notional === "number" && Number.isFinite(r.notional) && r.notional > 0
    ? `~$${Math.round(r.notional).toLocaleString("en-US")}` : "size not reported";
  return `${side}${inst} · ${size}`;
}

export function shadowRowTime(r: ShadowEntryRow): string {
  return fmtClockET(r.time).slice(0, 5);
}
