// activeTicker — the one symbol the Trade world is looking at.
//
// Human 2026-10-29: "if you look for a ticker symbol and then you might want
// the stock vs. options info on it, and trade information" — one search
// should follow the user across Options & Volatility, Smart Money,
// Structure, Research (the stock), Planner (the trade), News and, for the
// owner, the Dossier. Each page seeds its own input from here and writes
// back when the user picks a new name; the shell shows the symbol in a
// context bar with one-click jumps between those views.
//
// Session-scoped: sessionStorage keeps the symbol across a reload, a new
// tab starts clean. Same shape as lib/units.ts and lib/tradeBackdrop.ts.

export const ACTIVE_TICKER_KEY = "vt-active-ticker";

// Symbols the analyze endpoints accept: letters, digits, dot and dash,
// 1–10 characters, upper case (BRK.B, RDS-A, SPY).
const VALID = /^[A-Z][A-Z0-9.\-]{0,9}$/;

export function normalizeTicker(t: string | null | undefined): string | null {
  if (!t) return null;
  const u = String(t).trim().toUpperCase();
  return VALID.test(u) ? u : null;
}

function readStored(): string | null {
  try {
    return normalizeTicker(sessionStorage.getItem(ACTIVE_TICKER_KEY));
  } catch {
    return null; // no DOM (tests), private mode, or storage disabled
  }
}

let current: string | null = readStored();
const listeners = new Set<() => void>();
let storageWarned = false;

function writeStored(next: string | null): void {
  try {
    if (next) sessionStorage.setItem(ACTIVE_TICKER_KEY, next);
    else sessionStorage.removeItem(ACTIVE_TICKER_KEY);
  } catch (err) {
    // Storage unavailable (private mode, disabled, no DOM): the in-memory
    // value still works for this page load. Say so once instead of hiding it
    // (ci counter commented_empty_catch: log, don't swallow).
    if (!storageWarned) {
      storageWarned = true;
      console.debug("activeTicker: sessionStorage unavailable, symbol kept in memory only", err);
    }
  }
}

export function getActiveTicker(): string | null {
  return current;
}

/** Set (or clear with null) the active symbol; invalid input clears nothing and is ignored. */
export function setActiveTicker(t: string | null | undefined): void {
  const next = t === null ? null : normalizeTicker(t);
  if (t !== null && next === null) return; // garbage never replaces a good symbol
  if (next === current) return;
  current = next;
  writeStored(next);
  listeners.forEach((fn) => fn());
}

export function subscribeActiveTicker(fn: () => void): () => void {
  listeners.add(fn);
  return () => { listeners.delete(fn); };
}
