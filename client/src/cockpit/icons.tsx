// Cockpit chrome icons — the panel header controls and the float window's
// title bar draw from one set. 2026-10-29: moved here from bot/panels.tsx
// (which re-exports it as ICONS, so its call sites stand) because FloatWin
// moved into the shared engine and the engine imports nothing from bot/.
export const ICONS = {
  hold: <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" aria-hidden="true"><path d="M8 5v14M16 5v14" /></svg>,
  play: <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="none" aria-hidden="true"><path d="M7 4l12 8-12 8z" /></svg>,
  pin: <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true"><path d="M12 17v5M8 3h8l-1 7 3 3H6l3-3z" /></svg>,
  win: <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true"><rect x="3" y="8" width="13" height="13" rx="1.5" /><path d="M8 8V3h13v13h-5" /></svg>,
  zoom: <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true"><path d="M4 10V4h6M20 14v6h-6M20 4l-7 7M4 20l7-7" /></svg>,
  dock: <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true"><path d="M4 14v6h6M20 10V4h-6M4 20l7-7M20 4l-7 7" /></svg>,
  close: <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" /></svg>,
  pop: <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true"><path d="M14 4h6v6M20 4l-9 9M11 5H4v15h15v-7" /></svg>,
  arrow: <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" /></svg>,
};
