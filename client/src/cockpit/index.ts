// client/src/cockpit — the shared panel engine (2026-10-29). A cockpit page
// (the bot's today, the Trade page's next) binds its own panel ids, titles,
// grid templates and pop-out prefix to this vocabulary; the engine owns the
// rest: tiers, flags, float windows, pop-outs and their channel, hold
// snapshots, FLIP on frameCore (Law I). Nothing here imports from bot/.
export * from "./zoneModel";
export * from "./popout";
export * from "./flipGeometry";
export * from "./icons";
export * from "./useNow";
export * from "./useTier";
export * from "./useFlip";
export * from "./FloatWin";
export * from "./useCockpitLayout";
