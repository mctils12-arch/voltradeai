# Data / Access Wishlist — human reviews weekly

- **Historical options prices** (EOD chains + marks, ~2016→present) to backtest
  the options leg honestly. Without it, only the equity/ETF logic can be
  validated — and the options leg is the suspected main performance drag.
  Candidates: ORATS, CBOE DataShop, historicaloptiondata.com.

- **Persist the max-drawdown high-water mark** (`state.equityPeak`,
  server/bot.ts:359/862/2482): in-memory only today, so every
  deploy/restart re-bases the drawdown kill switch from current equity —
  frequent autonomous deploys silently defang it. Proposal: save/restore
  equityPeak via the existing /data/voltrade state files. Touches frozen
  kill-switch machinery -> needs explicit human approval (this entry).
  Evidence: /api/health shows equityPeak 0 after today's deploys.
- **Read-only diagnostics access for autonomous sessions**: all diagnostic
  routes are owner-cookie gated (auth.ts — frozen). Options for human:
  (a) paste /api/bot/audit + /api/bot/ml-status JSON into sessions when
  diagnosis is needed, (b) approve a scoped read-only token path in
  auth.ts, or (c) a nightly job that snapshots key state JSON into the
  repo. Until then KNOWN BROKEN #3/#4 can only be verified by the human.
