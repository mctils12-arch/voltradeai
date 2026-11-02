// clock_skew_preload.cjs — shift the wall clock forward for a test run.
//   SKEW_DAYS=30 node --require ./scripts/clock_skew_preload.cjs ...
// Only the REAL-clock reads shift (Date.now(), new Date() with no args);
// explicit-argument Dates are untouched, so fixtures with fixed dates keep
// meaning what they say while "now" moves away from them. A test that fails
// under skew but passes unskewed has a calendar time-bomb (a fixture date that
// ages out of a windowed code path).
"use strict";
const days = Number(process.env.SKEW_DAYS || 0);
if (Number.isFinite(days) && days !== 0) {
  const off = days * 86400000;
  const RealDate = Date;
  class SkewDate extends RealDate {
    constructor(...a) { if (a.length === 0) super(RealDate.now() + off); else super(...a); }
    static now() { return RealDate.now() + off; }
  }
  globalThis.Date = SkewDate;
}
