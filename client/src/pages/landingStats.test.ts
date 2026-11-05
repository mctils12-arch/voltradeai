// The landing hero's counters always resolve to a designed state (2026-11-05;
// DESIGN.md loading-state rule). Live: the "—" placeholder, drawn in the
// accent blue at number size, read as a decorative bar for as long as the
// stats request hung.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { statViews, LOADING_VIEW, STAT_FETCH_TIMEOUT_MS, STAT_RETRY_MS, STAT_MAX_TRIES } from "./landingStats";

const here = path.dirname(fileURLToPath(import.meta.url));

test("a full answer: every counter is a number, no retry", () => {
  const v = statViews({ layers_live: 18, streams_recording: 20, observations: 1284550, observations_pending: false });
  assert.equal(v.layers.state, "ready");
  assert.equal(v.layers.text, (18).toLocaleString());
  assert.equal(v.streams.text, (20).toLocaleString());
  assert.equal(v.observations.state, "ready");
  assert.equal(v.observations.text, (1284550).toLocaleString());
  assert.equal(v.retry, false);
});

test("the server's first count still running: layers and streams show, observations say counting… and retry", () => {
  const v = statViews({ layers_live: 18, streams_recording: 20, observations: null, observations_pending: true });
  assert.equal(v.layers.state, "ready");
  assert.equal(v.observations.state, "pending");
  assert.equal(v.observations.text, "counting…");
  assert.equal(v.retry, true);
});

test("a failed or timed-out request says retrying… while attempts remain, then unavailable — words, never a bare dash", () => {
  const again = statViews(null, true);
  for (const k of ["layers", "streams", "observations"] as const) {
    assert.equal(again[k].state, "retrying");
    assert.equal(again[k].text, "retrying…");
  }
  assert.equal(again.retry, true);
  const last = statViews(null, false);
  for (const k of ["layers", "streams", "observations"] as const) {
    assert.equal(last[k].state, "unavailable");
    assert.equal(last[k].text, "unavailable");
  }
  assert.equal(last.retry, false, "the ladder is bounded");
});

test("a count still running after the last attempt stays counting… (still true) and schedules nothing", () => {
  const v = statViews({ layers_live: 18, streams_recording: 20, observations: null, observations_pending: true }, false);
  assert.equal(v.observations.text, "counting…");
  assert.equal(v.retry, false);
});

test("zero, negative, NaN or missing counts are never shown as a number", () => {
  const v = statViews({ layers_live: 0, streams_recording: Number.NaN, observations: -5 });
  assert.equal(v.layers.state, "unavailable");
  assert.equal(v.streams.state, "unavailable");
  assert.equal(v.observations.state, "unavailable", "not pending: the server did not say a count is running");
  assert.equal(v.layers.text, "unavailable");
  assert.equal(v.retry, false);
});

test("the timed ladder resolves within DESIGN.md's 30 s: the first designed state lands when the first request ends", () => {
  assert.equal(LOADING_VIEW.state, "loading");
  assert.ok(STAT_FETCH_TIMEOUT_MS <= 10_000, "a hung request becomes 'unavailable' quickly, never an eternal placeholder");
  assert.ok(STAT_MAX_TRIES >= 2 && STAT_MAX_TRIES <= 5, "bounded retries");
  assert.ok(STAT_RETRY_MS >= 5_000, "no hammering of a cold server");
});

test("wiring: the hero applies statViews with a timed-out fetch; the markup starts every counter in the loading state; CSS styles the non-number states", () => {
  const landing = fs.readFileSync(path.join(here, "landing.tsx"), "utf8");
  assert.ok(landing.includes('import { statViews'), "landing.tsx uses landingStats.statViews");
  assert.ok(/fetch\("\/api\/data\/platform\/stats", \{ signal: ctl\.signal \}\)/.test(landing), "the stats fetch is abortable");
  assert.ok(landing.includes("window.setTimeout(() => ctl.abort(), STAT_FETCH_TIMEOUT_MS)"), "and aborted on a timer");
  assert.ok(landing.includes("cleanups.push(() => { window.clearTimeout(statTimer); statCtl?.abort(); });"),
    "leaving the page cancels the pending retry and the request in flight");
  const body = fs.readFileSync(path.join(here, "_landing_body.html.txt"), "utf8");
  for (const id of ["di-layers", "di-streams", "di-samples"]) {
    assert.ok(body.includes(`id="${id}" data-state="loading"`), `${id} starts in the loading state`);
  }
  const css = fs.readFileSync(path.join(here, "_landing_styles.css.txt"), "utf8");
  assert.ok(css.includes('.dataintel-num[data-state="unavailable"]'), "the unavailable dash is styled as status");
  assert.ok(css.includes('.dataintel-num[data-state="pending"]'), "counting… is styled as status");
  assert.ok(css.includes('.dataintel-num[data-state="retrying"]'), "retrying… is styled as status");
  assert.ok(landing.includes("statViews(d, statTries < STAT_MAX_TRIES)"), "the view knows whether attempts remain");
  assert.ok(/data-state="loading"\][^{]*\{[^}]*var\(--muted\)/.test(css), "the loading dash is muted, not the accent blue");
});
