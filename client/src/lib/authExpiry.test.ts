import { test } from "node:test";
import assert from "node:assert/strict";
import {
  apiPath, isAuthEndpoint, isSessionExpiry, authErrorKind,
  onSessionExpired, notifySessionExpired, checkSessionExpiry, SESSION_EXPIRED_NOTICE,
} from "./authExpiry";

test("apiPath strips origin, query and hash", () => {
  assert.equal(apiPath("/api/forecast"), "/api/forecast");
  assert.equal(apiPath("https://voltrade.example/api/forecast?x=1"), "/api/forecast");
  assert.equal(apiPath("/api/bot/audit#a"), "/api/bot/audit");
});

test("the sign-in endpoints are excluded (no loop)", () => {
  for (const u of ["/api/auth/login", "/api/auth/me", "/api/auth/logout", "/api/auth/signup", "/api/auth/reset-confirm", "/api/auth"]) {
    assert.equal(isAuthEndpoint(u), true, u);
    assert.equal(isSessionExpiry(u, 401, '{"error":"Not authenticated"}'), false, u);
  }
  // a failed login is a 401 — never an "expired session"
  assert.equal(isSessionExpiry("/api/auth/login", 401, '{"error":"Invalid credentials"}'), false);
  assert.equal(isAuthEndpoint("/api/authors"), false);
});

test("session-shaped 401s from app endpoints are expiries", () => {
  assert.equal(isSessionExpiry("/api/forecast", 401, '{"error":"Not authenticated"}'), true);
  assert.equal(isSessionExpiry("/api/bot/status", 401, '{"error":"Session expired"}'), true);
  assert.equal(isSessionExpiry("/api/watchlist", 401, '{"error":"Authentication required"}'), true);
});

test("other answers are not expiries", () => {
  assert.equal(isSessionExpiry("/api/forecast", 403, '{"error":"Owner only"}'), false);
  assert.equal(isSessionExpiry("/api/forecast", 500, '{"error":"Not authenticated"}'), false);
  assert.equal(isSessionExpiry("/api/forecast", 200, ""), false);
  // API-key / diag-token 401s are a different thing
  assert.equal(isSessionExpiry("/api/v1/signals", 401, '{"error":"invalid or missing API key"}'), false);
  assert.equal(isSessionExpiry("/api/bot/diag", 401, '{"error":"unauthorized"}'), false);
  // non-API URLs (static assets) never count
  assert.equal(isSessionExpiry("/assets/x.js", 401, "Not authenticated"), false);
});

test("authErrorKind reads apiRequest's '<status>: <body>' errors", () => {
  assert.equal(authErrorKind(new Error('401: {"error":"Not authenticated"}')), "unauthorized");
  assert.equal(authErrorKind(new Error('403: {"error":"Owner only"}')), "forbidden");
  assert.equal(authErrorKind(new Error("503: down")), null);
  assert.equal(authErrorKind(new TypeError("Failed to fetch")), null);
  assert.equal(authErrorKind(null), null);
});

test("checkSessionExpiry dispatches only real expiries, to every listener", () => {
  const seen: string[] = [];
  const offA = onSessionExpired((u) => seen.push("a:" + u));
  const offB = onSessionExpired(() => { throw new Error("boom"); }); // must not stop the rest
  const offC = onSessionExpired((u) => seen.push("c:" + u));
  const warn = console.warn; console.warn = () => {};
  try {
    assert.equal(checkSessionExpiry("/api/auth/me", 401, "Not authenticated"), false);
    assert.equal(checkSessionExpiry("/api/forecast", 503, "Not authenticated"), false);
    assert.deepEqual(seen, []);
    assert.equal(checkSessionExpiry("/api/forecast?w=1", 401, '{"error":"Not authenticated"}'), true);
    assert.deepEqual(seen, ["a:/api/forecast", "c:/api/forecast"]);
    offA(); offB(); offC();
    notifySessionExpired("/api/x");
    assert.equal(seen.length, 2, "unsubscribed listeners are not called");
  } finally { console.warn = warn; }
  assert.match(SESSION_EXPIRED_NOTICE, /session expired/i);
});
