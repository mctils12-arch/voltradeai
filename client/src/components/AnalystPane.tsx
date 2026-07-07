import { useEffect, useRef, useState } from "react";
import { Send, X, MessageSquareText } from "lucide-react";

/**
 * AnalystPane — ANALYST CONSOLE W6 (CLIENT half): the chat pane for
 * POST /api/analyst (server/analyst.ts). Lazy-loaded from datamap.tsx so a
 * closed pane costs zero bundle/runtime (zero-cost-when-off spirit); it
 * never polls anything — the ONLY network call is the explicit user send.
 *
 * Honesty rules (console charter): every server state renders distinctly —
 * awaiting_key / budget_exhausted / 429 busy / 401 sign-in / 400 / 502 /
 * success — and success surfaces the full honesty machinery: source chips
 * (tool + freshness), a collapsible tool trace, and the usage/budget/model
 * footline. Map commands run through callbacks the PARENT owns so
 * toggle_layer drives the exact same state the layer panel uses.
 */

export interface AnalystMapCommand {
  command: "fly_to" | "toggle_layer";
  lat?: number; lon?: number; zoom?: number;
  layer?: string; on?: boolean;
}

interface SourceChip { tool: string; params: any; freshness?: string | null }
interface ToolTraceEntry { tool: string; input: any; ok: boolean; summary: string }
interface SuccessEnvelope {
  answer: string;
  tool_trace: ToolTraceEntry[];
  map_commands: AnalystMapCommand[];
  sources: SourceChip[];
  usage: { input_tokens: number; output_tokens: number; requests: number };
  budget: { spent_today: number; daily_limit: number };
  model: string;
  note?: string;
  transport_error?: string;
}

type Resp =
  | { kind: "ok"; env: SuccessEnvelope; execNotes: string[] }
  | { kind: "awaiting_key"; reason?: string }
  | { kind: "budget"; resets?: string; budget?: { spent_today: number; daily_limit: number } }
  | { kind: "busy"; retryAfterSec?: number }
  | { kind: "auth" }
  | { kind: "bad"; error?: string }
  | { kind: "transport"; error?: string; budget?: { spent_today: number; daily_limit: number } }
  | { kind: "network"; error: string };

interface ChatItem { id: number; question: string; resp: Resp | null }

// Session history survives close/open (the pane unmounts when closed so it
// costs nothing; module state inside this lazy chunk keeps the transcript).
let sessionHistory: ChatItem[] = [];
let seq = 0;

const QUESTION_MAX = 2000;

const SUGGESTIONS = [
  "What changed near Cushing, OK this week?",
  "Any active severe-weather alerts right now?",
];

/** Relative age for a freshness timestamp; falls back to the raw string. */
function fmtFresh(s?: string | null): string | null {
  if (!s) return null;
  const t = Date.parse(s);
  if (!Number.isFinite(t)) return String(s).slice(0, 19);
  const sec = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (sec < 90) return `${sec}s ago`;
  if (sec < 5400) return `${Math.round(sec / 60)}m ago`;
  if (sec < 129600) return `${Math.round(sec / 3600)}h ago`;
  return `${Math.round(sec / 86400)}d ago`;
}

function fmtParams(params: any): string {
  try {
    const s = JSON.stringify(params);
    return s === "{}" || s === "null" ? "" : s;
  } catch { return ""; }
}

export default function AnalystPane({ onClose, onMapCommand }: {
  onClose: () => void;
  /** Executes one validated map command on the LIVE map (parent owns the
   *  map ref + the same layer-toggle state the panel uses); returns the
   *  human-readable "→ ..." note rendered in the chat. */
  onMapCommand: (cmd: AnalystMapCommand) => string;
}) {
  const [items, setItems] = useState<ChatItem[]>(() => sessionHistory);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [traceOpen, setTraceOpen] = useState<Record<number, boolean>>({});
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const commit = (next: ChatItem[]) => { sessionHistory = next; setItems(next); };

  // keep the newest exchange in view as answers land
  useEffect(() => {
    try { endRef.current?.scrollIntoView({ block: "end" }); } catch {}
  }, [items, busy]);

  const send = async () => {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    const id = ++seq;
    commit([...sessionHistory, { id, question: q, resp: null }]);
    setBusy(true);
    const settle = (resp: Resp) =>
      commit(sessionHistory.map((it) => (it.id === id ? { ...it, resp } : it)));
    try {
      const r = await fetch("/api/analyst", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const body: any = await r.json().catch(() => null);
      if (r.status === 401) settle({ kind: "auth" });
      else if (r.status === 429) settle({ kind: "busy", retryAfterSec: body?.retry_after_sec });
      else if (r.status === 400) settle({ kind: "bad", error: body?.error });
      else if (r.status === 502) settle({ kind: "transport", error: body?.error, budget: body?.budget });
      else if (r.ok && body?.awaiting_key) settle({ kind: "awaiting_key", reason: body?.reason });
      else if (r.ok && body?.budget_exhausted) settle({ kind: "budget", resets: body?.resets, budget: body?.budget });
      else if (r.ok && typeof body?.answer === "string") {
        // execute map commands on the live map, note each one in the chat
        const execNotes = (Array.isArray(body.map_commands) ? body.map_commands : [])
          .map((cmd: AnalystMapCommand) => { try { return onMapCommand(cmd); } catch { return "map command failed to execute"; } });
        settle({ kind: "ok", env: body as SuccessEnvelope, execNotes });
      } else {
        settle({ kind: "network", error: `unexpected response (HTTP ${r.status}) — nothing was answered` });
      }
    } catch {
      settle({ kind: "network", error: "request failed (network) — nothing was answered" });
    } finally {
      setBusy(false);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  const loginHref = `/login?next=${encodeURIComponent("/app#/data")}`;

  const renderResp = (it: ChatItem) => {
    const r = it.resp;
    if (!r) {
      // perceived-performance rule: designed skeleton, never a bare wait
      return (
        <div className="vt-analyst-answer" aria-label="Analyst is querying platform data">
          <div className="vt-analyst-skel" />
          <div className="vt-analyst-skel" style={{ width: "82%" }} />
          <div className="vt-analyst-skel" style={{ width: "64%" }} />
          <span className="vt-analyst-skel-note">querying platform data tools…</span>
        </div>
      );
    }
    switch (r.kind) {
      case "auth":
        return (
          <div className="vt-analyst-state" data-state="auth">
            <b>Sign in required.</b> The analyst spends a real daily token
            budget, so questions need an account.
            <a className="vt-analyst-login" href={loginHref}>Sign in →</a>
          </div>
        );
      case "awaiting_key":
        return (
          <div className="vt-analyst-state" data-state="awaiting_key">
            <b>Awaiting API key.</b> The analyst activates when the API key is
            added — no key, no fake answers.
            {r.reason && <span className="vt-analyst-state-sub">{r.reason}</span>}
          </div>
        );
      case "budget":
        return (
          <div className="vt-analyst-state" data-state="budget">
            <b>Daily token budget spent</b>
            {r.budget && <> — {r.budget.spent_today.toLocaleString()} of {r.budget.daily_limit.toLocaleString()} tokens used</>}.
            An honest stop, not a silent degrade.
            {r.resets && <span className="vt-analyst-state-sub">Resets {r.resets.replace("T", " ").replace(":00Z", " UTC")}</span>}
          </div>
        );
      case "busy":
        return (
          <div className="vt-analyst-state" data-state="busy">
            <b>Analyst at capacity</b> (2 concurrent questions max) — try again
            in {r.retryAfterSec ?? 10}s.
          </div>
        );
      case "bad":
        return (
          <div className="vt-analyst-state" data-state="bad">
            <b>Question rejected:</b> {r.error || "invalid question"}
          </div>
        );
      case "transport":
        return (
          <div className="vt-analyst-state" data-state="transport">
            <b>Upstream failure</b> — {r.error || "the model transport failed"}.
            No answer is invented in its place.
            {r.budget && (
              <span className="vt-analyst-state-sub">
                budget today {r.budget.spent_today.toLocaleString()}/{r.budget.daily_limit.toLocaleString()} tokens
              </span>
            )}
          </div>
        );
      case "network":
        return (
          <div className="vt-analyst-state" data-state="network">
            <b>Request failed:</b> {r.error}
          </div>
        );
      case "ok": {
        const { env, execNotes } = r;
        const tokens = (env.usage?.input_tokens || 0) + (env.usage?.output_tokens || 0);
        const open = !!traceOpen[it.id];
        return (
          <div className="vt-analyst-answer">
            <p className="vt-analyst-answer-text">{env.answer}</p>
            {execNotes.map((n, i) => (
              <div key={i} className="vt-analyst-exec" role="status">→ {n}</div>
            ))}
            {Array.isArray(env.sources) && env.sources.length > 0 && (
              <div className="vt-analyst-chips" aria-label="Data sources used">
                {env.sources.map((s, i) => {
                  const age = fmtFresh(s.freshness);
                  return (
                    <span key={i} className="vt-analyst-chip"
                          title={`${s.tool} ${fmtParams(s.params)}${s.freshness ? ` · data as of ${s.freshness}` : ""}`}>
                      {s.tool}{age ? <i> · {age}</i> : null}
                    </span>
                  );
                })}
              </div>
            )}
            {Array.isArray(env.tool_trace) && env.tool_trace.length > 0 && (
              <div className="vt-analyst-trace">
                <button className="vt-analyst-trace-btn" aria-expanded={open}
                        onClick={() => setTraceOpen((s) => ({ ...s, [it.id]: !s[it.id] }))}>
                  <span className={`vt-layer-group-chev${open ? "" : " closed"}`}>▾</span>
                  how I got this — {env.tool_trace.length} tool call{env.tool_trace.length === 1 ? "" : "s"}
                </button>
                {open && env.tool_trace.map((t, i) => (
                  <div key={i} className="vt-analyst-trace-row">
                    <i style={{ background: t.ok ? "var(--accent-green)" : "var(--accent-red)" }} />
                    <span className="vt-analyst-trace-tool">{t.tool}</span>
                    <span className="vt-analyst-trace-sum">{t.summary}</span>
                  </div>
                ))}
              </div>
            )}
            {env.note && <div className="vt-analyst-note">{env.note}</div>}
            {env.transport_error && <div className="vt-analyst-note">{env.transport_error}</div>}
            <div className="vt-analyst-foot">
              {tokens.toLocaleString()} tokens · today{" "}
              {env.budget?.spent_today?.toLocaleString?.() ?? "?"}/{env.budget?.daily_limit?.toLocaleString?.() ?? "?"}
              {" · "}{env.model}
            </div>
          </div>
        );
      }
    }
  };

  return (
    <div className="vt-analyst-panel" data-vt-analyst-panel role="dialog" aria-label="Analyst">
      <div className="vt-analyst-head">
        <span className="vt-analyst-title">
          <MessageSquareText size={14} /> Analyst
          <span className="vt-analyst-title-sub">answers only from platform data</span>
        </span>
        <button className="vt-icon-btn" aria-label="Close analyst" onClick={onClose}>
          <X size={15} />
        </button>
      </div>
      <div className="vt-analyst-msgs">
        {items.length === 0 && (
          <div className="vt-analyst-intro">
            <p>
              Ask about facilities, fleets, flows, and grids. Answers come
              strictly from the platform's own data tools — every figure cites
              its source, and "I don't have that" is a valid answer.
            </p>
            <div className="vt-analyst-suggest">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="vt-analyst-suggest-btn"
                        onClick={() => { setInput(s); try { inputRef.current?.focus(); } catch {} }}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {items.map((it) => (
          <div key={it.id} className="vt-analyst-item">
            <div className="vt-analyst-q">{it.question}</div>
            {renderResp(it)}
          </div>
        ))}
        <div ref={endRef} />
      </div>
      <div className="vt-analyst-inputrow">
        <textarea
          ref={inputRef}
          className="vt-analyst-input"
          data-vt-analyst-input
          rows={2}
          maxLength={QUESTION_MAX}
          placeholder="Ask the analyst…"
          aria-label="Question for the analyst"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
        />
        <button
          className="vt-analyst-send"
          data-vt-analyst-send
          aria-label="Send question"
          disabled={busy || !input.trim()}
          onClick={send}
        >
          <Send size={16} />
        </button>
      </div>
      {input.length > QUESTION_MAX - 200 && (
        <div className="vt-analyst-charcount">{input.length}/{QUESTION_MAX}</div>
      )}
    </div>
  );
}
