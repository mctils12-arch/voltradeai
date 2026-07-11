import { useEffect, useState } from "react";
import { Link } from "wouter";
import { apiRequest } from "@/lib/queryClient";

/**
 * /account/api-keys — self-serve preview API key management (PLATFORM
 * INTEGRATION P3, research/platform_program.md). Every key issued here is
 * the fixed "dev" preview tier — free, instant, bound to the logged-in
 * account. No billing, no checkout, no pricing enablement: this replaces
 * "join the waitlist and wait for an email" with an instant preview key at
 * the SAME limits, nothing more. Full docs live on /developers.
 */

interface AuthMe { authenticated: boolean; email?: string }
interface ApiKeySummary {
  id: number;
  label: string;
  keyPreview: string;
  createdAt: number;
  lastUsedAt: number | null;
  revokedAt: number | null;
  usageToday: number;
}
interface KeysResponse {
  keys: ApiKeySummary[];
  max_keys: number;
  tier: string;
  limits: { perMinute: number; perDay: number };
}

function fmtDate(ms: number | null): string {
  if (!ms) return "never";
  return new Date(ms).toLocaleString();
}

/** apiRequest() throws `"<status>: <bodyText>"` — pull the JSON error field
 *  out of that body when present, falling back to the raw message. */
function extractErrorMessage(err: unknown, fallback: string): string {
  const raw = err instanceof Error ? err.message : "";
  const bodyStart = raw.indexOf(": {");
  if (bodyStart === -1) return raw || fallback;
  try {
    const parsed = JSON.parse(raw.slice(bodyStart + 2));
    return parsed?.error || raw || fallback;
  } catch {
    return raw || fallback;
  }
}

export default function ApiKeysPage() {
  const [me, setMe] = useState<AuthMe | null>(null);
  const [data, setData] = useState<KeysResponse | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [label, setLabel] = useState("");
  const [creating, setCreating] = useState(false);
  const [createErr, setCreateErr] = useState<string | null>(null);
  const [justCreated, setJustCreated] = useState<{ key: string; label: string } | null>(null);
  const [copied, setCopied] = useState(false);
  const [revokingId, setRevokingId] = useState<number | null>(null);

  const loadKeys = () => {
    apiRequest("GET", "/api/account/api-keys")
      .then((r) => r.json())
      .then((d) => { setData(d); setLoadErr(null); })
      .catch(() => setLoadErr("Couldn't load your keys — try refreshing."));
  };

  useEffect(() => {
    apiRequest("GET", "/api/auth/me")
      .then((r) => r.json())
      .then((d) => { setMe(d); if (d.authenticated) loadKeys(); })
      .catch(() => setMe({ authenticated: false }));
  }, []);

  const createKey = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);
    setCreateErr(null);
    try {
      const r = await apiRequest("POST", "/api/account/api-keys", { label });
      const d = await r.json();
      setJustCreated({ key: d.key, label: d.label });
      setLabel("");
      loadKeys();
    } catch (err) {
      setCreateErr(extractErrorMessage(err, "Couldn't create a key — try again."));
    } finally {
      setCreating(false);
    }
  };

  const revoke = async (id: number) => {
    setRevokingId(id);
    try {
      await apiRequest("DELETE", `/api/account/api-keys/${id}`);
      loadKeys();
    } catch {
      setLoadErr("Couldn't revoke that key — try again.");
    } finally {
      setRevokingId(null);
    }
  };

  const copyKey = (key: string) => {
    navigator.clipboard?.writeText(key).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    }).catch(() => {});
  };

  if (!me) {
    return (
      <div className="vt-dev-page vt-keys-page">
        <p className="vt-dev-caption">loading your account…</p>
      </div>
    );
  }

  if (!me.authenticated) {
    return (
      <div className="vt-dev-page vt-keys-page">
        <header className="vt-dev-hero">
          <h1>My API keys</h1>
          <p>Log in to generate preview API keys bound to your account — free during the preview, no waitlist required.</p>
        </header>
        <Link href="/login" className="vt-keys-login-link">Log in →</Link>
      </div>
    );
  }

  const activeCount = data ? data.keys.filter((k) => !k.revokedAt).length : 0;

  return (
    <div className="vt-dev-page vt-keys-page">
      <header className="vt-dev-hero">
        <h1>My API keys</h1>
        <p>
          Preview keys for the /api/v1 data product — free, instant, capped at{" "}
          {data ? `${data.limits.perMinute.toLocaleString()} requests/min and ${data.limits.perDay.toLocaleString()}/day` : "the preview limits"}.
          Full endpoint reference and examples live on <Link href="/developers">/developers</Link>.
        </p>
      </header>

      <section className="vt-dev-section">
        <h2>Generate a new key <span className="vt-dev-badge">preview · free</span></h2>
        {data && activeCount >= data.max_keys ? (
          <p className="vt-dev-caption">
            You've reached the preview limit of {data.max_keys} active keys — revoke one below to make room.
          </p>
        ) : (
          <form className="vt-keys-create" onSubmit={createKey}>
            <input
              type="text" placeholder='Label (e.g. "local dev")' value={label}
              maxLength={60} aria-label="Key label"
              onChange={(e) => setLabel(e.target.value)}
            />
            <button type="submit" disabled={creating}>{creating ? "Generating…" : "Generate key"}</button>
          </form>
        )}
        {createErr && <p className="vt-dev-wl-err">{createErr}</p>}

        {justCreated && (
          <div className="vt-keys-reveal">
            <p className="vt-keys-reveal-warn">Save this key now — it will not be shown again.</p>
            <div className="vt-dev-code-row">
              <pre className="vt-dev-code vt-dev-code-inline">{justCreated.key}</pre>
              <button type="button" className="vt-dev-copy-btn" onClick={() => copyKey(justCreated.key)}>
                {copied ? "Copied" : "Copy"}
              </button>
            </div>
          </div>
        )}
      </section>

      <section className="vt-dev-section">
        <h2>Your keys</h2>
        {loadErr && <p className="vt-dev-wl-err">{loadErr}</p>}
        {!data && !loadErr && <p className="vt-dev-caption">loading your keys…</p>}
        {data && data.keys.length === 0 && (
          <p className="vt-dev-caption">No keys yet — generate one above to start calling /api/v1.</p>
        )}
        {data && data.keys.length > 0 && (
          <div className="vt-dev-table-wrap">
            <table className="vt-dev-table">
              <thead>
                <tr>
                  <th>Label</th><th>Key</th><th>Created</th><th>Last used</th><th>Today</th><th>Status</th><th aria-label="actions" />
                </tr>
              </thead>
              <tbody>
                {data.keys.map((k) => (
                  <tr key={k.id}>
                    <td>{k.label}</td>
                    <td><code>{k.keyPreview}</code></td>
                    <td>{fmtDate(k.createdAt)}</td>
                    <td>{fmtDate(k.lastUsedAt)}</td>
                    <td>{k.usageToday.toLocaleString()} req</td>
                    <td>{k.revokedAt ? "revoked" : "active"}</td>
                    <td>
                      {!k.revokedAt && (
                        <button
                          type="button" className="vt-keys-revoke-btn"
                          disabled={revokingId === k.id}
                          onClick={() => revoke(k.id)}
                        >
                          {revokingId === k.id ? "Revoking…" : "Revoke"}
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
