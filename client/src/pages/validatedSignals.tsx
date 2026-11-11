// ValidatedSignalsView — the Signals PRODUCT hub (#/data/validated-signals):
// the nav's "Signals — validated signal products" leaf (PLATFORM INTEGRATION
// PROGRAM), which sat disabled as "soon" from 2026-07-11 while two gate-2-
// passed signals were already live on /data and /api/v1. Reads
// /api/data/signal-products (server/signalProducts.ts): the ladder decides
// which roots are products (gate >= 2 passed), a hand-maintained wiring
// table says where each one's live board / preview JSON / keyed endpoint /
// license mark live, and the existing live summaries are reduced to one
// headline each. Distinct from signalLadder.tsx (#/data/signals), which is
// the honesty ledger of EVERY root incl. killed/failed — this page is the
// storefront, and it says plainly that gate 2 is not gate 3.
// PREMIUM EXPERIENCE STANDARD (c): every card carries freshness (relative
// age of its reading), provenance (ladder record + source_ref) and the
// license posture a paying customer needs before they redistribute.
// Mobile-first (390px primary). Reuses .vt-filings/.vt-quality/.vt-ladder
// shells + the new .vt-sigprod-* tokens-only additions in index.css.
import { useCallback, useEffect, useState } from "react";
import { ArrowLeft, RefreshCw, Sparkles, ShieldCheck, Copy, Check } from "lucide-react";
import { relativeAge, licensePosture, curlFor } from "@/lib/signalProducts";

interface LiveReading { state: "live" | "warming_up" | "unavailable" | "unwired"; generated_at: string | null; headline: string }
interface ProductRow {
  root_id: string; product: string; measures: string; validated_channel: string; cadence: string;
  data_route: string; v1_path: string; detail_route: string;
  name: string; category: string; status: string; current_gate: number; last_update_date: string; source_ref: string;
  tradeable: false;
  license: { resell: string; attribution: string; note: string };
  live: LiveReading;
}
interface InValidationRow { id: string; name: string; status: string; current_gate: number; last_update_date: string; note: string }
interface Payload {
  generated_at: string;
  ladder_compiled: string;
  products: ProductRow[];
  unwired: string[];
  in_validation: InValidationRow[];
  gate_definitions: { gate: number; name: string; test: string }[];
  honesty: string;
}

const STATE_LABEL: Record<LiveReading["state"], string> = {
  live: "live", warming_up: "warming up", unavailable: "unavailable", unwired: "not wired",
};

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  const onCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setDone(true);
      window.setTimeout(() => setDone(false), 1400);
    } catch {
      // Clipboard can be denied in insecure contexts / iframes — the endpoint
      // is already selectable text beside the button, so nothing else to do.
      setDone(false);
    }
  }, [text]);
  return (
    <button type="button" className="vt-sigprod-copy" aria-label="Copy endpoint" onClick={() => void onCopy()}>
      {done ? <Check size={13} /> : <Copy size={13} />}
    </button>
  );
}

function ProductCard({ p, now, origin }: { p: ProductRow; now: number; origin: string }) {
  const posture = licensePosture(p.license.resell);
  const curl = curlFor(origin, p.v1_path);
  return (
    <article className="vt-sigprod-card" data-vt-sigprod={p.root_id}>
      <div className="vt-sigprod-card-top">
        <div style={{ minWidth: 0 }}>
          <div className="vt-sigprod-name">{p.product}</div>
          <div className="vt-sigprod-research">{p.name}</div>
        </div>
        <span className="vt-ladder-badge vt-ladder-badge-pass">gate {p.current_gate} passed</span>
      </div>

      <div className="vt-sigprod-live" title={p.live.generated_at ?? undefined}>
        <span className={`vt-sigprod-dot vt-sigprod-dot-${p.live.state}`} aria-hidden="true" />
        <span>{p.live.headline}</span>
        <span className="vt-sigprod-age">{STATE_LABEL[p.live.state]} · {relativeAge(p.live.generated_at, now)}</span>
      </div>

      <div>
        <div className="vt-sigprod-label">What it measures</div>
        <div className="vt-sigprod-text">{p.measures}</div>
      </div>
      <div>
        <div className="vt-sigprod-label">What gate 2 validated — and only that</div>
        <div className="vt-sigprod-text">{p.validated_channel}</div>
      </div>

      <div className="vt-sigprod-chips">
        <span className={`vt-sigprod-chip vt-sigprod-chip-${posture.cls}`} title={p.license.note}>{posture.label}</span>
        <span className="vt-sigprod-chip" title={posture.blurb}>{p.license.attribution || "attribution missing"}</span>
        <span className="vt-sigprod-chip">not tradeable · gate 3 not passed</span>
      </div>

      <div>
        <div className="vt-sigprod-label">API · x-api-key</div>
        <div className="vt-sigprod-endpoint">
          <code>{curl}</code>
          <CopyButton text={curl} />
        </div>
      </div>

      <div className="vt-sigprod-actions">
        <a className="vt-sigprod-btn" href={p.detail_route}>Open live board →</a>
        <a className="vt-sigprod-btn vt-sigprod-btn-quiet" href={p.data_route} target="_blank" rel="noreferrer">Preview JSON</a>
        <a className="vt-sigprod-btn vt-sigprod-btn-quiet" href="#/data/signals">Ladder record</a>
        <a className="vt-sigprod-btn vt-sigprod-btn-quiet" href="/developers">API docs</a>
      </div>

      <div className="vt-ladder-row-foot">
        <span>{p.cadence}</span>
        <span>·</span>
        <span>ladder updated {p.last_update_date}</span>
        <span>·</span>
        <span className="vt-ladder-row-src" title={p.source_ref}>source: {p.source_ref}</span>
      </div>
    </article>
  );
}

export default function ValidatedSignalsView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<Payload | null>(null);
  const [error, setError] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [now, setNow] = useState(() => Date.now());

  const load = useCallback(async () => {
    setRefreshing(true);
    try {
      const r = await fetch("/api/data/signal-products");
      if (!r.ok) throw new Error(String(r.status));
      const d = (await r.json()) as Payload;
      if (!Array.isArray(d.products)) throw new Error("shape");
      setData(d);
      setNow(Date.now());
      setError(false);
    } catch {
      setError(true);
    } finally {
      setRefreshing(false);
    }
  }, []);
  useEffect(() => { void load(); }, [load]);

  const origin = typeof window !== "undefined" ? window.location.origin : "https://voltradeai.com";
  const count = data?.products.length ?? 0;

  return (
    <div className="vt-filings-page" role="region" aria-label="Validated signal products">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Sparkles size={16} />
        <div>
          <div className="vt-filings-title">Signals</div>
          <div className="vt-filings-sub">
            {data ? `${count} validated signal product${count === 1 ? "" : "s"} · gate 2 passed · none tradeable yet`
              : error ? "catalog unavailable — retry below" : "loading…"}
          </div>
        </div>
        <button className="vt-icon-btn" aria-label="Refresh catalog" onClick={() => void load()}
                style={{ marginLeft: "auto" }} disabled={refreshing}>
          <RefreshCw size={15} className={refreshing ? "vt-spin" : undefined} />
        </button>
      </div>

      <div className="vt-streams-body om-sb">
        {error && !data && (
          <div className="vt-filings-state">
            Could not load the signal catalog. <button type="button" className="vt-graph-example" onClick={() => void load()}>Retry</button>
          </div>
        )}

        {data && (
          <>
            <div className="vt-sigprod-banner" role="note">
              <ShieldCheck size={16} />
              <div><b>How to read this page.</b> {data.honesty}</div>
            </div>

            {data.unwired.length > 0 && (
              <div className="vt-gridstress-banner" role="alert">
                <ShieldCheck size={16} />
                <div>
                  <b>{data.unwired.length} validated root{data.unwired.length === 1 ? "" : "s"} not yet wired as a product:</b>{" "}
                  {data.unwired.join(", ")} — the ladder says gate 2 passed, but no live board or endpoint is registered here yet.
                </div>
              </div>
            )}

            <section className="vt-quality-section">
              <div className="vt-quality-section-head">
                <span>Products</span>
                <span className="vt-quality-section-sub">{count} · live reading, license posture, endpoint</span>
              </div>
              {count === 0 ? (
                <div className="vt-filings-state">No root has passed gate 2 yet — the ladder dashboard shows what is being tested.</div>
              ) : (
                <div className="vt-sigprod-grid">
                  {data.products.map((p) => <ProductCard key={p.root_id} p={p} now={now} origin={origin} />)}
                </div>
              )}
            </section>

            <section className="vt-quality-section">
              <div className="vt-quality-section-head">
                <span>In validation</span>
                <span className="vt-quality-section-sub">{data.in_validation.length} at gate 2 pending · not products</span>
              </div>
              {data.in_validation.length === 0 ? (
                <div className="vt-filings-state">Nothing is mid-test at gate 2 right now.</div>
              ) : (
                <div className="vt-ladder-list">
                  {data.in_validation.map((r) => (
                    <div key={r.id} className="vt-ladder-row" data-vt-sigprod-pending={r.id}>
                      <div className="vt-ladder-row-top">
                        <span className="vt-ladder-row-name">{r.name}</span>
                        <span className="vt-ladder-badge vt-ladder-badge-pending">gate 2 pending</span>
                      </div>
                      <div className="vt-ladder-row-note">{r.note}</div>
                      <div className="vt-ladder-row-foot">
                        <span>updated {r.last_update_date}</span>
                        <span>·</span>
                        <a href="#/data/signals" className="vt-graph-example">full ladder record →</a>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>

            <section className="vt-quality-section">
              <div className="vt-quality-section-head">
                <span>How a signal earns a place here</span>
                <span className="vt-quality-section-sub">the five-gate validation ladder · a product needs gate 2 · the bot needs gate 3+</span>
              </div>
              <div className="vt-sigprod-gates">
                {data.gate_definitions.map((g) => (
                  <div key={g.gate} className="vt-sigprod-gate">
                    <span className={`vt-sigprod-gate-num${g.gate <= 2 ? " vt-sigprod-gate-num-on" : ""}`}>{g.gate}</span>
                    <span><b>{g.name}</b> — {g.test}</span>
                  </div>
                ))}
              </div>
            </section>

            <div className="vt-filings-sub" style={{ padding: "4px 0 8px" }}>
              Catalog generated {relativeAge(data.generated_at, now)} from the ladder snapshot compiled {data.ladder_compiled}; live readings
              carry their own age above. Full ladder incl. killed and failed roots: <a href="#/data/signals">Signal strength</a>.
            </div>
          </>
        )}
      </div>
    </div>
  );
}
