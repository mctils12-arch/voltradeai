// layerRegistry — the /data layers panel's ONE registry load
// (/api/data/layers), made retryable.
//
// [REPAIR 2026-10-14] The panel's every toggle row comes from this one
// response. It used to be a bare one-shot fetch whose .catch set the list to
// [] for the whole session: one failure at page start (Chrome restoring the
// tab before the network is up, a deploy switching containers, a blip) and
// every layer group silently vanished. The panel went straight from the
// dashboard launchers to CELESTIAL with no error. Meanwhile the map's own
// layers kept polling and recovered, so the page looked alive with its
// controls missing. The human hit this on startup. Now any failure (network
// error, non-2xx, non-JSON, empty list) THROWS, so runResilientLoad retries
// it with backoff. The panel says it is retrying instead of rendering
// nothing.
//
// DOM/React-free and fetch-injectable so tsx --test exercises it directly.

export interface LayerRegistryPayload<T> {
  layers: T[];
  /** the server's package version; null when the response omits it */
  serverVersion: string | null;
}

/** Pure: validate a parsed registry body. Throws on any shape that would
 *  leave the panel empty — an empty registry is never a real state (the
 *  static datacore/layers.json always has rows), so it is retried, not shown. */
export function parseLayerRegistry<T = unknown>(body: unknown): LayerRegistryPayload<T> {
  const b = body as { layers?: unknown; server_version?: unknown } | null;
  if (!b || !Array.isArray(b.layers)) throw new Error("layer registry: no layers array");
  if (b.layers.length === 0) throw new Error("layer registry: empty layers array");
  return {
    layers: b.layers as T[],
    serverVersion: b.server_version != null && b.server_version !== "" ? String(b.server_version) : null,
  };
}

type FetchLike = (url: string, init?: { signal?: AbortSignal }) => Promise<{
  ok: boolean; status: number; json: () => Promise<unknown>;
}>;

/** One attempt: fetch + status check + parse. Throws on every failure mode
 *  so runResilientLoad's backoff takes over. */
export async function fetchLayerRegistry<T = unknown>(
  fetchFn: FetchLike, signal?: AbortSignal,
): Promise<LayerRegistryPayload<T>> {
  const r = await fetchFn("/api/data/layers", { signal });
  if (!r.ok) throw new Error(`layer registry: HTTP ${r.status}`);
  return parseLayerRegistry<T>(await r.json());
}
