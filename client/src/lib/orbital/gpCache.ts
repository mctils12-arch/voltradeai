// Persistent GP/SATCAT catalog cache (IndexedDB) — CelesTrak POLITENESS +
// resilience.
//
// PRODUCTION OUTAGE 2026-07-18 (root cause): every page reload with the
// satellites layer on refetched the FULL ~13MB GP catalog — the only cache
// was in-memory per session. A day of heavy use tripped CelesTrak's
// documented over-fetch IP blocking, and the layer died for that user with
// "still retrying automatically…" (each retry another full fetch, making
// the block worse). That is both a self-inflicted outage and a violation
// of the free-community-feed politeness discipline the aircraft tiling
// implements upstream.
//
// POLICY NOW:
//   fresh (< TTL): serve from IndexedDB, ZERO network — reloads stop
//     touching CelesTrak entirely inside the TTL window.
//   stale: refetch; on success persist the new catalog.
//   fetch FAILURE with a last-good catalog present: serve the STALE
//     catalog and surface its age honestly (SGP4 propagation degrades
//     gracefully — the satellite cards already carry the element-age
//     uncertainty note). Freeze-over-fiction does not apply here: aged
//     REAL elements are real data, labeled.
//   fetch failure with nothing cached: the caller's resilient-load backoff
//     keeps retrying (that path now only exists for first-ever visitors
//     during a CelesTrak outage/block).
//
// Failure-safe: any IndexedDB error (private mode, quota, corrupt DB)
// degrades to "no cache" — never breaks the layer.

const DB_NAME = "vt-orbital";
const STORE = "catalogs";
const DB_VERSION = 1;

export interface CachedCatalog<T> {
  at: number; // epoch ms the catalog was fetched
  data: T;
}

function openDb(): Promise<IDBDatabase | null> {
  return new Promise((resolve) => {
    try {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        try {
          if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE);
        } catch { /* degraded below */ }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
      req.onblocked = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

export async function idbGetCatalog<T>(key: string): Promise<CachedCatalog<T> | null> {
  const db = await openDb();
  if (!db) return null;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE, "readonly");
      const req = tx.objectStore(STORE).get(key);
      req.onsuccess = () => {
        const v = req.result;
        resolve(v && typeof v.at === "number" && v.data ? (v as CachedCatalog<T>) : null);
        db.close();
      };
      req.onerror = () => { resolve(null); db.close(); };
    } catch {
      resolve(null);
      try { db.close(); } catch { /* already closed */ }
    }
  });
}

export async function idbSetCatalog<T>(key: string, data: T, at: number = Date.now()): Promise<void> {
  const db = await openDb();
  if (!db) return;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).put({ at, data }, key);
      tx.oncomplete = () => { resolve(); db.close(); };
      tx.onerror = () => { resolve(); db.close(); };
    } catch {
      resolve();
      try { db.close(); } catch { /* already closed */ }
    }
  });
}

// ── pure policy (unit-tested) ───────────────────────────────────────────────

export type CatalogPlan = "use-cached" | "refetch";

/** Fresh-cache decision: inside the TTL the cache is authoritative and the
 *  network is never touched (the politeness core of this module). */
export function catalogPlan(nowMs: number, cachedAtMs: number | null, ttlMs: number): CatalogPlan {
  if (cachedAtMs != null && nowMs - cachedAtMs < ttlMs) return "use-cached";
  return "refetch";
}

/** Honest status-note suffix when serving a stale catalog after a failed
 *  refetch. Age is stated plainly; the uncertainty claim mirrors the
 *  element-age note the satellite cards already carry. */
export function staleCatalogNote(ageMs: number): string {
  const h = ageMs / 3_600_000;
  const age = h < 48 ? `${h.toFixed(h < 10 ? 1 : 0)}h` : `${(h / 24).toFixed(1)}d`;
  return `catalog ${age} old — CelesTrak unreachable right now; positions propagate from the last real elements (orbit uncertainty grows with element age)`;
}
