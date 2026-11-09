// volumeUsage — READ-ONLY breakdown of what fills the persistent volume
// (2026-11-09). Why: the archive offload status shows the local archive at
// ~0.6 GB yet the volume sits at ~1.0 GB free, under the 1 GiB global-scope
// guard, so ~1 GB is NOT archive and nothing could name it. Pure + injectable
// fs (node:test safe). Bounded walk: depth/entry caps, no symlink following,
// returns names + byte counts only (never file contents).
import fs from "node:fs";
import path from "node:path";

export const VOLUME_MAX_DEPTH = 3;
export const VOLUME_MAX_ENTRIES = 200_000;

export interface VolumeNode { path: string; bytes: number; files: number }
export interface VolumeUsage {
  root: string;
  totalBytes: number;
  totalFiles: number;
  truncated: boolean;
  top: VolumeNode[];
  freeBytes: number | null;
}

export function volumeUsage(
  root: string,
  opts: { maxDepth?: number; maxEntries?: number; topN?: number } = {},
  fsx: Pick<typeof fs, "readdirSync" | "lstatSync" | "statfsSync"> = fs,
): VolumeUsage {
  const maxDepth = opts.maxDepth ?? VOLUME_MAX_DEPTH;
  const maxEntries = opts.maxEntries ?? VOLUME_MAX_ENTRIES;
  const nodes = new Map<string, VolumeNode>();
  let seen = 0;
  let truncated = false;
  let totalBytes = 0;
  let totalFiles = 0;

  const bump = (rel: string, bytes: number) => {
    const n = nodes.get(rel) ?? { path: rel, bytes: 0, files: 0 };
    n.bytes += bytes; n.files += 1;
    nodes.set(rel, n);
  };

  const walk = (dir: string, rel: string[]) => {
    let names: string[];
    try { names = fsx.readdirSync(dir) as string[]; } catch { return; }
    for (const name of names) {
      if (seen >= maxEntries) { truncated = true; return; }
      seen++;
      const full = path.join(dir, name);
      let st: fs.Stats;
      try { st = fsx.lstatSync(full) as fs.Stats; } catch { continue; }
      if (st.isSymbolicLink()) continue;
      if (st.isDirectory()) { walk(full, [...rel, name]); continue; }
      totalBytes += st.size; totalFiles++;
      // attribute the file to each ancestor directory up to maxDepth
      const parts = rel.length ? rel : ["(root files)"];
      for (let d = 1; d <= Math.min(parts.length, maxDepth); d++) bump(parts.slice(0, d).join("/"), st.size);
    }
  };
  walk(root, []);

  let freeBytes: number | null = null;
  try { const s: any = (fsx as any).statfsSync(root); freeBytes = Number(s.bavail) * Number(s.bsize); } catch { freeBytes = null; }
  const top = [...nodes.values()].sort((a, b) => b.bytes - a.bytes).slice(0, opts.topN ?? 40);
  return { root: path.basename(root), totalBytes, totalFiles, truncated, top, freeBytes };
}
