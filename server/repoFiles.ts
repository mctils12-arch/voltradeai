/**
 * repoFiles.ts — resolve repo data files that must be READ FROM DISK at
 * runtime (not statically imported/bundled).
 *
 * [REPAIR 2026-07-07] The production image (frozen Dockerfile) ships
 * dist/, content/, *.py, strategies/, alphadesk/ — NOT the repo datacore/
 * tree. Every server-side `process.cwd()/datacore/...` disk read therefore
 * silently found nothing on Railway: /api/data/platform/stats served
 * sentinel2_last_reading:null since the audit-defect-#9 repair shipped
 * (null was a designed value, so nothing looked broken), and
 * /api/data/streams served an empty inventory on deploy day. Fix class,
 * not instance: script/build.ts now copies the runtime-needed datacore
 * files into dist/datacore/, and this resolver falls back to that bundled
 * copy when the repo tree is absent. Statically-imported JSON (layers.json,
 * entity_spine.json, …) was never affected — esbuild inlines those.
 */
import fs from "fs";
import path from "path";

/** Resolve a repo-relative data path: prefer the working tree (dev,
 *  tests, CI), fall back to the dist/ copy (production image). When
 *  neither exists, the direct path is returned so callers can surface
 *  the miss honestly — never silently. */
export function repoDataPath(rel: string, cwd: string = process.cwd()): string {
  const direct = path.resolve(cwd, rel);
  if (fs.existsSync(direct)) return direct;
  const bundled = path.resolve(cwd, "dist", rel);
  if (fs.existsSync(bundled)) return bundled;
  return direct;
}
