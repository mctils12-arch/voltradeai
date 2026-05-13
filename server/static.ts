import express, { type Express } from "express";
import fs from "fs";
import path from "path";

export function serveStatic(app: Express) {
  const distPath = path.resolve(__dirname, "public");
  if (!fs.existsSync(distPath)) {
    throw new Error(
      `Could not find the build directory: ${distPath}, make sure to build the client first`,
    );
  }

  /**
   * Cache strategy:
   *   - Hashed assets (e.g. /assets/*.js, *.css with content hash in filename)
   *     → cache aggressively forever (1 year, immutable)
   *   - index.html → never cache; always fetch fresh so users get the latest
   *     bundle references after a deploy. Without this, returning users see
   *     stale HTML pointing at JS chunks that no longer exist on the server,
   *     which produces "Failed to fetch dynamically imported module" errors.
   */
  app.use(
    express.static(distPath, {
      setHeaders: (res, filePath) => {
        // Vite emits hashed filenames under /assets/ — they're immutable
        if (filePath.includes(`${path.sep}assets${path.sep}`) ||
            filePath.includes("/assets/")) {
          res.setHeader("Cache-Control", "public, max-age=31536000, immutable");
        }
        // index.html (and any HTML at root) must never be cached
        else if (filePath.endsWith(".html")) {
          res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
          res.setHeader("Pragma", "no-cache");
          res.setHeader("Expires", "0");
        }
      },
    })
  );

  // SPA fallback: any unmatched route returns a fresh index.html.
  // No-cache headers ensure the user always gets the latest bundle references.
  app.use("/{*path}", (_req, res) => {
    res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
    res.setHeader("Pragma", "no-cache");
    res.setHeader("Expires", "0");
    res.sendFile(path.resolve(distPath, "index.html"));
  });
}
