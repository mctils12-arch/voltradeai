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

  // Landing page at /bot
  const landingPath = path.resolve(__dirname, "..", "landing");
  if (fs.existsSync(landingPath)) {
    app.use("/bot", express.static(landingPath));
    app.get("/bot", (_req, res) => {
      res.sendFile(path.resolve(landingPath, "index.html"));
    });
  }

  // Main app served at root and all sub-routes
  app.use(express.static(distPath));

  // fall through to app index.html for any unmatched route
  app.use("/{*path}", (_req, res) => {
    res.sendFile(path.resolve(distPath, "index.html"));
  });
}
