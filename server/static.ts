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

  // Landing page at root
  const landingPath = path.resolve(__dirname, "..", "landing");
  if (fs.existsSync(landingPath)) {
    app.use("/landing", express.static(landingPath));
    app.get("/", (_req, res) => {
      res.sendFile(path.resolve(landingPath, "index.html"));
    });
  }

  // Dashboard app at /app and all sub-routes
  app.use(express.static(distPath));
  app.get("/app", (_req, res) => {
    res.sendFile(path.resolve(distPath, "index.html"));
  });

  // fall through to dashboard index.html for any unmatched route
  app.use("/{*path}", (_req, res) => {
    res.sendFile(path.resolve(distPath, "index.html"));
  });
}
