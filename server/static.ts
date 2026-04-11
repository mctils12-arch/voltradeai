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

  // Dashboard app
  app.use(express.static(distPath));

  // fall through to dashboard index.html if the file doesn't exist
  app.use("/{*path}", (_req, res) => {
    res.sendFile(path.resolve(distPath, "index.html"));
  });
}
