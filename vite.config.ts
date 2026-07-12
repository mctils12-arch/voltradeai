import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "client", "src"),
      "@shared": path.resolve(import.meta.dirname, "shared"),
      "@assets": path.resolve(import.meta.dirname, "attached_assets"),
    },
  },
  root: path.resolve(import.meta.dirname, "client"),
  // Absolute, not relative ("./"): the app is always served from the
  // domain root (server/static.ts's SPA fallback + express.static both
  // assume that), and a relative base resolves against the CURRENT URL's
  // path — for any 2+-segment route (e.g. /newsletter/:slug) the browser
  // requested a nonexistent nested asset path instead of /assets/*, which
  // 404'd to the SPA fallback and got refused as a script (wrong MIME
  // type) -> blank white screen. See open_questions.md OPS GOTCHAS.
  base: "/",
  build: {
    outDir: path.resolve(import.meta.dirname, "dist/public"),
    emptyOutDir: true,
  },
  server: {
    fs: {
      strict: true,
      deny: ["**/.*"],
    },
  },
});
