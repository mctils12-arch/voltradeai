import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { readFileSync } from "fs";

// THE RELEASE VERSION TRAVELS IN index.html, NOT IN THE HASHED JS (map speed,
// 2026-11-19). The client compared `import { version } from package.json`
// with the registry's server_version to detect an open tab running an old
// bundle. That import put the version string inside the entry chunk, and
// every PR bumps the version — so every deploy renamed the 2.1 MB entry
// chunk (607 KB on the wire) even when no client file had changed: 167 of
// the 194 merges to main in the 14 days before this change touched nothing
// the bundle is built from. index.html is served no-store (server/static.ts)
// and is deployed together with the bundle it names, so a <meta> there
// states the same fact at no cache cost: a server-only deploy now leaves
// every hashed asset name unchanged and a returning visitor downloads no JS.
// Read by client/src/lib/clientVersion.ts.
function clientVersionMeta(): Plugin {
  const { version } = JSON.parse(readFileSync(path.resolve(import.meta.dirname, "package.json"), "utf-8"));
  return {
    name: "vt-client-version-meta",
    transformIndexHtml() {
      return [{ tag: "meta", attrs: { name: "vt-client-version", content: String(version) }, injectTo: "head" }];
    },
  };
}

export default defineConfig({
  plugins: [react(), clientVersionMeta()],
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
    rollupOptions: {
      output: {
        // DEPLOY-STABLE VENDOR CHUNKS (map speed, 2026-11-19). Rollup put its
        // CommonJS interop helper in the entry chunk, so the lazily loaded
        // maplibre-gl chunk (1.05 MB, 283 KB on the wire) began with
        // `import{g}from"./index-<hash>.js"`. The entry's hash changes on
        // every deploy (it embeds package.json's version, bumped by every PR),
        // so maplibre's own content hash changed with it: byte-identical
        // library, new file name, ~14 times a day. Every returning visitor
        // re-downloaded it (0.40 s measured on production, on the path to the
        // first map frame) and V8 threw its compiled-code cache away. With
        // the helper in a chunk of its own, a vendor chunk imports only that
        // chunk and keeps its name until the library itself changes.
        // scripts/visual_check.mjs asserts it on the built assets.
        manualChunks(id) {
          if (id.includes("commonjsHelpers")) return "cjs-interop";
        },
      },
    },
  },
  server: {
    fs: {
      strict: true,
      deny: ["**/.*"],
    },
  },
});
