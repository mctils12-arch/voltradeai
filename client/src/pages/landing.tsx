import { useEffect, useRef } from "react";
import { useLocation } from "wouter";

// Vite supports importing files as raw strings with the ?raw suffix.
// This keeps the landing markup, styles, and scripts in plain files
// (easy to edit) while bundling them into the React component.
import landingStyles from "./_landing_styles.css.txt?raw";
import landingBody from "./_landing_body.html.txt?raw";
import landingScript from "./_landing_script.js.txt?raw";

const D3_CDN = "https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js";
const TOPOJSON_CDN = "https://cdnjs.cloudflare.com/ajax/libs/topojson/3.0.2/topojson.min.js";

/** Load an external script once. Resolves when ready. */
function loadScriptOnce(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) {
      // Already injected — assume loaded
      resolve();
      return;
    }
    const s = document.createElement("script");
    s.src = src;
    s.async = false;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(s);
  });
}

export default function LandingPage() {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [, navigate] = useLocation();

  useEffect(() => {
    let cancelled = false;
    let cleanups: Array<() => void> = [];

    async function bootLanding() {
      // 1. Load D3 + topojson from CDN
      try {
        await loadScriptOnce(D3_CDN);
        await loadScriptOnce(TOPOJSON_CDN);
      } catch (err) {
        console.error("Landing page: failed to load D3/topojson", err);
        return;
      }
      if (cancelled) return;

      // 2. Run the landing's IIFE script. It manipulates the DOM by id (#world-canvas, #cities, #arcs-svg, etc.)
      //    Wrap it in an extra IIFE just in case.
      try {
        // eslint-disable-next-line no-new-func
        const fn = new Function(landingScript);
        fn();
      } catch (err) {
        console.error("Landing page: script execution failed", err);
      }

      // 3. Intercept clicks on internal [data-route] links so wouter handles them
      //    (without a full page reload).
      const root = rootRef.current;
      if (root) {
        const handler = (e: MouseEvent) => {
          const target = (e.target as HTMLElement).closest("a[data-route]") as HTMLAnchorElement | null;
          if (!target) return;
          const route = target.getAttribute("data-route");
          if (!route) return;
          e.preventDefault();
          navigate(route);
        };
        root.addEventListener("click", handler);
        cleanups.push(() => root.removeEventListener("click", handler));
      }
    }

    bootLanding();

    // ── DATA INTELLIGENCE section wiring (additive, directive 2026-07-04).
    // Lives HERE, not in the D3-gated landing script: live stats and the
    // waitlist form must work even if the globe's CDN scripts fail —
    // graceful degradation means content degrades to placeholders, never
    // to a dead form.
    const num = (id: string, v: number) => {
      const el = document.getElementById(id);
      if (el && Number.isFinite(v)) el.textContent = v.toLocaleString();
    };
    fetch("/api/data/layers").then((r) => r.json()).then((d) => {
      const live = (d.layers || []).filter((l: any) => l.status === "live").length;
      if (live > 0) num("di-layers", live);
    }).catch(() => {});
    fetch("/api/data/archive/stats").then((r) => r.json()).then((d) => {
      const kinds = d.kinds || {};
      const names = Object.keys(kinds);
      const samples = names.reduce((s, k) => s + (kinds[k]?.samples || 0), 0);
      if (names.length > 0) num("di-streams", names.length);
      if (samples > 0) num("di-samples", samples);
    }).catch(() => {});
    const form = document.getElementById("dataintel-waitlist");
    if (form) {
      const onSubmit = (e: Event) => {
        e.preventDefault();
        const input = document.getElementById("di-email") as HTMLInputElement | null;
        const msg = document.getElementById("di-waitlist-msg");
        const email = input?.value?.trim() || "";
        if (!email) { if (msg) msg.textContent = "Enter an email address."; return; }
        if (msg) msg.textContent = "Joining…";
        fetch("/api/waitlist", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, source: "landing" }),
        }).then((r) => r.json()).then((d) => {
          if (!msg) return;
          if (d?.ok) {
            msg.textContent = d.status === "duplicate"
              ? "Already on the list — you're set."
              : "You're on the list. No billing, just early-access news.";
            if (input) input.value = "";
          } else {
            msg.textContent = d?.error || "Something went wrong — try again.";
          }
        }).catch(() => { if (msg) msg.textContent = "Network hiccup — try again."; });
      };
      form.addEventListener("submit", onSubmit);
      cleanups.push(() => form.removeEventListener("submit", onSubmit));
    }

    // ── HERO GLOBE (directive 2026-07-05, #data-intel only): live rotating
    // globe as the section background — REAL aircraft/vessels/sites on a
    // land silhouette from OUR self-hosted boundaries (no external tiles).
    // Gated on: section approaching the viewport (no cost until scrolled
    // near), WebGL availability, and maplibre's lazy chunk loading. Any
    // failure leaves the styled dark backdrop — never a janky globe.
    // Rotation pauses off-screen, on hidden tabs, and under
    // prefers-reduced-motion. interactive:false + pointer-events:none —
    // the hero can never hijack scroll.
    (() => {
      const slot = document.getElementById("di-globe");
      if (!slot) return;
      const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
      let booted = false;
      let inView = false;
      let raf = 0;
      let refreshIv = 0;
      let globeMap: any = null;

      const webglOk = () => {
        try {
          const c = document.createElement("canvas");
          return Boolean(c.getContext("webgl2") || c.getContext("webgl"));
        } catch { return false; }
      };

      const bootGlobe = async () => {
        if (booted || cancelled) return;
        booted = true;
        if (!webglOk()) return; // graceful: backdrop stays
        try {
          const maplibregl = (await import("maplibre-gl")).default;
          if (cancelled) return;
          globeMap = new maplibregl.Map({
            container: slot,
            interactive: false,
            attributionControl: false,
            style: {
              version: 8,
              projection: { type: "globe" } as any,
              sources: {},
              layers: [{ id: "space", type: "background", paint: { "background-color": "#04070f" } }],
            } as any,
            center: [-55, 24],
            zoom: 1.05,
          });
          globeMap.on("load", async () => {
            if (cancelled) return;
            const safeJson = (u: string) => fetch(u).then((r) => r.json()).catch(() => null);
            const [land, sites, aircraft, vessels] = await Promise.all([
              safeJson("/api/data/boundaries"),
              safeJson("/api/data/sites"),
              safeJson("/api/data/aircraft?lamin=-85&lamax=85&lomin=-180&lomax=180"),
              safeJson("/api/data/vessels?lamin=-85&lamax=85&lomin=-180&lomax=180"),
            ]);
            if (cancelled || !globeMap) return;
            if (land?.features) {
              globeMap.addSource("di-land", { type: "geojson", data: land });
              globeMap.addLayer({
                id: "di-land-fill", type: "fill", source: "di-land",
                paint: { "fill-color": "#12233f", "fill-outline-color": "#23406b" },
              });
            }
            const pts = (rows: any[], la: string, lo: string) => ({
              type: "FeatureCollection",
              features: (rows || []).map((r: any) => ({
                type: "Feature",
                geometry: { type: "Point", coordinates: [r[lo], r[la]] },
                properties: {},
              })),
            });
            if (aircraft?.aircraft?.length) {
              globeMap.addSource("di-air", { type: "geojson", data: pts(aircraft.aircraft.slice(0, 400), "lat", "lon") });
              globeMap.addLayer({ id: "di-air-p", type: "circle", source: "di-air",
                paint: { "circle-radius": 1.4, "circle-color": "#4d9fff", "circle-opacity": 0.9 } });
            }
            if (vessels?.vessels?.length) {
              globeMap.addSource("di-sea", { type: "geojson", data: pts(vessels.vessels.slice(0, 300), "lat", "lon") });
              globeMap.addLayer({ id: "di-sea-p", type: "circle", source: "di-sea",
                paint: { "circle-radius": 1.4, "circle-color": "#4ade80", "circle-opacity": 0.9 } });
            }
            if (sites?.sites?.length) {
              globeMap.addSource("di-sites", { type: "geojson", data: pts(sites.sites, "lat", "lon") });
              globeMap.addLayer({ id: "di-sites-p", type: "circle", source: "di-sites",
                paint: { "circle-radius": 3, "circle-color": "#fbb24c",
                         "circle-stroke-color": "rgba(251,178,76,0.35)", "circle-stroke-width": 3 } });
            }
            // live movement: refresh aircraft positions while the hero is on
            // screen (30s — one light request; the visual motion between
            // refreshes comes from the rotation)
            refreshIv = window.setInterval(async () => {
              if (!inView || document.hidden || cancelled || !globeMap) return;
              const d = await safeJson("/api/data/aircraft?lamin=-85&lamax=85&lomin=-180&lomax=180");
              const src: any = globeMap.getSource("di-air");
              if (d?.aircraft?.length && src) src.setData(pts(d.aircraft.slice(0, 400), "lat", "lon"));
            }, 30_000);
            if (!reduced) {
              const spin = () => {
                if (cancelled || !globeMap) return;
                if (inView && !document.hidden) {
                  const c = globeMap.getCenter();
                  globeMap.setCenter([c.lng + 0.02, c.lat]);
                }
                raf = requestAnimationFrame(spin);
              };
              raf = requestAnimationFrame(spin);
            }
          });
        } catch {
          /* graceful: styled backdrop stays */
        }
      };

      const io = new IntersectionObserver((entries) => {
        for (const e of entries) {
          inView = e.isIntersecting;
          if (e.isIntersecting) bootGlobe();
        }
      }, { rootMargin: "400px" });
      const section = document.getElementById("data-intel");
      if (section) io.observe(section);
      cleanups.push(() => {
        io.disconnect();
        if (raf) cancelAnimationFrame(raf);
        if (refreshIv) window.clearInterval(refreshIv);
        try { globeMap?.remove(); } catch {}
        globeMap = null;
      });
    })();

    return () => {
      cancelled = true;
      cleanups.forEach((fn) => fn());
    };
  }, [navigate]);

  return (
    <div ref={rootRef} className="vt-landing-root">
      {/* Inline the page-specific styles. Scoping isn't needed because this
          page is rendered exclusively at the "/" route. */}
      <style dangerouslySetInnerHTML={{ __html: landingStyles }} />
      <div dangerouslySetInnerHTML={{ __html: landingBody }} />
    </div>
  );
}
