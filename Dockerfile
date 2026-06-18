# ══════════════════════════════════════════════════════════════════
# Stage 1: Build (React client + esbuild server bundle + node_modules)
# ══════════════════════════════════════════════════════════════════
FROM node:20-slim AS builder

# build-essential needed for native Node addons (better-sqlite3)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Node packages (full — includes devDependencies for build)
COPY package.json package-lock.json ./
RUN npm ci --prefer-offline

# Copy source + build (React via Vite + server via esbuild)
COPY . .
RUN npm run build

# Re-install production-only deps for the runtime copy
RUN rm -rf node_modules && npm ci --prefer-offline --omit=dev

# ══════════════════════════════════════════════════════════════════
# Stage 2: Production runtime (no build tools, no devDependencies)
# ══════════════════════════════════════════════════════════════════
FROM node:20-slim

# Python runtime + libgomp (required by LightGBM for OpenMP threading)
# numpy/scipy/lightgbm all ship pre-built manylinux wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python packages ─────────────────────────────────────────────
COPY requirements.txt ./
RUN python3 -m pip install -r requirements.txt \
    --break-system-packages --quiet \
    && python3 -m pip cache purge 2>/dev/null || true

# ── Node production deps (pre-built with native addons from builder) ──
COPY --from=builder /app/node_modules ./node_modules
COPY package.json ./

# ── Copy built artifacts from builder stage ─────────────────────
# dist/ contains both server bundle (index.cjs) and client (public/).
# The landing page is now part of the React app (no separate landing/ folder).
COPY --from=builder /app/dist ./dist

# ── Newsletter content (markdown archive read by server/newsletter.ts) ─
# Files live at /app/content/newsletter/*.md at runtime.
COPY content/ ./content/

# ── Copy Python source (needed at runtime) ──────────────────────
COPY *.py ./
COPY strategies/ ./strategies/
# AlphaDesk research engine package — the /api/research route runs
# `python3 -m alphadesk` from /app/alphadesk, so the whole package must ship.
COPY alphadesk/ ./alphadesk/

# ── Copy backtest data needed for trade_feedback seeding ────────
# (alpha audit 2026-05-03: voltrade_daemon auto-seeds the Kelly
# bucket stats from this file on first startup if trade_feedback.json
# is missing or sparse)
COPY backtest_10yr_results.json ./

# ── Copy daemon supervisor script (used by Railway startCommand) ─
COPY run_with_daemon.sh ./
RUN chmod +x run_with_daemon.sh

# ── Pre-compile Python bytecode ─────────────────────────────────
RUN python3 -m compileall -q . 2>/dev/null || true

EXPOSE 3000

# Node heap capped at 512MB (sufficient for Express + child_process orchestration)
# Python processes manage their own memory separately
ENV NODE_OPTIONS="--max-old-space-size=512"

# Constrain Python thread-hungry libraries (OpenBLAS, MKL, OpenMP)
# Railway containers have limited PIDs — 2 threads is plenty for scoring math
ENV OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OMP_NUM_THREADS=2 \
    NUMEXPR_MAX_THREADS=2 \
    VECLIB_MAXIMUM_THREADS=2

# Tell glibc malloc to return memory to the OS more aggressively
# (default threshold is 128KB — lower to 0 to release pages faster)
ENV MALLOC_TRIM_THRESHOLD_=0

CMD ["node", "dist/index.cjs"]
