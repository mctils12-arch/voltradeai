FROM node:20-slim

# ── System deps ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python packages (cached layer — reruns only when requirements.txt changes) ──
COPY requirements.txt ./
RUN python3 -m pip install -r requirements.txt \
    --break-system-packages --quiet

# ── Node packages (cached layer — reruns only when package-lock changes) ──
COPY package.json package-lock.json ./
RUN npm ci --prefer-offline

# ── Copy source + build ───────────────────────────────────────────────────
COPY . .
RUN npm run build

# ── Pre-compile Python bytecode (faster first-scan startup) ───────────────
RUN python3 -m compileall -q . 2>/dev/null || true

EXPOSE 3000

# Increased Node heap for parallel Python subprocess management (Pro plan)
CMD ["node", "--max-old-space-size=2048", "dist/index.cjs"]
