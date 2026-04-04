FROM node:20-slim

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
RUN python3 -m pip install yfinance scipy numpy scikit-learn joblib --break-system-packages --quiet

# Install Node packages
COPY package.json package-lock.json ./
RUN npm install

# Copy source
COPY . .

# Build the app
RUN npm run build

EXPOSE 3000

CMD ["node", "dist/index.cjs"]
