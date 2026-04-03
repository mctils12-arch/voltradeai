#!/bin/bash
set -e

# Install Python packages at startup if not present
python3 -c "import yfinance" 2>/dev/null || {
  echo "Installing Python packages..."
  python3 -m pip install yfinance scipy numpy --user --quiet 2>/dev/null || \
  pip3 install yfinance scipy numpy --user --quiet 2>/dev/null || \
  curl https://bootstrap.pypa.io/get-pip.py | python3 - && python3 -m pip install yfinance scipy numpy --user --quiet
}

echo "Starting server..."
node dist/index.cjs
