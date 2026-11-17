#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
export VITE_INFERENCE_MODE="browser"
command -v node >/dev/null || { echo "Install Node.js 20.19+ or 22+ from https://nodejs.org"; exit 1; }
node -e 'const [a,b]=process.versions.node.split(".").map(Number);process.exit(a>20||(a===20&&b>=19)?0:1)' || { echo "Update Node.js to 20.19+ or 22+."; exit 1; }
command -v npm >/dev/null || { echo "npm is required (included with Node.js)."; exit 1; }
if [[ ! -x frontend/node_modules/.bin/vite ]]; then npm ci --prefix frontend; fi
backend_pid=""
cleanup() { if [[ -n "$backend_pid" ]]; then kill "$backend_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT INT TERM
if [[ "${1:-}" == "--with-backend" ]]; then
  command -v python3 >/dev/null || { echo "Install Python 3.11 or 3.12 for the optional backend."; exit 1; }
  if [[ ! -x backend/.venv/bin/python ]]; then python3 -m venv backend/.venv; fi
  backend/.venv/bin/python -m pip install -r backend/requirements.txt
  (cd backend && .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 9000) &
  backend_pid=$!
  export VITE_API_URL="http://127.0.0.1:9000"
  export VITE_INFERENCE_MODE="api"
fi
printf 'TeaLeafAI: http://localhost:5173 — keep this terminal open. Ctrl+C stops it.\n'
cd frontend
npm run dev -- --port 5173 --strictPort --open
