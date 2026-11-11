#!/usr/bin/env bash
# Start the chatbot and share it through a free Cloudflare quick tunnel.
# The public https://....trycloudflare.com link is printed below; it changes every run. Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
: "${NVIDIA_API_KEY:?Put NVIDIA_API_KEY=nvapi-... in .env (see .env.example)}"
[ -n "${CHAT_PASSWORD:-}" ] || echo "WARNING: CHAT_PASSWORD is not set, so anyone with the link can use your NVIDIA quota."

python3 -m uvicorn app:app --port 8000 &
server=$!
trap 'kill $server 2>/dev/null' EXIT
for _ in $(seq 300); do  # the first start builds the index, which can take minutes
  kill -0 "$server" 2>/dev/null || { echo "Server exited before it was ready." >&2; exit 1; }
  curl -s -o /dev/null localhost:8000 && break
  sleep 1
done
cloudflared tunnel --no-autoupdate --url http://localhost:8000
