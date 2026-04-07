#!/usr/bin/env bash
# run_test.sh — Start server + inference in one shell
set -e
cd "$(dirname "$0")"

HF_TOKEN=$(cat /mnt/Windows-SSD/Users/yvavi/yeet/coding/projects/meta/IncidentDiagnosisEnv/.env | tr -d '[:space:]')
export HF_TOKEN
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-72B-Instruct}"
export API_BASE_URL="${API_BASE_URL:-https://router.huggingface.co/v1}"
export ENV_BASE_URL="http://localhost:8889"

echo "=== Starting IncidentDiagnosisEnv server on port 8889 ==="
uv run uvicorn server.app:app --host 0.0.0.0 --port 8889 --log-level warning &
SERVER_PID=$!

# Wait for server ready
for i in $(seq 1 15); do
  if curl -sf http://localhost:8889/health > /dev/null 2>&1 || curl -sf http://localhost:8889/tasks > /dev/null 2>&1; then
    echo "Server up after ${i}s"
    break
  fi
  sleep 1
done

echo ""
echo "=== /tasks endpoint ==="
curl -s http://localhost:8889/tasks
echo ""

echo ""
echo "=== Running inference (all 3 tasks) ==="
uv run python inference.py 2>&1
EXIT_CODE=$?

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
exit $EXIT_CODE
