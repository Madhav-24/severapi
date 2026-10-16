#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
IMAGE="construction-ai-inference:1.0.0"
BUILD_IMAGE="${BUILD_IMAGE:-true}"

if [[ ! -f .env ]]; then
  echo "ERROR: .env is missing. Copy .env.example, configure secrets and GPU0_UUID, then chmod 600 .env." >&2
  exit 1
fi

docker compose config --quiet
CONFIG="$(docker compose config --format json)"
GPU0_UUID="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["construction-ai-api"]["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"][0])' <<< "$CONFIG")"
MODEL_PATH="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["construction-ai-api"]["environment"]["MODEL_PATH"])' <<< "$CONFIG")"
case "$MODEL_PATH" in
  /models/*) MODEL_FILE="$(realpath -e "models/${MODEL_PATH#/models/}")" ;;
  *) echo "ERROR: MODEL_PATH must be located under /models." >&2; exit 1 ;;
esac
MODELS_DIR="$(realpath -e models)"
case "$MODEL_FILE" in
  "$MODELS_DIR"/*) ;;
  *) echo "ERROR: MODEL_PATH resolves outside the mounted models directory." >&2; exit 1 ;;
esac
if [[ ! -f "$MODEL_FILE" ]]; then
  echo "ERROR: configured model not found: $MODEL_FILE" >&2
  exit 1
fi
export GPU0_UUID
export MODEL_FILE
: "${GPU0_UUID:?GPU0_UUID is required}"
bash scripts/check_host.sh
if [[ "$BUILD_IMAGE" == "true" ]]; then
  docker compose build construction-ai-api
elif ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "ERROR: requested local image is unavailable: $IMAGE" >&2
  exit 1
fi
bash scripts/check_container_gpu.sh "$IMAGE"
docker compose up -d --no-build construction-ai-api

echo "Waiting for readiness..."
HEALTH_URL="http://${INTERNAL_BIND_ADDRESS:-127.0.0.1}:${HOST_PORT:-8000}/health/ready"
for _ in $(seq 1 60); do
  if curl --fail --silent "$HEALTH_URL" >/dev/null; then
    echo "Service is ready."
    exit 0
  fi
  sleep 2
done

echo "ERROR: service did not become ready. Inspect: docker compose logs --tail=200 construction-ai-api" >&2
exit 1
