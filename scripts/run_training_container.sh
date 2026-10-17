#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env.training ]]; then
  echo "ERROR: .env.training is missing. Copy .env.training.example and configure it." >&2
  exit 1
fi

set -a
source ./.env.training
set +a

: "${TRAINING_API_KEY:?Set a private TRAINING_API_KEY in .env.training}"
: "${TRAINING_GPU_UUID:?Set TRAINING_GPU_UUID to the dedicated GPU UUID}"
if [[ ${#TRAINING_API_KEY} -lt 32 ]]; then
  echo "ERROR: TRAINING_API_KEY must be at least 32 characters." >&2
  exit 1
fi
if [[ ! "$TRAINING_GPU_UUID" =~ ^GPU-[A-Za-z0-9-]+$ ]]; then
  echo "ERROR: TRAINING_GPU_UUID must be a GPU UUID reported by nvidia-smi." >&2
  exit 1
fi
if ! nvidia-smi --query-gpu=uuid --format=csv,noheader | grep -Fxq "$TRAINING_GPU_UUID"; then
  echo "ERROR: TRAINING_GPU_UUID is not visible on this host." >&2
  exit 1
fi

IMAGE="construction-yolo-training:1.0.0"
CONTAINER="construction_yolo_training"
HOST_BIND_ADDRESS="${TRAINING_HOST_BIND_ADDRESS:-127.0.0.1}"
HOST_PORT="${TRAINING_PORT:-8001}"
if [[ ! "$HOST_PORT" =~ ^[0-9]+$ ]] || (( HOST_PORT < 1 || HOST_PORT > 65535 )); then
  echo "ERROR: TRAINING_PORT must be between 1 and 65535." >&2
  exit 1
fi

docker build -f Dockerfile.training -t "$IMAGE" .
docker volume create yolo-training-runs >/dev/null
docker run -d \
  --name "$CONTAINER" \
  --restart unless-stopped \
  --gpus "device=$TRAINING_GPU_UUID" \
  --env-file .env.training \
  --env TRAINING_DATA_DIR=/training_runs \
  --env TRAINING_GPU_DEVICE=0 \
  --publish "$HOST_BIND_ADDRESS:$HOST_PORT:8001" \
  --mount source=yolo-training-runs,target=/training_runs \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=1g \
  --shm-size 1g \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  "$IMAGE" >/dev/null

READY=false
for _ in $(seq 1 60); do
  if curl --fail --silent --output /dev/null "http://$HOST_BIND_ADDRESS:$HOST_PORT/health/ready"; then
    READY=true
    break
  fi
  sleep 2
done
if [[ "$READY" != true ]]; then
  echo "ERROR: Training API did not become ready; inspect: docker logs --tail=100 $CONTAINER" >&2
  exit 1
fi

echo "Training API started: http://$HOST_BIND_ADDRESS:$HOST_PORT"
echo "Interactive docs: http://$HOST_BIND_ADDRESS:$HOST_PORT/docs"
echo "Check startup: docker logs --tail=100 $CONTAINER"