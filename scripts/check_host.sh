#!/usr/bin/env bash
set -euo pipefail

MODEL_FILE="${MODEL_FILE:-models/construction-safety.pt}"
IMAGE="${IMAGE:-construction-ai-inference:1.0.0}"

section() { printf '\n== %s ==\n' "$1"; }
section "Operating system and kernel"
cat /etc/os-release | grep -E '^(NAME|VERSION|VERSION_ID)=' || true
uname -r

section "Docker"
docker --version
docker compose version

section "NVIDIA driver and GPUs"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is not available; host GPU state cannot be verified." >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used --format=csv
nvidia-smi

section "Disk space"
df -h .

section "Model file"
if [[ -f "$MODEL_FILE" ]]; then
  ls -lh "$MODEL_FILE"
else
  echo "ERROR: Custom model not found: $MODEL_FILE" >&2
  echo "Provide the compatible construction model; no generic weights will be downloaded." >&2
  exit 1
fi

section "Required local image"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker image inspect "$IMAGE" --format '{{.Id}}'
else
  echo "Image not present locally: $IMAGE"
  echo "A build or approved offline image import is required before deployment."
fi

section "Optional external registry connectivity"
registry_status="$(curl --connect-timeout 3 --max-time 5 -sS -o /dev/null -w '%{http_code}' https://registry-1.docker.io/v2/ 2>/dev/null || true)"
if [[ "$registry_status" == "200" || "$registry_status" == "401" ]]; then
  echo "Docker Hub registry endpoint reachable from this host (HTTP $registry_status)."
else
  echo "Docker Hub registry endpoint unavailable; use the approved offline transfer workflow."
fi
