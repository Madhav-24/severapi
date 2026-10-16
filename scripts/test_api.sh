#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
API_KEY="${API_KEY:?Set API_KEY without placing it in shell history where possible}"
IMAGE_FILE="${1:?Usage: scripts/test_api.sh /path/to/test-image.jpg}"

curl --fail --silent --show-error "$BASE_URL/health/live"
printf '\n'
curl --fail --silent --show-error "$BASE_URL/health/ready"
printf '\n'
curl --fail --silent --show-error "$BASE_URL/health/gpu"
printf '\n'
curl --fail --silent --show-error \
  --config <(printf 'header = "X-API-Key: %s"\n' "$API_KEY") \
  "$BASE_URL/v1/model/info"
printf '\n'
curl --fail --silent --show-error \
  --config <(printf 'header = "X-API-Key: %s"\n' "$API_KEY") \
  -F "file=@${IMAGE_FILE}" \
  "$BASE_URL/v1/predict"
printf '\n'
