#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-construction-ai-inference:1.0.0}"
GPU0_UUID="${GPU0_UUID:?Set GPU0_UUID to the host GPU 0 UUID}"

docker run --rm \
  --gpus "device=${GPU0_UUID}" \
  -e NVIDIA_VISIBLE_DEVICES="$GPU0_UUID" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  "$IMAGE" \
  python -c 'import os, subprocess, torch; selected=os.environ["NVIDIA_VISIBLE_DEVICES"]; ids=subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).strip().splitlines(); n=torch.cuda.device_count(); print({"visible_count":n,"cuda":torch.cuda.is_available(),"device":torch.cuda.get_device_name(0) if n else None,"visible_uuids":ids}); assert torch.cuda.is_available() and n == 1 and len(ids) == 1 and ids[0].strip().lower() == selected.lower(), "Expected exactly the selected production GPU UUID"'
