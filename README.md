# Construction Safety Image Inference API

A single-process FastAPI service for image-based object detection using an
operator-supplied Ultralytics YOLO model. It does not download weights, open
camera streams, infer PPE violations, or claim readiness without the model and
the configured production GPU.

## Architecture

`app/main.py` configures request IDs, safe JSON errors, CORS, lifecycle, and
logging. The `health` and `predictions` routers expose health and inference
routes; `model_manager.py` owns the single model instance and GPU verification.
Image inference is offloaded from the async event loop and protected by a
bounded, fail-fast concurrency gate. One Uvicorn worker is intentional: more
workers would each load a separate model copy.

## Files and model requirement

```text
app/                 API, configuration, model lifecycle, inference, security
models/              Operator-supplied model mount (not part of the image)
tests/               CPU-only tests with mocked inference
scripts/             Host, container GPU, API, and deployment checks
compose.yaml         Single-container deployment, GPU UUID isolation
Dockerfile           Pinned PyTorch/CUDA runtime base
```

Place the approved custom model at `models/construction-safety.pt` (or choose
another file beneath `models/` using `MODEL_PATH=/models/<filename>`). It must
be compatible with the pinned Ultralytics runtime and trained for the intended
construction classes. No generic weights are substituted if it is missing.
Treat weights as trusted executable artifacts: verify their origin and
checksum before deploying because loading a PyTorch checkpoint can execute
serialized code.
Readiness stays HTTP 503 until the file loads successfully. Confirm the
model's actual class names at `/v1/model/info`; the service does not turn
object detections into PPE compliance decisions.

## API

| Method | Route | Auth | Purpose |
|---|---|---|---|
| GET | `/health/live` | No | Process liveness only |
| GET | `/health/ready` | No | Model readiness; HTTP 503 when unavailable |
| GET | `/health/gpu` | No | CUDA and visible device measurements |
| GET | `/v1/model/info` | `X-API-Key` | Loaded model and inference configuration |
| POST | `/v1/predict` | `X-API-Key` | One JPEG, PNG, or WebP image upload |

Interactive API documentation is available at `/docs` (OpenAPI schema:
`/openapi.json`). The docs page describes the multipart upload and API-key
header; use the key only on a trusted local/internal connection, and do not
make the documentation endpoint publicly accessible.

Prediction accepts multipart field `file` and optional `confidence` (0.05–1)
and `iou` (0.1–0.95) query parameters. Responses contain request ID, model,
image dimensions, elapsed processing time, and detections with `x1,y1,x2,y2`
pixel coordinates relative to the original image. Invalid images return a
consistent JSON error. Uploads are processed in memory/spooled request
temporary storage and are never saved as application files. The default
maximum upload is 10 MiB and default inference concurrency is one; excess
inference requests fail with 503 rather than queue without bound.
Even when the admission cap is raised, a lock serializes calls on the single
model instance; it does not create simultaneous GPU execution streams.
Prediction requests must include a `Content-Length`; requests without it are
rejected before multipart parsing.
Inference is bounded by `INFERENCE_TIMEOUT_SECONDS` (default 120). A timeout
returns HTTP 504, but an already-running GPU operation cannot safely be
interrupted; its admission slot remains occupied until it exits, preventing
overlapping requests from exceeding the configured limit.

Example (after deployment):

```json
{
  "request_id": "generated-uuid",
  "model": "construction-safety-model",
  "image": {"width": 1920, "height": 1080},
  "processing_time_ms": 125.5,
  "predictions": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.9632,
      "box_coordinates": {"x1": 120.5, "y1": 80.0, "x2": 340.2, "y2": 450.7}
    }
  ]
}
```

Values are illustrative; actual classes/results depend exclusively on the
supplied model. Set `MODEL_VERSION` to the approved weights release label; the
default `unspecified` does not claim a training/version history.

## Intern YOLO training API

`app.training_api:app` is a separate API for interns to submit one YOLO object-
detection training job at a time. It does not change the inference service on
port 8000. The training API uses `yolov8n.pt` as its pretrained starting model;
Ultralytics downloads that checkpoint on the first job if it is not cached.

Prepare the protected training environment on the GPU host:

```bash
cp .env.training.example .env.training
chmod 600 .env.training
openssl rand -hex 32
editor .env.training
nvidia-smi -L
```

Set the generated value as `TRAINING_API_KEY`, choose the dedicated GPU's exact
UUID for `TRAINING_GPU_UUID`, and select the host interface in
`TRAINING_HOST_BIND_ADDRESS`. The default `127.0.0.1` is host-only. For interns
on an approved VPN/internal network, use the GPU host's VPN/internal IP only
after firewall rules restrict the port to those clients. Do not publish this
training API to the public internet; use an internal TLS proxy for remote
access. Never commit `.env.training` or share its API key outside the approved
team.

Start the CUDA container (Docker NVIDIA GPU support must already be working):

```bash
bash scripts/run_training_container.sh
```

The script prints the API and docs URLs. For the default settings, the docs
page is `http://127.0.0.1:8001/docs`; remote interns need the approved internal
host address in that URL. Check `/health/ready` before sharing it. This is a
separate GPU process from inference; choose a GPU not in use by another
workload. The first image build and first model download require registry and
internet access, respectively.

Interns upload one `.zip` dataset through `POST /v1/training/jobs` with
`X-API-Key`. The ZIP must contain `data.yaml` at its root, image directories
referenced by its relative `train` and `val` entries, and YOLO-format labels.
For example:

```text
data.yaml
images/train/*.jpg
images/val/*.jpg
labels/train/*.txt
labels/val/*.txt
```

`names` in `data.yaml` must define the class names. Epochs, image size, and
batch are optional form fields (defaults: 50, 640, and 16). Uploads are capped
at 2 GiB and extracted data at 20 GiB. The job status endpoint reports
`queued`, `running`, `succeeded`, or `failed`; when successful,
`GET /v1/training/jobs/{job_id}/model` downloads `best.pt`. Job status is held
in memory, so download completed models before restarting the service. The
training container stores run output in the persistent `yolo-training-runs`
Docker volume.

## Production deployment (Ubuntu 24.04)

Run commands from the project directory on the approved Linux host. Do not
deploy from this Windows development environment. Production commands below
do not install or alter host kernel, NVIDIA driver, CUDA, Docker, or unrelated
containers.

### 1. Inspect the host and model

```bash
bash scripts/check_host.sh
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used --format=csv
uname -r
df -h .
```

The host script reports OS, actual running kernel, Docker/Compose, driver/GPU
identity and memory, storage, model presence, local image presence, and
best-effort registry connectivity. It exits if `nvidia-smi` or the custom model
is unavailable. The listed kernel version is not assumed.

### 2. Confirm model and select GPU 0

Use the actual GPU UUID reported for host index `0`:

```bash
nvidia-smi -L
```

Never use GPU 1's UUID. Check GPU 1's existing use before deploying; this
service does not stop or modify other GPU workloads. The container reservation
and `NVIDIA_VISIBLE_DEVICES` both select GPU 0 by UUID. The inference process
requires exactly one visible CUDA device and uses it as `cuda:0` inside the
container. Where PyTorch exposes a device UUID, startup additionally compares
it with `GPU0_UUID`; `/health/gpu` reports whether that comparison was
available and matched.

### 3. Configure secrets and interface

```bash
cp .env.example .env
openssl rand -hex 32
editor .env
chmod 600 .env
```

Set the generated random value as `API_KEY`, the host GPU 0 UUID as
`GPU0_UUID`, and use `MODEL_ID` and `MODEL_VERSION` to identify the actual
model and weights release. `MODEL_PATH` must resolve under `/models`, which
maps read-only to the host's `models/` folder. Startup remains unready unless
both the API key and model/GPU configuration are present.
The API key must be at least 32 characters; the `openssl rand -hex 32` command
generates a 64-character key.

The published interface defaults to loopback (`127.0.0.1`). To permit
authorised VPN clients directly, set `INTERNAL_BIND_ADDRESS` to the approved
VPN/internal address and configure both host and upstream firewalls to allow
TCP `HOST_PORT` (default 8000) only from authorised VPN client ranges. Do not
expose this API to the public internet. Prefer an internal TLS reverse proxy;
API keys alone are bearer credentials and do not provide user identity,
rotation policy, or granular authorization. Use enterprise identity/mTLS at a
trusted gateway for stronger controls. Generate and rotate keys through an
approved secrets process; never commit `.env`.

### 4. Check Docker GPU support and validate Compose

```bash
docker info --format '{{json .Runtimes}}'
nvidia-smi -L
docker compose config --quiet
docker compose config --format json | python3 -c 'import json,sys; s=json.load(sys.stdin)["services"]["construction-ai-api"]; print({"ports":s["ports"],"gpu_device_ids":s["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"],"NVIDIA_VISIBLE_DEVICES":s["environment"]["NVIDIA_VISIBLE_DEVICES"]})'
```

Docker's NVIDIA runtime must already be configured; this project will not
modify the driver or Docker daemon. The definitive container visibility check
is run against the built/loaded image in the next step. The filtered Compose
output deliberately avoids printing the configured API key.

### 5. Build, verify GPU isolation, and start

```bash
docker compose build construction-ai-api
export GPU0_UUID="$(docker compose config --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["construction-ai-api"]["environment"]["GPU0_UUID"])')"
bash scripts/check_container_gpu.sh construction-ai-inference:1.0.0
docker compose up -d --no-build construction-ai-api
docker compose ps
docker compose logs --tail=100 construction-ai-api
```

Alternatively, run the guarded deployment helper, which checks config, builds
or uses a preloaded local image (`BUILD_IMAGE=false`), runs the isolated GPU
check, starts the service, and waits for readiness. Do not run it until the
model, UUID, and protected `.env` values have been verified:

```bash
bash scripts/deploy.sh
# Offline after docker load: BUILD_IMAGE=false bash scripts/deploy.sh
```

The base image and Python dependencies need to be available to the build
machine. No offline build is claimed or tested here. To transfer an approved
image from a connected build machine, build there and then:

```bash
docker save construction-ai-inference:1.0.0 -o construction-ai-inference.tar
# Transfer the archive through the approved internal process.
docker load -i construction-ai-inference.tar
```

Also transfer the model separately through the approved process and verify its
checksum before deployment. On an offline target, use `docker compose up -d
--no-build`; the target must already have the matching image and model.

### 6. Verify API, inference, and GPU isolation

```bash
curl --fail http://127.0.0.1:8000/health/live
curl --fail http://127.0.0.1:8000/health/ready
curl http://127.0.0.1:8000/health/gpu
read -rsp 'API key: ' API_KEY; printf '\n'; export API_KEY
bash scripts/test_api.sh /path/to/approved-test-image.jpg
unset API_KEY
docker exec construction_ai_service nvidia-smi -L
nvidia-smi
docker stats construction_ai_service
docker inspect construction_ai_service --format '{{json .State.Health}}'
```

Readiness must be successful before inference. In the container's `nvidia-smi
-L`, confirm exactly one GPU appears and its UUID equals host GPU 0's UUID;
host GPU 1 must not appear. Confirm GPU 1's workload/memory remains unaffected
on the host. If GPU UUID introspection is unavailable through PyTorch, treat
the startup mapping as unverified until container inspection confirms it.

Run a modest, controlled load test with an approved image set and concurrency
matching the intended clients. Measure latency and throughput on this actual
model and hardware; do not infer capacity from a different model or benchmark.
Monitor:

```bash
docker compose logs --follow construction-ai-api
docker stats construction_ai_service
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv
df -h
docker inspect construction_ai_service --format 'restarts={{.RestartCount}} status={{.State.Status}}'
```

The host is a single point of failure. `restart: unless-stopped` is process
recovery, not high availability.

## Environment variables

| Name | Default | Notes |
|---|---|---|
| `API_KEY` | required | Secret; sent only in `X-API-Key` |
| `GPU0_UUID` | required | Host GPU 0 UUID (`GPU-...`) |
| `MODEL_PATH` | `/models/construction-safety.pt` | Select a file under the read-only `/models` mount |
| `MODEL_ID` | `construction-safety-model` | Safe identifier shown to clients |
| `MODEL_VERSION` | `unspecified` | Operator-supplied weights release label |
| `INFERENCE_DEVICE` | `cuda:0` | Must stay `cuda:0` within isolated container |
| `CONFIDENCE_THRESHOLD` | `0.25` | Default confidence (0.05–1) |
| `IOU_THRESHOLD` | `0.45` | Default NMS IoU (0.1–0.95) |
| `IMAGE_SIZE` | `640` | Inference image size, bounded 320–1280 |
| `INFERENCE_TIMEOUT_SECONDS` | `120` | API deadline, bounded 1–3600 seconds |
| `MAX_UPLOAD_BYTES` | `10485760` | Upload limit, bounded to 50 MiB |
| `MAX_INFLIGHT_INFERENCES` | `1` | Bounded concurrency, 1–8 |
| `HOST_PORT` | `8000` | Published host port |
| `INTERNAL_BIND_ADDRESS` | `127.0.0.1` | Host bind IP; configure firewall before changing |
| `CORS_ALLOWED_ORIGINS` | empty | Explicit comma-separated origins; wildcard rejected |

## Operations, model updates, and rollback

Restart the service:

```bash
docker compose restart construction-ai-api
```

Update weights by placing a verified model file in `models/` while retaining
the prior model backup outside the active mount, then restart and verify
`/health/ready`, `/v1/model/info`, test inference, and GPU isolation before
switching client traffic. For rollback, restore the previous verified weights
to the configured filename, restart the same service, and repeat readiness
and inference checks. Never overwrite the only copy; keep protected, checksum
verified backups and the previous container image tag.

Use Docker's bounded json-file rotation for container logs. Logs include
request IDs, endpoint, duration, errors and model identifier. They exclude
uploaded images and API-key headers. Do not enable request-body/access-header
logging in a proxy. Monitor host disk space, container restarts, GPU VRAM,
latency, and 5xx/503 responses.

## Troubleshooting

- **Readiness 503:** check logs and model path/mount, file compatibility, GPU
  UUID, NVIDIA container runtime, and the host's GPU 0 visibility.
- **CUDA unavailable or more than one GPU visible:** do not serve traffic;
  validate GPU UUID selection with `docker compose config` and `nvidia-smi -L`
  inside the container. Do not change the host driver without approval.
- **No model file:** provide the correct custom weights; the service never
  downloads generic weights.
- **401:** check the API key value and `X-API-Key` header without printing it.
- **413/415/400:** check upload size, supported JPEG/PNG/WebP encoding, and
  that content type matches image bytes.
- **503 inference capacity:** concurrency is intentionally bounded. A larger
  value should be set only after testing sustained VRAM and latency with the
  production model.
- **Image build/network unavailable:** build on an approved connected
  development machine and transfer the image and model through approved
  internal channels. The workflow has not been verified offline.

## Testing and local development

```bash
python -m pip install -r requirements.txt
python -m pip install pytest httpx
python -m pytest -q
```

Tests mock model inference and require no GPU or weights. They verify API
validation, response shape, auth, model readiness, size limits, and safe
inference errors. They do not constitute a real model/CUDA test.

## Future RTSP video architecture

Keep this deployment image-only. For a future video service, add one managed
reader per configured RTSP camera, sampled frames into bounded per-camera
queues, and a central GPU inference scheduler; apply tracking (for example
ByteTrack) and temporal PPE association/verification before deduplicated
alerts are persisted and integrated with the safety dashboard. Manage camera
credentials separately and never log stream URLs containing credentials.
Do not open RTSP streams at this stage. Kafka, Kubernetes, Redis, and a
database are unnecessary until measured scale or durability requirements
justify their operational cost.
#   s e v e r a p i 
 
 