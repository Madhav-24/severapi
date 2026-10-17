# Codebase File-by-File Guide

This guide explains the purpose and behavior of every hand-maintained project
file in this repository. It describes the implementation as it exists now;
it is not a substitute for the production deployment checklist in
[README.md](./README.md).

## Project overview

The project is a FastAPI image-inference service. It accepts JPEG, PNG, and
WebP uploads, checks API keys and request limits, sends decoded RGB images to
one Ultralytics YOLO model, then returns object classes, confidence scores,
and pixel bounding boxes. Docker Compose limits the production container to
the host GPU selected by UUID.

The service does not provide a model or download one. Until an approved
compatible model is supplied, configured, and loaded on a CUDA-enabled Linux
host, readiness reports unavailable and predictions cannot run. CPU-only
tests use a fake model manager and do not demonstrate actual YOLO/GPU
inference.

## Request and startup flow

1. `app.main:app` is the ASGI entry point used by Uvicorn.
2. The application lifespan checks that an API key is configured and asks
   `ModelManager` to load the configured weights once.
3. The request middleware assigns a request ID, rejects unauthenticated or
   excessive inference requests early, and records duration and status.
4. The health router reports liveness, model readiness, or measured CUDA
   status. The inference router exposes model metadata and image prediction.
5. For predictions, image bytes are size- and format-checked, decoded in
   memory, and passed to the model on a worker thread. A single bounded slot
   and the model's inference lock prevent concurrent GPU work from growing
   without control.
6. Responses use Pydantic schemas and errors are returned in a consistent
   JSON shape. Logs go to stdout for Docker collection.

## How to run the project

There are two different ways to run this repository:

- **Automated tests** use a fake model manager. They run without GPU hardware
  or model weights and are the recommended way to develop API behavior.
- **Real inference** requires the approved model file, a CUDA-enabled NVIDIA
  GPU, a compatible PyTorch/CUDA runtime, a configured API key, and the
  production GPU UUID. The production path is Docker Compose on the approved
  Linux host.

### Run the tests locally

From the repository root, install the Python dependencies in your virtual
environment and run:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest httpx
python -m pytest -q
```

Or use `make test` if pytest and httpx are already installed. Tests use
`tests/conftest.py`'s `FakeManager`; passing tests do not prove that the model
weights, CUDA, GPU selection, or production deployment work.

### Start a local API process

`make run` starts one Uvicorn worker on `127.0.0.1:8000`, for access from the
same computer only. It does not load `.env` itself. To load local settings,
create a private `.env` using `.env.example`, set `MODEL_PATH` to the actual
model file path on this computer, set `API_KEY` and `GPU0_UUID`, and run:

```bash
python -m uvicorn app.main:app --env-file .env --host 127.0.0.1 --port 8000 --workers 1
```

The installed `uvicorn[standard]` extra provides `.env` support. Local real
inference still requires the model and compatible CUDA/PyTorch setup; this
project intentionally rejects CPU inference and requires exactly one visible
GPU. If startup cannot load the model or validate CUDA, the process may still
serve `/docs` and `/health/live`, but `/health/ready` reports HTTP 503 and
inference is unavailable. For GPU-free development, use the mocked tests
instead.

When the local process is running, open `http://127.0.0.1:8000/docs` on the
same computer. `127.0.0.1` always refers to the computer where the browser is
running: it works for a browser on the API host but does not reach a different
VPN-connected server.

### Deploy and access the API on the server

Follow the production checklist in [README.md](./README.md). In summary, on
the approved Linux server:

1. Provide the approved model file under `models/`, confirm Docker GPU support
   and the host's GPU 0 UUID, and copy `.env.example` to `.env`.
2. Set a generated API key, the GPU 0 UUID, model identity/version, and
   interface settings in `.env`; run `chmod 600 .env` to restrict permissions.
3. Keep the default `INTERNAL_BIND_ADDRESS=127.0.0.1` if only the server itself
   should connect. For VPN clients, set it to the server's approved VPN-facing
   IP address and have the network administrator allow `HOST_PORT` only from
   authorized VPN ranges. Do not expose the API to the public internet.
4. Run `bash scripts/deploy.sh`, then verify readiness and GPU selection as
   described in `README.md`.

On the server, `http://127.0.0.1:8000/docs` refers to the server itself. On a
VPN-connected laptop, use `http://<server-vpn-hostname-or-ip>:8000/docs` with
the real server VPN address instead. For example, if the server's approved
VPN address were `10.20.30.40`, the URL would be
`http://10.20.30.40:8000/docs`. That address is illustrative; obtain the
actual hostname/IP from the server or network administrator. A browser
`ERR_CONNECTION_REFUSED` or timeout from the laptop usually means it is using
the laptop's own `127.0.0.1`, the service is still bound to server loopback,
or the VPN/firewall does not allow the port.

## API endpoint reference

| Method | Path | Authentication | Purpose and input |
|---|---|---|---|
| `GET` | `/docs` | None | Interactive Swagger UI for exploring and calling the API. The page itself is not protected, so limit network access to trusted users. |
| `GET` | `/openapi.json` | None | Machine-readable OpenAPI schema used by Swagger UI and API tooling. |
| `GET` | `/health/live` | None | Process liveness; does not check model readiness. |
| `GET` | `/health/ready` | None | Model readiness. Returns HTTP 503 until the model is loaded and available. |
| `GET` | `/health/gpu` | None | CUDA/GPU visibility, identity, and memory diagnostic fields. |
| `GET` | `/v1/model/info` | `X-API-Key` | Loaded model ID/version/classes and inference configuration; returns 503 if not ready. |
| `POST` | `/v1/predict` | `X-API-Key` | Multipart image inference. Send image as `file`; optional query parameters: `confidence` (0.05–1.0) and `iou` (0.1–0.95). Returns image size, processing time, and detection list. |

Supported upload formats are JPEG, PNG, and WebP. The inference endpoints
require a valid API key, while the health endpoints are unauthenticated for
service monitoring. The service does not make a detection into a PPE or
construction-safety compliance decision; returned classes come from the
operator-provided model.

## Runtime configuration fields

The application reads settings from environment variables in `app/config.py`.
Docker Compose reads `.env` and passes the application settings into the
container. For direct local Uvicorn runs, use `--env-file .env` or otherwise
set the environment variables; merely creating `.env` does not make Python
read it.

| Variable | Default | Purpose and constraints |
|---|---|---|
| `API_KEY` | Required; empty by default | Shared bearer secret sent as `X-API-Key` to protected endpoints. Startup requires at least 32 characters. Keep it private; the example file intentionally leaves it blank. |
| `GPU0_UUID` | Required; empty by default | UUID of the approved host GPU 0. Required for model loading; Compose also uses it to expose only that GPU to the container. |
| `MODEL_PATH` | `/models/construction-safety.pt` | Model checkpoint path. In Compose it must resolve under the read-only `/models` mount, backed by the repository's `models/` directory. For direct local execution, use the real path on that computer. |
| `MODEL_ID` | `construction-safety-model` | Safe, client-visible identifier for the model. |
| `MODEL_VERSION` | `unspecified` | Operator-supplied label identifying the deployed weights release. |
| `INFERENCE_DEVICE` | `cuda:0` | Container-visible inference device. The app rejects other values; Compose isolates the selected host GPU and it appears in-container as device 0. |
| `CONFIDENCE_THRESHOLD` | `0.25` | Default minimum detection confidence; valid range is `0.05`–`1.0`. A request may override it with the `confidence` query parameter. |
| `IOU_THRESHOLD` | `0.45` | Default non-maximum-suppression intersection-over-union threshold; valid range is `0.1`–`0.95`. A request may override it with `iou`. |
| `IMAGE_SIZE` | `640` | YOLO inference image size; valid range is `320`–`1280`. |
| `INFERENCE_TIMEOUT_SECONDS` | `120` | Request wait deadline; valid range is `1`–`3600`. A timed-out GPU operation is not forcibly cancelled and keeps its inference slot until it finishes. |
| `MAX_UPLOAD_BYTES` | `10485760` (10 MiB) | Maximum image payload size; valid range is 1 byte–50 MiB. |
| `MAX_INFLIGHT_INFERENCES` | `1` | Bounded inference admission limit; valid range is `1`–`8`. The single model lock still serializes actual model calls. Requests over capacity fail fast with HTTP 503. |
| `CORS_ALLOWED_ORIGINS` | Empty | Optional comma-separated browser origins, for example `https://internal.example`. Wildcard `*` is rejected. CORS controls browser cross-origin access; it does not publish a network port or replace authentication/firewalls. |
| `HOST_PORT` | `8000` | Host-side published TCP port in Compose. The container listens on port 8000. |
| `INTERNAL_BIND_ADDRESS` | `127.0.0.1` | Host interface address used by Compose to publish the port. Loopback is local to the server; use only an approved VPN/internal interface address for remote clients, with firewall restrictions. |

`NVIDIA_VISIBLE_DEVICES` and `NVIDIA_DRIVER_CAPABILITIES` are configured by
Compose from the selected GPU and are not user-tunable application settings.
The application does not download or substitute model weights.

## Application source: `app/`

### `app/__init__.py`

Marks `app` as a Python package and provides a short package description. It
contains no application behavior.

### `app/main.py`

Creates and configures the FastAPI application.

- `create_app(settings, model_manager)` supports normal configuration and
  dependency injection for tests.
- The lifespan checks for an API key of at least 32 characters, loads the
  model at startup, and closes it at shutdown. Model-load failures are logged
  and leave the service unready rather than falsely reporting success.
- Swagger UI is available at `/docs`; its OpenAPI document is at
  `/openapi.json`. ReDoc is disabled.
- CORS middleware is added only when explicit origins are configured.
- Request middleware creates a UUID request ID, authenticates and checks
  `Content-Length` and inference capacity before parsing an inference upload,
  sets the `X-Request-ID` response header, and logs request timing/status.
- Exception handlers convert HTTP, validation, and unexpected errors to
  client-safe JSON.
- The health and inference routers are registered at the end.
- `app = create_app()` is the module-level object Uvicorn serves.

### `app/config.py`

Defines the immutable `Settings` dataclass. It reads configuration from
environment variables, including model path/name/version, API key, GPU UUID,
inference device, thresholds, image size, request limits, timeout, and CORS
origins.

`_bounded_int` and `_bounded_float` parse numeric environment values and reject
non-numeric or out-of-range values at import/configuration time. `Settings`
is frozen so settings cannot be mutated after construction. `__post_init__`
also ensures inference uses container device `cuda:0`, rejects wildcard CORS,
and validates configuration ranges. See [Runtime configuration fields](#runtime-configuration-fields)
for every setting, its default, and its purpose.

### `app/logging_config.py`

Configures application logging to stdout. `JsonFormatter` emits JSON with
UTC timestamp, severity, logger name, message, and selected safe metadata such
as request ID, endpoint, duration, status, model ID, and error category.
Exception details are included in server logs for diagnosis. Uvicorn access
logs are reduced to avoid unnecessary request logging.

### `app/model_manager.py`

Owns the single loaded YOLO model and all model/GPU operations.

- `ModelUnavailableError` represents expected startup or readiness failures.
- `ModelManager.load()` checks that the configured model exists, a GPU UUID
  is configured, CUDA is available, exactly one GPU is visible, and—when
  PyTorch exposes its UUID—that it matches the configured UUID. It then loads
  the supplied Ultralytics YOLO model, moves it to the configured CUDA device,
  and records the model's actual class names.
- `ready` and `failure` expose safe lifecycle state to health checks.
- `record_failure()` clears readiness after a startup failure.
- `predict()` serializes model calls with a thread lock, invokes YOLO with the
  configured image size and request thresholds, and maps returned detections
  to class names and `x1,y1,x2,y2` coordinates.
- `info()` returns public model/configuration metadata, not filesystem paths.
- `gpu_status()` queries PyTorch for CUDA availability, visible device count,
  device name/UUID, and allocated/reserved VRAM. If inspection fails, it
  logs the exception and reports CUDA as unavailable.
- `close()` waits for any active locked inference, releases the model
  reference, clears cached CUDA memory, and logs shutdown.

The model is not downloaded or replaced with generic weights. Only trusted,
checksum-verified weights should be loaded.

### `app/schemas.py`

Defines the Pydantic API data models:

- `ErrorDetail`: `code` (stable machine-readable error category), `message`
  (safe client-facing description), and `request_id` (correlates with logs).
- `ErrorResponse`: wraps an `ErrorDetail` under the JSON field `error`.
- `BoxCoordinates`: `x1`, `y1`, `x2`, and `y2`, the bounding-box corners in
  pixels relative to the original image.
- `Prediction`: `class_id` (model class index), `class_name` (model class
  label), `confidence` (score from 0 to 1), and `box_coordinates`.
- `ImageSize`: original image `width` and `height`, in pixels.
- `PredictionResponse`: `request_id`, configured `model` identifier, original
  `image` dimensions, `processing_time_ms`, and a list of `predictions` (empty
  when no objects are detected).
- `ModelInfo`: `model`, `model_version`, integer-keyed `classes`, `device`,
  `image_size`, `confidence_threshold`, `iou_threshold`, and `model_loaded`.
- `HealthResponse`: liveness `status`.
- `ReadinessResponse`: readiness `status`, `model_loaded`, and optional
  `reason`.
- `GpuResponse`: `cuda_available`, `device_name`, `device_uuid`,
  `configured_host_gpu_uuid`, `identity_verified`, `container_device_index`,
  `device_count`, `allocated_memory_bytes`, `reserved_memory_bytes`, and
  optional `error`. Nullable fields may be `null` when a value is unavailable
  or cannot be verified.

These schemas document and validate response formats; they do not perform
inference.

### `app/security.py`

Provides `valid_api_key()`, which rejects absent values and compares supplied
and configured API keys with `hmac.compare_digest` to avoid a
timing-sensitive normal string comparison. `request_id()` safely obtains a
request ID from request state, or returns `"unknown"` if none was assigned.

### `app/routers/__init__.py`

Marks the router directory as a Python package. It contains no route behavior.

### `app/routers/health.py`

Defines three unauthenticated operational endpoints:

- `GET /health/live` returns `{"status":"alive"}` to show the process is
  responding; it does not assert model or GPU health.
- `GET /health/ready` returns `{"status":"ready","model_loaded":true}` only
  when the model manager is ready. Otherwise it returns HTTP 503 with
  `status:"not_ready"`, `model_loaded:false`, and a failure `reason`.
- `GET /health/gpu` returns the `GpuResponse` fields with measured
  CUDA/device status. This is diagnostic information, not proof by itself that
  the API is ready to infer.

### `app/routers/predictions.py`

Defines authenticated inference routes and image validation.

- `GET /v1/model/info` requires the `X-API-Key` header and returns the
  `ModelInfo` fields only when the model is ready.
- `POST /v1/predict` accepts the multipart `file` field and optional bounded
  `confidence` and `iou` query parameters. It verifies authentication and
  readiness, enforces size/media type limits, decodes a real JPEG/PNG/WebP,
  converts it to RGB, and passes it to inference on a background worker.
- Inference has a configured deadline. If the request times out, ongoing GPU
  work cannot safely be interrupted, so its inference slot remains held until
  work finishes.
- Upload and decoded image resources are closed when safe; inference errors
  are logged internally and returned without raw exception details.
- `_decode_image()` verifies image bytes and declared MIME type and treats
  decompression-bomb warnings as invalid input.
- `_log_late_inference_failure()` records errors from GPU work that completes
  after the caller timed out.

Both `/v1/model/info` and `/v1/predict` require a valid API key. In Swagger UI,
use the endpoint's `X-API-Key` header input. A prediction request is multipart
form data with one `file` image; optional query parameters are `confidence`
(0.05–1.0) and `iou` (0.1–0.95). The JSON prediction response includes:

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

The values above are illustrative and depend on the supplied model. Common
responses include HTTP 400 for an empty/invalid image, 401 for a missing or
invalid API key, 411 when `Content-Length` is absent, 413 for an oversized
upload, 415 for an unsupported declared media type, 422 for invalid query
parameters, 503 when the model is unready or capacity is full, 504 for an
inference timeout, and 500 for an inference failure. Errors include a request
ID so the corresponding server log entry can be located.

## Automated tests: `tests/`

Tests use FastAPI's `TestClient`; `FakeManager` in `conftest.py` supplies
predictable mocked detections so the suite does not need model weights or a
GPU.

### `tests/__init__.py`

Marks the tests directory as a Python package so test helpers can be imported.

### `tests/conftest.py`

Provides shared fixtures and support code:

- `TEST_API_KEY` is a deliberately test-only key.
- `FakeManager` mimics model lifecycle, metadata, GPU state, and prediction.
- `png_bytes()` creates a tiny in-memory PNG for request tests.
- The `client` fixture creates an app with test settings and a ready fake
  manager.

### `tests/test_health.py`

Tests process liveness, Swagger UI and the multipart upload schema, readiness,
the missing-weights failure, and the GPU endpoint's unavailable state.

### `tests/test_predictions.py`

Tests successful response shape, authentication, empty/malformed/unsupported
uploads, MIME mismatch, upload size limits, admission saturation, missing
model behavior, timeout handling, sanitized inference errors, and query
threshold validation.

### `tests/test_security.py`

Tests that model information requires an API key and that a valid key can
retrieve the information.

Run the mocked suite using `python -m pytest -q` or `make test`; see
[Run the tests locally](#run-the-tests-locally) for setup.

### `tests/__pycache__/` and `.pytest_cache/`

These are generated Python/pytest caches, not source code. They can be
recreated by running Python or tests and should not be edited or committed.

## Container and dependency files

### `Dockerfile`

Builds the service image on the pinned PyTorch CUDA runtime base. It installs
the pinned Python requirements, creates a non-root user, copies only the
application package, exposes port 8000, defines a readiness health check, and
starts one Uvicorn worker. Model weights are deliberately kept outside the
image and mounted read-only by Compose.

### `compose.yaml`

Defines the single production service `construction-ai-api` with container
name `construction_ai_service`.

- Publishes configurable host port 8000, loopback-bound by default.
- Requires `API_KEY` and `GPU0_UUID` from `.env`.
- Sets the model and inference settings.
- Selects the GPU UUID through both the Docker device reservation and
  `NVIDIA_VISIBLE_DEVICES`.
- Mounts `models/` read-only, uses a read-only root filesystem and a temporary
  filesystem, drops Linux capabilities, and enables `no-new-privileges`.
- Applies CPU/memory limits, Docker health checks, log rotation, restart
  behavior, and a graceful stop period.

Compose syntax validation does not prove GPU access; the container check on
the Linux server must verify the actual UUID.

### `requirements.txt`

Pins the runtime Python libraries: FastAPI, Uvicorn, multipart form handling,
Pillow image decoding, and Ultralytics. PyTorch is supplied by the selected
PyTorch/CUDA base image rather than installed as a second incompatible build.
Test-only packages such as pytest/httpx are installed separately for local
tests and are not part of the runtime requirements.

### `Makefile`

Provides short developer commands:

- `make test` runs the pytest suite.
- `make run` starts a single local Uvicorn worker bound to loopback.

### `.env.example`

Template for environment variables. Copy this to `.env` on the target host,
then supply a strong API key and the actual production GPU 0 UUID. Never use
the placeholder UUID or a development key in production.

### `.env`

Local/private runtime configuration created from the example. It is ignored
by Git and must not be committed, pasted into support messages, or printed
when showing resolved Compose settings. This guide intentionally does not
read or reproduce its contents.

### `.gitignore`

Excludes secrets, Python virtual environments/caches, test caches, logs, and
model files from Git while retaining the `.gitkeep` placeholders.

### `.dockerignore`

Excludes Git metadata, `.env`, developer environments/caches, tests, logs,
local model files, datasets, and bytecode from the Docker build context. This
keeps secrets/data out of the image build and the model external.

## Operational scripts: `scripts/`

These scripts target Bash on the Linux server. They have not been verified by
running them on the production host.

### `scripts/check_host.sh`

Prints OS and kernel details, Docker/Compose versions, NVIDIA driver/GPU UUID
and memory information, disk space, custom model presence, local image
presence, and a best-effort check of Docker Hub reachability. It exits if
`nvidia-smi` or the expected model is missing. It does not install or modify
host software.

### `scripts/check_container_gpu.sh`

Starts a temporary container from a specified image with only the supplied
GPU UUID exposed. It checks CUDA availability, visible device count, and UUID
reported by `nvidia-smi`; it fails if the device mapping is not exactly the
selected GPU.

### `scripts/test_api.sh`

Uses curl to exercise liveness, readiness, GPU status, authenticated model
info, and one image prediction. It expects `API_KEY` to be set in the
environment and the image path as its argument. Set `BASE_URL` to the API
origin when it is not `http://127.0.0.1:8000`; for example, a VPN client must
use the server's reachable VPN hostname/IP. The API key is passed through a
temporary curl config stream to avoid putting it directly in curl's argument
list.

### `scripts/deploy.sh`

Provides a guarded Linux deployment flow. It requires `.env`, validates
Compose, extracts the selected GPU and model path from resolved configuration,
ensures the model resolves inside the mounted `models/` directory, runs host
checks, builds (or uses a preloaded image with `BUILD_IMAGE=false`), checks
GPU visibility, starts the service, then waits for readiness. It does not
install drivers, alter the kernel, or stop unrelated containers.

### `dockerrun.txt`

Contains two manual Docker commands (`docker info` and a basic `docker build`)
from an earlier/simple workflow. The supported deployment process is
`scripts/deploy.sh` or the validated Compose steps in `README.md`; a bare
`docker build` alone does not configure the API key, model mount, GPU UUID,
port binding, or production hardening.

## Supporting directories and documentation

### `models/.gitkeep`

Keeps the otherwise-empty `models/` folder in source control. Actual
operator-supplied model weights are ignored by Git and Docker build context;
Compose mounts this folder read-only at `/models`.

### `logs/.gitkeep`

Keeps the otherwise-empty `logs/` folder. The current application writes logs
to stdout for Docker log rotation rather than saving uploaded images or
application log files here.

### `config/README.md`

Explains how to set protected runtime configuration, keep secrets out of
Compose/source control, and bind the API only to an approved internal/VPN
interface with firewall restrictions.

### `README.md`

The main operator guide: service architecture, endpoints, model requirements,
environment variables, Linux deployment, GPU UUID verification, offline
image-transfer options, API testing, monitoring, troubleshooting, rollback,
security, and the future RTSP design.

### `all_code_explain.md`

This file: a code-oriented explanation of the purpose and interactions of the
project files.

## Directories that are not source

`app/__pycache__/`, `app/routers/__pycache__/`, and `tests/__pycache__/`
contain Python bytecode generated during imports or tests. `.pytest_cache/`
contains pytest's local run metadata. They are ignored by Git and should not
be maintained manually.

The private `.env` and operator model weights are deliberately excluded from
this explanation. Never paste their contents into documentation or commit
them to source control.
