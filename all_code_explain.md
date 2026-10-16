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
values outside the allowed ranges. `Settings.__post_init__` also ensures
inference uses container device `cuda:0`, rejects wildcard CORS, and validates
configuration ranges.

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

- `ErrorDetail` and `ErrorResponse`: common error payload.
- `BoxCoordinates`, `Prediction`, `ImageSize`, and `PredictionResponse`:
  image inference result, including confidence bounds and original-image pixel
  coordinates.
- `ModelInfo`: model identity/version, classes, device, and inference settings.
- `HealthResponse`, `ReadinessResponse`, and `GpuResponse`: health endpoint
  response shapes.

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

- `GET /health/live` returns `alive` to show the process is responding; it
  does not assert model or GPU health.
- `GET /health/ready` returns readiness only when the model manager is ready;
  otherwise it returns HTTP 503 and a reason.
- `GET /health/gpu` returns the model manager's measured CUDA/device status.

### `app/routers/predictions.py`

Defines authenticated inference routes and image validation.

- `GET /v1/model/info` requires the `X-API-Key` header and returns model
  metadata only when the model is ready.
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

Run the mocked suite with:

```bash
python -m pytest -q
```

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
running them on the production host in this Windows workspace.

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
environment and the image path as its argument. The API key is passed through
a temporary curl config stream to avoid putting it directly in curl's
argument list.

### `scripts/deploy.sh`

Provides a guarded Linux deployment flow. It requires `.env`, validates
Compose, extracts the selected GPU and model path from resolved configuration,
ensures the model resolves inside the mounted `models/` directory, runs host
checks, builds (or uses a preloaded image with `BUILD_IMAGE=false`), checks
GPU visibility, starts the service, then waits for readiness. It does not
install drivers, alter the kernel, or stop unrelated containers.

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
