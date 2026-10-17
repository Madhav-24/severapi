# YOLO Training API Link Report

## What the Link Is

The training API is a FastAPI service for submitting YOLO object-detection
training jobs to the GPU server. It is separate from the inference API on port
8000. Interns use it to upload a labeled dataset, start a training run, check
whether it finished, and download the resulting `best.pt` model.

The default address is `http://127.0.0.1:8001`. That address works only on the
GPU server itself. For interns to connect remotely, the operator must configure
`TRAINING_HOST_BIND_ADDRESS` to the approved VPN/internal interface address,
restrict port `8001` with the host firewall, and share that internal link. Do
not expose this API directly to the public internet. Use an internal TLS proxy
for remote access where available.

Once started, the launcher prints the API URL and the interactive docs URL.
For example:

```text
API:  http://10.20.30.40:8001
Docs: http://10.20.30.40:8001/docs
```

The example IP is illustrative. Use the actual approved address printed by the
GPU host. The docs page lets an intern inspect and call the routes from a
browser; it does not replace the API key or network restrictions.

## Routes

In the commands below, replace the example IP address with the approved API
base URL and replace `<JOB_ID>` with the ID returned when a job is submitted.

| Method and path | API key? | Purpose |
| --- | --- | --- |
| `GET /health/live` | No | Confirms the API process is running. It does not verify the GPU. |
| `GET /health/ready` | No | Confirms a valid training key is configured and the selected CUDA GPU is available. Returns the GPU name when ready. |
| `GET /health/gpu` | No | Reports whether CUDA is available and, when available, the configured visible device index and GPU name. |
| `POST /v1/training/jobs` | Yes | Uploads a dataset ZIP and queues one training job. |
| `GET /v1/training/jobs/<JOB_ID>` | Yes | Returns the job state and basic run settings. |
| `GET /v1/training/jobs/<JOB_ID>/model` | Yes | Downloads the successful job's `best.pt` artifact. |
| `GET /docs` | No | Interactive Swagger API documentation. |
| `GET /redoc` | No | Alternative generated API documentation. |
| `GET /openapi.json` | No | Machine-readable OpenAPI description. |

The health routes are diagnostic endpoints, not proof that a model has trained
or that a dataset is valid. A job can be submitted only while the configured
GPU is available. Only one training job can run at a time; another submission
while it is occupied receives HTTP `503`.

## What Interns Need

The intern needs:

- The approved internal API URL.
- The `TRAINING_API_KEY`, shared privately by the operator.
- A YOLO object-detection dataset packaged as one `.zip` file.
- Network access to the GPU host on the approved port.

The API key goes in the `X-API-Key` HTTP header. Do not put it in the URL,
dataset, screenshots, source code, or a shared notebook. Anyone who has the
key and network access can submit GPU jobs and download their model artifacts.

## Dataset ZIP Format

Put `data.yaml` at the root of the ZIP. Its `train` and `val` entries must
point to image directories inside the ZIP, and `names` must list the dataset's
class names. For example:

```text
dataset.zip
|-- data.yaml
|-- images/
|   |-- train/
|   |   `-- image001.jpg
|   `-- val/
|       `-- image101.jpg
`-- labels/
    |-- train/
    |   `-- image001.txt
    `-- val/
        `-- image101.txt
```

Example `data.yaml`:

```yaml
train: images/train
val: images/val
names:
  0: helmet
  1: safety_vest
```

Label files must follow YOLO detection format: one row per object, containing
`class_id x_center y_center width height`, with coordinates normalized between
0 and 1. Image and label basenames must match. The API checks the ZIP structure,
required YAML fields, and that the train/validation directories contain images;
Ultralytics performs additional dataset validation during training.

The default maximum ZIP upload is 2 GiB and the extracted dataset limit is
20 GiB. The ZIP must not contain absolute paths, `..` path traversal, special
files, or duplicate paths. Do not upload data unless you have permission to
process it on this server.

## Submit a Training Job

First, check the service and GPU:

```bash
API='http://10.20.30.40:8001'  # Replace with the approved GPU server URL.
curl --fail "$API/health/ready"
curl "$API/health/gpu"
```

Read the API key without echoing it, then submit the ZIP. The optional form
fields shown below use the service defaults explicitly: 50 epochs, 640 image
size, and batch size 16. Allowed ranges are 1–300 epochs, image size 320–1280,
and batch size 1–64.

```bash
read -rsp 'Training API key: ' TRAINING_API_KEY; printf '\n'
curl --fail-with-body -X POST "$API/v1/training/jobs" \
  -H "X-API-Key: $TRAINING_API_KEY" \
  -F 'file=@dataset.zip;type=application/zip' \
  -F 'epochs=50' \
  -F 'image_size=640' \
  -F 'batch=16'
unset TRAINING_API_KEY
```

The successful response has HTTP `202 Accepted` and includes a `job_id`, a
starting status, and a `status_url`. Save the `job_id`. The server trains from
`yolov8n.pt` by default; on the first run, Ultralytics may need internet access
to download that starting checkpoint. Training uses the GPU selected by the
operator, not a GPU chosen by the intern.

## Check Progress and Download the Model

Use the returned ID to check the job:

```bash
read -rsp 'Training API key: ' TRAINING_API_KEY; printf '\n'
curl --fail "$API/v1/training/jobs/<JOB_ID>" \
  -H "X-API-Key: $TRAINING_API_KEY"
unset TRAINING_API_KEY
```

The status is one of `queued`, `running`, `succeeded`, or `failed`. The current
API returns job state, timestamps, chosen epochs/image size/batch, whether a
model exists, and an error category on failure. It does not stream logs or
report per-epoch progress. Ask the operator to inspect container logs for
training diagnostics.

After the status is `succeeded`, download the model:

```bash
read -rsp 'Training API key: ' TRAINING_API_KEY; printf '\n'
curl --fail "$API/v1/training/jobs/<JOB_ID>/model" \
  -H "X-API-Key: $TRAINING_API_KEY" \
  -o best.pt
unset TRAINING_API_KEY
```

Keep the downloaded model in the team's approved storage and record which
dataset and training settings produced it. The server retains run output in a
persistent Docker volume. Job status is held in memory, however, so download
the model before an API restart; old job IDs will no longer be queryable after
restart.

## Common Responses

| HTTP status | Meaning |
| --- | --- |
| `202` | Job accepted. Poll its status URL. |
| `400` | Upload is not a valid ZIP archive. |
| `401` | Missing or invalid API key. |
| `411` | Request did not include `Content-Length`. |
| `413` | Upload or expanded archive exceeds its configured limit. |
| `415` | Upload filename is not a `.zip` file. |
| `422` | Required upload field or training settings are invalid. |
| `503` | GPU unavailable or another training job is active. |
| `404` | Job ID is unknown, or the model artifact is not ready/available. |

If `/health/live` works but `/health/ready` fails, do not submit training jobs.
Contact the operator to check the configured GPU, CUDA/PyTorch container, and
training API key. If readiness succeeds but training fails, check the dataset
layout and ask the operator to review the container logs.