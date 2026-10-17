.PHONY: test run run-training

test:
	python -m pytest -q

run:
	python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1

run-training:
	@test -f .env.training || { echo "Create .env.training from .env.training.example first." >&2; exit 1; }
	set -a; . ./.env.training; set +a; exec .venv/bin/python -m uvicorn app.training_api:app --host "$${TRAINING_HOST_BIND_ADDRESS:-127.0.0.1}" --port "$${TRAINING_PORT:-8001}" --workers 1
