.PHONY: test run

test:
	python -m pytest -q

run:
	python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
