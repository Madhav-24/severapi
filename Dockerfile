FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp

WORKDIR /opt/construction-ai

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --uid 10001 --create-home --home-dir /home/app appuser

COPY --chown=appuser:appuser app ./app
USER 10001:10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "30"]
