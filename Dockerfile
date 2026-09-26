FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch wheel keeps the image ~4x smaller than the default CUDA build
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[model,serve]"

RUN useradd --create-home appuser
USER appuser

# mount a fine-tuned checkpoint here, e.g. -v $(pwd)/checkpoints/indicbart-hindi:/model
ENV TEXTSUMM_CHECKPOINT=/model
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "textsumm.api:app", "--host", "0.0.0.0", "--port", "8000"]
