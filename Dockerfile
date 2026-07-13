FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY sanguine ./sanguine

RUN pip install --no-cache-dir -e ".[server]"

EXPOSE 8000
CMD ["uvicorn", "sanguine.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
