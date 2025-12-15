FROM python:3.11-slim

WORKDIR /app

# libs for torchaudio reading wav
RUN apt-get update && apt-get install -y --no-install-recommends     libsndfile1 sox   && rm -rf /var/lib/apt/lists/*

COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt


COPY src ./src
# models copied after you train/evaluate locally
COPY models ./models

RUN mkdir -p /app/data

ENV MODEL_PATH=/app/models/model.pt
ENV METRICS_PATH=/app/models/metrics.json
ENV DATA_ROOT=/app/data

EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "src.webapp.app:app"]
