# Football Match Analyzer — v0.11
# CPU-only. GPU would need the nvidia runtime + a CUDA torch wheel.
FROM python:3.11-slim

WORKDIR /app

# system deps for opencv + matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY matchanalyzer/ matchanalyzer/
COPY models/ models/
COPY tools/ tools/

# uploads + reports live here (mount a volume to keep them)
VOLUME /app/uploads
EXPOSE 8000

CMD ["uvicorn", "matchanalyzer.web:app", "--host", "0.0.0.0", "--port", "8000"]
