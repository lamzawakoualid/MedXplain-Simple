FROM python:3.11-slim

LABEL maintainer="Yasser Daoud <yasser.daoud@edu.uiz.ac.ma>"
LABEL description="MedXplain-Simple — Medical VQA with Explainable AI"
LABEL version="2.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    MEDXPLAIN_DEMO_MODE=true

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Create necessary directories
RUN mkdir -p medxplain_db/data medxplain_db/reports static

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.backend_api:app", "--host", "0.0.0.0", "--port", "8000"]
