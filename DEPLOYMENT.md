# MedXplain-Simple — Deployment Guide

## Prerequisites

- Docker 24+ and Docker Compose
- Python 3.9+ (for local development)
- NVIDIA GPU with 16+ GB VRAM (for full mode; not needed for demo)

## Demo Mode (No GPU)

Demo mode uses a mock pipeline that returns realistic responses without loading models.

```bash
# Option 1: Docker
docker-compose up --build

# Option 2: Local
pip install -r requirements.txt
MEDXPLAIN_DEMO_MODE=true uvicorn app.backend_api:app --host 0.0.0.0 --port 8000
```

Access the UI at `http://localhost:8000`.

## Full Mode (GPU Required)

### Local Deployment

```bash
# Install dependencies (including Medra-4B support)
pip install -r requirements-medra.txt

# Set environment
export MEDXPLAIN_DEMO_MODE=false
export MEDXPLAIN_DEVICE=cuda

# Optional: set model checkpoint paths
export BLIP2_CHECKPOINT=./checkpoints/blip2/best
export MEDRA_CHECKPOINT=./checkpoints/medra/best

# Start server
uvicorn app.backend_api:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
# With GPU support
MEDXPLAIN_DEMO_MODE=false MEDXPLAIN_DEVICE=cuda docker-compose up --build
```

For NVIDIA GPU in Docker, ensure `nvidia-container-toolkit` is installed and add to `docker-compose.yml`:

```yaml
services:
  medxplain-api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Environment Variables

| Variable | Default | Options |
|----------|---------|---------|
| `MEDXPLAIN_DEMO_MODE` | `true` | `true` / `false` |
| `MEDXPLAIN_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `MEDXPLAIN_ENABLE_MEDRA` | `true` | `true` / `false` |
| `MEDXPLAIN_ENABLE_GRADCAM` | `true` | `true` / `false` |
| `BLIP2_CHECKPOINT` | — | Path to LoRA weights |
| `MEDRA_CHECKPOINT` | — | Path to LoRA weights |
| `GEMINI_API_KEY` | — | Google Gemini API key |

## Health Check

```bash
curl http://localhost:8000/health
```

## Performance

- **Demo mode**: ~300ms per query
- **Full mode (GPU)**: Target < 5 seconds per query
- **Memory**: ~2 GB (demo), ~8-16 GB (full with quantisation)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Enable 4-bit quantisation (`MEDXPLAIN_ENABLE_MEDRA=true`) |
| Slow inference | Use GPU (`MEDXPLAIN_DEVICE=cuda`) |
| Model not found | Check checkpoint paths and HuggingFace cache |
| DICOM errors | Ensure `pydicom` is installed |
