# MedXplain-Simple

**Design and Training of a Multimodal Vision-Language Model for Medical Visual Question Answering**

MedXplain-Simple is a fully functional, containerised medical imaging analysis platform that combines multiple AI components for explainable medical VQA.

## Architecture

The system integrates:
- **BLIP-2** — Core Vision-Language Model for medical VQA
- **Medra-4B** — Specialised medical LLM for clinical reasoning
- **DenseNet / ResNet** — Classical CNN visual feature extractors
- **ViT / Swin Transformer** — Transformer-based visual encoders
- **Grad-CAM** — Explainable AI heatmap generation
- **SSIM Comparator** — Longitudinal image comparison

### Dual-Model Inference Pipeline

```
Medical Image + Question
        │
        ▼
   ┌─────────┐     ┌──────────┐
   │  BLIP-2  │────▶│ Medra-4B │
   │   VQA    │     │ Clinical │
   │  Engine  │     │ Reasoning│
   └────┬─────┘     └────┬─────┘
        │                 │
        ▼                 ▼
   ┌─────────┐     ┌──────────┐
   │ Grad-CAM│     │ Combined │
   │ Heatmap │     │  Output  │
   └─────────┘     └──────────┘
```

## Quick Start

### Demo Mode (No GPU Required)

```bash
# Clone the repository
git clone https://github.com/your-username/MedXplain-Simple.git
cd MedXplain-Simple

# Install dependencies
pip install -r requirements.txt

# Run in demo mode
MEDXPLAIN_DEMO_MODE=true uvicorn app.backend_api:app --host 0.0.0.0 --port 8000

# Open http://localhost:8000 in your browser
```

### Docker Deployment

```bash
# Build and run
docker-compose up --build

# Or with custom settings
MEDXPLAIN_DEMO_MODE=false MEDXPLAIN_DEVICE=cuda docker-compose up --build
```

### Full Mode (GPU Required)

```bash
# Install all dependencies including Medra-4B
pip install -r requirements-medra.txt

# Run with full model loading
MEDXPLAIN_DEMO_MODE=false uvicorn app.backend_api:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
MedXplain-Simple/
├── app/
│   ├── backend_api.py          # FastAPI server — REST endpoints
│   ├── backend.py              # Core inference pipeline
│   └── medxplain_ui.html       # Vanilla JS/HTML5 frontend
├── models/
│   ├── vqa_model.py            # Abstract VQA model interfaces + BLIP-2
│   ├── medra_llm.py            # Medra-4B clinical reasoning wrapper
│   ├── cnn_models/             # DenseNet + ResNet encoders
│   └── transformer_models/     # ViT + Swin Transformer encoders
├── explainability/
│   ├── grad_cam.py             # Grad-CAM heatmap generation
│   └── comparator.py           # SSIM longitudinal comparison
├── training/
│   ├── trainer.py              # Core training loop + checkpointing
│   ├── train_blip2.py          # BLIP-2 LoRA fine-tuning
│   ├── train_medra.py          # Medra-4B LoRA fine-tuning
│   └── metrics.py              # BLEU, METEOR, CIDEr, BERTScore
├── data_ingestion/
│   ├── dataset.py              # PyTorch Dataset + DataLoader
│   └── dicom_pipeline.py       # DICOM file processing
├── medical_vqa_infrastructure/
│   ├── config.yaml             # Full system configuration
│   └── clinical_features.py    # Clinical knowledge modules
├── medxplain_db/               # Local JSON storage
├── scripts/
│   └── test_reformulation.py   # Gemini query reformulation tests
├── pdf_generator.py            # Clinical PDF report engine
├── Dockerfile                  # Container image spec
├── docker-compose.yml          # Multi-container orchestration
├── requirements.txt            # Core Python dependencies
└── requirements-medra.txt      # Additional Medra-4B dependencies
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend UI |
| `/health` | GET | Health check |
| `/analyze` | POST | Image + question → answer + heatmap + reasoning |
| `/report` | POST | Generate clinical PDF report |
| `/compare` | POST | SSIM longitudinal comparison |

## Training

### Phase 1: Vision-Language Alignment

```bash
python -m training.train_blip2 \
    --phase 1 \
    --epochs 5 \
    --lr 1e-4 \
    --data-index data/vqa_index.json \
    --image-root data/images/
```

### Phase 2: VQA Fine-Tuning with LoRA

```bash
python -m training.train_blip2 \
    --phase 2 \
    --epochs 10 \
    --lr 2e-5 \
    --lora-r 16 \
    --data-index data/vqa_index.json \
    --image-root data/images/
```

### Medra-4B Fine-Tuning

```bash
python -m training.train_medra \
    --epochs 3 \
    --lr 1e-4 \
    --data-index data/medra_train.json \
    --output-dir checkpoints/medra
```

## Evaluation Targets

| Metric | Target | Description |
|--------|--------|-------------|
| VQA Accuracy | > 65% | Overall question answering |
| BLEU-4 | > 0.25 | Answer generation quality |
| METEOR | > 0.30 | Recall-oriented complement |
| CIDEr | > 0.50 | Consensus-based metric |
| BERTScore F1 | > 0.75 | Semantic similarity |
| Clinical F1 | > 0.60 | Pathology detection |
| Grad-CAM Coverage | ≥ 90% | Anatomically relevant heatmaps |
| Response Latency | < 5s | End-to-end per query |

## Technology Stack

- **Backend**: FastAPI + Uvicorn
- **Frontend**: Vanilla JS / HTML5 / CSS3
- **VLM**: BLIP-2 (Salesforce)
- **Medical LLM**: Medra-4B
- **Visual Encoders**: DenseNet, ResNet, ViT, Swin (via timm)
- **PEFT**: LoRA (Hugging Face PEFT)
- **Quantisation**: bitsandbytes (4-bit)
- **Explainability**: pytorch-grad-cam
- **Medical Imaging**: pydicom
- **PDF Reports**: ReportLab
- **Deployment**: Docker + Docker Compose

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEDXPLAIN_DEMO_MODE` | `true` | Use mock pipeline (no GPU) |
| `MEDXPLAIN_DEVICE` | `auto` | Device: auto, cpu, cuda |
| `MEDXPLAIN_ENABLE_MEDRA` | `true` | Enable Medra-4B reasoning |
| `MEDXPLAIN_ENABLE_GRADCAM` | `true` | Enable Grad-CAM heatmaps |
| `BLIP2_CHECKPOINT` | — | Path to BLIP-2 LoRA weights |
| `MEDRA_CHECKPOINT` | — | Path to Medra-4B LoRA weights |
| `GEMINI_API_KEY` | — | Google Gemini API key |

## Ethical Considerations

- **Clinical Disclaimer**: This is a research prototype, NOT intended for clinical use.
- **Patient Privacy**: Only de-identified, open-access datasets used.
- **Bias Transparency**: Dataset distributions documented; limitations communicated.
- **Explainability by Design**: Grad-CAM and attention visualisation are mandatory outputs.

## Author

**OUALID LAMZAWAK** — Master's in AI and Data Analytics, University Ibn Zohr  
Supervised by **Prof. Yassine Oukdach**

## License

Academic use. All source datasets verified for open-access licensing.
