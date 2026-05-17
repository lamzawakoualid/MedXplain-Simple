# MedXplain-Simple — Architecture Documentation

## System Overview

MedXplain-Simple follows a **layered three-tier architecture**:

```
┌────────────────────────────────────────────────┐
│              Frontend (Vanilla JS)              │
│  medxplain_ui.html — Image upload, Q&A, Heatmap│
└─────────────────────┬──────────────────────────┘
                      │ HTTP REST
┌─────────────────────▼──────────────────────────┐
│           Backend API (FastAPI + Uvicorn)        │
│  backend_api.py — /analyze, /report, /compare   │
│  backend.py     — MedXplainPipeline orchestrator│
└─────────────────────┬──────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────┐
│              AI Model Modules                    │
│  ┌──────────┐ ┌──────────┐ ┌───────────────┐   │
│  │  BLIP-2  │ │ Medra-4B │ │ CNN/ViT/Swin  │   │
│  │   VQA    │ │ Clinical │ │   Encoders    │   │
│  │  Engine  │ │ Reasoning│ │               │   │
│  └────┬─────┘ └────┬─────┘ └───────────────┘   │
│       │             │                            │
│  ┌────▼─────┐ ┌────▼─────┐                     │
│  │ Grad-CAM │ │   SSIM   │                     │
│  │ Heatmaps │ │Comparator│                     │
│  └──────────┘ └──────────┘                     │
└─────────────────────────────────────────────────┘
```

## Component Details

### 1. BLIP-2 Vision-Language Model (`models/vqa_model.py`)

- **Base**: Salesforce BLIP-2 (bootstrapped language-image pre-training)
- **Input**: 224x224 medical image + tokenised question (max 128 tokens)
- **Output**: Free-form answer text with confidence estimates
- **Fine-tuning**: LoRA adaptation on medical VQA datasets

### 2. Medra-4B Medical LLM (`models/medra_llm.py`)

- **Base**: Medra-4B Doctor Assistant LLM
- **Role**: Enriches BLIP-2 answers with clinical context
- **Features**: 4-bit quantisation, LoRA fine-tuning, fully local inference

### 3. Visual Encoders (`models/cnn_models/`, `models/transformer_models/`)

- **CNN**: DenseNet-121, ResNet-50 (via timm)
- **Transformer**: ViT-Base/16, Swin Transformer Tiny (via timm)
- **Role**: Auxiliary visual feature extraction

### 4. Explainability (`explainability/`)

- **Grad-CAM**: Heatmaps showing influential image regions
- **SSIM Comparator**: Structural similarity for longitudinal tracking

### 5. Training Pipeline (`training/`)

Two-phase LoRA fine-tuning strategy:
- **Phase 1**: Vision-language alignment (freeze LLM, train Q-Former)
- **Phase 2**: VQA task fine-tuning (LoRA on full pipeline)

### 6. Data Ingestion (`data_ingestion/`)

- **Dataset**: PyTorch Dataset for image-QA triplets (JSON/CSV index)
- **DICOM Pipeline**: Native .dcm reading with windowing and normalisation

## Inference Flow

```
1. User uploads image + question via frontend
2. FastAPI receives multipart form data
3. DICOM pipeline converts .dcm → PIL Image (if needed)
4. BLIP-2 generates VQA answer from image + question
5. Medra-4B enriches with clinical reasoning + differential
6. Grad-CAM produces heatmap from visual encoder
7. Results returned as JSON (+ optional PDF report)
```

## Deployment

Fully containerised via Docker Compose:
- Single service with FastAPI + all AI models
- HuggingFace model cache volume for persistence
- Health checks every 30 seconds
- Configurable via environment variables
