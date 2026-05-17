# MedXplain-Simple — API Documentation

## Base URL

```
http://localhost:8000
```

## Endpoints

### GET `/health`

Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "demo_mode": true,
    "timestamp": 1717000000.0
}
```

---

### POST `/analyze`

Analyse a medical image with a clinical question.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | Yes | Medical image (JPEG, PNG, DICOM) |
| `question` | string | Yes | Clinical question about the image |
| `generate_heatmap` | boolean | No | Generate Grad-CAM heatmap (default: true) |

**Response:**
```json
{
    "vqa_answer": "The image shows findings consistent with...",
    "clinical_reasoning": "Key features include...",
    "differential_diagnosis": [
        "Primary condition based on imaging features",
        "Alternative diagnosis requiring clinical correlation"
    ],
    "recommendations": [
        "Clinical correlation recommended",
        "Follow-up imaging in 3-6 months"
    ],
    "confidence": 0.78,
    "heatmap_base64": "iVBORw0KGgo...",
    "heatmap_coverage": 0.42,
    "processing_time_ms": 1234.5,
    "model_used": "BLIP-2 + Medra-4B"
}
```

---

### POST `/report`

Generate a clinical PDF report.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | Yes | Medical image |
| `question` | string | Yes | Clinical question |
| `patient_id` | string | No | Patient identifier (default: ANONYMOUS) |
| `modality` | string | No | Imaging modality |
| `body_part` | string | No | Body part examined |

**Response:** PDF file download (`application/pdf`)

---

### POST `/compare`

SSIM-based longitudinal image comparison.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image_baseline` | file | Yes | Baseline (earlier) scan |
| `image_followup` | file | Yes | Follow-up (later) scan |

**Response:**
```json
{
    "ssim_score": 0.8742,
    "summary": "SSIM Score: 0.8742 — Minor changes detected.\nChanged regions identified: 2.",
    "changed_regions": [
        {
            "id": 1,
            "bbox": [50, 80, 120, 150],
            "area_pixels": 245,
            "mean_difference": 0.412,
            "severity": "moderate"
        }
    ],
    "overlay_base64": "iVBORw0KGgo..."
}
```

## Error Responses

All errors follow this format:

```json
{
    "detail": "Description of the error"
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Invalid request (bad image, missing fields) |
| 500 | Internal server error |

## CORS

All origins are allowed by default. Configure via the `api.cors_origins` setting in `config.yaml`.
