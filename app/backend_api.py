"""
MedXplain-Simple — FastAPI server with all REST endpoints for analysis,
report generation, and VQA.

Endpoints:
  POST /analyze       — Image + question → answer + heatmap + reasoning
  POST /report        — Generate a clinical PDF report
  POST /compare       — SSIM longitudinal comparison of two images
  GET  /health        — Health check
  GET  /              — Serve the frontend UI
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MedXplain-Simple API",
    description="Medical Visual Question Answering with Explainable AI",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global pipeline instance (lazy-loaded)
# ---------------------------------------------------------------------------
_pipeline = None
_demo_mode = os.getenv("MEDXPLAIN_DEMO_MODE", "true").lower() == "true"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
REPORTS_DIR = PROJECT_ROOT / "medxplain_db" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if _demo_mode:
            from app.backend import MockPipeline
            _pipeline = MockPipeline()
        else:
            from app.backend import MedXplainPipeline
            _pipeline = MedXplainPipeline(
                blip2_checkpoint=_get_env_path("BLIP2_CHECKPOINT"),
                medra_checkpoint=_get_env_path("MEDRA_CHECKPOINT"),
                device=os.getenv("MEDXPLAIN_DEVICE", "auto"),
                enable_medra=os.getenv("MEDXPLAIN_ENABLE_MEDRA", "true").lower() == "true",
                enable_gradcam=os.getenv("MEDXPLAIN_ENABLE_GRADCAM", "true").lower() == "true",
            )
        _pipeline.load()
    return _pipeline


def _get_env_path(key: str) -> Optional[Path]:
    val = os.getenv(key)
    return Path(val) if val else None


# ---------------------------------------------------------------------------
# Mount static files
# ---------------------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def serve_frontend():
    """Serve the MedXplain-Simple frontend UI."""
    ui_path = PROJECT_ROOT / "app" / "medxplain_ui.html"
    if ui_path.exists():
        return FileResponse(str(ui_path), media_type="text/html")
    return JSONResponse({"message": "MedXplain-Simple API is running. UI not found."})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "model_loaded": pipeline.is_loaded,
        "demo_mode": _demo_mode,
        "timestamp": time.time(),
    }


@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(..., description="Medical image file (JPEG, PNG, or DICOM)"),
    question: str = Form(..., description="Clinical question about the image"),
    generate_heatmap: bool = Form(True, description="Generate Grad-CAM heatmap"),
):
    """Analyse a medical image with a clinical question.

    Returns VQA answer, clinical reasoning, differential diagnosis,
    recommendations, and an optional Grad-CAM heatmap overlay.
    """
    start = time.time()

    try:
        contents = await image.read()
        filename = image.filename or ""

        if filename.lower().endswith(".dcm"):
            pil_image = _read_dicom_bytes(contents)
        else:
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    pipeline = get_pipeline()
    result = pipeline.analyze(
        image=pil_image,
        question=question,
        generate_heatmap=generate_heatmap,
    )

    return {
        "vqa_answer": result.vqa_answer,
        "clinical_reasoning": result.clinical_reasoning,
        "differential_diagnosis": result.differential_diagnosis,
        "recommendations": result.recommendations,
        "confidence": result.confidence,
        "heatmap_base64": result.heatmap_base64,
        "heatmap_coverage": result.heatmap_coverage,
        "processing_time_ms": result.processing_time_ms,
        "model_used": result.model_used,
    }


@app.post("/report")
async def generate_report(
    image: UploadFile = File(..., description="Medical image"),
    question: str = Form(..., description="Clinical question"),
    patient_id: str = Form("ANONYMOUS", description="Patient ID"),
    modality: str = Form("", description="Imaging modality"),
    body_part: str = Form("", description="Body part examined"),
):
    """Generate a clinical PDF report for a medical image analysis."""
    try:
        contents = await image.read()
        filename = image.filename or ""

        if filename.lower().endswith(".dcm"):
            pil_image = _read_dicom_bytes(contents)
        else:
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    pipeline = get_pipeline()
    result = pipeline.analyze(image=pil_image, question=question)

    from pdf_generator import PDFReportGenerator, ReportData

    report_data = ReportData(
        patient_id=patient_id,
        modality=modality,
        body_part=body_part,
        clinical_question=question,
        vqa_answer=result.vqa_answer,
        clinical_reasoning=result.clinical_reasoning,
        differential_diagnosis=result.differential_diagnosis,
        recommendations=result.recommendations,
        confidence=result.confidence,
        heatmap_base64=result.heatmap_base64,
        model_used=result.model_used,
        processing_time_ms=result.processing_time_ms,
    )

    generator = PDFReportGenerator()
    pdf_bytes = generator.generate(report_data)

    report_path = REPORTS_DIR / f"report_{patient_id}_{int(time.time())}.pdf"
    report_path.write_bytes(pdf_bytes)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=MedXplain_Report_{patient_id}.pdf"
        },
    )


@app.post("/compare")
async def compare_images(
    image_baseline: UploadFile = File(..., description="Baseline (earlier) scan"),
    image_followup: UploadFile = File(..., description="Follow-up (later) scan"),
):
    """Compare two medical images using SSIM for longitudinal analysis."""
    try:
        base_bytes = await image_baseline.read()
        follow_bytes = await image_followup.read()
        img_base = Image.open(io.BytesIO(base_bytes)).convert("RGB")
        img_follow = Image.open(io.BytesIO(follow_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image(s): {e}")

    from explainability.comparator import SSIMComparator

    comparator = SSIMComparator()
    result = comparator.compare(img_base, img_follow)

    overlay_b64 = None
    if result.overlay_image is not None:
        buf = io.BytesIO()
        result.overlay_image.save(buf, format="PNG")
        overlay_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "ssim_score": result.ssim_score,
        "summary": result.summary,
        "changed_regions": result.changed_regions,
        "overlay_base64": overlay_b64,
    }


def _read_dicom_bytes(data: bytes) -> Image.Image:
    """Read DICOM from raw bytes."""
    import tempfile
    from data_ingestion.dicom_pipeline import DICOMPipeline

    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as f:
        f.write(data)
        f.flush()
        pipeline = DICOMPipeline()
        image, _ = pipeline.read(f.name)

    return image
