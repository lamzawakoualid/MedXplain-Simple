"""
MedXplain-Simple — Core inference pipeline.

Orchestrates BLIP-2 VQA, Medra-4B clinical reasoning, and Grad-CAM
explainability into a unified analysis pipeline.
"""

from __future__ import annotations

import io
import base64
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Complete result from the MedXplain inference pipeline."""

    vqa_answer: str
    clinical_reasoning: str
    differential_diagnosis: list[str]
    recommendations: list[str]
    confidence: float
    heatmap_base64: Optional[str]
    heatmap_coverage: float
    processing_time_ms: float
    model_used: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MedXplainPipeline:
    """Core inference pipeline combining BLIP-2, Medra-4B, and Grad-CAM.

    The dual-model pipeline works as follows:
    1. BLIP-2 receives the medical image + question and generates a VQA answer.
    2. Medra-4B takes the VQA answer and enriches it with clinical reasoning.
    3. Grad-CAM generates a heatmap from the visual encoder for explainability.
    """

    def __init__(
        self,
        blip2_checkpoint: Optional[Path] = None,
        medra_checkpoint: Optional[Path] = None,
        device: str = "auto",
        enable_medra: bool = True,
        enable_gradcam: bool = True,
    ):
        self.device = self._resolve_device(device)
        self.enable_medra = enable_medra
        self.enable_gradcam = enable_gradcam
        self.blip2_checkpoint = blip2_checkpoint
        self.medra_checkpoint = medra_checkpoint

        self._blip2 = None
        self._medra = None
        self._gradcam = None
        self._loaded = False

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def load(self) -> None:
        """Load all model components."""
        logger.info("Loading MedXplain pipeline on %s...", self.device)
        start = time.time()

        self._load_blip2()

        if self.enable_medra:
            self._load_medra()

        if self.enable_gradcam:
            self._setup_gradcam()

        self._loaded = True
        elapsed = time.time() - start
        logger.info("Pipeline loaded in %.1f seconds.", elapsed)

    def _load_blip2(self) -> None:
        from models.vqa_model import BLIP2VQAModel

        self._blip2 = BLIP2VQAModel(
            device=str(self.device),
            use_lora=self.blip2_checkpoint is not None,
            lora_checkpoint=self.blip2_checkpoint,
        )
        self._blip2.load()
        logger.info("BLIP-2 loaded.")

    def _load_medra(self) -> None:
        from models.medra_llm import MedraLLM

        self._medra = MedraLLM(
            device=str(self.device),
            use_quantisation=True,
            lora_checkpoint=self.medra_checkpoint,
        )
        self._medra.load()
        logger.info("Medra-4B loaded.")

    def _setup_gradcam(self) -> None:
        if self._blip2 is None or not self._blip2.is_loaded:
            logger.warning("Cannot set up Grad-CAM: BLIP-2 not loaded.")
            return
        from explainability.grad_cam import MedicalGradCAM

        visual_encoder = self._blip2.get_visual_encoder()
        target_layer = list(visual_encoder.children())[-1]
        self._gradcam = MedicalGradCAM(
            model=visual_encoder,
            target_layer=target_layer,
        )
        logger.info("Grad-CAM initialised.")

    def analyze(
        self,
        image: Image.Image,
        question: str,
        generate_heatmap: bool = True,
    ) -> AnalysisResult:
        """Run the full analysis pipeline on an image-question pair.

        Parameters
        ----------
        image : PIL.Image
            The medical image to analyse.
        question : str
            The clinical question about the image.
        generate_heatmap : bool
            Whether to generate a Grad-CAM heatmap.

        Returns
        -------
        AnalysisResult
            Complete analysis with VQA answer, clinical reasoning, and heatmap.
        """
        if not self._loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        start = time.time()
        image = image.convert("RGB")

        # Step 1: BLIP-2 VQA
        from models.vqa_model import VQAInput

        vqa_input = VQAInput(image=image, question=question)
        vqa_output = self._blip2.predict(vqa_input)

        # Step 2: Medra-4B clinical reasoning
        clinical_reasoning = ""
        differential: list[str] = []
        recommendations: list[str] = []

        if self.enable_medra and self._medra is not None and self._medra.is_loaded:
            try:
                medra_response = self._medra.generate(
                    vqa_answer=vqa_output.answer,
                    question=question,
                )
                clinical_reasoning = medra_response.clinical_reasoning
                differential = medra_response.differential_diagnosis
                recommendations = medra_response.recommendations
            except Exception as e:
                logger.error("Medra-4B inference failed: %s", e)
                clinical_reasoning = f"Clinical reasoning unavailable: {e}"

        # Step 3: Grad-CAM heatmap
        heatmap_b64 = None
        coverage = 0.0

        if generate_heatmap and self._gradcam is not None:
            try:
                from torchvision import transforms

                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                input_tensor = transform(image).unsqueeze(0).to(self.device)

                overlay, raw_heatmap = self._gradcam.generate_and_overlay(
                    input_tensor=input_tensor,
                    original_image=image,
                )
                coverage = self._gradcam.compute_coverage(raw_heatmap)

                buf = io.BytesIO()
                overlay.save(buf, format="PNG")
                heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as e:
                logger.error("Grad-CAM generation failed: %s", e)

        elapsed_ms = (time.time() - start) * 1000

        return AnalysisResult(
            vqa_answer=vqa_output.answer,
            clinical_reasoning=clinical_reasoning,
            differential_diagnosis=differential,
            recommendations=recommendations,
            confidence=vqa_output.confidence,
            heatmap_base64=heatmap_b64,
            heatmap_coverage=coverage,
            processing_time_ms=elapsed_ms,
            model_used="BLIP-2 + Medra-4B" if self.enable_medra else "BLIP-2",
            metadata={
                "device": str(self.device),
                "medra_enabled": self.enable_medra,
                "gradcam_enabled": self.enable_gradcam,
            },
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class MockPipeline:
    """Mock pipeline for development and testing without GPU.

    Returns realistic-looking demo responses so the frontend
    can be developed and tested without loading actual models.
    """

    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        self._loaded = True
        logger.info("Mock pipeline loaded (demo mode).")

    def analyze(
        self,
        image: Image.Image,
        question: str,
        generate_heatmap: bool = True,
    ) -> AnalysisResult:
        if not self._loaded:
            raise RuntimeError("Pipeline not loaded.")

        time.sleep(0.3)

        heatmap_b64 = None
        if generate_heatmap:
            heatmap_b64 = self._generate_demo_heatmap(image)

        return AnalysisResult(
            vqa_answer=self._generate_demo_answer(question),
            clinical_reasoning=(
                "The image shows findings consistent with the described condition. "
                "Key features include altered density patterns in the region of interest. "
                "The findings should be correlated with clinical history and prior imaging."
            ),
            differential_diagnosis=[
                "Primary condition based on imaging features",
                "Alternative diagnosis requiring clinical correlation",
                "Less likely differential to consider",
            ],
            recommendations=[
                "Clinical correlation with patient history recommended",
                "Follow-up imaging in 3-6 months may be considered",
                "Specialist consultation if symptoms persist",
            ],
            confidence=0.78,
            heatmap_base64=heatmap_b64,
            heatmap_coverage=0.42,
            processing_time_ms=300.0,
            model_used="Demo Mode (Mock Pipeline)",
            metadata={"mode": "demo"},
        )

    def _generate_demo_answer(self, question: str) -> str:
        q_lower = question.lower()
        if any(w in q_lower for w in ("abnormal", "finding", "patholog")):
            return (
                "The image shows potential abnormal findings in the examined region. "
                "There appears to be an area of increased density that may indicate "
                "a pathological process requiring further clinical evaluation."
            )
        if any(w in q_lower for w in ("normal", "healthy")):
            return (
                "The image demonstrates largely normal anatomical structures. "
                "No significant acute abnormalities are readily identified, though "
                "clinical context should always be considered."
            )
        return (
            "Based on the visual analysis, the image shows features relevant to "
            "the clinical question. The findings require correlation with clinical "
            "history and potentially additional diagnostic workup."
        )

    def _generate_demo_heatmap(self, image: Image.Image) -> str:
        img = image.convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0

        h, w = arr.shape[:2]
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        sigma = min(h, w) / 4
        heatmap = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
        heatmap = heatmap / heatmap.max()

        import matplotlib
        cmap = matplotlib.colormaps.get_cmap("jet")
        heatmap_colored = cmap(heatmap)[:, :, :3].astype(np.float32)

        alpha = 0.4
        overlay = (1 - alpha) * arr + alpha * heatmap_colored
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

        result = Image.fromarray(overlay)
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @property
    def is_loaded(self) -> bool:
        return self._loaded
