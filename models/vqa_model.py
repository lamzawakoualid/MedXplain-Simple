"""
MedXplain-Simple — Abstract base classes and interfaces for VQA model implementations.

Defines the contract that all VQA models (BLIP-2, Medra-4B, etc.) must satisfy
so the backend inference pipeline can orchestrate them uniformly.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image


@dataclass
class VQAInput:
    """Standardised input for all VQA models."""

    image: Image.Image
    question: str
    max_answer_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VQAOutput:
    """Standardised output from all VQA models."""

    answer: str
    confidence: float = 0.0
    raw_logits: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVQAModel(abc.ABC):
    """Abstract base class for Vision-Language VQA models."""

    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self._model: Any = None
        self._processor: Any = None

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @abc.abstractmethod
    def load(self, checkpoint_path: Optional[Path] = None) -> None:
        """Load model weights from HuggingFace Hub or a local checkpoint."""

    @abc.abstractmethod
    def predict(self, vqa_input: VQAInput) -> VQAOutput:
        """Run inference on a single image-question pair."""

    @abc.abstractmethod
    def get_visual_encoder(self) -> torch.nn.Module:
        """Return the visual encoder sub-module (needed for Grad-CAM)."""

    def to(self, device: torch.device) -> "BaseVQAModel":
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


class BaseVisualEncoder(abc.ABC):
    """Abstract base class for standalone visual feature extractors (CNN / ViT)."""

    def __init__(self, model_name: str, num_classes: int, device: str = "auto"):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = BaseVQAModel._resolve_device(device)
        self._model: Any = None

    @abc.abstractmethod
    def load(self, checkpoint_path: Optional[Path] = None) -> None:
        """Load pre-trained or fine-tuned weights."""

    @abc.abstractmethod
    def extract_features(self, image: Image.Image) -> torch.Tensor:
        """Return a feature vector for the given image."""

    @abc.abstractmethod
    def classify(self, image: Image.Image) -> dict[str, float]:
        """Return class-probability mapping."""

    @abc.abstractmethod
    def get_target_layer(self) -> torch.nn.Module:
        """Return the layer to hook for Grad-CAM."""

    def to(self, device: torch.device) -> "BaseVisualEncoder":
        self.device = device
        if self._model is not None:
            self._model.to(device)
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


class BLIP2VQAModel(BaseVQAModel):
    """BLIP-2 Vision-Language Model for Medical VQA.

    Uses Salesforce BLIP-2 with optional LoRA fine-tuning for
    medical domain adaptation.  Generates free-form textual answers
    given a medical image and a clinical question.
    """

    HF_MODEL_ID = "Salesforce/blip2-opt-2.7b"

    def __init__(
        self,
        device: str = "auto",
        use_lora: bool = True,
        lora_checkpoint: Optional[Path] = None,
    ):
        super().__init__(model_name=self.HF_MODEL_ID, device=device)
        self.use_lora = use_lora
        self.lora_checkpoint = lora_checkpoint

    def load(self, checkpoint_path: Optional[Path] = None) -> None:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self._processor = Blip2Processor.from_pretrained(self.HF_MODEL_ID)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.HF_MODEL_ID,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)

        lora_path = checkpoint_path or self.lora_checkpoint
        if self.use_lora and lora_path and lora_path.exists():
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, str(lora_path))

        self._model.eval()

    def predict(self, vqa_input: VQAInput) -> VQAOutput:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self._processor(
            images=vqa_input.image,
            text=vqa_input.question,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=vqa_input.max_answer_length,
                temperature=vqa_input.temperature,
                top_p=vqa_input.top_p,
                do_sample=vqa_input.temperature > 0,
            )

        answer = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return VQAOutput(
            answer=answer,
            confidence=0.0,
            metadata={"model": self.model_name},
        )

    def get_visual_encoder(self) -> torch.nn.Module:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")
        return self._model.vision_model
