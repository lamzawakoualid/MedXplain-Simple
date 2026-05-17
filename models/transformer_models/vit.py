"""
MedXplain-Simple — Vision Transformer (ViT) visual encoder for medical imaging.

Complementary transformer-based visual encoder using the timm library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import timm
import torch
import torch.nn as nn
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from models.vqa_model import BaseVisualEncoder


class MedicalViT(BaseVisualEncoder):
    """ViT-Base/16 fine-tuned for medical image classification."""

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        device: str = "auto",
    ):
        super().__init__(model_name="vit_base_patch16_224", num_classes=num_classes, device=device)
        self.pretrained = pretrained
        self._transform = None

    def load(self, checkpoint_path: Optional[Path] = None) -> None:
        self._model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=self.pretrained,
            num_classes=self.num_classes,
        ).to(self.device)

        config = resolve_data_config({}, model=self._model)
        self._transform = create_transform(**config)

        if checkpoint_path and checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self._model.load_state_dict(state_dict, strict=False)

        self._model.eval()

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        if self._transform is None:
            raise RuntimeError("Model not loaded.")
        img = image.convert("RGB")
        tensor = self._transform(img).unsqueeze(0)
        return tensor.to(self.device)

    def extract_features(self, image: Image.Image) -> torch.Tensor:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        tensor = self._preprocess(image)
        features = self._model.forward_features(tensor)
        cls_token = features[:, 0]
        return cls_token.detach()

    def classify(self, image: Image.Image) -> dict[str, float]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        tensor = self._preprocess(image)
        with torch.no_grad():
            logits = self._model(tensor)
        probs = torch.softmax(logits, dim=-1).squeeze()
        return {f"class_{i}": float(probs[i]) for i in range(len(probs))}

    def get_target_layer(self) -> nn.Module:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")
        return self._model.blocks[-1].norm1
