"""
MedXplain-Simple — Grad-CAM heatmap generation for AI prediction transparency.

Generates class-activation heatmaps highlighting image regions that influenced
each prediction, providing clinicians with visual explanations of model reasoning.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class MedicalGradCAM:
    """Grad-CAM heatmap generator for medical imaging models.

    Wraps pytorch-grad-cam to produce clinically meaningful heatmaps
    overlaid on the original medical image.
    """

    COLORMAP_JET = 2  # cv2.COLORMAP_JET equivalent

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._cam = None

    def _ensure_cam(self) -> None:
        if self._cam is not None:
            return
        try:
            from pytorch_grad_cam import GradCAM

            self._cam = GradCAM(
                model=self.model,
                target_layers=[self.target_layer],
            )
        except ImportError:
            raise ImportError(
                "pytorch-grad-cam is required. Install via: pip install pytorch-grad-cam"
            )

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        image_size: tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """Generate a Grad-CAM heatmap for the given input.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Preprocessed image tensor of shape (1, C, H, W).
        target_class : int, optional
            Target class index. If None, uses the predicted class.
        image_size : tuple
            Output heatmap size (H, W).

        Returns
        -------
        np.ndarray
            Heatmap as a float array of shape (H, W) in [0, 1].
        """
        self._ensure_cam()
        targets = None
        if target_class is not None:
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            targets = [ClassifierOutputTarget(target_class)]

        grayscale_cam = self._cam(input_tensor=input_tensor, targets=targets)
        heatmap = grayscale_cam[0]
        return heatmap

    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: Image.Image,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> Image.Image:
        """Overlay a Grad-CAM heatmap on the original image.

        Parameters
        ----------
        heatmap : np.ndarray
            Heatmap array of shape (H, W) in [0, 1].
        original_image : PIL.Image
            The original medical image.
        alpha : float
            Transparency factor for the overlay.
        colormap : str
            Matplotlib colormap name.

        Returns
        -------
        PIL.Image
            Image with heatmap overlay.
        """
        import matplotlib
        import matplotlib.cm as cm

        img = original_image.convert("RGB").resize(
            (heatmap.shape[1], heatmap.shape[0])
        )
        img_array = np.array(img, dtype=np.float32) / 255.0

        cmap = matplotlib.colormaps.get_cmap(colormap)
        heatmap_colored = cmap(heatmap)[:, :, :3]

        overlay = (1 - alpha) * img_array + alpha * heatmap_colored.astype(np.float32)
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(overlay)

    def generate_and_overlay(
        self,
        input_tensor: torch.Tensor,
        original_image: Image.Image,
        target_class: Optional[int] = None,
        alpha: float = 0.4,
    ) -> tuple[Image.Image, np.ndarray]:
        """Generate heatmap and overlay in one call.

        Returns
        -------
        tuple[PIL.Image, np.ndarray]
            (overlay_image, raw_heatmap)
        """
        heatmap = self.generate_heatmap(input_tensor, target_class)
        overlay = self.overlay_heatmap(heatmap, original_image, alpha)
        return overlay, heatmap

    @staticmethod
    def heatmap_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
        """Convert a PIL image to bytes for API responses."""
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        return buffer.getvalue()

    @staticmethod
    def compute_coverage(heatmap: np.ndarray, threshold: float = 0.3) -> float:
        """Compute the fraction of the image covered by significant activations.

        Parameters
        ----------
        heatmap : np.ndarray
            Heatmap in [0, 1].
        threshold : float
            Minimum activation to consider significant.

        Returns
        -------
        float
            Coverage ratio in [0, 1].
        """
        return float(np.mean(heatmap >= threshold))
