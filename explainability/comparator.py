"""
MedXplain-Simple — SSIM-based longitudinal image comparison for disease progression tracking.

Compares two medical images (e.g. follow-up scans) using the Structural
Similarity Index (SSIM) and produces difference maps highlighting regions
of change.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of SSIM-based longitudinal comparison."""

    ssim_score: float
    difference_map: np.ndarray
    changed_regions: list[dict]
    summary: str
    overlay_image: Optional[Image.Image] = None


class SSIMComparator:
    """SSIM-based longitudinal image comparison.

    Compares two medical scans (baseline vs follow-up) to detect and
    visualise structural changes over time.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (256, 256),
        win_size: int = 7,
        threshold: float = 0.3,
    ):
        self.target_size = target_size
        self.win_size = win_size
        self.threshold = threshold

    def compare(
        self,
        image_baseline: Image.Image,
        image_followup: Image.Image,
    ) -> ComparisonResult:
        """Compare two medical images using SSIM.

        Parameters
        ----------
        image_baseline : PIL.Image
            The baseline (earlier) scan.
        image_followup : PIL.Image
            The follow-up (later) scan.

        Returns
        -------
        ComparisonResult
            SSIM score, difference map, and change analysis.
        """
        from skimage.metrics import structural_similarity

        arr_base = self._preprocess(image_baseline)
        arr_follow = self._preprocess(image_followup)

        ssim_score, ssim_map = structural_similarity(
            arr_base,
            arr_follow,
            full=True,
            win_size=self.win_size,
            data_range=1.0,
        )

        diff_map = 1.0 - ssim_map
        changed_regions = self._detect_changed_regions(diff_map)

        summary = self._generate_summary(ssim_score, changed_regions)
        overlay = self._create_overlay(image_followup, diff_map)

        return ComparisonResult(
            ssim_score=float(ssim_score),
            difference_map=diff_map.astype(np.float32),
            changed_regions=changed_regions,
            summary=summary,
            overlay_image=overlay,
        )

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.convert("L").resize(self.target_size)
        arr = np.array(img, dtype=np.float64) / 255.0
        return arr

    def _detect_changed_regions(
        self,
        diff_map: np.ndarray,
    ) -> list[dict]:
        """Identify contiguous regions of significant change."""
        from scipy import ndimage

        binary = diff_map > self.threshold
        labeled, num_features = ndimage.label(binary)

        regions = []
        for i in range(1, num_features + 1):
            region_mask = labeled == i
            area = int(np.sum(region_mask))
            if area < 10:
                continue

            coords = np.where(region_mask)
            y_min, y_max = int(coords[0].min()), int(coords[0].max())
            x_min, x_max = int(coords[1].min()), int(coords[1].max())
            mean_diff = float(diff_map[region_mask].mean())

            severity = "mild"
            if mean_diff > 0.6:
                severity = "significant"
            elif mean_diff > 0.4:
                severity = "moderate"

            regions.append({
                "id": i,
                "bbox": [x_min, y_min, x_max, y_max],
                "area_pixels": area,
                "mean_difference": round(mean_diff, 4),
                "severity": severity,
            })

        regions.sort(key=lambda r: r["mean_difference"], reverse=True)
        return regions

    def _generate_summary(
        self,
        ssim_score: float,
        regions: list[dict],
    ) -> str:
        if ssim_score > 0.95:
            status = "No significant changes detected"
        elif ssim_score > 0.85:
            status = "Minor changes detected"
        elif ssim_score > 0.70:
            status = "Moderate changes detected"
        else:
            status = "Significant changes detected"

        parts = [
            f"SSIM Score: {ssim_score:.4f} — {status}.",
            f"Changed regions identified: {len(regions)}.",
        ]

        for region in regions[:3]:
            parts.append(
                f"  Region {region['id']}: {region['severity']} change "
                f"(mean diff={region['mean_difference']:.3f}, "
                f"area={region['area_pixels']}px)."
            )

        return "\n".join(parts)

    def _create_overlay(
        self,
        image: Image.Image,
        diff_map: np.ndarray,
        alpha: float = 0.5,
    ) -> Image.Image:
        """Create a visual overlay highlighting changes."""
        img = image.convert("RGB").resize(self.target_size)
        img_array = np.array(img, dtype=np.float32) / 255.0

        red_overlay = np.zeros_like(img_array)
        red_overlay[:, :, 0] = diff_map.astype(np.float32)

        mask = diff_map > self.threshold
        result = img_array.copy()
        result[mask] = (1 - alpha) * img_array[mask] + alpha * red_overlay[mask]
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(result)

    @staticmethod
    def result_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        return buffer.getvalue()
