"""
MedXplain-Simple — DICOM format reading, processing, and pixel-level normalisation.

Handles native .dcm file reading, windowing, and conversion to PIL images
suitable for model inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DICOMMetadata:
    """Extracted metadata from a DICOM file."""

    patient_id: str
    study_date: str
    modality: str
    body_part: str
    study_description: str
    series_description: str
    rows: int
    columns: int
    pixel_spacing: Optional[tuple[float, float]]
    window_center: Optional[float]
    window_width: Optional[float]
    extra: dict


class DICOMPipeline:
    """DICOM file processing pipeline.

    Reads .dcm files, extracts metadata, applies windowing,
    and converts pixel data to normalised PIL Images.
    """

    def __init__(self, default_window: tuple[float, float] = (400.0, 1500.0)):
        self.default_window_center, self.default_window_width = default_window

    def read(self, path: str | Path) -> tuple[Image.Image, DICOMMetadata]:
        """Read a DICOM file and return (image, metadata).

        Parameters
        ----------
        path : str or Path
            Path to the .dcm file.

        Returns
        -------
        tuple[Image.Image, DICOMMetadata]
            Processed PIL image and extracted metadata.
        """
        import pydicom

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DICOM file not found: {path}")

        ds = pydicom.dcmread(str(path))
        metadata = self._extract_metadata(ds)
        pixel_array = self._get_pixel_array(ds)
        windowed = self._apply_windowing(pixel_array, metadata)
        image = self._to_pil(windowed)

        return image, metadata

    def _extract_metadata(self, ds) -> DICOMMetadata:
        def safe_get(attr: str, default: str = "") -> str:
            val = getattr(ds, attr, default)
            return str(val) if val else default

        pixel_spacing = None
        if hasattr(ds, "PixelSpacing") and ds.PixelSpacing:
            pixel_spacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))

        wc = float(ds.WindowCenter) if hasattr(ds, "WindowCenter") and ds.WindowCenter else None
        ww = float(ds.WindowWidth) if hasattr(ds, "WindowWidth") and ds.WindowWidth else None

        if isinstance(wc, (list, type(None))):
            wc = float(wc[0]) if wc else None
        if isinstance(ww, (list, type(None))):
            ww = float(ww[0]) if ww else None

        return DICOMMetadata(
            patient_id=safe_get("PatientID", "UNKNOWN"),
            study_date=safe_get("StudyDate"),
            modality=safe_get("Modality"),
            body_part=safe_get("BodyPartExamined"),
            study_description=safe_get("StudyDescription"),
            series_description=safe_get("SeriesDescription"),
            rows=int(getattr(ds, "Rows", 0)),
            columns=int(getattr(ds, "Columns", 0)),
            pixel_spacing=pixel_spacing,
            window_center=wc,
            window_width=ww,
            extra={
                "manufacturer": safe_get("Manufacturer"),
                "institution": safe_get("InstitutionName"),
                "bits_stored": int(getattr(ds, "BitsStored", 0)),
            },
        )

    def _get_pixel_array(self, ds) -> np.ndarray:
        pixel_array = ds.pixel_array.astype(np.float64)

        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        pixel_array = pixel_array * slope + intercept

        return pixel_array

    def _apply_windowing(
        self,
        pixel_array: np.ndarray,
        metadata: DICOMMetadata,
    ) -> np.ndarray:
        wc = metadata.window_center or self.default_window_center
        ww = metadata.window_width or self.default_window_width

        lower = wc - ww / 2
        upper = wc + ww / 2

        windowed = np.clip(pixel_array, lower, upper)
        windowed = (windowed - lower) / (upper - lower)

        return windowed

    @staticmethod
    def _to_pil(normalised: np.ndarray) -> Image.Image:
        uint8 = (normalised * 255).astype(np.uint8)
        if uint8.ndim == 2:
            return Image.fromarray(uint8, mode="L").convert("RGB")
        return Image.fromarray(uint8)

    def batch_read(self, directory: str | Path) -> list[tuple[Image.Image, DICOMMetadata]]:
        """Read all DICOM files in a directory."""
        directory = Path(directory)
        results = []
        for dcm_path in sorted(directory.glob("**/*.dcm")):
            try:
                results.append(self.read(dcm_path))
            except Exception as e:
                logger.warning("Failed to read %s: %s", dcm_path, e)
        return results
