"""
MedXplain-Simple — Clinical feature modules and differential diagnosis layers.

Provides structured clinical knowledge for enriching VQA outputs
with domain-specific context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


PATHOLOGY_INFO: dict[str, dict[str, str]] = {
    "Atelectasis": {
        "description": "Partial or complete collapse of the lung or a section of the lung.",
        "severity": "moderate",
        "followup": "CT scan may be warranted for persistent cases.",
    },
    "Cardiomegaly": {
        "description": "Enlargement of the heart, often indicating underlying cardiac disease.",
        "severity": "moderate",
        "followup": "Echocardiography recommended for further evaluation.",
    },
    "Consolidation": {
        "description": "Region of lung tissue filled with liquid instead of air.",
        "severity": "moderate-high",
        "followup": "Clinical correlation and possibly CT for extent assessment.",
    },
    "Edema": {
        "description": "Excess fluid accumulation in the lungs (pulmonary edema).",
        "severity": "high",
        "followup": "Urgent clinical assessment; echocardiography and BNP levels.",
    },
    "Effusion": {
        "description": "Abnormal accumulation of fluid in the pleural space.",
        "severity": "moderate",
        "followup": "Thoracentesis may be needed for large effusions.",
    },
    "Emphysema": {
        "description": "Destruction of air sacs (alveoli) leading to breathing difficulty.",
        "severity": "moderate",
        "followup": "Pulmonary function tests; CT for detailed assessment.",
    },
    "Fibrosis": {
        "description": "Scarring of lung tissue causing progressive breathing difficulty.",
        "severity": "moderate-high",
        "followup": "High-resolution CT; pulmonary function tests.",
    },
    "Mass": {
        "description": "Abnormal growth or lesion in the lung requiring investigation.",
        "severity": "high",
        "followup": "Urgent CT with contrast; possible biopsy.",
    },
    "Nodule": {
        "description": "Small rounded opacity in the lung, often incidental.",
        "severity": "low-moderate",
        "followup": "Size-dependent follow-up; Fleischner criteria for management.",
    },
    "Pneumonia": {
        "description": "Infection of the lung parenchyma causing inflammation.",
        "severity": "moderate-high",
        "followup": "Clinical assessment; sputum culture; follow-up imaging.",
    },
    "Pneumothorax": {
        "description": "Presence of air in the pleural space causing lung collapse.",
        "severity": "high",
        "followup": "Urgent assessment; chest tube may be required.",
    },
    "No Finding": {
        "description": "No significant pathological findings identified.",
        "severity": "none",
        "followup": "Routine follow-up as clinically indicated.",
    },
}


@dataclass
class ClinicalContext:
    """Structured clinical context for a detected finding."""

    pathology: str
    description: str
    severity: str
    followup_recommendation: str
    confidence: float
    region: Optional[str] = None


def get_clinical_context(
    pathology: str,
    confidence: float = 0.0,
    region: Optional[str] = None,
) -> ClinicalContext:
    """Look up clinical context for a detected pathology."""
    info = PATHOLOGY_INFO.get(pathology, {
        "description": f"Detected finding: {pathology}",
        "severity": "unknown",
        "followup": "Clinical correlation recommended.",
    })

    return ClinicalContext(
        pathology=pathology,
        description=info["description"],
        severity=info["severity"],
        followup_recommendation=info["followup"],
        confidence=confidence,
        region=region,
    )


def generate_differential(
    primary_finding: str,
    confidence: float,
) -> list[ClinicalContext]:
    """Generate a differential diagnosis list based on the primary finding."""
    differentials: dict[str, list[str]] = {
        "Consolidation": ["Pneumonia", "Atelectasis", "Edema"],
        "Mass": ["Nodule", "Consolidation", "Fibrosis"],
        "Effusion": ["Edema", "Consolidation", "Pneumonia"],
        "Pneumonia": ["Consolidation", "Edema", "Atelectasis"],
        "Nodule": ["Mass", "Fibrosis", "Pneumonia"],
        "Edema": ["Effusion", "Consolidation", "Pneumonia"],
    }

    alt_names = differentials.get(primary_finding, [])
    results = [get_clinical_context(primary_finding, confidence)]

    for i, name in enumerate(alt_names[:3]):
        alt_conf = max(0.1, confidence - 0.15 * (i + 1))
        results.append(get_clinical_context(name, alt_conf))

    return results
