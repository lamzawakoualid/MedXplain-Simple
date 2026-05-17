"""
MedXplain-Simple — Unit tests for Gemini prompt reformulation engine.

Tests the clinical query reformulation pipeline that uses Google Gemini API
to optimise questions for better model performance.

Usage:
    python -m scripts.test_reformulation
    python -m pytest scripts/test_reformulation.py -v
"""

from __future__ import annotations

import logging
import os
import unittest
from typing import Optional

logger = logging.getLogger(__name__)


class ClinicalQueryReformulator:
    """Reformulates clinical queries using Google Gemini API for optimal model performance.

    Transforms user questions into structured clinical prompts that
    elicit better responses from the BLIP-2 + Medra-4B pipeline.
    """

    REFORMULATION_PROMPT = (
        "You are a medical imaging specialist. Reformulate the following clinical "
        "question to be more precise and structured for a medical VQA system. "
        "The reformulated question should:\n"
        "1. Be specific about what visual findings to look for.\n"
        "2. Reference relevant anatomical structures.\n"
        "3. Use standard radiology terminology.\n"
        "4. Be concise but comprehensive.\n\n"
        "Original question: {question}\n\n"
        "Reformulated question:"
    )

    TEMPLATES = {
        "abnormality": (
            "Identify and describe any abnormal findings visible in this "
            "{modality} image of the {body_part}. Include location, size "
            "estimation, and morphological characteristics."
        ),
        "diagnosis": (
            "Based on the visible imaging features in this {modality} of the "
            "{body_part}, what is the most likely diagnosis? List key "
            "supporting findings."
        ),
        "comparison": (
            "Compare this {modality} image with the prior study. Identify any "
            "interval changes in the {body_part}, noting progression, "
            "regression, or new findings."
        ),
        "normal": (
            "Evaluate this {modality} image of the {body_part} and confirm "
            "whether all structures appear within normal limits. Note any "
            "variants or borderline findings."
        ),
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._model = None

    def _init_gemini(self) -> bool:
        if not self.api_key:
            logger.warning("No Gemini API key provided. Using template fallback.")
            return False
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel("gemini-pro")
            return True
        except ImportError:
            logger.warning("google-generativeai not installed.")
            return False
        except Exception as e:
            logger.warning("Failed to initialise Gemini: %s", e)
            return False

    def reformulate(
        self,
        question: str,
        modality: str = "radiological",
        body_part: str = "examined region",
    ) -> str:
        """Reformulate a clinical question for optimal VQA performance.

        Tries Gemini API first, falls back to template-based reformulation.
        """
        if self._model is None and not self._init_gemini():
            return self._template_reformulate(question, modality, body_part)

        try:
            prompt = self.REFORMULATION_PROMPT.format(question=question)
            response = self._model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning("Gemini reformulation failed: %s. Using template.", e)
            return self._template_reformulate(question, modality, body_part)

    def _template_reformulate(
        self,
        question: str,
        modality: str,
        body_part: str,
    ) -> str:
        q_lower = question.lower()

        if any(w in q_lower for w in ("abnormal", "finding", "wrong", "issue")):
            template = self.TEMPLATES["abnormality"]
        elif any(w in q_lower for w in ("diagnos", "condition", "disease", "what is")):
            template = self.TEMPLATES["diagnosis"]
        elif any(w in q_lower for w in ("compar", "change", "progress", "prior")):
            template = self.TEMPLATES["comparison"]
        elif any(w in q_lower for w in ("normal", "healthy", "clear")):
            template = self.TEMPLATES["normal"]
        else:
            return question

        return template.format(modality=modality, body_part=body_part)

    def get_template(self, category: str) -> Optional[str]:
        return self.TEMPLATES.get(category)


class TestClinicalQueryReformulator(unittest.TestCase):
    """Unit tests for the ClinicalQueryReformulator."""

    def setUp(self):
        self.reformulator = ClinicalQueryReformulator(api_key=None)

    def test_abnormality_detection(self):
        result = self.reformulator.reformulate(
            "Are there any abnormal findings?",
            modality="X-Ray",
            body_part="chest",
        )
        self.assertIn("abnormal", result.lower())
        self.assertIn("X-Ray", result)
        self.assertIn("chest", result)

    def test_diagnosis_query(self):
        result = self.reformulator.reformulate(
            "What is the diagnosis?",
            modality="CT",
            body_part="brain",
        )
        self.assertIn("diagnosis", result.lower())
        self.assertIn("CT", result)

    def test_comparison_query(self):
        result = self.reformulator.reformulate(
            "How does this compare to the prior study?",
            modality="MRI",
            body_part="spine",
        )
        self.assertIn("compare", result.lower())

    def test_normal_query(self):
        result = self.reformulator.reformulate(
            "Is this image normal?",
            modality="X-Ray",
            body_part="chest",
        )
        self.assertIn("normal", result.lower())

    def test_passthrough_unknown(self):
        original = "Show me the pixel intensity distribution."
        result = self.reformulator.reformulate(original)
        self.assertEqual(result, original)

    def test_templates_exist(self):
        for category in ("abnormality", "diagnosis", "comparison", "normal"):
            template = self.reformulator.get_template(category)
            self.assertIsNotNone(template)
            self.assertIn("{modality}", template)

    def test_empty_question(self):
        result = self.reformulator.reformulate("")
        self.assertEqual(result, "")

    def test_no_api_key_fallback(self):
        reformulator = ClinicalQueryReformulator(api_key=None)
        result = reformulator.reformulate("What abnormalities are present?")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


if __name__ == "__main__":
    unittest.main()
