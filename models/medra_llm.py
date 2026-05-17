"""
MedXplain-Simple — Medra-4B integration wrapper with specialised clinical knowledge.

Medra-4B is a domain-specific medical LLM used as a secondary reasoning engine.
It enriches BLIP-2 answers with clinical context, differential diagnoses,
and treatment recommendations.  Supports 4-bit quantisation for local deployment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class MedraResponse:
    """Response from Medra-4B clinical reasoning."""

    clinical_reasoning: str
    differential_diagnosis: list[str]
    recommendations: list[str]
    confidence: float
    raw_text: str
    metadata: dict[str, Any]


class MedraLLM:
    """Wrapper around Medra-4B Doctor Assistant LLM.

    Features
    --------
    - 4-bit quantisation via bitsandbytes for low-VRAM deployment.
    - LoRA adapter support for domain-specific fine-tuning.
    - Structured clinical output parsing.
    - Fully local inference — no patient data leaves the server.
    """

    HF_MODEL_ID = "Henrychur/Medra-4B"

    SYSTEM_PROMPT = (
        "You are a medical AI assistant specialising in radiology and medical imaging. "
        "Given an image analysis from a VQA model, provide:\n"
        "1. Clinical reasoning explaining the findings.\n"
        "2. Differential diagnosis (list possible conditions).\n"
        "3. Recommendations for follow-up or further investigation.\n"
        "Be precise, evidence-based, and always note that this is AI-assisted analysis "
        "requiring clinical validation."
    )

    def __init__(
        self,
        device: str = "auto",
        use_quantisation: bool = True,
        lora_checkpoint: Optional[Path] = None,
    ):
        self.device = self._resolve_device(device)
        self.use_quantisation = use_quantisation
        self.lora_checkpoint = lora_checkpoint
        self._model: Any = None
        self._tokenizer: Any = None

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def load(self, checkpoint_path: Optional[Path] = None) -> None:
        """Load Medra-4B with optional 4-bit quantisation and LoRA adapter."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading Medra-4B tokenizer from %s", self.HF_MODEL_ID)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.HF_MODEL_ID,
            trust_remote_code=True,
        )

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }

        if self.use_quantisation and self.device.type == "cuda":
            try:
                from transformers import BitsAndBytesConfig

                quantisation_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                load_kwargs["quantization_config"] = quantisation_config
                logger.info("4-bit quantisation enabled via bitsandbytes.")
            except ImportError:
                logger.warning("bitsandbytes not available; loading without quantisation.")
                load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = (
                torch.float16 if self.device.type == "cuda" else torch.float32
            )

        logger.info("Loading Medra-4B model...")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.HF_MODEL_ID,
            **load_kwargs,
        )

        if not self.use_quantisation:
            self._model.to(self.device)

        lora_path = checkpoint_path or self.lora_checkpoint
        if lora_path and lora_path.exists():
            from peft import PeftModel

            logger.info("Loading LoRA adapter from %s", lora_path)
            self._model = PeftModel.from_pretrained(self._model, str(lora_path))

        self._model.eval()
        logger.info("Medra-4B loaded successfully on %s.", self.device)

    def generate(
        self,
        vqa_answer: str,
        question: str,
        image_description: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> MedraResponse:
        """Generate clinical reasoning based on BLIP-2 VQA output.

        Parameters
        ----------
        vqa_answer : str
            The answer produced by BLIP-2 for the medical image.
        question : str
            The original clinical question.
        image_description : str, optional
            Additional description of the medical image.
        max_new_tokens : int
            Maximum tokens to generate.
        temperature : float
            Sampling temperature.
        top_p : float
            Nucleus sampling threshold.

        Returns
        -------
        MedraResponse
            Structured clinical reasoning output.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        prompt = self._build_prompt(vqa_answer, question, image_description)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated_text = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        return self._parse_response(generated_text)

    def _build_prompt(
        self,
        vqa_answer: str,
        question: str,
        image_description: str,
    ) -> str:
        parts = [
            f"### System:\n{self.SYSTEM_PROMPT}\n",
            f"### Clinical Question:\n{question}\n",
        ]
        if image_description:
            parts.append(f"### Image Description:\n{image_description}\n")
        parts.append(f"### VQA Model Analysis:\n{vqa_answer}\n")
        parts.append(
            "### Instructions:\n"
            "Based on the above, provide:\n"
            "1. **Clinical Reasoning**: Explain the findings.\n"
            "2. **Differential Diagnosis**: List possible conditions.\n"
            "3. **Recommendations**: Suggest follow-up actions.\n\n"
            "### Response:\n"
        )
        return "\n".join(parts)

    def _parse_response(self, text: str) -> MedraResponse:
        clinical_reasoning = ""
        differential: list[str] = []
        recommendations: list[str] = []

        sections = text.split("\n")
        current_section = ""
        for line in sections:
            lower = line.strip().lower()
            if "clinical reasoning" in lower or "findings" in lower:
                current_section = "reasoning"
                continue
            elif "differential" in lower:
                current_section = "differential"
                continue
            elif "recommendation" in lower:
                current_section = "recommendations"
                continue

            stripped = line.strip()
            if not stripped:
                continue

            if current_section == "reasoning":
                clinical_reasoning += stripped + " "
            elif current_section == "differential":
                cleaned = stripped.lstrip("- •0123456789.)")
                if cleaned:
                    differential.append(cleaned.strip())
            elif current_section == "recommendations":
                cleaned = stripped.lstrip("- •0123456789.)")
                if cleaned:
                    recommendations.append(cleaned.strip())

        if not clinical_reasoning:
            clinical_reasoning = text

        return MedraResponse(
            clinical_reasoning=clinical_reasoning.strip(),
            differential_diagnosis=differential,
            recommendations=recommendations,
            confidence=0.0,
            raw_text=text,
            metadata={"model": self.HF_MODEL_ID},
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
