"""
MedXplain-Simple — Evaluation metrics: BLEU, METEOR, CIDEr, BERTScore.

Provides a unified interface for computing NLG metrics on VQA predictions
against reference answers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Container for evaluation metric scores."""

    bleu_1: float = 0.0
    bleu_4: float = 0.0
    meteor: float = 0.0
    cider: float = 0.0
    rouge_l: float = 0.0
    bert_score_f1: float = 0.0
    accuracy: float = 0.0
    clinical_f1: float = 0.0
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        return {
            "bleu_1": self.bleu_1,
            "bleu_4": self.bleu_4,
            "meteor": self.meteor,
            "cider": self.cider,
            "rouge_l": self.rouge_l,
            "bert_score_f1": self.bert_score_f1,
            "accuracy": self.accuracy,
            "clinical_f1": self.clinical_f1,
        }


class VQAMetrics:
    """Unified metric computation for medical VQA evaluation."""

    def __init__(self, use_bertscore: bool = True):
        self.use_bertscore = use_bertscore
        self._nltk_ready = False

    def _ensure_nltk(self) -> None:
        if self._nltk_ready:
            return
        import nltk

        for resource in ("punkt", "punkt_tab", "wordnet"):
            try:
                nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
        self._nltk_ready = True

    def compute_bleu(
        self,
        predictions: list[str],
        references: list[str],
    ) -> tuple[float, float]:
        """Compute BLEU-1 and BLEU-4 scores."""
        import sacrebleu

        refs = [[r] for r in references]

        bleu_1 = sacrebleu.corpus_bleu(
            predictions,
            list(zip(*refs)),
            max_ngram_order=1,
        ).score / 100.0

        bleu_4 = sacrebleu.corpus_bleu(
            predictions,
            list(zip(*refs)),
            max_ngram_order=4,
        ).score / 100.0

        return bleu_1, bleu_4

    def compute_meteor(
        self,
        predictions: list[str],
        references: list[str],
    ) -> float:
        """Compute METEOR score."""
        self._ensure_nltk()
        from nltk.translate.meteor_score import meteor_score as nltk_meteor

        scores = []
        for pred, ref in zip(predictions, references):
            score = nltk_meteor([ref.split()], pred.split())
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    def compute_rouge_l(
        self,
        predictions: list[str],
        references: list[str],
    ) -> float:
        """Compute ROUGE-L F1 score."""
        from rouge_score.rouge_scorer import RougeScorer

        scorer = RougeScorer(["rougeL"], use_stemmer=True)
        scores = []
        for pred, ref in zip(predictions, references):
            result = scorer.score(ref, pred)
            scores.append(result["rougeL"].fmeasure)

        return sum(scores) / len(scores) if scores else 0.0

    def compute_cider(
        self,
        predictions: list[str],
        references: list[str],
    ) -> float:
        """Compute CIDEr score (simplified TF-IDF based)."""
        from collections import Counter
        import math

        def ngrams(text: str, n: int) -> list[str]:
            tokens = text.lower().split()
            return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

        n_val = 4
        doc_freq: Counter = Counter()
        num_docs = len(references)

        for ref in references:
            seen = set()
            for n in range(1, n_val + 1):
                for ng in ngrams(ref, n):
                    if ng not in seen:
                        doc_freq[ng] += 1
                        seen.add(ng)

        def tfidf_vec(text: str) -> dict[str, float]:
            vec: dict[str, float] = {}
            all_ng: list[str] = []
            for n in range(1, n_val + 1):
                all_ng.extend(ngrams(text, n))
            tf = Counter(all_ng)
            for ng, count in tf.items():
                idf = math.log(max(1.0, num_docs / (1.0 + doc_freq.get(ng, 0))))
                vec[ng] = count * idf
            return vec

        def cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
            common = set(a) & set(b)
            dot = sum(a[k] * b[k] for k in common)
            norm_a = math.sqrt(sum(v * v for v in a.values())) or 1.0
            norm_b = math.sqrt(sum(v * v for v in b.values())) or 1.0
            return dot / (norm_a * norm_b)

        scores = []
        for pred, ref in zip(predictions, references):
            vec_p = tfidf_vec(pred)
            vec_r = tfidf_vec(ref)
            scores.append(cosine_sim(vec_p, vec_r))

        return sum(scores) / len(scores) * 10.0 if scores else 0.0

    def compute_bertscore(
        self,
        predictions: list[str],
        references: list[str],
    ) -> float:
        """Compute BERTScore F1."""
        if not self.use_bertscore:
            return 0.0
        try:
            from bert_score import score as bert_score_fn

            _, _, f1 = bert_score_fn(
                predictions,
                references,
                lang="en",
                verbose=False,
            )
            return float(f1.mean())
        except ImportError:
            logger.warning("bert-score not installed. Skipping BERTScore.")
            return 0.0

    def compute_accuracy(
        self,
        predictions: list[str],
        references: list[str],
    ) -> float:
        """Compute exact-match and soft VQA accuracy."""
        correct = 0
        for pred, ref in zip(predictions, references):
            pred_clean = pred.strip().lower()
            ref_clean = ref.strip().lower()
            if pred_clean == ref_clean:
                correct += 1
            elif ref_clean in pred_clean or pred_clean in ref_clean:
                correct += 0.5
        return correct / len(predictions) if predictions else 0.0

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
        question_types: Optional[list[str]] = None,
    ) -> MetricResult:
        """Run all metrics and return a comprehensive result."""
        bleu_1, bleu_4 = self.compute_bleu(predictions, references)
        meteor = self.compute_meteor(predictions, references)
        rouge_l = self.compute_rouge_l(predictions, references)
        cider = self.compute_cider(predictions, references)
        bertscore = self.compute_bertscore(predictions, references)
        accuracy = self.compute_accuracy(predictions, references)

        per_category: dict[str, dict[str, float]] = {}
        if question_types:
            categories: dict[str, tuple[list[str], list[str]]] = {}
            for pred, ref, qtype in zip(predictions, references, question_types):
                if qtype not in categories:
                    categories[qtype] = ([], [])
                categories[qtype][0].append(pred)
                categories[qtype][1].append(ref)

            for cat, (preds, refs) in categories.items():
                per_category[cat] = {
                    "accuracy": self.compute_accuracy(preds, refs),
                    "count": len(preds),
                }

        return MetricResult(
            bleu_1=bleu_1,
            bleu_4=bleu_4,
            meteor=meteor,
            cider=cider,
            rouge_l=rouge_l,
            bert_score_f1=bertscore,
            accuracy=accuracy,
            per_category=per_category,
        )
