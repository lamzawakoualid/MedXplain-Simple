"""
MedXplain-Simple — PyTorch Dataset and DataLoader classes for training and inference.

Handles loading of medical VQA triplets (image, question, answer)
with support for multiple data formats and augmentation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class MedicalVQADataset(Dataset):
    """PyTorch Dataset for medical VQA triplets.

    Supports both JSON and CSV index formats with fields:
    image_path, question, answer, question_type, modality,
    anatomical_region, source_dataset.
    """

    DEFAULT_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    def __init__(
        self,
        index_path: str | Path,
        image_root: str | Path,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        tokenizer: Optional[Any] = None,
        max_question_length: int = 128,
        max_answer_length: int = 64,
    ):
        self.index_path = Path(index_path)
        self.image_root = Path(image_root)
        self.split = split
        self.transform = transform or self.DEFAULT_TRANSFORM
        self.tokenizer = tokenizer
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length

        self.data = self._load_index()
        logger.info(
            "Loaded %d samples for split '%s' from %s",
            len(self.data),
            self.split,
            self.index_path,
        )

    def _load_index(self) -> list[dict[str, Any]]:
        if self.index_path.suffix == ".json":
            with open(self.index_path) as f:
                all_data = json.load(f)
        elif self.index_path.suffix == ".csv":
            df = pd.read_csv(self.index_path)
            all_data = df.to_dict("records")
        else:
            raise ValueError(f"Unsupported index format: {self.index_path.suffix}")

        if "split" in (all_data[0] if all_data else {}):
            return [d for d in all_data if d.get("split") == self.split]
        return all_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.data[idx]
        image_path = self.image_root / record["image_path"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning("Failed to load image %s: %s. Using blank.", image_path, e)
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        image_tensor = self.transform(image)

        item: dict[str, Any] = {
            "image": image_tensor,
            "question": record["question"],
            "answer": record["answer"],
            "image_path": str(record["image_path"]),
        }

        for key in ("question_type", "modality", "anatomical_region", "source_dataset"):
            if key in record:
                item[key] = record[key]

        if self.tokenizer is not None:
            q_enc = self.tokenizer(
                record["question"],
                max_length=self.max_question_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            a_enc = self.tokenizer(
                record["answer"],
                max_length=self.max_answer_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            item["question_ids"] = q_enc["input_ids"].squeeze(0)
            item["question_mask"] = q_enc["attention_mask"].squeeze(0)
            item["answer_ids"] = a_enc["input_ids"].squeeze(0)
            item["answer_mask"] = a_enc["attention_mask"].squeeze(0)

        return item


def create_dataloaders(
    index_path: str | Path,
    image_root: str | Path,
    batch_size: int = 16,
    num_workers: int = 4,
    tokenizer: Optional[Any] = None,
) -> dict[str, DataLoader]:
    """Create train/val/test dataloaders from an index file."""
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = MedicalVQADataset(
            index_path=index_path,
            image_root=image_root,
            split=split,
            tokenizer=tokenizer,
        )
        if len(dataset) == 0:
            continue
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )
    return loaders
