"""
MedXplain-Simple — BLIP-2 LoRA fine-tuning on medical VQA datasets.

Two-phase training strategy:
  Phase 1: Vision-language alignment — freeze LLM backbone, train Q-Former.
  Phase 2: VQA task fine-tuning — LoRA adapters on the full model.

Usage:
    python -m training.train_blip2 --config configs/blip2_train.yaml
    python -m training.train_blip2 --phase 1 --epochs 5 --lr 1e-4
    python -m training.train_blip2 --phase 2 --epochs 10 --lr 2e-5 --lora-r 16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BLIP-2 LoRA Fine-Tuning")

    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="./checkpoints/blip2")
    parser.add_argument("--data-index", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    config = vars(args).copy()
    if args.config:
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)
        config.update(yaml_cfg)
    return config


def setup_model(config: dict):
    """Load BLIP-2 with LoRA configuration."""
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    from peft import LoraConfig, get_peft_model, TaskType

    logger.info("Loading BLIP-2 model and processor...")
    model_id = "Salesforce/blip2-opt-2.7b"
    processor = Blip2Processor.from_pretrained(model_id)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    phase = config.get("phase", 1)
    if phase == 1:
        logger.info("Phase 1: Freezing LLM, training vision + Q-Former")
        for param in model.language_model.parameters():
            param.requires_grad = False
        for param in model.vision_model.parameters():
            param.requires_grad = True
        for param in model.qformer.parameters():
            param.requires_grad = True
    else:
        logger.info("Phase 2: Applying LoRA to full model")
        lora_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, processor


def setup_data(config: dict, processor):
    """Create data loaders for training."""
    from data_ingestion.dataset import MedicalVQADataset
    from torch.utils.data import DataLoader

    train_dataset = MedicalVQADataset(
        index_path=config["data_index"],
        image_root=config["image_root"],
        split="train",
        tokenizer=processor.tokenizer,
    )

    val_dataset = MedicalVQADataset(
        index_path=config["data_index"],
        image_root=config["image_root"],
        split="val",
        tokenizer=processor.tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    ) if len(val_dataset) > 0 else None

    return train_loader, val_loader


def main() -> None:
    args = parse_args()
    config = load_config(args)

    torch.manual_seed(config.get("seed", 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.get("seed", 42))

    model, processor = setup_model(config)

    train_loader, val_loader = setup_data(config, processor)

    from training.trainer import Trainer, TrainingConfig

    training_config = TrainingConfig(
        output_dir=config.get("output_dir", "./checkpoints/blip2"),
        num_epochs=config.get("epochs", 5),
        learning_rate=config.get("lr", 2e-5),
        batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("grad_accum", 8),
        fp16=torch.cuda.is_available(),
        phase=config.get("phase", 1),
        use_lora=False,  # Already applied in setup_model for phase 2
        lora_r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        seed=config.get("seed", 42),
    )

    trainer = Trainer(
        model=model,
        config=training_config,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=processor.tokenizer,
    )

    state = trainer.train()
    logger.info(
        "Training complete. Best loss: %.4f, Total steps: %d",
        state.best_loss,
        state.global_step,
    )

    with open(Path(config["output_dir"]) / "final_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)


if __name__ == "__main__":
    main()
