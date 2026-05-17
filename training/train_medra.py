"""
MedXplain-Simple — Medra-4B LoRA fine-tuning for clinical domain adaptation.

Fine-tunes Medra-4B with LoRA adapters on clinical QA pairs
using bitsandbytes 4-bit quantisation to reduce VRAM requirements.

Usage:
    python -m training.train_medra --data-index data/medra_train.json --output-dir checkpoints/medra
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Medra-4B LoRA Fine-Tuning")

    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--use-4bit", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="./checkpoints/medra")
    parser.add_argument("--data-index", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
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
    """Load Medra-4B with 4-bit quantisation and LoRA."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    model_id = "Henrychur/Medra-4B"

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"trust_remote_code": True}

    if config.get("use_4bit", True) and torch.cuda.is_available():
        logger.info("Enabling 4-bit quantisation...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )

    logger.info("Loading Medra-4B model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    if config.get("use_4bit", True) and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


class MedraTextDataset(torch.utils.data.Dataset):
    """Simple text dataset for Medra-4B fine-tuning."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        question = item.get("question", "")
        answer = item.get("answer", "")
        context = item.get("context", "")

        prompt = f"### Question:\n{question}\n"
        if context:
            prompt += f"### Context:\n{context}\n"
        prompt += f"### Answer:\n{answer}"

        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def main() -> None:
    args = parse_args()
    config = load_config(args)

    torch.manual_seed(config.get("seed", 42))

    model, tokenizer = setup_model(config)

    train_dataset = MedraTextDataset(
        config["data_index"],
        tokenizer,
        max_length=config.get("max_length", 1024),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 2),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    from training.trainer import Trainer, TrainingConfig

    training_config = TrainingConfig(
        output_dir=config.get("output_dir", "./checkpoints/medra"),
        num_epochs=config.get("epochs", 3),
        learning_rate=config.get("lr", 1e-4),
        batch_size=config.get("batch_size", 2),
        gradient_accumulation_steps=config.get("grad_accum", 16),
        fp16=torch.cuda.is_available(),
        use_lora=False,  # Already applied above
        seed=config.get("seed", 42),
    )

    trainer = Trainer(
        model=model,
        config=training_config,
        train_loader=train_loader,
        tokenizer=tokenizer,
    )

    state = trainer.train()
    logger.info("Medra-4B training complete. Best loss: %.4f", state.best_loss)

    output_dir = Path(config["output_dir"])
    with open(output_dir / "final_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)


if __name__ == "__main__":
    main()
