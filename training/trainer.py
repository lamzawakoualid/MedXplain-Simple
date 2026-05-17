"""
MedXplain-Simple — Core training loop, optimisation, and checkpoint management.

Implements a two-phase training strategy:
  Phase 1: Vision-language alignment (freeze LLM, train Q-Former / projection).
  Phase 2: VQA task fine-tuning (LoRA on the full pipeline).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    output_dir: str = "./checkpoints"
    num_epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    fp16: bool = True
    save_every_n_epochs: int = 1
    eval_every_n_steps: int = 500
    log_every_n_steps: int = 50
    seed: int = 42
    phase: int = 1  # 1 = alignment, 2 = VQA fine-tuning
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            k: v for k, v in self.__dict__.items() if not k.startswith("_")
        }


@dataclass
class TrainingState:
    """Mutable training state for checkpointing."""

    epoch: int = 0
    global_step: int = 0
    best_loss: float = float("inf")
    best_metric: float = 0.0
    history: list[dict[str, float]] = field(default_factory=list)


class Trainer:
    """Core training loop for MedXplain-Simple models.

    Supports:
    - Two-phase LoRA fine-tuning (alignment then VQA).
    - Mixed-precision training with gradient accumulation.
    - Periodic checkpointing and metric logging.
    - TensorBoard / W&B experiment tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = TrainingState()

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_lora()
        self._setup_optimiser()
        self._setup_scaler()
        self._setup_logger()

    def _setup_lora(self) -> None:
        if not self.config.use_lora:
            self.model.to(self.device)
            return

        try:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        except ImportError:
            logger.warning("PEFT not available; training all parameters.")

        self.model.to(self.device)

    def _setup_optimiser(self) -> None:
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimiser = AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        total_steps = len(self.train_loader) * self.config.num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimiser,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.01,
        )

    def _setup_scaler(self) -> None:
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.config.fp16 and self.device.type == "cuda")

    def _setup_logger(self) -> None:
        self.tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb_writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))
        except ImportError:
            logger.info("TensorBoard not available; skipping.")

    def train(self) -> TrainingState:
        """Run the full training loop."""
        logger.info(
            "Starting Phase %d training: %d epochs, %d steps/epoch",
            self.config.phase,
            self.config.num_epochs,
            len(self.train_loader),
        )

        torch.manual_seed(self.config.seed)
        self.model.train()

        for epoch in range(self.state.epoch, self.config.num_epochs):
            self.state.epoch = epoch
            epoch_loss = self._train_epoch(epoch)

            val_loss = None
            if self.val_loader is not None:
                val_loss = self._validate()

            self.state.history.append({
                "epoch": epoch,
                "train_loss": epoch_loss,
                "val_loss": val_loss,
                "lr": self.optimiser.param_groups[0]["lr"],
            })

            logger.info(
                "Epoch %d/%d — train_loss=%.4f, val_loss=%s",
                epoch + 1,
                self.config.num_epochs,
                epoch_loss,
                f"{val_loss:.4f}" if val_loss is not None else "N/A",
            )

            if epoch % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch}")

            check_loss = val_loss if val_loss is not None else epoch_loss
            if check_loss < self.state.best_loss:
                self.state.best_loss = check_loss
                self._save_checkpoint("best")

        self._save_checkpoint("final")
        self._save_history()

        if self.tb_writer:
            self.tb_writer.close()

        return self.state

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimiser.zero_grad()

        for step, batch in enumerate(self.train_loader):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.amp.autocast("cuda", enabled=self.config.fp16 and self.device.type == "cuda"):
                outputs = self.model(**self._prepare_model_inputs(batch))
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                loss = loss / self.config.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimiser)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.scaler.step(self.optimiser)
                self.scaler.update()
                self.scheduler.step()
                self.optimiser.zero_grad()
                self.state.global_step += 1

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            if step % self.config.log_every_n_steps == 0:
                avg = total_loss / num_batches
                logger.info(
                    "  Step %d/%d — loss=%.4f, lr=%.2e",
                    step,
                    len(self.train_loader),
                    avg,
                    self.optimiser.param_groups[0]["lr"],
                )
                if self.tb_writer:
                    self.tb_writer.add_scalar(
                        "train/loss", avg, self.state.global_step
                    )

        return total_loss / max(num_batches, 1)

    def _prepare_model_inputs(self, batch: dict[str, Any]) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        for key in ("input_ids", "attention_mask", "pixel_values", "labels",
                     "question_ids", "question_mask", "answer_ids", "image"):
            if key in batch:
                inputs[key] = batch[key]
        if "answer_ids" in batch and "labels" not in inputs:
            inputs["labels"] = batch["answer_ids"]
        return inputs

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with torch.amp.autocast("cuda", enabled=self.config.fp16 and self.device.type == "cuda"):
                outputs = self.model(**self._prepare_model_inputs(batch))
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

            total_loss += loss.item()
            num_batches += 1

        self.model.train()
        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, tag: str) -> None:
        ckpt_dir = self.output_dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(ckpt_dir))
        else:
            torch.save(self.model.state_dict(), ckpt_dir / "model.pt")

        if self.tokenizer and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(str(ckpt_dir))

        meta = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "best_loss": self.state.best_loss,
            "config": self.config.to_dict(),
        }
        with open(ckpt_dir / "training_state.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved checkpoint: %s", ckpt_dir)

    def _save_history(self) -> None:
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(self.state.history, f, indent=2)
