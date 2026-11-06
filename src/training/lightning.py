"""Train and evaluate multimodal classifiers with Lightning.
MMModel handles losses, metrics, modality dropout and checkpoint loading for the training script.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from itertools import chain
from typing import Any

import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Metric

from src.data.collate import modality_available
from src.data.datasets import modality_prefix
from src.metrics import build_binary_metrics
from src.models.factory import build_fusion_model


class MMModel(L.LightningModule):
    """Handle training, evaluation and prediction for a multimodal binary classifier.

    If no model is supplied, build one from the fusion method, configuration and feature dimensions.
    The model accepts a batch dictionary and returns logits of shape ``[batch]`` or ``[batch, 1]``,
    either directly or under ``logits`` in a dictionary with optional auxiliary outputs.
    """

    def __init__(
        self, model: nn.Module | None = None, learning_rate: float = 3e-4,
        weight_decay: float = 1e-4, pos_weight: torch.Tensor | float | None = None,
        scheduler_patience: int | None = None, scheduler_monitor: str = "val_mcc",
        fusion_method: str = "concat", fusion_config: dict[str, Any] | None = None,
        feature_dims: dict[str, int] | None = None,
        modality_dropout_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.fusion_method = fusion_method.lower()
        self.fusion_config = dict(fusion_config or {})
        self.modality_dropout_config = dict(modality_dropout_config or {})
        # train.py passes dropout settings from train.modality_dropout via modality_dropout_config
        # Warn if they were left under model.modality_dropout, where they have no effect
        if "modality_dropout" in self.fusion_config:
            warnings.warn(
                "fusion_config carries a 'modality_dropout' key, which is the pre-refactor "
                "location and is IGNORED. Move it to the config's `train:` section (it "
                "reaches MMModel as modality_dropout_config); otherwise modality dropout "
                "stays disabled regardless of what that block says.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.feature_dims = dict(feature_dims or {})
        if model is None:
            self.model = build_fusion_model(
                self.fusion_method, self.fusion_config, self.feature_dims,
            )
        else:
            self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        self.scheduler_monitor = scheduler_monitor
        self.pos_weight = self._normalise_pos_weight(pos_weight)
        loss_pos_weight = (
            None if self.pos_weight is None else torch.tensor(self.pos_weight, dtype=torch.float32)
        )
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=loss_pos_weight)
        self.stage_metrics = nn.ModuleDict(
            {self._metric_key(stage): build_binary_metrics() for stage in ("train", "val", "test")}
        )
        self.save_hyperparameters(
            {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "scheduler_patience": self.scheduler_patience,
                "scheduler_monitor": self.scheduler_monitor,
                "fusion_method": self.fusion_method,
                "fusion_config": self.fusion_config,
                "feature_dims": self.feature_dims,
                "pos_weight": self.pos_weight,
            }
        )

        if self.fusion_method == "m3care":
            self._m3care_epoch_loss_sum: dict[str, dict[str, float]] = {"train": {}, "val": {}}
            self._m3care_epoch_sample_count: dict[str, dict[str, float]] = {"train": {}, "val": {}}
            self._m3care_epoch_total_samples: dict[str, int] = {"train": 0, "val": 0}
            # Send epoch averages directly to loggers so Lightning does not average them again
            self._m3care_epoch_reported_values: dict[str, dict[str, float]] = {
                "train": {}, "val": {},
            }

        self.float()

    def setup(self, stage: str) -> None:
        """Require float32 precision in both the trainer and model."""
        if self._trainer is not None and self.trainer.precision != "32-true":
            raise ValueError('MMModel requires Trainer(precision="32-true").')
        self._check_float32({})

    @staticmethod
    def _check_float32_tensors(value: Any, name: str) -> None:
        """Require float32 for floating tensors, including those in nested batches or outputs."""
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() and value.dtype != torch.float32:
                raise TypeError(f"{name} must use torch.float32, got {value.dtype}.")
        elif isinstance(value, Mapping):
            for key, item in value.items():
                MMModel._check_float32_tensors(item, f"{name}.{key}")
        elif isinstance(value, list | tuple):
            for index, item in enumerate(value):
                MMModel._check_float32_tensors(item, f"{name}[{index}]")

    def _check_float32(self, batch: Mapping[str, Any]) -> None:
        """Check that model parameters, buffers and batch tensors use float32 without autocast."""
        if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
            raise RuntimeError("MMModel requires float32 execution with autocast disabled.")
        for name, value in chain(self.named_parameters(), self.named_buffers()):
            self._check_float32_tensors(value, name)
        self._check_float32_tensors(batch, "batch")

    def _reset_m3care_epoch_accumulators(self, stage: str) -> None:
        """Clear M3Care loss totals and saved values for the next epoch."""
        if self.fusion_method != "m3care":
            return
        if stage not in self._m3care_epoch_loss_sum:
            return

        self._m3care_epoch_loss_sum[stage].clear()
        self._m3care_epoch_sample_count[stage].clear()
        self._m3care_epoch_total_samples[stage] = 0
        self._m3care_epoch_reported_values[stage].clear()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Rename the old missing-modality embedding key before loading a checkpoint."""
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping):
            return

        legacy_key = "model.fusion.missing_embeddings"
        current_key = "model.fusion.missing_modality_embeddings"
        if legacy_key in state_dict and current_key not in state_dict:
            state_dict[current_key] = state_dict.pop(legacy_key)

    def load_pretrained_weights(
        self, checkpoint_path: str, strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Load model weights from a Lightning checkpoint or a plain state dictionary.

        Leave optimiser, scheduler, training progress and metrics unchanged. With ``strict=True``,
        the weights must match the model exactly. Return lists of missing and unexpected keys.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, Mapping):
            state_dict = checkpoint
        else:
            raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")

        if not isinstance(state_dict, Mapping):
            raise TypeError(
                "Checkpoint state_dict must be a mapping of parameter names to tensors."
            )

        # A Lightning checkpoint includes wrapper state as well as model weights
        model_state_dict: dict[str, torch.Tensor] = {}
        has_model_prefix = any(
            isinstance(key, str) and key.startswith("model.") for key in state_dict
        )

        if has_model_prefix:
            for key, value in state_dict.items():
                if not isinstance(key, str):
                    continue
                if key.startswith("model."):
                    model_state_dict[key[len("model.") :]] = value
        else:
            model_state_dict = dict(state_dict)

        if not model_state_dict:
            raise ValueError(f"No model parameters were found in checkpoint: {checkpoint_path}")

        # Apply the checkpoint key rename after removing the model prefix
        legacy_key = "fusion.missing_embeddings"
        current_key = "fusion.missing_modality_embeddings"
        if legacy_key in model_state_dict and current_key not in model_state_dict:
            model_state_dict[current_key] = model_state_dict.pop(legacy_key)

        missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=strict)
        return list(missing_keys), list(unexpected_keys)

    def on_train_epoch_start(self) -> None:
        """Reset training loss totals and keep frozen encoders in their required state."""
        if self.fusion_method == "m3care":
            self._reset_m3care_epoch_accumulators("train")

        apply_frozen_state = getattr(self.model, "apply_frozen_modality_encoder_state", None)
        if callable(apply_frozen_state):
            apply_frozen_state()

    def on_validation_epoch_start(self) -> None:
        """Reset M3Care validation loss totals."""
        if self.fusion_method == "m3care":
            self._reset_m3care_epoch_accumulators("val")

    def _accumulate_subset_loss(
        self, stage: str, key: str, loss_value: torch.Tensor,
        sample_count: torch.Tensor | float | int,
    ) -> None:
        """Weight each batch loss by the number of samples used to compute it."""
        if stage not in self._m3care_epoch_loss_sum:
            return

        count = float(sample_count.item() if torch.is_tensor(sample_count) else sample_count)
        if count <= 0:
            return

        value = float(loss_value.detach().cpu().item())
        self._m3care_epoch_loss_sum[stage][key] = (
            self._m3care_epoch_loss_sum[stage].get(key, 0.0) + value * count
        )
        self._m3care_epoch_sample_count[stage][key] = (
            self._m3care_epoch_sample_count[stage].get(key, 0.0) + count
        )

    def _get_accumulated_subset_stats(self, stage: str, key: str) -> tuple[float, float] | None:
        """Return the sample-weighted mean loss and count, or None when no samples contributed."""
        count = self._m3care_epoch_sample_count[stage].get(key, 0.0)
        if count <= 0.0:
            return None

        loss_sum = self._m3care_epoch_loss_sum[stage].get(key, 0.0)
        return loss_sum / count, count

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        logits = self._extract_logits(self._model_output(batch))
        return logits.squeeze(-1)

    def training_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def predict_step(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Return predictions, available identifiers and labels, and supported auxiliary outputs."""
        model_output = self._model_output(batch, include_aux=True)
        logits = self._extract_logits(model_output).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).to(dtype=torch.int64)
        output = {"probability": probabilities.detach(), "prediction": predictions.detach()}
        identifier_column = "hadm_id" if "hadm_id" in batch else "ed_stay_id"
        if identifier_column in batch:
            output[identifier_column] = batch[identifier_column].detach()
        if "label" in batch:
            output["label"] = batch["label"].detach()
        if isinstance(model_output, Mapping):
            if "gate_weights" in model_output:
                output["gate_weights"] = model_output["gate_weights"].detach()
            if "router_output_norms" in model_output:
                output["router_output_norms"] = model_output["router_output_norms"].detach()
            if "aux_loss" in model_output:
                output["aux_loss"] = model_output["aux_loss"].detach()
            if "imputation_attentions" in model_output:
                output["imputation_attentions"] = {
                    key: [tensor.detach().cpu() for tensor in value]
                    for key, value in model_output["imputation_attentions"].items()
                }
            if "fusion_attentions" in model_output:
                fusion_attentions = model_output["fusion_attentions"]
                output["fusion_attentions"] = {
                    key: (value.detach().cpu() if torch.is_tensor(value) else value)
                    for key, value in fusion_attentions.items()
                }
        return output

    def configure_optimizers(self) -> Any:
        """Optimise trainable parameters and optionally reduce the learning rate on a plateau."""
        optimiser = torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.scheduler_patience is None:
            return optimiser

        scheduler = ReduceLROnPlateau(optimiser, mode="max", patience=self.scheduler_patience)
        return {
            "optimizer": optimiser,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.scheduler_monitor,
                "interval": "epoch",
            },
        }

    def _compute_modality_balanced_epoch_stats(
        self, stage: str, modalities: list[str], key_template: str,
    ) -> dict[str, float] | None:
        """Average loss and sample count equally across modalities that contributed samples."""
        modality_losses: list[float] = []
        modality_counts: list[float] = []

        for modality in modalities:
            key = key_template.format(modality=modality)
            stats = self._get_accumulated_subset_stats(stage, key)
            if stats is None:
                continue

            loss_value, count = stats
            modality_losses.append(loss_value)
            modality_counts.append(count)

        if not modality_losses:
            return None

        return {
            "loss": sum(modality_losses) / len(modality_losses),
            "mean_sample_count": sum(modality_counts) / len(modality_counts),
        }

    def _compute_modality_balanced_epoch_fraction(
        self, stage: str, modalities: list[str], key_template: str,
    ) -> float | None:
        """Average sample fractions equally across modalities with non-zero counts."""
        total_samples = self._m3care_epoch_total_samples[stage]
        if total_samples <= 0:
            return None

        fractions: list[float] = []
        for modality in modalities:
            key = key_template.format(modality=modality)
            count = self._m3care_epoch_sample_count[stage].get(key, 0.0)
            if count <= 0.0:
                continue
            fractions.append(count / total_samples)

        if not fractions:
            return None

        return sum(fractions) / len(fractions)

    def _log_m3care_epoch_value(self, name: str, value: float) -> None:
        """Store a completed M3Care epoch value for later logging."""
        if self.fusion_method != "m3care":
            return
        if not hasattr(self, "_m3care_epoch_reported_values"):
            return

        stage = "train" if name.startswith("train_") else "val"
        self._m3care_epoch_reported_values[stage][name] = float(value)

    def _log_m3care_epoch_flag(self, name: str, enabled: bool) -> None:
        """Record whether an M3Care option is enabled as 1 or 0."""
        self._log_m3care_epoch_value(name, 1.0 if enabled else 0.0)

    def _flush_m3care_epoch_values(self, stage: str) -> None:
        """Write completed M3Care epoch values directly to CSV and W&B loggers."""
        if self.fusion_method != "m3care":
            return

        trainer = getattr(self, "_trainer", None)
        if trainer is None or not hasattr(self, "_m3care_epoch_reported_values"):
            return

        values = self._m3care_epoch_reported_values.get(stage, {})
        if not values:
            return

        active_loggers = tuple(trainer.loggers)
        if not active_loggers:
            return

        epoch = int(trainer.current_epoch)
        global_step = int(trainer.global_step)

        # Include the epoch so train.py can merge training and validation rows in the CSV
        csv_payload = dict(values)
        csv_payload["epoch"] = epoch

        # Use Lightning's global step for both loggers
        for active_logger in active_loggers:
            if isinstance(active_logger, CSVLogger):
                active_logger.log_metrics(csv_payload, step=global_step)

        for active_logger in active_loggers:
            if isinstance(active_logger, WandbLogger):
                wandb_payload = dict(values)
                wandb_payload["epoch"] = epoch
                active_logger.log_metrics(wandb_payload, step=global_step)

    def _log_m3care_epoch_subset_losses(self, stage: str) -> None:
        if self.fusion_method != "m3care":
            return
        if not hasattr(self.model, "imputation"):
            return

        observed_stats = None
        missing_stats = None
        reconstruction_stats = None
        target_nce_stats = None

        observed_weighted_loss = 0.0
        missing_weighted_loss = 0.0
        reconstruction_weighted_loss = 0.0
        target_nce_weighted_loss = 0.0

        imputation = self.model.imputation
        contrastive_modalities = list(getattr(imputation, "contrastive_modalities", []))
        if not contrastive_modalities:
            return

        # Alignment loss for modalities with observed inputs
        if bool(getattr(imputation, "enable_observed_contrastive", False)):
            observed_stats = self._compute_modality_balanced_epoch_stats(
                stage, contrastive_modalities, "observed_{modality}_contrastive_loss",
            )
            observed_fraction = (
                self._compute_modality_balanced_epoch_fraction(
                    stage, contrastive_modalities, "observed_{modality}_contrastive_loss",
                )
                if observed_stats is not None
                else None
            )
            if observed_stats is not None:
                observed_loss = observed_stats["loss"]
                observed_count = observed_stats["mean_sample_count"]
                observed_weighted_loss = (
                    float(imputation.observed_contrastive_weight) * observed_loss
                )

                self._log_m3care_epoch_value(f"{stage}_observed_contrastive_loss", observed_loss)
                self._log_m3care_epoch_value(
                    f"{stage}_weighted_observed_contrastive_loss", observed_weighted_loss,
                )
                self._log_m3care_epoch_value(
                    f"{stage}_observed_contrastive_sample_count", observed_count,
                )
                if observed_fraction is not None:
                    self._log_m3care_epoch_value(
                        f"{stage}_observed_contrastive_sample_fraction", observed_fraction,
                    )

            if imputation.enable_observed_contrastive:
                self._log_m3care_epoch_flag(f"{stage}_observed_contrastive_enabled", True)

        # Alignment loss for modalities with missing inputs
        if bool(getattr(imputation, "enable_missing_contrastive", False)):
            missing_stats = self._compute_modality_balanced_epoch_stats(
                stage, contrastive_modalities, "missing_{modality}_contrastive_loss",
            )
            missing_fraction = (
                self._compute_modality_balanced_epoch_fraction(
                    stage, contrastive_modalities, "missing_{modality}_contrastive_loss",
                )
                if missing_stats is not None
                else None
            )
            if missing_stats is not None:
                missing_loss = missing_stats["loss"]
                missing_count = missing_stats["mean_sample_count"]
                missing_weighted_loss = float(imputation.missing_contrastive_weight) * missing_loss

                self._log_m3care_epoch_value(f"{stage}_missing_contrastive_loss", missing_loss)
                self._log_m3care_epoch_value(
                    f"{stage}_weighted_missing_contrastive_loss", missing_weighted_loss,
                )
                self._log_m3care_epoch_value(
                    f"{stage}_missing_contrastive_sample_count", missing_count,
                )
                if missing_fraction is not None:
                    self._log_m3care_epoch_value(
                        f"{stage}_missing_contrastive_sample_fraction", missing_fraction,
                    )

            if imputation.enable_missing_contrastive:
                self._log_m3care_epoch_flag(f"{stage}_missing_contrastive_enabled", True)

        # Reconstruction and alignment losses against learned targets
        if bool(getattr(imputation, "enable_learned_target_supervision", False)):
            reconstruction_stats = self._compute_modality_balanced_epoch_stats(
                stage, contrastive_modalities, "learned_target_{modality}_reconstruction_loss",
            )
            reconstruction_fraction = (
                self._compute_modality_balanced_epoch_fraction(
                    stage, contrastive_modalities, "learned_target_{modality}_reconstruction_loss",
                )
                if reconstruction_stats is not None
                else None
            )
            if reconstruction_stats is not None:
                reconstruction_loss = reconstruction_stats["loss"]
                reconstruction_count = reconstruction_stats["mean_sample_count"]
                reconstruction_weighted_loss = (
                    float(imputation.learned_target_loss_weight) * reconstruction_loss
                )

                self._log_m3care_epoch_value(
                    f"{stage}_learned_target_reconstruction_loss", reconstruction_loss,
                )
                self._log_m3care_epoch_value(
                    f"{stage}_weighted_learned_target_reconstruction_loss",
                    reconstruction_weighted_loss,
                )
                self._log_m3care_epoch_value(
                    f"{stage}_learned_target_reconstruction_sample_count", reconstruction_count,
                )
                if reconstruction_fraction is not None:
                    self._log_m3care_epoch_value(
                        f"{stage}_learned_target_reconstruction_sample_fraction",
                        reconstruction_fraction,
                    )

            target_nce_stats = self._compute_modality_balanced_epoch_stats(
                stage, contrastive_modalities, "learned_target_{modality}_contrastive_loss",
            )
            target_nce_fraction = (
                self._compute_modality_balanced_epoch_fraction(
                    stage, contrastive_modalities, "learned_target_{modality}_contrastive_loss",
                )
                if target_nce_stats is not None
                else None
            )
            if target_nce_stats is not None:
                target_nce_loss = target_nce_stats["loss"]
                target_nce_count = target_nce_stats["mean_sample_count"]
                target_nce_weighted_loss = (
                    float(imputation.learned_target_contrastive_weight) * target_nce_loss
                )

                self._log_m3care_epoch_value(
                    f"{stage}_learned_target_contrastive_loss", target_nce_loss,
                )
                self._log_m3care_epoch_value(
                    f"{stage}_weighted_learned_target_contrastive_loss", target_nce_weighted_loss,
                )
                self._log_m3care_epoch_value(
                    f"{stage}_learned_target_contrastive_sample_count", target_nce_count,
                )
                if target_nce_fraction is not None:
                    self._log_m3care_epoch_value(
                        f"{stage}_learned_target_contrastive_sample_fraction", target_nce_fraction,
                    )

            if imputation.enable_learned_target_supervision:
                self._log_m3care_epoch_flag(f"{stage}_learned_target_supervision_enabled", True)

        weighted_contrastive_loss = 0.0
        if observed_stats is not None:
            weighted_contrastive_loss += observed_weighted_loss
        if missing_stats is not None:
            weighted_contrastive_loss += missing_weighted_loss
        if imputation.enable_observed_contrastive or imputation.enable_missing_contrastive:
            self._log_m3care_epoch_value(
                f"{stage}_weighted_contrastive_loss", weighted_contrastive_loss,
            )

        weighted_learned_target_loss = 0.0
        if reconstruction_stats is not None:
            weighted_learned_target_loss += reconstruction_weighted_loss
        if target_nce_stats is not None:
            weighted_learned_target_loss += target_nce_weighted_loss
        if imputation.enable_learned_target_supervision:
            self._log_m3care_epoch_value(
                f"{stage}_weighted_learned_target_loss", weighted_learned_target_loss,
            )

    def _finalise_m3care_epoch_stats(self, stage: str) -> None:
        """Compute M3Care epoch averages from stored totals and send them to the loggers."""
        if self.fusion_method != "m3care":
            return

        if stage not in {"train", "val"}:
            raise ValueError(f"Unsupported M3Care epoch stage: {stage!r}")

        self._log_m3care_epoch_subset_losses(stage)
        self._flush_m3care_epoch_values(stage)

    def _shared_step(self, batch: Mapping[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Compute the stage loss and metrics, adding auxiliary loss only during training."""
        batch = self._conditional_training_modality_dropout(batch, stage)
        output = self._model_output(batch, include_aux=stage != "test")
        logits = self._extract_logits(output).squeeze(-1)
        labels = batch["label"].to(dtype=logits.dtype)
        loss = self.loss_fn(logits, labels)
        aux_loss = self._extract_aux_loss(output)
        aux_loss_components = (
            self._extract_aux_loss_components(output) if self.fusion_method == "m3care" else {}
        )
        if stage == "train" and aux_loss is not None:
            loss = loss + aux_loss
        batch_size = int(labels.numel())
        probabilities = torch.sigmoid(logits.detach())
        target = batch["label"].reshape(-1).to(dtype=torch.int64)
        metrics = self.stage_metrics[self._metric_key(stage)]
        metrics.update(probabilities.reshape(-1), target)

        self._log_value(
            f"{stage}_loss", loss, batch_size=batch_size, on_epoch=True, on_step=False,
            prog_bar=stage != "test",
        )

        if aux_loss is not None:
            self._log_value(
                f"{stage}_aux_loss", aux_loss, batch_size=batch_size, on_epoch=True, on_step=False,
                prog_bar=stage != "test",
            )

        # M3Care loss averages use sample counts because the eligible subset varies by batch
        if self.fusion_method == "m3care" and aux_loss_components:
            self._m3care_epoch_total_samples[stage] += batch_size

            imputation = getattr(self.model, "imputation", None)
            if imputation is None:
                self._log_metrics(metrics, stage, batch_size)
                return loss

            contrastive_modalities = list(getattr(imputation, "contrastive_modalities", []))

            observed_enabled = bool(
                aux_loss_components.get("observed_contrastive_enabled", loss.new_zeros(())).item()
            )
            missing_enabled = bool(
                aux_loss_components.get("missing_contrastive_enabled", loss.new_zeros(())).item()
            )
            learned_target_enabled = bool(
                aux_loss_components.get(
                    "learned_target_supervision_enabled", loss.new_zeros(())
                ).item()
            )

            # Skip disabled losses so they do not appear as zero-valued series in W&B
            if observed_enabled:
                for modality in contrastive_modalities:
                    loss_key = f"observed_{modality}_contrastive_loss"
                    count_key = f"observed_{modality}_contrastive_sample_count"

                    if loss_key in aux_loss_components and count_key in aux_loss_components:
                        count = aux_loss_components[count_key]
                        loss_value = aux_loss_components[loss_key]

                        if int(count.item()) >= imputation.min_contrastive_samples:
                            self._accumulate_subset_loss(stage, loss_key, loss_value, count)

            if missing_enabled:
                for modality in contrastive_modalities:
                    loss_key = f"missing_{modality}_contrastive_loss"
                    count_key = f"missing_{modality}_contrastive_sample_count"

                    if loss_key in aux_loss_components and count_key in aux_loss_components:
                        count = aux_loss_components[count_key]
                        loss_value = aux_loss_components[loss_key]

                        if int(count.item()) >= imputation.min_contrastive_samples:
                            self._accumulate_subset_loss(stage, loss_key, loss_value, count)

            if learned_target_enabled:
                for modality in contrastive_modalities:
                    loss_key = f"learned_target_{modality}_reconstruction_loss"
                    count_key = f"learned_target_{modality}_reconstruction_sample_count"

                    if loss_key in aux_loss_components and count_key in aux_loss_components:
                        count = aux_loss_components[count_key]
                        loss_value = aux_loss_components[loss_key]
                        if int(count.item()) > 0:
                            self._accumulate_subset_loss(stage, loss_key, loss_value, count)

            if learned_target_enabled:
                for modality in contrastive_modalities:
                    loss_key = f"learned_target_{modality}_contrastive_loss"
                    count_key = f"learned_target_{modality}_contrastive_sample_count"

                    if loss_key in aux_loss_components and count_key in aux_loss_components:
                        count = aux_loss_components[count_key]
                        loss_value = aux_loss_components[loss_key]

                        if int(count.item()) >= imputation.min_contrastive_samples:
                            self._accumulate_subset_loss(stage, loss_key, loss_value, count)

        self._log_metrics(metrics, stage, batch_size)
        return loss

    def _log_metrics(self, metrics: Mapping[str, Metric], stage: str, batch_size: int) -> None:
        metric_key = self._metric_key(stage)
        for name, metric in metrics.items():
            self._log_value(
                f"{stage}_{name}", metric, batch_size=batch_size, on_epoch=True, on_step=False,
                prog_bar=name in {"auroc", "mcc"} and stage != "test",
                metric_attribute=f"stage_metrics.{metric_key}.{name}",
            )

    def _log_value(self, name: str, value: Any, **kwargs: Any) -> None:
        if getattr(self, "_trainer", None) is None:
            return

        self.log(name, value, sync_dist=True, **kwargs)

    def _model_output(
        self, batch: Mapping[str, torch.Tensor], include_aux: bool = False,
    ) -> torch.Tensor | Mapping[str, torch.Tensor]:
        self._check_float32(batch)
        if include_aux and bool(getattr(self.model, "supports_aux_output", False)):
            output = self.model(batch, return_aux=True)
        else:
            output = self.model(batch)
        self._check_float32_tensors(output, "output")
        return output

    def _conditional_training_modality_dropout(
        self, batch: Mapping[str, torch.Tensor], stage: str,
    ) -> Mapping[str, torch.Tensor]:
        """Hide selected available modalities during training without changing the original batch."""
        dropout_cfg = self.modality_dropout_config
        if (
            stage != "train"
            or not self.training
            or not self.model.training
            or not isinstance(dropout_cfg, Mapping)
            or not bool(dropout_cfg.get("enabled", False))
        ):
            return batch

        assert stage == "train", "Modality dropout must only run during training."
        edited: dict[str, torch.Tensor] | None = None
        modalities = list(dropout_cfg.get("modalities", ("ecg", "cxr")))
        if "notes_prob" in dropout_cfg and "notes" not in modalities:
            modalities.append("notes")
        for modality in modalities:
            prob = float(dropout_cfg.get(f"{modality}_prob", 0.0))
            batch_modality = self._resolve_dropout_modality(batch, modality)
            if prob <= 0.0:
                continue
            if prob > 1.0:
                raise ValueError(f"modality_dropout.{modality}_prob must be <= 1.0.")

            available = self._modality_available(batch, batch_modality)
            drop_mask = self._class_balanced_dropout_mask(batch, available, prob)
            if not bool(drop_mask.any().item()):
                continue

            if edited is None:
                edited = dict(batch)
            self._drop_modality_for_mask(edited, batch_modality, drop_mask)

        if edited is None:
            return batch
        return edited

    @staticmethod
    def _resolve_dropout_modality(batch: Mapping[str, torch.Tensor], modality: str) -> str:
        """Use notes_prev for notes dropout when the batch contains previous-admission notes."""
        if modality == "notes" and "notes_prev_feats" in batch:
            return "notes_prev"
        return modality

    @staticmethod
    def _class_balanced_dropout_mask(
        batch: Mapping[str, torch.Tensor], available: torch.Tensor, prob: float,
    ) -> torch.Tensor:
        """Choose available samples to drop separately within each label class."""
        labels = batch["label"].to(device=available.device).reshape(-1)
        drop_mask = torch.zeros_like(available, dtype=torch.bool)

        for class_mask in (labels.lt(0.5), labels.ge(0.5)):
            candidates = (available & class_mask).nonzero(as_tuple=False).flatten()
            count = MMModel._stochastic_dropout_count(
                int(candidates.numel()), prob, available.device,
            )
            if count <= 0:
                continue

            selected = torch.randperm(int(candidates.numel()), device=available.device)[:count]
            drop_mask[candidates[selected]] = True

        return drop_mask

    @staticmethod
    def _stochastic_dropout_count(candidate_count: int, prob: float, device: torch.device) -> int:
        """Round the expected dropout count up with probability equal to its fractional part."""
        expected = prob * candidate_count
        count = int(expected)
        fractional = expected - count
        if fractional > 0.0 and bool(torch.rand((), device=device).lt(fractional).item()):
            count += 1
        return min(count, candidate_count)

    @staticmethod
    def _drop_modality_for_mask(
        batch: dict[str, torch.Tensor], modality: str, drop_mask: torch.Tensor,
    ) -> None:
        prefix = modality_prefix(modality)
        for suffix in ("missing", "lengths", "feats", "time"):
            key = f"{prefix}_{suffix}"
            if key not in batch:
                continue
            batch[key] = batch[key].clone()

        missing_key = f"{prefix}_missing"
        lengths_key = f"{prefix}_lengths"
        feats_key = f"{prefix}_feats"
        time_key = f"{prefix}_time"
        if missing_key in batch:
            batch[missing_key][drop_mask] = 1
        if lengths_key in batch:
            batch[lengths_key][drop_mask] = 0
        if feats_key in batch:
            batch[feats_key][drop_mask] = 0
        if time_key in batch:
            batch[time_key][drop_mask] = 0

    @staticmethod
    def _modality_available(batch: Mapping[str, torch.Tensor], modality: str) -> torch.Tensor:
        return modality_available(batch, modality)

    @staticmethod
    def _extract_logits(output: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(output, Mapping):
            try:
                return output["logits"]
            except KeyError as exc:
                raise ValueError("Model output dict must include 'logits'.") from exc
        return output

    @staticmethod
    def _extract_aux_loss(output: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor | None:
        if not isinstance(output, Mapping):
            return None
        return output.get("aux_loss")

    @staticmethod
    def _extract_aux_loss_components(
        output: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        if not isinstance(output, Mapping):
            return {}
        components = output.get("aux_loss_components")
        if not isinstance(components, Mapping):
            return {}
        return components

    @staticmethod
    def _metric_key(stage: str) -> str:
        return f"{stage}_metrics"

    @staticmethod
    def _normalise_pos_weight(pos_weight: torch.Tensor | float | None) -> float | None:
        if pos_weight is None:
            return None
        if isinstance(pos_weight, (int, float)):
            return float(pos_weight)
        return float(pos_weight.detach().cpu().item())
