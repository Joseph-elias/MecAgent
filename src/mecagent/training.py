from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class TrainingConfig:
    label_smoothing: float = 0.1
    scheduled_sampling_start: float = 0.0
    scheduled_sampling_end: float = 0.25
    curriculum_warmup_epochs: int = 5


def cadquery_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int, label_smoothing: float = 0.1) -> torch.Tensor:
    return F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        targets.view(-1),
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )


def scheduled_sampling_ratio(epoch: int, total_epochs: int, cfg: TrainingConfig) -> float:
    progress = min(max(epoch / max(total_epochs, 1), 0.0), 1.0)
    return cfg.scheduled_sampling_start + progress * (cfg.scheduled_sampling_end - cfg.scheduled_sampling_start)


def curriculum_max_ops(epoch: int, cfg: TrainingConfig, min_ops: int = 1, max_ops: int = 12) -> int:
    if cfg.curriculum_warmup_epochs <= 0:
        return max_ops
    alpha = min(epoch / cfg.curriculum_warmup_epochs, 1.0)
    return int(min_ops + alpha * (max_ops - min_ops))
