from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(slots=True)
class DecodeState:
    tokens: torch.Tensor
    score: float
    finished: bool


def _apply_token_constraints(logits: torch.Tensor, forbidden_ids: list[int] | None = None) -> torch.Tensor:
    if not forbidden_ids:
        return logits
    constrained = logits.clone()
    constrained[..., forbidden_ids] = -torch.inf
    return constrained


def decode_top_k(
    model: nn.Module,
    image: torch.Tensor,
    bos_token_id: int,
    eos_token_id: int,
    max_length: int,
    temperature: float = 0.8,
    top_k: int = 20,
    forbidden_ids: list[int] | None = None,
) -> torch.Tensor:
    model.eval()
    seq = torch.tensor([[bos_token_id]], device=image.device)

    for _ in range(max_length - 1):
        logits = model(image, seq)[:, -1, :] / max(temperature, 1e-5)
        logits = _apply_token_constraints(logits, forbidden_ids)
        values, indices = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
        probs = F.softmax(values, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        next_token = indices.gather(1, sampled)
        seq = torch.cat([seq, next_token], dim=1)
        if next_token.item() == eos_token_id:
            break

    return seq.squeeze(0)


def decode_beam_search(
    model: nn.Module,
    image: torch.Tensor,
    bos_token_id: int,
    eos_token_id: int,
    max_length: int,
    beam_size: int = 4,
    forbidden_ids: list[int] | None = None,
) -> torch.Tensor:
    model.eval()
    beams = [DecodeState(tokens=torch.tensor([[bos_token_id]], device=image.device), score=0.0, finished=False)]

    for _ in range(max_length - 1):
        candidates: list[DecodeState] = []
        for beam in beams:
            if beam.finished:
                candidates.append(beam)
                continue

            logits = model(image, beam.tokens)[:, -1, :]
            logits = _apply_token_constraints(logits, forbidden_ids)
            log_probs = F.log_softmax(logits, dim=-1)
            values, indices = torch.topk(log_probs, k=beam_size, dim=-1)

            for k in range(values.shape[-1]):
                token = indices[:, k : k + 1]
                score = beam.score + values[0, k].item()
                tokens = torch.cat([beam.tokens, token], dim=1)
                done = token.item() == eos_token_id
                candidates.append(DecodeState(tokens=tokens, score=score, finished=done))

        beams = sorted(candidates, key=lambda x: x.score / (x.tokens.shape[1] ** 0.7), reverse=True)[:beam_size]
        if all(b.finished for b in beams):
            break

    return beams[0].tokens.squeeze(0)
