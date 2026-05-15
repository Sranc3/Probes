"""Model + loss utilities with the corrected DPO/GRPO formulation.

Key differences from Plan_gpt55:

1. ``completion_token_logprobs`` returns the *per-generated-token* logprobs as a
   tensor (shape ``(B, T_gen)``) plus a mask. This lets us:
   - sum to get *standard* DPO sequence log-likelihood (not length-averaged);
   - take *per-token* importance ratios for PPO/GRPO so clipping actually engages.

2. ``locate_answer_span`` finds the canonical answer span (gold alias) inside
   the completion at the token level. The DPO loss can then be applied to only
   that span, concentrating gradient on the factual answer rather than the
   verbose surrounding text.

3. ``per_token_dpo_logits`` computes the standard DPO logit using *summed*
   log-likelihood over the chosen answer span (or full completion if no span
   given). This is the formulation reviewers expect when seeing ``beta=0.1``.

4. ``ppo_loss_per_token`` applies clipping at the token level and returns
   averaged loss over generated tokens, matching the original PPO formulation.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Sequence

import torch
import torch.nn.functional as F


def build_prompt(tokenizer: Any, question: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n\nQuestion: {question}\nAnswer:"


def encode_pair(tokenizer: Any, prompt: str, completion: str, device: str) -> dict[str, torch.Tensor]:
    """Tokenize prompt + completion, returning prompt length to slice generated tokens."""
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    full_ids = tokenizer(prompt + completion, return_tensors="pt", add_special_tokens=False)
    prompt_len = int(prompt_ids["input_ids"].shape[-1])
    input_ids = full_ids["input_ids"].to(device)
    attention_mask = full_ids.get("attention_mask", torch.ones_like(input_ids)).to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_len": prompt_len,
        "completion_len": int(input_ids.shape[-1]) - prompt_len,
    }


def completion_token_logprobs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-token logprobs over the completion tokens and the gen mask.

    Shapes:
        token_logprobs: ``(B, T_full - 1)`` per-position predicted-label logprob
        completion_mask: same shape, 1 where position is a completion (generated) token
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    completion_mask = torch.zeros_like(token_logprobs)
    # Position i in token_logprobs is the prediction of input_ids[:, i+1].
    # Generated tokens occupy input positions [prompt_len, T_full-1], i.e.
    # logit-positions [prompt_len-1, T_full-2].
    completion_mask[:, max(prompt_len - 1, 0) :] = 1.0
    completion_mask = completion_mask * attention_mask[:, 1:].float()
    return token_logprobs, completion_mask


def reference_token_logprobs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same as completion_token_logprobs but with PEFT adapter disabled (frozen base)."""
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            with torch.no_grad():
                lp, mask = completion_token_logprobs(model, input_ids, attention_mask, prompt_len)
    else:
        with torch.no_grad():
            lp, mask = completion_token_logprobs(model, input_ids, attention_mask, prompt_len)
    return lp.detach(), mask.detach()


def locate_answer_span(
    tokenizer: Any,
    prompt: str,
    completion: str,
    canonical_answer_text: str | None,
    completion_len: int,
    prompt_len: int,
) -> tuple[int, int]:
    """Return ``(start, end)`` indices (in the *logit position* space) covering the
    canonical answer span inside the completion.

    The returned indices are absolute positions within ``token_logprobs`` of
    ``completion_token_logprobs``, suitable for slicing
    ``token_logprobs[:, start:end]``.

    If the canonical answer cannot be located, returns the full completion span
    so the caller falls back to standard DPO over the full completion.
    """
    full_start = max(prompt_len - 1, 0)
    full_end = full_start + completion_len
    if not canonical_answer_text:
        return full_start, full_end
    # Try matching span by tokenizing with leading-space variant first (LLM
    # tokenizers usually merge a leading space with the next word).
    candidates = []
    for variant in (" " + canonical_answer_text.strip(), canonical_answer_text.strip()):
        ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if ids:
            candidates.append(ids)
    if not candidates:
        return full_start, full_end
    # Search inside completion ids for one of the variants.
    full_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
    completion_ids = full_ids[prompt_len:]
    for ids in candidates:
        L = len(ids)
        for offset in range(len(completion_ids) - L + 1):
            if completion_ids[offset : offset + L] == ids:
                start = full_start + offset
                end = start + L
                return start, end
    return full_start, full_end


def sequence_logprob_sum(
    token_logprobs: torch.Tensor,
    mask: torch.Tensor,
    span: tuple[int, int] | None = None,
    use_mean: bool = False,
) -> torch.Tensor:
    """Sum per-token logprobs inside the span (or the entire generated region).

    If ``use_mean=True`` the result is divided by the (masked) token count -
    this exactly reproduces Plan_gpt55's diluted DPO loss and is provided so
    we can run an ablation that isolates the effect of the loss formulation.
    """
    if span is None:
        sliced = token_logprobs
        sliced_mask = mask
    else:
        start, end = span
        end = min(end, token_logprobs.shape[-1])
        start = max(min(start, end), 0)
        if end <= start:
            sliced = token_logprobs
            sliced_mask = mask
        else:
            sliced = token_logprobs[:, start:end]
            sliced_mask = mask[:, start:end]
    summed = (sliced * sliced_mask).sum(dim=-1)
    if use_mean:
        return summed / sliced_mask.sum(dim=-1).clamp_min(1.0)
    return summed


def dpo_pair_logits(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    pi_margin = chosen_logp - rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp
    return beta * (pi_margin - ref_margin)


def dpo_loss(logits: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    return -F.logsigmoid(logits).mean() * float(weight)


def per_token_ppo_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Standard PPO clipped surrogate loss applied per token.

    Args:
        new_logp: per-token logprobs from current policy ``(B, T)``.
        old_logp: detached per-token logprobs at sampling time ``(B, T)``.
        advantages: per-sequence advantages broadcast to ``(B, 1)``.
        mask: completion mask ``(B, T)``.
        clip_eps: PPO clip epsilon (e.g. 0.2).
    """
    advantages = advantages.detach().to(new_logp.dtype)
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)
    ratios = torch.exp(new_logp - old_logp)
    clipped = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps)
    unclipped_obj = ratios * advantages
    clipped_obj = clipped * advantages
    surrogate = -torch.minimum(unclipped_obj, clipped_obj)
    masked = surrogate * mask
    denom = mask.sum().clamp_min(1.0)
    loss = masked.sum() / denom

    with torch.no_grad():
        clip_frac = ((ratios > 1.0 + clip_eps) | (ratios < 1.0 - clip_eps)).float() * mask
        diag = {
            "ppo_mean_ratio": float((ratios * mask).sum().item() / max(1.0, mask.sum().item())),
            "ppo_max_ratio": float(ratios.masked_fill(mask == 0, float("-inf")).max().item()),
            "ppo_clip_frac": float(clip_frac.sum().item() / max(1.0, mask.sum().item())),
        }
    return loss, diag


def per_token_kl(
    new_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Approximate KL via squared logprob diff per token (matches Plan_gpt55 style
    so beta_kl scale is comparable, but applied at token granularity)."""
    diff = (new_logp - ref_logp) * mask
    denom = mask.sum().clamp_min(1.0)
    return (diff * diff).sum() / denom


def stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    return ((sum((x - mu) ** 2 for x in values) / (len(values) - 1)) ** 0.5)
