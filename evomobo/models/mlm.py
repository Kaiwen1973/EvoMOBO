from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch import LongTensor
from torch.nn import functional as F

from evomobo import transforms as gfp_transforms
from evomobo.models.shared_elements import check_early_stopping
from evomobo.wandb_utils import wandb_log
from evomobo.utils import str_to_tokens

logger = logging.getLogger(__name__)


class _Compose:

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def _get_device(module: torch.nn.Module) -> torch.device:
    if hasattr(module, "device"):
        try:
            return torch.device(module.device)
        except Exception:
            pass
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def sample_tokens(
    base_tokens,
    logit_batch,
    enc_tokenizer,
    replacement: bool = False,
    temp: float = 1.0,
    sample: bool = True,
):
    logits = logit_batch if isinstance(logit_batch, torch.Tensor) else torch.as_tensor(logit_batch)
    logits = logits.to(dtype=torch.float32)

    if base_tokens is not None and not isinstance(base_tokens, torch.Tensor):
        base_tokens = torch.as_tensor(base_tokens, dtype=torch.long, device=logits.device)
    elif isinstance(base_tokens, torch.Tensor):
        base_tokens = base_tokens.to(device=logits.device, dtype=torch.long)

    t = float(temp)
    if t <= 0:
        raise ValueError(f"temp must be > 0, got {temp}")
    logits = logits / t

    logits = logits.clone()
    special_ids = torch.as_tensor(enc_tokenizer.special_idxs, dtype=torch.long, device=logits.device)
    if special_ids.numel() > 0:
        logits.index_fill_(-1, special_ids, -1e9)

    if (not replacement) and (base_tokens is not None):
        bt = base_tokens
        if bt.dim() == logits.dim() - 1:
            bt = bt.unsqueeze(-1)  
        logits.scatter_(-1, bt, -1e9)

    # Restrict decoding to the protein alphabet.
    sampling_ids = torch.as_tensor(
        [enc_tokenizer.convert_token_to_id(tok) for tok in enc_tokenizer.sampling_vocab],
        dtype=torch.long,
        device=logits.device,
    )
    sampling_mask = torch.zeros(logits.size(-1), dtype=torch.bool, device=logits.device)
    sampling_mask[sampling_ids] = True
    sampling_logits = logits.masked_fill(~sampling_mask, -1e9)

    dist = torch.distributions.Categorical(logits=sampling_logits)
    token_samples = dist.sample() if sample else None
    entropy = dist.entropy()

    return token_samples, entropy


def sample_mask(
    token_batch: LongTensor,
    tokenizer,
    mask_ratio: float = 0.125,
    mask_size: Optional[int] = None,
) -> LongTensor:
    if token_batch.dim() != 2:
        raise ValueError(f"token_batch must be [B, L], got shape {tuple(token_batch.shape)}")

    bsz, seqlen = token_batch.shape
    if seqlen <= 0:
        raise ValueError("token_batch has empty sequence length.")

    if mask_size is None:
        mask_size = int(math.ceil(seqlen * float(mask_ratio)))
    mask_size = int(mask_size)

    if mask_size <= 0:
        mask_size = 1

    device = token_batch.device
    special = torch.as_tensor(tokenizer.special_idxs, dtype=torch.long, device=device).view(-1, 1, 1)

    is_non_special = token_batch.ne(special).prod(dim=0).to(dtype=torch.float32)

    valid_counts = is_non_special.sum(dim=-1)
    min_valid = int(valid_counts.min().item()) if valid_counts.numel() else 0

    if min_valid <= 0:
        raise ValueError(
            "sample_mask: found at least one sequence with no maskable tokens "
            "(likely all special/pad). Check tokenizer.special_idxs or input batch."
        )

    if mask_size > min_valid:
        mask_size = min_valid

    denom = is_non_special.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    mask_weights = is_non_special / denom

    mask_idxs = torch.multinomial(mask_weights, mask_size, replacement=False)
    return mask_idxs


def enumerate_windows(
    base_seqs,
    tokenizer,
    mask_size: int,
    allowed_token_positions: Optional[Set[int]] = None,
) -> Dict[int, List[np.ndarray]]:
    if mask_size is None or int(mask_size) <= 0:
        raise ValueError(f"mask_size must be positive int, got {mask_size}")

    window_mask_idxs: Dict[int, List[np.ndarray]] = {}
    allowed = (
        None
        if allowed_token_positions is None
        else {int(pos) for pos in allowed_token_positions}
    )

    for idx, seq in enumerate(base_seqs):
        window_mask_idxs[idx] = []

        tokens = str_to_tokens(np.array([seq]), tokenizer)
        if tokens.dim() != 2 or tokens.size(0) != 1:
            continue

        mask_size_eff = min(int(mask_size), int(tokens.shape[-1]) - 2)
        if mask_size_eff <= 0:
            continue

        residue_stop = int(tokens.shape[-1]) - 1
        if allowed is None:
            offset = np.random.randint(1, mask_size_eff + 1)
            starts = range(offset, residue_stop, mask_size_eff)
        else:
            invalid = sorted(pos for pos in allowed if pos < 1 or pos >= residue_stop)
            if invalid:
                raise ValueError(
                    f"Mutable positions map outside sequence {idx} token range "
                    f"[1, {residue_stop - 1}]: {invalid}"
                )
            starts = sorted(allowed)

        for mask_start in starts:
            if allowed is None and mask_start + mask_size_eff >= residue_stop:
                mask_start = residue_stop - mask_size_eff

            window = tuple(range(mask_start, mask_start + mask_size_eff))
            if window[-1] >= residue_stop:
                continue
            if allowed is not None and not set(window).issubset(allowed):
                continue

            mask_idxs_np = np.asarray(window, dtype=np.int64).reshape(1, -1)
            if not any(
                np.array_equal(mask_idxs_np, existing)
                for existing in window_mask_idxs[idx]
            ):
                window_mask_idxs[idx].append(mask_idxs_np)

        if not window_mask_idxs[idx]:
            position_text = (
                "the configured mutable positions"
                if allowed is not None
                else "the sequence"
            )
            raise ValueError(
                f"No valid mutation window of size {mask_size_eff} can be formed "
                f"from {position_text} for sequence {idx}."
            )

    return window_mask_idxs


def evaluate_windows(
    base_seqs,
    encoder,
    mask_size: int,
    replacement: bool = True,
    encoder_obj: str = "mlm",
    window_mask_idxs: Optional[Dict[int, List[np.ndarray]]] = None,
) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, List[float]]]:
    if window_mask_idxs is None:
        window_mask_idxs = enumerate_windows(
            base_seqs,
            encoder.tokenizer,
            mask_size,
        )

    device = _get_device(encoder)
    window_entropy: Dict[int, List[float]] = {}

    for idx, seq in enumerate(base_seqs):
        window_entropy[idx] = []

        tokens = str_to_tokens(np.array([seq]), encoder.tokenizer)
        if tokens.dim() != 2 or tokens.size(0) != 1:
            continue

        tokens_dev = tokens.to(device=device, dtype=torch.long)
        for mask_idxs_np in window_mask_idxs[idx]:
            mask_idxs_t = torch.from_numpy(mask_idxs_np).to(
                device=device,
                dtype=torch.long,
            )

            with torch.no_grad():
                masked_inputs = tokens_dev.clone()
                masked_inputs[0, mask_idxs_t[0]] = encoder.tokenizer.masking_idx

                tgt_tok_logits, tgt_mask = encoder.logits_from_tokens(masked_inputs)

                if encoder_obj == "mlm":
                    _, logit_entropy = sample_tokens(
                        tokens_dev,
                        tgt_tok_logits,
                        encoder.tokenizer,
                        replacement=replacement,
                        sample=False,
                    )
                    ent_masked = logit_entropy.gather(
                        1,
                        mask_idxs_t.to(logit_entropy.device),
                    )
                elif encoder_obj == "lanmt":
                    _, ent_full = encoder.sample_tgt_tok_idxs(
                        tgt_tok_logits,
                        tgt_mask,
                        temp=1.0,
                        sample=False,
                    )
                    ent_masked = ent_full.gather(
                        1,
                        mask_idxs_t.to(ent_full.device),
                    )
                else:
                    raise ValueError(f"Unsupported encoder_obj: {encoder_obj}")

            window_entropy[idx].append(float(ent_masked.mean().item()))

    return window_mask_idxs, window_entropy


def mlm_train_step(model, optimizer, token_batch, mask_ratio: float, loss_scale: float = 1.0):
    optimizer.zero_grad(set_to_none=True)

    token_batch = token_batch.to(model.device)
    mask_idxs = sample_mask(token_batch, model.tokenizer, mask_ratio)

    masked_token_batch = token_batch.clone()
    masked_token_batch.scatter_(1, mask_idxs.to(masked_token_batch.device), model.tokenizer.masking_idx)

    logits, _ = model.logits_from_tokens(masked_token_batch)
    vocab_size = int(logits.shape[-1])

    gather_idx = mask_idxs.to(logits.device).unsqueeze(-1).expand(-1, -1, vocab_size)
    masked_logits = logits.gather(1, gather_idx).reshape(-1, vocab_size)

    masked_tokens = token_batch.gather(1, mask_idxs.to(token_batch.device)).reshape(-1).to(model.device)

    loss = float(loss_scale) * F.cross_entropy(masked_logits, masked_tokens)
    loss.backward()
    optimizer.step()

    return loss, masked_logits, masked_tokens


def mlm_train_epoch(model, optimizer, train_loader, mask_ratio: float):
    metrics = {"train_loss": 0.0, "train_perplexity": 0.0}
    model.train()

    denom = max(1, len(train_loader))
    for minibatch in train_loader:
        token_batch = minibatch[0] if isinstance(minibatch, tuple) else minibatch
        loss, masked_logits, masked_tokens = mlm_train_step(model, optimizer, token_batch, mask_ratio)

        log_prob = F.log_softmax(masked_logits, dim=-1)
        true_log_prob = log_prob.gather(1, masked_tokens.view(-1, 1))
        mean_log2p = (true_log_prob / math.log(2.0)).mean()
        perplexity = 2 ** (-mean_log2p)

        metrics["train_perplexity"] += float(perplexity.detach().item()) / denom
        metrics["train_loss"] += float(loss.detach().item()) / denom

    return metrics


def mlm_eval_epoch(model, eval_loader, mask_ratio: float, split: str):
    metrics = {"perplexity": 0.0}
    model.eval()

    denom = max(1, len(eval_loader))
    with torch.no_grad():
        for minibatch in eval_loader:
            token_batch = minibatch[0] if isinstance(minibatch, tuple) else minibatch
            token_batch = token_batch.to(model.device)

            mask_idxs = sample_mask(token_batch, model.tokenizer, mask_ratio)

            masked_token_batch = token_batch.clone()
            masked_token_batch.scatter_(1, mask_idxs.to(masked_token_batch.device), model.tokenizer.masking_idx)

            logits, _ = model.logits_from_tokens(masked_token_batch)
            vocab_size = int(logits.shape[-1])

            gather_idx = mask_idxs.to(logits.device).unsqueeze(-1).expand(-1, -1, vocab_size)
            masked_logits = logits.gather(1, gather_idx).reshape(-1, vocab_size)

            masked_tokens = token_batch.gather(1, mask_idxs.to(token_batch.device)).reshape(-1).to(model.device)

            log_prob = F.log_softmax(masked_logits, dim=-1)
            true_log_prob = log_prob.gather(1, masked_tokens.view(-1, 1))
            mean_log2p = (true_log_prob / math.log(2.0)).mean()
            perplexity = 2 ** (-mean_log2p)

            metrics["perplexity"] += float(perplexity.detach().item()) / denom

    return {f"{split}_{k}": float(v) for k, v in metrics.items()}


def fit_masked_language_model(
    model,
    train_seqs,
    num_epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    mask_ratio: float,
    max_shift: int,
    weights=None,
    log_prefix: str = "",
):
    from evomobo import dataset as gfp_dataset

    train_transform = []
    if int(max_shift) > 0:
        train_transform.append(gfp_transforms.SequenceTranslation(int(max_shift)))
    train_transform.append(gfp_transforms.StringToLongTensor(model.tokenizer))
    train_transform = _Compose(train_transform)

    train_dataset = gfp_dataset.TransformTensorDataset([train_seqs], train_transform)

    if weights is None:
        loader_kwargs = dict(batch_size=int(batch_size), shuffle=True)
    else:
        sampler = torch.utils.data.WeightedRandomSampler(weights, int(batch_size), replacement=True)
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=int(batch_size), drop_last=False)
        loader_kwargs = dict(batch_sampler=batch_sampler)

    collate_fn = lambda x: gfp_transforms.padding_collate_fn(x, model.tokenizer.padding_idx)
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, **loader_kwargs)

    optimizer = torch.optim.Adam(model.param_groups(lr))
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=math.ceil(int(patience) / 2))

    records = []
    best_score, best_epoch, best_weights = None, 0, None

    model.requires_grad_(True)
    for epoch in range(int(num_epochs)):
        metrics = mlm_train_epoch(model, optimizer, train_loader, float(mask_ratio))
        lr_sched.step(metrics["train_loss"])

        best_score, best_epoch, best_weights, stop = check_early_stopping(
            model=model,
            best_score=best_score,
            best_epoch=best_epoch,
            best_weights=best_weights,
            curr_score=metrics["train_loss"],
            curr_epoch=epoch + 1,
            patience=int(patience),
            save_weights=True,
        )

        metrics.update({"best_score": best_score, "best_epoch": best_epoch})
        log_metrics = {"/".join((log_prefix, k)): v for k, v in metrics.items()} if log_prefix else metrics
        wandb_log(log_metrics)

        records.append(metrics)
        if stop:
            break

    if best_weights is not None:
        model.load_state_dict(best_weights, strict=False)
    model.requires_grad_(False)

    return records
