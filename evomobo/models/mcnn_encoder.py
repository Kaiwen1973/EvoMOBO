from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn

from evomobo.models.lm_elements import (
    PositionalEncoding,
    FunctionHead,
    LengthHead,
    LengthTransform,
)
from evomobo.models.masked_layers import Apply, mResidualBlock


def _get_vocab_size(tokenizer) -> int:
    for attr in ("vocab_size",):
        v = getattr(tokenizer, attr, None)
        if isinstance(v, int) and v > 0:
            return v

    for attr in ("full_vocab", "vocab", "idx_to_token"):
        v = getattr(tokenizer, attr, None)
        if v is None:
            continue
        try:
            n = len(v)
            if n > 0:
                return int(n)
        except Exception:
            pass

    raise ValueError(
        "[mCNN] Cannot infer vocab size from tokenizer. "
        "Expected tokenizer to have vocab_size/full_vocab/vocab/idx_to_token."
    )


class Permute(nn.Module):

    def __init__(self, *dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


class mCNN(nn.Module):

    def __init__(
        self,
        tokenizer,
        max_len: int,
        embed_dim: int = 64,
        latent_dim: int = 16,
        out_dim: int = 64,
        kernel_size: int = 5,
        p: float = 0.0,
        layernorm: bool = True,
        max_len_delta: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.embed_dim = int(embed_dim)
        self.latent_dim = int(latent_dim)
        self.out_dim = int(out_dim)
        self.max_len_delta = int(max_len_delta)

        for attr in ("padding_idx", "eos_idx"):
            if not hasattr(tokenizer, attr):
                raise ValueError(f"tokenizer missing required attr: {attr}")

        vocab_size = _get_vocab_size(tokenizer)

        self.embedding = nn.Embedding(
            vocab_size, self.embed_dim, padding_idx=tokenizer.padding_idx
        )
        self.pos_encoder = PositionalEncoding(
            self.embed_dim, p, self.max_len, batch_first=True
        )

        self.encoder = nn.Sequential(
            Apply(Permute(0, 2, 1)),
            mResidualBlock(self.embed_dim, self.embed_dim, kernel_size, layernorm, p),
            mResidualBlock(self.embed_dim, self.embed_dim, kernel_size, layernorm, p),
            mResidualBlock(self.embed_dim, self.latent_dim, kernel_size, layernorm, p),
            Apply(Permute(0, 2, 1)),
        )

        self.decoder = nn.Sequential(
            Apply(Permute(0, 2, 1)),
            mResidualBlock(2 * self.latent_dim, self.embed_dim, kernel_size, layernorm, p),
            mResidualBlock(self.embed_dim, self.embed_dim, kernel_size, layernorm, p),
            mResidualBlock(self.embed_dim, self.embed_dim, kernel_size, layernorm, p),
            Apply(Permute(0, 2, 1)),
        )

        self.length_transform = LengthTransform()
        self.function_head = FunctionHead(
            self.latent_dim, self.out_dim, kernel_size, layernorm, p, None, type="conv"
        )
        self.length_head = LengthHead(self.latent_dim, self.max_len_delta)

        self.lm_head = nn.Linear(self.embed_dim, vocab_size)

    def _truncate(self, tok_idxs: torch.Tensor) -> torch.Tensor:
        if tok_idxs.size(1) > self.max_len:
            tok_idxs = tok_idxs[:, : self.max_len]
        return tok_idxs

    def enc_tok_features(self, src_tok_idxs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_tok_idxs = self._truncate(src_tok_idxs).long().to(self.embedding.weight.device)

        src_tok_features = self.embedding(src_tok_idxs) * math.sqrt(self.embed_dim)
        src_tok_features = self.pos_encoder(src_tok_features)

        src_mask = (src_tok_idxs != self.tokenizer.padding_idx).float()
        src_tok_features, src_mask = self.encoder((src_tok_features, src_mask))
        return src_tok_features, src_mask

    def dec_tok_features(
        self,
        src_tok_features: torch.Tensor,
        src_mask: torch.Tensor,
        lat_tok_features: Optional[torch.Tensor] = None,
        tgt_lens: Optional[torch.Tensor] = None,
    ):
        if lat_tok_features is None:
            lat_tok_features, _ = self.function_head(
                src_tok_features, padding_mask=src_mask, pooling_mask=src_mask
            )

        len_delta_logits = self.length_head(src_tok_features, src_mask)

        if tgt_lens is None:
            src_lens = src_mask.float().sum(-1)
            tgt_lens = self.length_head.sample(src_lens, len_delta_logits)

        tgt_tok_features, tgt_mask = self.length_transform(
            src_tok_features=torch.cat([src_tok_features, lat_tok_features], dim=-1),
            src_mask=src_mask,
            tgt_lens=tgt_lens,
        )
        tgt_tok_features, tgt_mask = self.decoder((tgt_tok_features, tgt_mask))
        return lat_tok_features, tgt_tok_features, tgt_mask, len_delta_logits

    def tgt_tok_logits(self, tgt_tok_features: torch.Tensor) -> torch.Tensor:
        flat = tgt_tok_features.flatten(end_dim=-2)
        logits = self.lm_head(flat)
        return logits.view(*tgt_tok_features.shape[:-1], -1)

    def get_pooling_mask(
        self,
        tok_idxs: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        tok_idxs = tok_idxs[:, : padding_mask.size(1)].to(padding_mask.device)
        return padding_mask.float() * tok_idxs.ne(self.tokenizer.eos_idx).float()

    def forward(self, src_tok_idxs: torch.Tensor) -> torch.Tensor:
        src_tok_idxs = self._truncate(src_tok_idxs).long()
        src_tok_features, src_mask = self.enc_tok_features(src_tok_idxs)

        pooling_mask = self.get_pooling_mask(src_tok_idxs, src_mask)
        _, pooled_features = self.function_head(src_tok_features, src_mask, pooling_mask)
        return pooled_features

    def param_groups(self, lr, weight_decay: float = 0.0, pretrain_lr=None):
        shared_group = dict(params=[], lr=lr, weight_decay=weight_decay, betas=(0.0, 1e-2))
        other_group = dict(params=[], lr=lr, weight_decay=weight_decay)

        shared_names = {"embedding", "pos_encoder", "encoder", "function_head"}
        for p_name, param in self.named_parameters():
            prefix = p_name.split(".", 1)[0]
            if prefix in shared_names:
                shared_group["params"].append(param)
            else:
                other_group["params"].append(param)

        return shared_group, other_group
