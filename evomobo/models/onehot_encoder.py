from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from evomobo.models.lm_elements import FunctionHead
from evomobo.models.masked_layers import mResidualBlock


class OneHotEncoderLmHeadForLambo(nn.Module):
    """One-hot token encoder with a directly compatible trainable MLM head."""

    def __init__(
        self,
        tokenizer,
        max_len: int,
        out_dim: int = 32,
        kernel_size: int = 5,
        p: float = 0.0,
        layernorm: bool = True,
        logit_scale: float = 5.0,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.vocab_size = len(tokenizer.full_vocab)
        self.latent_dim = self.vocab_size

        self.function_head = FunctionHead(
            self.vocab_size,
            int(out_dim),
            int(kernel_size),
            bool(layernorm),
            float(p),
            None,
            type="conv",
        )
        self.decoder = mResidualBlock(
            self.vocab_size,
            self.vocab_size,
            int(kernel_size),
            bool(layernorm),
            float(p),
        )
        self.lm_head = nn.Linear(self.vocab_size, self.vocab_size)
        with torch.no_grad():
            self.lm_head.weight.copy_(
                torch.eye(self.vocab_size) * float(logit_scale)
            )
            self.lm_head.bias.zero_()

    def enc_tok_features(self, tok_idxs: torch.Tensor):
        tok_idxs = tok_idxs[:, : self.max_len].long().to(self.lm_head.weight.device)
        mask = tok_idxs.ne(self.tokenizer.padding_idx)
        features = F.one_hot(tok_idxs, num_classes=self.vocab_size).to(
            dtype=self.lm_head.weight.dtype
        )
        features = features * mask.unsqueeze(-1)
        return features, mask.float()

    def dec_tok_features(
        self,
        src_feat: torch.Tensor,
        src_mask: torch.Tensor,
        lat_feat: Optional[torch.Tensor] = None,
        tgt_lens: Optional[torch.Tensor] = None,
    ):
        if lat_feat is None:
            lat_feat, _ = self.function_head(
                src_feat,
                padding_mask=src_mask.float(),
                pooling_mask=src_mask.float(),
            )
        decoded, tgt_mask = self.decoder(
            (src_feat.permute(0, 2, 1), src_mask.float())
        )
        return lat_feat, decoded.permute(0, 2, 1), tgt_mask.bool(), None

    def tgt_tok_logits(self, tgt_feat: torch.Tensor) -> torch.Tensor:
        flat = tgt_feat.flatten(end_dim=-2)
        logits = self.lm_head(flat)
        return logits.view(*tgt_feat.shape[:-1], self.vocab_size)

    def get_pooling_mask(
        self,
        tok_idxs: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        tok_idxs = tok_idxs[:, : padding_mask.size(1)].to(padding_mask.device)
        return padding_mask.float() * tok_idxs.ne(self.tokenizer.eos_idx).float()

    def forward(self, tok_idxs: torch.Tensor) -> torch.Tensor:
        features, mask = self.enc_tok_features(tok_idxs)
        pooling_mask = self.get_pooling_mask(tok_idxs, mask)
        _, pooled = self.function_head(features, mask, pooling_mask)
        return pooled

    def param_groups(self, lr, weight_decay: float = 0.0, pretrain_lr=None):
        return (
            {
                "params": list(self.function_head.parameters()),
                "lr": lr,
                "weight_decay": weight_decay,
                "betas": (0.0, 1e-2),
            },
            {
                "params": list(self.decoder.parameters())
                + list(self.lm_head.parameters()),
                "lr": lr,
                "weight_decay": weight_decay,
            },
        )
