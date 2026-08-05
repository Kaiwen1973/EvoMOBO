from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from evomobo.models.lm_elements import FunctionHead, LengthHead, LengthTransform


class MaskedProteinLMAdapter(nn.Module):
    """Common LaMBO interface for pretrained masked protein LMs."""

    def __init__(
        self,
        tokenizer,
        max_len: int,
        latent_dim: int,
        out_dim: int,
        kernel_size: int,
        p: float,
        layernorm: bool,
        max_len_delta: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.latent_dim = int(latent_dim)
        self.function_head = FunctionHead(
            self.latent_dim,
            int(out_dim),
            int(kernel_size),
            bool(layernorm),
            float(p),
            None,
            type="conv",
        )
        self.length_head = LengthHead(self.latent_dim, int(max_len_delta))
        self.length_tf = LengthTransform()
        self.register_buffer(
            "external_to_model_ids",
            torch.empty(0, dtype=torch.long),
            persistent=True,
        )

    def set_external_to_model_ids(self, token_ids) -> None:
        mapping = torch.as_tensor(token_ids, dtype=torch.long)
        if mapping.numel() != len(self.tokenizer.full_vocab):
            raise ValueError(
                "External/model vocabulary map must contain one entry for each "
                f"external token; got {mapping.numel()} and {len(self.tokenizer.full_vocab)}."
            )
        self.external_to_model_ids = mapping

    def map_external_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        mapping = self.external_to_model_ids.to(token_ids.device)
        return mapping[token_ids.long()]

    def map_model_logits(self, logits: torch.Tensor) -> torch.Tensor:
        mapping = self.external_to_model_ids.to(logits.device)
        return logits.index_select(-1, mapping)

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
        return lat_feat, src_feat, src_mask.bool(), None

    def tgt_tok_logits(self, tgt_feat: torch.Tensor) -> torch.Tensor:
        flat = tgt_feat.flatten(end_dim=-2)
        model_logits = self.lm_head(flat)
        external_logits = self.map_model_logits(model_logits)
        return external_logits.view(*tgt_feat.shape[:-1], -1)

    def get_pooling_mask(
        self,
        tok_idxs: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        tok_idxs = tok_idxs[:, : padding_mask.size(1)].to(padding_mask.device)
        return padding_mask.float() * tok_idxs.ne(self.tokenizer.eos_idx).float()

    def forward(self, tok_idxs: torch.Tensor) -> torch.Tensor:
        features, mask = self.enc_tok_features(tok_idxs)
        tok_idxs = tok_idxs[:, : features.size(1)].to(features.device)
        pooling_mask = self.get_pooling_mask(tok_idxs, mask)
        _, pooled = self.function_head(features, mask, pooling_mask)
        return pooled

    def param_groups(self, lr, weight_decay: float = 0.0, pretrain_lr=None):
        head_params = list(self.function_head.parameters()) + list(self.lm_head.parameters())
        other_params = list(self.length_head.parameters()) + list(self.length_tf.parameters())
        excluded_ids = {id(param) for param in head_params + other_params}
        backbone_params = [
            param for param in self.parameters() if id(param) not in excluded_ids
        ]
        head_group = {
            "params": head_params,
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": (0.0, 1e-2),
        }
        other_group = {
            "params": other_params,
            "lr": lr,
            "weight_decay": weight_decay,
        }
        if pretrain_lr is None or float(pretrain_lr) == 0.0:
            return head_group, other_group
        backbone_group = {
            "params": backbone_params,
            "lr": pretrain_lr,
            "weight_decay": weight_decay,
        }
        return head_group, backbone_group, other_group
