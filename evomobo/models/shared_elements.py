from __future__ import annotations

import copy
import math
from typing import Callable, Optional

import torch
import esm
from torch import nn

from evomobo.models.lm_elements import FunctionHead, LengthHead, LengthTransform

def gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def get_activation_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    name = str(name).lower().strip()
    if name == "gelu":
        return gelu
    if name == "relu":
        return torch.nn.functional.relu
    if name == "swish":
        return swish
    raise KeyError(f"Unknown activation function: {name}")


class LayerNorm(nn.Module):
    def __init__(self, h: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(h))
        self.bias = nn.Parameter(torch.zeros(h))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.mean(-1, keepdim=True)
        v = (x - m).pow(2).mean(-1, keepdim=True)
        return self.weight * (x - m) / torch.sqrt(v + self.eps) + self.bias



class EsmOnlyMLMHead(nn.Module):

    def __init__(self, hidden: int, embed_tokens: nn.Embedding):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.activation = gelu
        self.layer_norm = LayerNorm(hidden)

        self.decoder = nn.Linear(hidden, embed_tokens.num_embeddings, bias=False)
        self.decoder.weight = embed_tokens.weight

        self.bias = nn.Parameter(torch.zeros(embed_tokens.num_embeddings))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        return self.decoder(x) + self.bias



class EsmEncoderLmHeadForLambo(nn.Module):

    def __init__(
        self,
        tokenizer,
        max_len: int,
        esm_variant: str = "esm2_t6_8M_UR50D",
        kernel_size: int = 5,
        p: float = 0.1,
        out_dim: int = 16,
        layernorm: bool = True,
        max_len_delta: int = 0,
        esm_dropout: bool = True,  
        **kwargs,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.esm_variant = str(esm_variant)
        self.esm_dropout = bool(esm_dropout)

        try:
            load_fn = getattr(esm.pretrained, self.esm_variant)
        except AttributeError as e:
            raise ValueError(
                f"Unknown esm_variant='{self.esm_variant}'. "
                f"Check spelling and availability under esm.pretrained."
            ) from e

        self.esm_model, _ = load_fn()
        self.esm_layer = int(self.esm_model.num_layers)
        self.latent_dim = int(self.esm_model.embed_dim)

        self.esm_model.requires_grad_(False)

        self.function_head = FunctionHead(
            self.latent_dim, out_dim, kernel_size, layernorm, p, None, type="conv"
        )
        self.length_head = LengthHead(self.latent_dim, max_len_delta)
        self.length_tf = LengthTransform()

        pretrained_lm_head = getattr(self.esm_model, "lm_head", None)
        if pretrained_lm_head is None:
            self.cls = EsmOnlyMLMHead(
                self.latent_dim,
                copy.deepcopy(self.esm_model.embed_tokens),
            )
        else:
            # Copy the decoder to protect frozen ESM embeddings.
            self.cls = copy.deepcopy(pretrained_lm_head)
        self.cls.requires_grad_(True)

        self.decoder = nn.Identity()

        self.esm_model.eval()

    def _sync_esm_mode(self) -> None:
        if self.training and self.esm_dropout:
            self.esm_model.train()
        else:
            self.esm_model.eval()

    @torch.no_grad()
    def enc_tok_features(self, tok_idxs: torch.Tensor):
        self._sync_esm_mode()

        dev = self.esm_model.embed_tokens.weight.device
        tok_idxs = tok_idxs.to(dev).long()

        max_tok_len = int(self.max_len)
        if tok_idxs.size(1) > max_tok_len:
            tok_idxs = tok_idxs[:, :max_tok_len]

        mask = tok_idxs.ne(self.tokenizer.padding_idx).float()

        out = self.esm_model(
            tok_idxs,
            repr_layers=[self.esm_layer],
            return_contacts=False,
        )
        feat = out["representations"][self.esm_layer]
        return feat, mask

    def dec_tok_features(
        self,
        src_feat: torch.Tensor,
        src_mask: torch.Tensor,
        lat_feat: Optional[torch.Tensor] = None,
        tgt_lens: Optional[torch.Tensor] = None,
    ):
        if lat_feat is None:
            padding_mask = src_mask.float()
            pooling_mask = padding_mask
            lat_feat, _ = self.function_head(
                src_feat,
                padding_mask=padding_mask,
                pooling_mask=pooling_mask,
            )

        tgt_tok_features = src_feat
        tgt_mask = src_mask.bool()
        return lat_feat, tgt_tok_features, tgt_mask, None

    def tgt_tok_logits(self, tgt_feat: torch.Tensor) -> torch.Tensor:
        flat = tgt_feat.flatten(end_dim=-2)
        logits = self.cls(flat)
        return logits.view(*tgt_feat.shape[:-1], -1)

    def get_pooling_mask(
        self,
        tok_idxs: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        tok_idxs = tok_idxs[:, : padding_mask.size(1)].to(padding_mask.device)
        return padding_mask.float() * tok_idxs.ne(self.tokenizer.eos_idx).float()

    def forward(self, tok_idxs: torch.Tensor) -> torch.Tensor:
        feat, mask = self.enc_tok_features(tok_idxs)

        tok_idxs = tok_idxs.to(feat.device).long()
        if tok_idxs.size(1) > feat.size(1):
            tok_idxs = tok_idxs[:, :feat.size(1)]

        pool_mask = self.get_pooling_mask(tok_idxs, mask)
        _, pooled = self.function_head(feat, mask, pool_mask)
        return pooled

    def param_groups(self, lr, weight_decay: float = 0.0, pretrain_lr: float = None):
        shared_group = dict(
            params=[],
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.0, 1e-2),
        )
        other_group = dict(
            params=[],
            lr=lr,
            weight_decay=weight_decay,
        )
        pretrained_group = dict(
            params=[],
            lr=pretrain_lr,
            weight_decay=weight_decay,
        )

        if pretrain_lr is None:
            shared_names = ["esm_model", "function_head"]
            pretrained_names = []
        else:
            shared_names = ["function_head", "cls"]
            pretrained_names = ["esm_model"]

        for n, p in self.named_parameters():
            root = n.split(".")[0]
            if root in shared_names:
                shared_group["params"].append(p)
            elif root in pretrained_names:
                pretrained_group["params"].append(p)
            else:
                other_group["params"].append(p)

        if pretrain_lr is None or pretrain_lr == 0 or len(pretrained_group["params"]) == 0:
            return shared_group, other_group
        return shared_group, pretrained_group, other_group



def check_early_stopping(
    model,
    best_score,
    best_epoch,
    best_weights,
    curr_score,
    curr_epoch,
    patience,
    tol: float = 1e-3,
    save_weights: bool = True,
):
    eps, stop = 1e-6, False
    if best_score is None or (best_score - curr_score) / (abs(best_score) + eps) > tol:
        best_score, best_epoch = curr_score, curr_epoch
    elif curr_epoch - best_epoch >= patience:
        stop = True

    if best_epoch == curr_epoch and save_weights and model is not None:
        del best_weights
        model.cpu()
        best_weights = copy.deepcopy(model.state_dict())
        if hasattr(model, "device"):
            model.to(model.device)
    return best_score, best_epoch, best_weights, stop
