import math

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from evomobo.models.masked_layers import mResidualBlock


class LanguageModel(nn.Module):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def __init__(self, model, batch_size, num_epochs, patience, lr, max_shift, mask_ratio, **kwargs):
        super().__init__()
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.lr = lr
        self.max_shift = max_shift
        self.tokenizer = model.tokenizer
        self.mask_ratio = mask_ratio

    def forward(self, inputs):
        if isinstance(inputs, np.ndarray):
            tok_idxs = self.str_to_tokens(inputs)
        else:
            tok_idxs = inputs
        return self.model(tok_idxs)

    def pool_features(self, src_tok_features, src_mask, src_tok_idxs=None):
        pooling_mask = src_mask
        if src_tok_idxs is not None and hasattr(self.model, "get_pooling_mask"):
            pooling_mask = self.model.get_pooling_mask(src_tok_idxs, src_mask)
        lat_tok_features, pooled_features = self.model.function_head(
            src_tok_features,
            padding_mask=src_mask,
            pooling_mask=pooling_mask,
        )
        return lat_tok_features, pooled_features

    def get_token_idx(self, token):
        return self.model.tokenizer.convert_token_to_id(token)

    def get_token(self, idx):
        return self.model.tokenizer.convert_id_to_token(idx)

    def get_token_features(self, src_tok_idxs):
        src_tok_features, src_mask = self.model.enc_tok_features(src_tok_idxs)
        return src_tok_features, src_mask

    def logits_from_tokens(self, src_tok_idxs):
        src_tok_features, src_mask = self.get_token_features(src_tok_idxs)
        tgt_tok_logits, tgt_mask = self.logits_from_features(src_tok_features, src_mask, lat_tok_features=None)
        return tgt_tok_logits, tgt_mask

    def logits_from_features(self, src_tok_features, src_mask, lat_tok_features, tgt_lens=None):
        lat_tok_features, tgt_tok_features, tgt_mask, _ = self.model.dec_tok_features(
            src_tok_features, src_mask, lat_tok_features, tgt_lens
        )
        tgt_tok_logits = self.model.tgt_tok_logits(tgt_tok_features)
        return tgt_tok_logits, tgt_mask

    def sample_tgt_tok_idxs(self, tgt_tok_logits, tgt_mask, temp=1., sample=True):
        batch_size, num_tokens = tgt_mask.shape
        tgt_mask = tgt_mask.to(device=tgt_tok_logits.device).bool()
        tgt_lens = tgt_mask.float().sum(-1).long()

        t = float(temp)
        if t <= 0:
            raise ValueError(f"temp must be > 0, got {temp}")
        logits = (tgt_tok_logits / t).clone()

        # Exclude special tokens from sampling and entropy.
        non_viable_idxs = torch.as_tensor(
            self.tokenizer.special_idxs,
            dtype=torch.long,
            device=logits.device,
        )
        if non_viable_idxs.numel() > 0:
            logits.index_fill_(-1, non_viable_idxs, -1e10)

        # Restrict decoding to the protein alphabet.
        sampling_ids = torch.as_tensor(
            [
                self.tokenizer.convert_token_to_id(tok)
                for tok in self.tokenizer.sampling_vocab
            ],
            dtype=torch.long,
            device=logits.device,
        )
        sampling_mask = torch.zeros(
            logits.size(-1),
            dtype=torch.bool,
            device=logits.device,
        )
        sampling_mask[sampling_ids] = True
        sampling_logits = logits.masked_fill(~sampling_mask, -1e10)

        tok_dist = torch.distributions.Categorical(logits=sampling_logits)
        tgt_tok_idxs = None
        if sample:
            sample_tok_idxs = tok_dist.sample()
            tgt_tok_idxs = torch.full(
                tgt_mask.size(),
                self.tokenizer.padding_idx,
                dtype=torch.long,
                device=logits.device,
            )
            tgt_tok_idxs = torch.where(tgt_mask, sample_tok_idxs, tgt_tok_idxs)

            if num_tokens > 0:
                tgt_tok_idxs[:, 0] = self.tokenizer.bos_idx
                eos_pos = (tgt_lens.clamp_min(1) - 1).clamp_max(num_tokens - 1)
                tgt_tok_idxs[
                    torch.arange(batch_size, device=logits.device),
                    eos_pos,
                ] = self.tokenizer.eos_idx

        logit_entropy = tok_dist.entropy()
        logit_entropy *= tgt_mask.float()

        return tgt_tok_idxs, logit_entropy

    def param_groups(self, *args, **kwargs):
        return self.model.param_groups(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000, batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len + 1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len + 1, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        if self.batch_first:
            pe = pe.transpose(1, 0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]

        return self.dropout(x)


class FunctionHead(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, layernorm, dropout_p, num_heads, type):
        super().__init__()
        if type == 'conv':
            self.att_layer = mResidualBlock(input_dim, input_dim, kernel_size, layernorm, dropout_p)
        elif type == 'mha':
            self.att_layer = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout_p, batch_first=True)
            self.dropout = nn.Dropout(dropout_p)
        else:
            raise ValueError
        self.type = type
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(
            self,
            tok_features: Tensor,
            padding_mask: Tensor,
            pooling_mask: Tensor,
    ):
        if self.type == 'conv':
            tok_features, _ = self.att_layer((tok_features.permute(0, 2, 1), padding_mask))
            tok_features = tok_features.permute(0, 2, 1)
        else:
            key_padding_mask = ~padding_mask.bool()
            tok_features, _ = self.att_layer(
                tok_features, tok_features, tok_features, key_padding_mask, need_weights=False
            )
            tok_features = self.dropout(F.gelu(tok_features))

        pooling_mask = pooling_mask.unsqueeze(-1).float()
        pooled_features = (pooling_mask * tok_features).sum(-2) / (pooling_mask.sum(-2) + 1e-6)
        pooled_features = self.linear(pooled_features)
        return tok_features, pooled_features


class LengthHead(nn.Module):
    def __init__(self, input_dim, max_len_delta):
        super().__init__()
        num_classes = max_len_delta * 2 + 1
        self.linear = nn.Linear(input_dim, num_classes)
        self.max_len_delta = max_len_delta

    def forward(self, tok_features, pooling_mask):
        pooling_mask = pooling_mask.unsqueeze(-1).float()
        pooled_features = (pooling_mask * tok_features).sum(-2) / (pooling_mask.sum(-2) + 1e-6)
        logits = self.linear(pooled_features)
        return logits

    def sample(self, src_lens, logits):
        if self.max_len_delta == 0:
            return src_lens
        tgt_len_dist = torch.distributions.Categorical(logits=logits)
        len_deltas = tgt_len_dist.sample()
        len_deltas -= self.max_len_delta
        return src_lens + len_deltas


class LengthTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter('lengthscale', nn.Parameter(torch.tensor(1.)))

    def forward(
            self,
            src_tok_features: Tensor,
            src_mask: Tensor,
            tgt_lens: Tensor,
    ):
        batch_size, num_src_tokens = src_mask.shape
        src_lens = src_mask.float().sum(-1)
        tgt_lens = tgt_lens.to(src_lens)

        if torch.all(src_lens == tgt_lens):
            return src_tok_features, src_mask.bool()

        src_arange = torch.arange(num_src_tokens).to(src_mask.device)
        src_arange = src_arange.expand(batch_size, -1).unsqueeze(-1).float()

        tgt_arange = torch.arange(tgt_lens.max()).to(src_mask.device)
        tgt_arange = tgt_arange.expand(batch_size, -1).unsqueeze(-2).float()

        len_ratio = src_lens / tgt_lens
        len_ratio = len_ratio.view(-1, 1, 1)

        sq_diff = (src_arange - len_ratio * tgt_arange) ** 2
        sq_diff = sq_diff.to(self.lengthscale)

        logits = -sq_diff / (2 * self.lengthscale ** 2)
        logits = src_mask[..., None] * logits - (1 - src_mask[..., None]) * 1e10
        weights = F.softmax(logits, dim=-2)
        weights = weights.unsqueeze(-1).to(src_tok_features)

        tgt_token_features = src_tok_features.unsqueeze(-2)
        tgt_token_features = weights * tgt_token_features
        tgt_token_features = tgt_token_features.sum(-3)

        tgt_mask = (tgt_arange.squeeze(-2) < tgt_lens.unsqueeze(-1))

        return tgt_token_features, tgt_mask
