from __future__ import annotations

import copy

import torch

from evomobo.models.plm_adapter_base import MaskedProteinLMAdapter


class EsmCEncoderLmHeadForLambo(MaskedProteinLMAdapter):
    """Official ESM-C adapter with external LaMBO token ids."""

    def __init__(
        self,
        tokenizer,
        max_len: int,
        model_name: str = "esmc_300m",
        out_dim: int = 32,
        kernel_size: int = 5,
        p: float = 0.0,
        layernorm: bool = True,
        max_len_delta: int = 0,
        **kwargs,
    ):
        try:
            from esm.models.esmc import ESMC
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "ESM-C requires the official Biohub esm package. See README.md "
                "for the separate ESM-C environment installation command."
            ) from exc

        pretrained = ESMC.from_pretrained(model_name)
        latent_dim = int(pretrained.embed.embedding_dim)
        super().__init__(
            tokenizer=tokenizer,
            max_len=max_len,
            latent_dim=latent_dim,
            out_dim=out_dim,
            kernel_size=kernel_size,
            p=p,
            layernorm=layernorm,
            max_len_delta=max_len_delta,
        )

        self.model_name = str(model_name)
        self.esmc = pretrained
        self.lm_head = copy.deepcopy(pretrained.sequence_head)
        self.esmc.requires_grad_(False)
        self.lm_head.requires_grad_(True)
        self.esmc.eval()

        model_tokenizer = pretrained.tokenizer
        bos_id = getattr(model_tokenizer, "cls_token_id", None)
        if bos_id is None:
            bos_id = getattr(model_tokenizer, "bos_token_id", None)
        eos_id = getattr(model_tokenizer, "eos_token_id", None)
        if eos_id is None:
            eos_id = getattr(model_tokenizer, "sep_token_id", None)
        special_ids = {
            "<cls>": bos_id,
            "<pad>": getattr(model_tokenizer, "pad_token_id", None),
            "<eos>": eos_id,
            "<unk>": getattr(model_tokenizer, "unk_token_id", None),
            "<mask>": getattr(model_tokenizer, "mask_token_id", None),
        }
        x_id = int(model_tokenizer.convert_tokens_to_ids("X"))
        mapping = []
        for token in tokenizer.full_vocab:
            if token in special_ids and special_ids[token] is not None:
                mapping.append(int(special_ids[token]))
            elif token in tokenizer.sampling_vocab or token == "X":
                mapping.append(int(model_tokenizer.convert_tokens_to_ids(token)))
            else:
                mapping.append(x_id)
        self.set_external_to_model_ids(mapping)

    @torch.no_grad()
    def enc_tok_features(self, tok_idxs: torch.Tensor):
        self.esmc.eval()
        device = next(self.esmc.parameters()).device
        tok_idxs = tok_idxs[:, : self.max_len].to(device).long()
        mask = tok_idxs.ne(self.tokenizer.padding_idx)
        model_ids = self.map_external_tokens(tok_idxs)
        outputs = self.esmc(sequence_tokens=model_ids, sequence_id=mask)
        return outputs.embeddings, mask.float()
