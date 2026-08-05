from __future__ import annotations

import copy

import torch

from evomobo.models.plm_adapter_base import MaskedProteinLMAdapter


class ProtBertEncoderLmHeadForLambo(MaskedProteinLMAdapter):
    """ProtBERT-BFD adapter with external LaMBO token ids."""

    def __init__(
        self,
        tokenizer,
        max_len: int,
        model_name: str = "Rostlab/prot_bert_bfd",
        out_dim: int = 32,
        kernel_size: int = 5,
        p: float = 0.0,
        layernorm: bool = True,
        max_len_delta: int = 0,
        local_files_only: bool = False,
        **kwargs,
    ):
        try:
            from transformers import BertForMaskedLM, BertTokenizer
        except ImportError as exc:
            raise ImportError(
                "ProtBERT requires transformers. Install the project requirements first."
            ) from exc

        model_tokenizer = BertTokenizer.from_pretrained(
            model_name,
            do_lower_case=False,
            local_files_only=bool(local_files_only),
        )
        pretrained = BertForMaskedLM.from_pretrained(
            model_name,
            local_files_only=bool(local_files_only),
        )
        latent_dim = int(pretrained.config.hidden_size)
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
        self.protbert = pretrained.bert
        self.lm_head = copy.deepcopy(pretrained.cls)
        del pretrained
        self.protbert.requires_grad_(False)
        self.lm_head.requires_grad_(True)
        self.protbert.eval()

        special_ids = {
            "<cls>": model_tokenizer.cls_token_id,
            "<pad>": model_tokenizer.pad_token_id,
            "<eos>": model_tokenizer.sep_token_id,
            "<unk>": model_tokenizer.unk_token_id,
            "<mask>": model_tokenizer.mask_token_id,
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
        self.protbert.eval()
        device = next(self.protbert.parameters()).device
        tok_idxs = tok_idxs[:, : self.max_len].to(device).long()
        attention_mask = tok_idxs.ne(self.tokenizer.padding_idx)
        model_ids = self.map_external_tokens(tok_idxs)
        outputs = self.protbert(
            input_ids=model_ids,
            attention_mask=attention_mask.long(),
            return_dict=True,
        )
        return outputs.last_hidden_state, attention_mask.float()
