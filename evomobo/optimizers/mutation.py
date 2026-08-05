import numpy as np

import torch

from pymoo.factory import get_mutation
from pymoo.core.mutation import Mutation

from evomobo import utils
from evomobo.models.mlm import sample_tokens


def _module_device(module):
    if hasattr(module, "device"):
        try:
            return torch.device(module.device)
        except Exception:
            pass
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _filter_logits_to_sampling_vocab(logits, tokenizer):
    allowed_ids = torch.as_tensor(
        [tokenizer.convert_token_to_id(tok) for tok in tokenizer.sampling_vocab],
        dtype=torch.long,
        device=logits.device,
    )
    allowed_mask = torch.zeros(logits.size(-1), dtype=torch.bool, device=logits.device)
    allowed_mask[allowed_ids] = True

    filtered_logits = logits.clone()
    filtered_logits[..., ~allowed_mask] = -1e10
    return filtered_logits


def get_mlm_mutation(mlm_obj, problem, cand_idx, res_idx):
    cand_idx = np.asarray(cand_idx, dtype=int).reshape(-1)
    res_idx = np.asarray(res_idx, dtype=int).reshape(-1)
    if cand_idx.shape[0] != res_idx.shape[0]:
        raise ValueError(
            f"cand_idx and res_idx must have same length, got {cand_idx.shape[0]} and {res_idx.shape[0]}"
        )

    seqs = [problem.candidate_pool[i].mutant_residue_seq for i in cand_idx]
    base_tok_idxs = utils.str_to_tokens(seqs, mlm_obj.tokenizer)

    seq_lens = torch.as_tensor([len(seq) for seq in seqs], dtype=torch.long)
    residue_pos = torch.as_tensor(res_idx, dtype=torch.long).clamp_min(0)
    residue_pos = torch.minimum(residue_pos, (seq_lens - 1).clamp_min(0))
    mask_idxs = (residue_pos + 1).view(-1, 1)

    device = _module_device(mlm_obj)
    mask_idxs = mask_idxs.to(device=device)
    base_tok_idxs = base_tok_idxs.to(device=device, dtype=torch.long)
    src_tok_idxs = base_tok_idxs.clone()
    src_tok_idxs.scatter_(1, mask_idxs, mlm_obj.tokenizer.masking_idx)

    with torch.no_grad():
        tgt_tok_logits, _ = mlm_obj.logits_from_tokens(src_tok_idxs)
        tgt_tok_logits = _filter_logits_to_sampling_vocab(tgt_tok_logits, mlm_obj.tokenizer)

    new_tok_idxs, _ = sample_tokens(
        base_tok_idxs, tgt_tok_logits, mlm_obj.tokenizer, replacement=False
    )

    new_tok_idxs = new_tok_idxs.gather(1, mask_idxs.to(new_tok_idxs.device)).reshape(-1)
    new_tok_idxs = new_tok_idxs.detach().cpu().tolist()

    sampling_vocab_lookup = {
        tok: idx for idx, tok in enumerate(mlm_obj.tokenizer.sampling_vocab)
    }
    new_toks = [mlm_obj.tokenizer.convert_id_to_token(t_idx) for t_idx in new_tok_idxs]
    sampling_vocab_idxs = np.array([
        sampling_vocab_lookup[tok] for tok in new_toks
    ], dtype=int)
    return sampling_vocab_idxs


class UniformMutation(Mutation):
    def __init__(self, tokenizer=None, mlm_obj=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.mlm_obj = mlm_obj

    def _do(self, problem, x, **kwargs):
        query_batches = problem.x_to_query_batches(x)
        batch_shape, num_vars = query_batches.shape[:-1], query_batches.shape[-1]
        flat_queries = query_batches.reshape(-1, num_vars)
        num_samples = flat_queries.shape[0]

        x0 = np.asarray(flat_queries[..., 0], dtype=int)
        seqs = [problem.candidate_pool[i].mutant_residue_seq for i in x0]

        x1 = np.random.randint(problem.xl[1], problem.xu[1], num_samples)
        x1 = np.array([idx % len(seq) for idx, seq in zip(x1, seqs)], dtype=int)

        if self.mlm_obj is None:
            x2 = np.random.randint(0, len(self.tokenizer.sampling_vocab), num_samples)
        else:
            x2 = get_mlm_mutation(self.mlm_obj, problem, x0, x1)

        x3 = np.random.randint(0, len(problem.op_types), num_samples)

        new_queries = np.stack([x0, x1, x2, x3], axis=-1).astype(int).reshape(*batch_shape, -1)
        new_x = problem.query_batches_to_x(new_queries)

        return new_x


class LocalMutation(Mutation):
    def __init__(self, eta, prob, tokenizer=None, mlm_obj=None):
        super().__init__()
        self.poly_mutation = get_mutation('int_pm', eta=eta, prob=prob)
        self.tokenizer = tokenizer
        self.mlm_obj = mlm_obj

    def _do(self, problem, x, **kwargs):
        query_batches = problem.x_to_query_batches(x)
        batch_shape, num_vars = query_batches.shape[:-1], query_batches.shape[-1]
        flat_queries = query_batches.reshape(-1, num_vars)
        num_samples = flat_queries.shape[0]

        x0 = np.asarray(flat_queries[..., 0], dtype=int)

        mut_x = self.poly_mutation._do(problem, x)
        mut_x = problem.x_to_query_batches(mut_x).reshape(-1, num_vars)
        x1 = np.asarray(mut_x[..., 1], dtype=int)

        for i, idx in enumerate(x0):
            num_tokens = len(self.tokenizer.encode(problem.candidate_pool[idx].mutant_residue_seq)) - 2
            x1[i] = min(num_tokens - 1, x1[i])

        if self.mlm_obj is None:
            x2 = np.random.randint(0, len(self.tokenizer.sampling_vocab), num_samples)
        else:
            x2 = get_mlm_mutation(self.mlm_obj, problem, x0, x1)

        x3 = np.random.randint(0, len(problem.op_types), num_samples)

        new_queries = np.stack([x0, x1, x2, x3], axis=-1).astype(int).reshape(*batch_shape, -1)
        new_x = problem.query_batches_to_x(new_queries)

        return new_x
