import math
import random
from collections import namedtuple
import itertools
from typing import Optional

import numpy as np
import torch
from Levenshtein._levenshtein import distance as edit_distance, editops as edit_ops

from scipy.stats import rankdata
from scipy.special import softmax

from cachetools import cached, LRUCache

from evomobo.transforms import padding_collate_fn


AMINO_ACIDS = [
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D",
    "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C",
]

RESIDUE_ALPHABET = (
    ["<cls>", "<pad>", "<eos>", "<unk>"]
    + AMINO_ACIDS
    + ["X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"]
)


class IntTokenizer:
    def __init__(
        self,
        non_special_vocab,
        full_vocab,
        padding_token="<pad>",
        masking_token="<mask>",
        bos_token="<cls>",
        eos_token="<eos>",
    ):
        self.non_special_vocab = non_special_vocab
        self.full_vocab = full_vocab

        self.lookup = {a: i for (i, a) in enumerate(full_vocab)}
        self.inverse_lookup = {i: a for (i, a) in enumerate(full_vocab)}

        self.special_vocab = {
            tok
            for tok in full_vocab
            if tok.startswith("<") or tok in {".", "-"}
        }

        self.padding_idx = self.lookup[padding_token]
        self.masking_idx = self.lookup[masking_token]
        self.bos_idx = self.lookup[bos_token]
        self.eos_idx = self.lookup[eos_token]

        self.sampling_vocab = non_special_vocab
        self.non_special_idxs = [
            self.convert_token_to_id(t) for t in non_special_vocab
        ]
        self.special_idxs = [
            self.convert_token_to_id(t) for t in self.special_vocab
        ]

    @cached(cache=LRUCache(maxsize=int(1e4)))
    def encode(self, seq):
        tokens = ["<cls>"]
        for char in seq:
            tokens.append(char)
        tokens.append("<eos>")
        return [self.convert_token_to_id(tok) for tok in tokens]

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            return self.convert_id_to_token(token_ids)

        tokens = []
        for t_id in token_ids:
            token = self.convert_id_to_token(t_id)
            if token in self.special_vocab and token not in ["<mask>", "<unk>"]:
                continue
            tokens.append(token)
        return "".join(tokens)

    def convert_id_to_token(self, token_id):
        if torch.is_tensor(token_id):
            token_id = token_id.item()
        assert isinstance(token_id, int)
        return self.inverse_lookup.get(token_id, "<unk>")

    def convert_token_to_id(self, token):
        unk_idx = self.lookup["<unk>"]
        return self.lookup.get(token, unk_idx)

    def set_sampling_vocab(self, sampling_vocab=None, max_ngram_size=1):
        if sampling_vocab is None:
            sampling_vocab = []
            for i in range(1, max_ngram_size + 1):
                prod_space = [self.non_special_vocab] * i
                for comb in itertools.product(*prod_space):
                    sampling_vocab.append("".join(comb))
        else:
            new_tokens = set(sampling_vocab) - set(self.full_vocab)
            self.full_vocab.extend(list(new_tokens))
            self.lookup = {a: i for (i, a) in enumerate(self.full_vocab)}
            self.inverse_lookup = {
                i: a for (i, a) in enumerate(self.full_vocab)
            }

        self.sampling_vocab = sampling_vocab


class ResidueTokenizer(IntTokenizer):
    def __init__(self):
        super().__init__(
            non_special_vocab=AMINO_ACIDS,
            full_vocab=RESIDUE_ALPHABET,
        )


class Expression(torch.nn.Module):
    def __init__(self, lambda_fn):
        super().__init__()
        self.fn = lambda_fn

    def forward(self, x):
        return self.fn(x)


def draw_bootstrap(*arrays, bootstrap_ratio=0.632, min_samples=1):
    num_data = arrays[0].shape[0]
    assert all(arr.shape[0] == num_data for arr in arrays)

    if num_data == 0:
        raise ValueError("Cannot draw bootstrap samples from empty arrays.")

    if bootstrap_ratio is None:
        num_samples = min_samples
    else:
        assert bootstrap_ratio < 1
        if num_data == 1:
            num_samples = min_samples
        else:
            num_samples = int(
                math.log(1 - bootstrap_ratio)
                / math.log(1 - 1 / num_data)
            )
            num_samples = max(min_samples, num_samples)

    idxs = random.choices(range(num_data), k=num_samples)
    res = [arr[idxs] for arr in arrays]
    return res


def to_tensor(*arrays, device=torch.device("cpu")):
    tensors = []
    for arr in arrays:
        if isinstance(arr, torch.Tensor):
            tensors.append(arr.to(device))
        else:
            tensors.append(torch.tensor(arr, device=device))
    if len(arrays) == 1:
        return tensors[0]
    return tensors


def batched_call(fn, arg_array, batch_size, *args, **kwargs):
    if batch_size is None or batch_size <= 0:
        batch_size = arg_array.shape[0]

    n = arg_array.shape[0]
    num_batches = max(1, (n + batch_size - 1) // batch_size)

    if isinstance(arg_array, np.ndarray):
        arg_batches = np.array_split(arg_array, num_batches)
    elif isinstance(arg_array, torch.Tensor):
        arg_batches = torch.chunk(arg_array, num_batches, dim=0)
    else:
        raise ValueError("arg_array must be a numpy array or torch tensor.")

    return [fn(batch, *args, **kwargs) for batch in arg_batches]


def mutation_list(src_str, tgt_str, tokenizer):
    src_token_id_seq = "".join(
        [chr(x) for x in tokenizer.encode(src_str)[1:-1]]
    )
    tgt_token_id_seq = "".join(
        [chr(x) for x in tokenizer.encode(tgt_str)[1:-1]]
    )
    if edit_distance(src_token_id_seq, tgt_token_id_seq) == 0:
        return []
    mutations = []
    tmp_tok_id_seq, trans_adj = src_token_id_seq, 0
    for op_name, pos_1, pos_2 in edit_ops(
        src_token_id_seq, tgt_token_id_seq
    ):
        tmp_pos = pos_1 + trans_adj
        if op_name == "delete":
            char_1 = ord(src_token_id_seq[pos_1])
            if pos_1 == len(src_token_id_seq) - 1:
                tmp_tok_id_seq = tmp_tok_id_seq[:-1]
            else:
                tmp_tok_id_seq = (
                    tmp_tok_id_seq[:tmp_pos]
                    + tmp_tok_id_seq[tmp_pos + 1:]
                )
            mutations.append(
                StringDeletion(char_1, tmp_pos, tokenizer)
            )
            trans_adj -= 1
        if op_name == "replace":
            char_1 = ord(src_token_id_seq[pos_1])
            char_2 = ord(tgt_token_id_seq[pos_2])
            if pos_1 == len(src_token_id_seq) - 1:
                tmp_tok_id_seq = tmp_tok_id_seq[:-1] + chr(char_2)
            else:
                tmp_tok_id_seq = (
                    tmp_tok_id_seq[:tmp_pos]
                    + chr(char_2)
                    + tmp_tok_id_seq[tmp_pos + 1:]
                )
            mutations.append(
                StringSubstitution(
                    char_1, tmp_pos, char_2, tokenizer
                )
            )
        if op_name == "insert":
            if pos_1 < len(src_token_id_seq):
                char_1 = ord(src_token_id_seq[pos_1])
            else:
                char_1 = None
            char_2 = ord(tgt_token_id_seq[pos_2])
            if pos_1 == len(src_token_id_seq) - 1:
                tmp_tok_id_seq = (
                    tmp_tok_id_seq[:-1]
                    + chr(char_2)
                    + tmp_tok_id_seq[-1]
                )
            else:
                tmp_tok_id_seq = (
                    tmp_tok_id_seq[:tmp_pos]
                    + chr(char_2)
                    + tmp_tok_id_seq[tmp_pos:]
                )
            mutations.append(
                StringInsertion(
                    char_1, tmp_pos, char_2, tokenizer
                )
            )
            trans_adj += 1
    tmp_str = tokenizer.decode(
        [ord(x) for x in tmp_tok_id_seq]
    ).replace(" ", "")
    assert tmp_tok_id_seq == tgt_token_id_seq
    assert tmp_str == tgt_str, f"{tgt_str}\n{tmp_str}"
    return mutations


class StringSubstitution:
    def __init__(self, old_token_idx, token_pos, new_token_idx, tokenizer):
        self.old_token_idx = int(old_token_idx)
        self.old_token = tokenizer.decode(self.old_token_idx)
        self.token_pos = int(token_pos)
        self.new_token_idx = int(new_token_idx)
        self.new_token = tokenizer.decode(self.new_token_idx)

    def __str__(self):
        prefix = (
            f"{self.token_pos}{self.old_token}-"
            f"{self.token_pos}{self.old_token}_"
        )
        return prefix + f"sub{self.new_token}"


class StringDeletion:
    def __init__(self, old_token_idx, token_pos, tokenizer):
        self.old_token_idx = int(old_token_idx)
        self.old_token = tokenizer.decode(self.old_token_idx)
        self.token_pos = int(token_pos)

    def __str__(self):
        prefix = (
            f"{self.token_pos}{self.old_token_idx}-"
            f"{self.token_pos}{self.old_token_idx}_"
        )
        return prefix + "del"


class StringInsertion:
    def __init__(self, old_token_idx, token_pos, new_token_idx, tokenizer):
        if old_token_idx is None:
            self.old_token_idx = None
            self.old_token = ""
        else:
            self.old_token_idx = int(old_token_idx)
            self.old_token = tokenizer.decode(self.old_token_idx)
        self.token_pos = int(token_pos)
        self.new_token_idx = int(new_token_idx)
        self.new_token = tokenizer.decode(self.new_token_idx)

    def __str__(self):
        prefix = (
            f"{self.token_pos}{self.old_token}-"
            f"{self.token_pos}{self.old_token}_"
        )
        return prefix + f"ins{self.new_token}"


def weighted_resampling(scores, k=1.0, num_samples=None):
    num_rows = scores.shape[0]
    scores = scores.reshape(num_rows, -1)
    ranks = rankdata(scores, method="dense", axis=0)
    ranks = ranks.max(axis=-1)
    weights = softmax(-np.log(ranks) / k)
    num_samples = num_rows if num_samples is None else num_samples
    resampled_idxs = np.random.choice(
        np.arange(num_rows), num_samples, replace=True, p=weights
    )
    return ranks, weights, resampled_idxs


fields = ("inputs", "targets")
defaults = (np.array([], dtype=object), np.array([], dtype=float))
DataSplit = namedtuple("DataSplit", fields, defaults=defaults)


def safe_np_cat(arrays, dtype=None, **kwargs):
    arrays = list(arrays)
    if len(arrays) == 0 or all((getattr(arr, "size", 0) == 0) for arr in arrays):
        return np.array([], dtype=dtype) if dtype is not None else np.array([])

    cat_arrays = [arr for arr in arrays if getattr(arr, "size", 0)]
    if dtype is not None:
        cat_arrays = [np.asarray(a, dtype=dtype) for a in cat_arrays]
    return np.concatenate(cat_arrays, **kwargs)


def update_splits(
    train_split: DataSplit,
    val_split: DataSplit,
    test_split: DataSplit,
    new_split: DataSplit,
    holdout_ratio: float = 0.2,
    *,
    shuffle_new: bool = True,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    filter_existing: bool = True,
    dedupe_new: bool = False,
):
    train_inputs, train_targets = train_split
    val_inputs, val_targets = val_split
    test_inputs, test_targets = test_split
    new_inputs, new_targets = new_split

    train_inputs = np.asarray(train_inputs, dtype=object)
    val_inputs = np.asarray(val_inputs, dtype=object)
    test_inputs = np.asarray(test_inputs, dtype=object)

    train_targets = np.asarray(train_targets, dtype=float)
    val_targets = np.asarray(val_targets, dtype=float)
    test_targets = np.asarray(test_targets, dtype=float)

    new_inputs = np.asarray(new_inputs, dtype=object)
    new_targets = np.asarray(new_targets, dtype=float)

    if new_inputs.size == 0:
        return train_split, val_split, test_split

    if rng is None and seed is not None:
        rng = np.random.default_rng(int(seed))

    def _permute(n: int) -> np.ndarray:
        if rng is not None:
            return rng.permutation(np.arange(n))
        return np.random.permutation(np.arange(n))

    def _binomial(n: int, p: float) -> int:
        if rng is not None:
            return int(rng.binomial(n, p))
        return int(np.random.binomial(n, p))

    if filter_existing:
        existing_inputs = safe_np_cat([train_inputs, val_inputs, test_inputs], dtype=object)
        if existing_inputs.size:
            mask_new = ~np.in1d(new_inputs, existing_inputs)
            new_inputs = new_inputs[mask_new]
            new_targets = new_targets[mask_new]

    if new_inputs.size == 0:
        return train_split, val_split, test_split

    if dedupe_new and new_inputs.size > 1:
        seen = set()
        keep = []
        for i, s in enumerate(new_inputs):
            ss = str(s)
            if ss not in seen:
                seen.add(ss)
                keep.append(i)
        if len(keep) != new_inputs.shape[0]:
            keep = np.asarray(keep, dtype=int)
            new_inputs = new_inputs[keep]
            new_targets = new_targets[keep]

    if shuffle_new and new_inputs.shape[0] > 1:
        perm = _permute(new_inputs.shape[0])
        new_inputs = new_inputs[perm]
        new_targets = new_targets[perm]

    unseen_inputs = safe_np_cat([test_inputs, new_inputs], dtype=object)
    unseen_targets = safe_np_cat([test_targets, new_targets], dtype=float)

    num_rows = int(train_inputs.shape[0] + val_inputs.shape[0] + unseen_inputs.shape[0])

    num_test = min(
        _binomial(num_rows, holdout_ratio / 2.0),
        int(unseen_inputs.shape[0]),
    )
    num_test = (
        max(int(test_inputs.shape[0]), num_test)
        if test_inputs.size
        else max(1, num_test)
    )

    test_split = DataSplit(
        unseen_inputs[:num_test], unseen_targets[:num_test]
    )

    resid_inputs = unseen_inputs[num_test:]
    resid_targets = unseen_targets[num_test:]

    resid_inputs = safe_np_cat([val_inputs, resid_inputs], dtype=object)
    resid_targets = safe_np_cat([val_targets, resid_targets], dtype=float)

    num_val = min(
        _binomial(num_rows, holdout_ratio / 2.0),
        int(resid_inputs.shape[0]),
    )
    num_val = (
        max(int(val_inputs.shape[0]), num_val)
        if val_inputs.size
        else max(1, num_val)
    )

    val_split = DataSplit(
        resid_inputs[:num_val], resid_targets[:num_val]
    )

    last_inputs = resid_inputs[num_val:]
    last_targets = resid_targets[num_val:]

    train_inputs = safe_np_cat([train_inputs, last_inputs], dtype=object)
    train_targets = safe_np_cat([train_targets, last_targets], dtype=float)
    train_split = DataSplit(train_inputs, train_targets)

    return train_split, val_split, test_split


def str_to_tokens(str_array, tokenizer):
    tokens = [
        torch.tensor(tokenizer.encode(x), dtype=torch.long)
        for x in str_array
    ]
    batch = padding_collate_fn(tokens, tokenizer.padding_idx)
    return batch


def tokens_to_str(tok_idx_array, tokenizer):
    if torch.is_tensor(tok_idx_array):
        token_rows = tok_idx_array.detach().cpu().tolist()
    else:
        token_rows = np.asarray(tok_idx_array).tolist()

    str_array = np.array(
        [
            tokenizer.decode(token_ids).replace(" ", "")
            for token_ids in token_rows
        ]
    )
    return str_array


def apply_mutation(base_seq, mut_pos, mut_res, op_type, tokenizer):
    if len(base_seq) == 0:
        raise ValueError("base_seq is empty, cannot apply mutation.")

    mut_pos = mut_pos % len(base_seq)

    if op_type == "sub":
        mut_seq = base_seq[:mut_pos] + mut_res + base_seq[mut_pos + 1:]
    elif op_type == "ins":
        mut_seq = base_seq[:mut_pos] + mut_res + base_seq[mut_pos:]
    elif op_type == "del":
        mut_seq = base_seq[:mut_pos] + base_seq[mut_pos + 1:]
    else:
        raise ValueError(f"unsupported operation: {op_type}")
    return mut_seq


def to_cuda(batch):
    if torch.cuda.is_available():
        return tuple([x.to("cuda") for x in batch])
    else:
        return batch
