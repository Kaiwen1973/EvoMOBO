import uuid

from evomobo.utils import StringSubstitution, StringDeletion, StringInsertion


class StringCandidate:
    def __init__(
        self,
        wild_seq,
        mutation_list,
        tokenizer,
        wild_name=None,
        dist_from_wild: float = 0.0,
    ):
        self.wild_residue_seq = wild_seq
        self.uuid = uuid.uuid4().hex
        self.wild_name = "unnamed" if wild_name is None else wild_name
        self.tokenizer = tokenizer
        self.mutant_residue_seq = self.apply_mutations(mutation_list, tokenizer)
        self.dist_from_wild = dist_from_wild

    def __len__(self):
        tok_idxs = self.tokenizer.encode(self.mutant_residue_seq)
        return len(tok_idxs)

    def apply_mutations(self, mutation_list, tokenizer):
        if len(mutation_list) == 0:
            return self.wild_residue_seq

        token_ids = tokenizer.encode(self.wild_residue_seq)[1:-1]

        for mutation_op in mutation_list:
            old_tok_idx = mutation_op.old_token_idx
            mut_pos = mutation_op.token_pos

            if mut_pos < len(token_ids):
                assert (
                    old_tok_idx == token_ids[mut_pos]
                ), f"Token mismatch at pos {mut_pos}: {old_tok_idx} != {token_ids[mut_pos]} ({mutation_op})"

            if isinstance(mutation_op, StringSubstitution):
                new_tok_idx = mutation_op.new_token_idx
                token_ids = (
                    token_ids[:mut_pos] + [new_tok_idx] + token_ids[mut_pos + 1 :]
                )
            elif isinstance(mutation_op, StringDeletion):
                token_ids = token_ids[:mut_pos] + token_ids[mut_pos + 1 :]
            elif isinstance(mutation_op, StringInsertion):
                new_tok_idx = mutation_op.new_token_idx
                token_ids = (
                    token_ids[:mut_pos] + [new_tok_idx] + token_ids[mut_pos:]
                )
            else:
                raise RuntimeError(f"unrecognized mutation op: {type(mutation_op)}")

        mutant_seq = tokenizer.decode(token_ids)
        return mutant_seq

    def new_candidate(self, mutation_list, tokenizer):
        cand_kwargs = dict(
            wild_seq=self.mutant_residue_seq,
            mutation_list=mutation_list,
            tokenizer=tokenizer,
            wild_name=self.wild_name,
            dist_from_wild=self.dist_from_wild + len(mutation_list),
        )
        return StringCandidate(**cand_kwargs)
