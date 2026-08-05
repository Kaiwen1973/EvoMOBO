from __future__ import annotations

import csv
import os
import random
import time

import hydra
import numpy as np
import pandas as pd
import torch

from evomobo.models.lanmt import corrupt_tok_idxs
from evomobo.models.mlm import evaluate_windows, sample_tokens
from evomobo.models.shared_elements import check_early_stopping
from evomobo.optimizers.evomobo import EvoMOBO
from evomobo.rng_utils import isolated_seed
from evomobo.utils import (
    DataSplit,
    str_to_tokens,
    tokens_to_str,
    update_splits,
    weighted_resampling,
)


class SingleObjectiveEvoMOBO(EvoMOBO):
    """Scalar-objective EvoMOBO using q-EI and no hypervolume machinery."""

    def __init__(
        self,
        *args,
        output_dir: str = "evomobo/assets/active_learning",
        offline_selection_mode: str = "rank1_batch_only",
        **kwargs,
    ):
        mode = str(offline_selection_mode).strip().lower()
        if mode not in {
            "batch_ei",
            "greedy_mean_ei",
            "rank1_batch_only",
        }:
            raise ValueError(
                "single-objective offline_selection_mode must be 'batch_ei', "
                "'greedy_mean_ei', or 'rank1_batch_only'; "
                f"got {offline_selection_mode!r}."
            )
        self.single_output_dir = str(output_dir)
        parent_mode = {
            "rank1_batch_only": "rank1_batch_only",
            "batch_ei": "ranked_batch",
            "greedy_mean_ei": "greedy_mean_hvi",
        }[mode]
        super().__init__(
            *args,
            offline_selection_mode=parent_mode,
            use_pool_csv_points=False,
            use_pareto_extra_csv_points=False,
            **kwargs,
        )
        self.offline_selection_mode = mode

    @staticmethod
    def _load_wild_type(path: str) -> str:
        if not os.path.exists(path):
            return ""
        with open(path, "r") as handle:
            return "".join(
                line.strip() for line in handle if not line.startswith(">")
            )

    @staticmethod
    def _mutation_name(mutant: str, wild_type: str) -> str:
        if not mutant or not wild_type:
            return "NA"
        mutations = [
            f"{wt}{idx + 1}{mut}"
            for idx, (wt, mut) in enumerate(zip(wild_type, mutant))
            if wt != mut
        ]
        return "WT" if not mutations else ",".join(mutations)

    def _round_output_dir(self, project_root: str) -> str:
        base = os.path.expanduser(self.single_output_dir)
        if not os.path.isabs(base):
            base = os.path.join(project_root, base)
        return self._begin_round_output(base)

    def _print_split(self, tag: str, split: DataSplit, wild_type: str, sign: float):
        if not self.print_splits:
            return
        inputs = np.asarray(split.inputs, dtype=object)
        targets = np.asarray(split.targets, dtype=float).reshape(-1, 1)
        self._log(f"\n[{tag}] N={len(inputs)}")
        limit = len(inputs) if self.print_splits_max_rows == 0 else min(
            len(inputs), self.print_splits_max_rows
        )
        for seq, target in zip(inputs[:limit], targets[:limit, 0]):
            seq = str(seq)
            self._log(
                f"{self._mutation_name(seq, wild_type)},{seq},{float(target * sign):.4f}"
            )

    def _optimize_candidate_batches(self, acq_fn, batch_size: int):
        if self.resampling_weight is None:
            weights = np.ones(len(self.active_targets), dtype=float)
            weights /= weights.sum()
        else:
            _, weights, _ = weighted_resampling(
                self.active_targets,
                k=self.resampling_weight,
            )

        base_cand_batches = []
        new_seq_batches = []
        new_seq_scores = []
        batch_entropy = []

        for _ in range(self.num_gens):
            base_idxs = np.random.choice(
                np.arange(len(self.active_candidates)),
                batch_size,
                p=weights,
                replace=True,
            )
            base_candidates = self.active_candidates[base_idxs]
            base_seqs = np.asarray(
                [candidate.mutant_residue_seq for candidate in base_candidates],
                dtype=object,
            )
            base_tok_idxs = str_to_tokens(base_seqs, self.encoder.tokenizer)
            base_mask = base_tok_idxs != self.encoder.tokenizer.padding_idx
            base_lens = base_mask.float().sum(-1).long()
            tgt_lens = None if self.bb_task.allow_len_change else base_lens

            window_idxs = self._enumerate_mutation_windows(base_seqs)
            if self.position_sampler == "entropy_method":
                with torch.no_grad():
                    window_idxs, window_entropy = evaluate_windows(
                        base_seqs,
                        self.encoder,
                        self.window_size,
                        replacement=True,
                        encoder_obj=self.encoder_obj,
                        window_mask_idxs=window_idxs,
                    )
                mask_idxs = self.sample_mutation_window(window_idxs, window_entropy)
            elif self.position_sampler == "uniform":
                mask_idxs = np.concatenate(
                    [random.choice(indices) for indices in window_idxs.values()]
                )
            else:
                raise ValueError(f"Unknown position_sampler: {self.position_sampler}")

            with torch.no_grad():
                device = self.surrogate_model.device
                src_tok_idxs = base_tok_idxs.clone().to(device)
                if self.latent_init == "perturb_pareto":
                    opt_features, src_mask = self.encoder.get_token_features(src_tok_idxs)
                    opt_features += 1e-3 * torch.randn_like(opt_features)
                elif self.encoder_obj == "lanmt":
                    src_tok_idxs = corrupt_tok_idxs(
                        src_tok_idxs,
                        self.encoder.tokenizer,
                        max_len_delta=None,
                        select_idxs=mask_idxs,
                    )
                    opt_features, src_mask = self.encoder.get_token_features(src_tok_idxs)
                elif self.encoder_obj == "mlm":
                    mask_idxs_t = torch.from_numpy(mask_idxs).to(
                        device=device,
                        dtype=torch.long,
                    )
                    src_tok_idxs_masked = src_tok_idxs.clone()
                    src_tok_idxs_masked.scatter_(
                        1,
                        mask_idxs_t,
                        self.encoder.tokenizer.masking_idx,
                    )
                    src_tok_features, src_mask = self.encoder.get_token_features(
                        src_tok_idxs_masked
                    )
                    feature_dim = src_tok_features.size(-1)
                    gather_idx = mask_idxs_t.unsqueeze(-1).expand(
                        -1,
                        -1,
                        feature_dim,
                    )
                    opt_features = src_tok_features.gather(1, gather_idx)
                else:
                    raise ValueError(f"Unknown encoder_obj: {self.encoder_obj}")

                opt_params = opt_features.clone().detach()
                opt_origin = opt_features.clone().detach()
                opt_params.requires_grad_(self.optimize_latent)

            optimizer = torch.optim.Adam(
                [opt_params],
                lr=self.lr,
                betas=(0.0, 1e-2),
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=self.patience,
            )
            best_score, best_step = None, 0
            best_seqs = base_seqs
            best_entropy = float("nan")

            for step_idx in range(self.num_opt_steps):
                opt_params.grad = None
                if self.encoder_obj == "lanmt":
                    latent_features, pooled_features = self.encoder.pool_features(
                        opt_params,
                        src_mask,
                        src_tok_idxs,
                    )
                    logits, tgt_mask = self.encoder.logits_from_features(
                        opt_params,
                        src_mask,
                        latent_features,
                        tgt_lens,
                    )
                    tgt_tok_idxs, entropy = self.encoder.sample_tgt_tok_idxs(
                        logits,
                        tgt_mask,
                        temp=1.0,
                    )
                    entropy_term = entropy[tgt_mask.bool()].mean()
                else:
                    device = self.surrogate_model.device
                    mask_idxs_t = torch.from_numpy(mask_idxs).to(
                        device=device,
                        dtype=torch.long,
                    )
                    current_features = src_tok_features.clone()
                    feature_dim = current_features.size(-1)
                    scatter_idx = mask_idxs_t.unsqueeze(-1).expand(
                        -1,
                        -1,
                        feature_dim,
                    )
                    current_features = current_features.scatter(
                        1,
                        scatter_idx,
                        opt_params,
                    )
                    latent_features, pooled_features = self.encoder.pool_features(
                        current_features,
                        src_mask,
                        src_tok_idxs_masked,
                    )
                    logits, tgt_mask = self.encoder.logits_from_features(
                        current_features,
                        src_mask,
                        latent_features,
                        tgt_lens,
                    )
                    sampled_tokens, entropy_full = sample_tokens(
                        base_tok_idxs.to(device),
                        logits,
                        self.encoder.tokenizer,
                        replacement=False,
                    )
                    sampled_mutations = sampled_tokens.gather(
                        1,
                        mask_idxs_t,
                    )
                    tgt_tok_idxs = src_tok_idxs.clone()
                    tgt_tok_idxs.scatter_(
                        1,
                        mask_idxs_t,
                        sampled_mutations,
                    )
                    entropy_mut = entropy_full.gather(1, mask_idxs_t)
                    entropy_term = entropy_mut.mean()

                latent_ei = acq_fn(pooled_features.unsqueeze(0))
                loss = -latent_ei.mean() + self.entropy_penalty * entropy_term
                if self.optimize_latent:
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        delta = opt_params - opt_origin
                        delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                        scale = torch.clamp(
                            self.latent_trust_radius / delta_norm,
                            max=1.0,
                        )
                        opt_params.copy_(opt_origin + delta * scale)
                    scheduler.step(loss)

                candidate_seqs = tokens_to_str(
                    tgt_tok_idxs,
                    self.encoder.tokenizer,
                )
                with torch.no_grad():
                    actual_ei = acq_fn(candidate_seqs[None, :]).mean().item()
                best_score, best_step, _, stop = check_early_stopping(
                    model=None,
                    best_score=best_score,
                    best_epoch=best_step,
                    best_weights=None,
                    curr_score=-actual_ei,
                    curr_epoch=step_idx + 1,
                    patience=self.patience,
                    save_weights=False,
                )
                if best_step == step_idx + 1:
                    best_seqs = candidate_seqs.copy()
                    best_entropy = float(entropy_term.item())
                if stop:
                    self._log(f"Early stopping at step {step_idx + 1}")
                    break

            base_cand_batches.append(base_candidates.copy())
            new_seq_batches.append(best_seqs.copy())
            new_seq_scores.append(float(best_score))
            batch_entropy.append(best_entropy)

        return (
            base_cand_batches,
            np.stack(new_seq_batches),
            np.asarray(new_seq_scores, dtype=float),
            np.asarray(batch_entropy, dtype=float),
        )

    @staticmethod
    def _rank_single_candidates(candidate_means, candidate_meta, best_f: float):
        candidate_means = np.asarray(candidate_means, dtype=float).reshape(-1)
        improvements = np.maximum(candidate_means - float(best_f), 0.0)
        order = sorted(
            range(candidate_means.shape[0]),
            key=lambda idx: (
                improvements[idx],
                candidate_means[idx],
                -int(candidate_meta[idx]["source_batch_rank"]),
                -idx,
            ),
            reverse=True,
        )
        return np.asarray(order, dtype=int), candidate_means, improvements

    def _collect_single_objective_suggestions(
        self,
        base_cand_batches,
        new_seq_batches,
        new_seq_scores,
        batch_entropy,
        all_seqs,
        baseline_targets,
        batch_size: int,
    ):
        candidate_seqs, candidate_meta = (
            self._collect_novel_feasible_candidate_pool(
                base_cand_batches=base_cand_batches,
                new_seq_batches=new_seq_batches,
                new_seq_scores=new_seq_scores,
                batch_entropy=batch_entropy,
                all_seqs=all_seqs,
            )
        )
        if candidate_seqs.size == 0:
            return candidate_seqs, []

        desired_n = (
            batch_size
            if self.offline_num_suggestions is None
            else int(self.offline_num_suggestions)
        )
        desired_n = max(1, min(desired_n, candidate_seqs.shape[0]))
        rank1_indices = [
            idx
            for idx, row in enumerate(candidate_meta)
            if int(row["source_batch_rank"]) == 1
        ]

        if self.offline_selection_mode == "rank1_batch_only":
            selected_indices = rank1_indices[: int(batch_size)]
            selected_meta = [
                {
                    **candidate_meta[idx],
                    "selection_mode": "rank1_batch_only",
                    "selection_rank": rank + 1,
                }
                for rank, idx in enumerate(selected_indices)
            ]
            self._log(
                "[offline] rank1_batch_only "
                f"requested_q={batch_size}, selected={len(selected_indices)}, "
                "backfill=disabled"
            )
            return (
                candidate_seqs[np.asarray(selected_indices, dtype=int)],
                selected_meta,
            )

        self.encoder.eval()
        self.surrogate_model.eval()
        with torch.inference_mode():
            posterior = self.surrogate_model.posterior(candidate_seqs)
        candidate_means = (
            posterior.mean.detach().cpu().numpy().reshape(-1)
        )
        baseline_targets = np.asarray(baseline_targets, dtype=float).reshape(-1)
        best_f = float(baseline_targets.max())
        ranked_indices, candidate_means, improvements = (
            self._rank_single_candidates(
                candidate_means,
                candidate_meta,
                best_f,
            )
        )

        if self.offline_selection_mode == "batch_ei":
            selected_indices = rank1_indices[:desired_n]
            selected_set = set(selected_indices)
            if (
                len(selected_indices) < desired_n
                and self.offline_backfill_from_ranked_batches
            ):
                for idx in ranked_indices:
                    idx = int(idx)
                    if idx in selected_set:
                        continue
                    if int(candidate_meta[idx]["source_batch_rank"]) == 1:
                        continue
                    selected_indices.append(idx)
                    selected_set.add(idx)
                    if len(selected_indices) >= desired_n:
                        break
            num_rank1 = sum(
                int(candidate_meta[idx]["source_batch_rank"]) == 1
                for idx in selected_indices
            )
            self._log(
                "[offline] batch_ei "
                f"rank1={num_rank1}, "
                f"mean_ei_backfill={len(selected_indices) - num_rank1}, "
                f"selected={len(selected_indices)}"
            )
        elif self.offline_selection_mode == "greedy_mean_ei":
            selected_indices = [
                int(idx) for idx in ranked_indices[:desired_n]
            ]
            num_positive = sum(
                improvements[idx] > 0.0 for idx in selected_indices
            )
            self._log(
                "[offline] greedy_mean_ei "
                f"candidate_pool={candidate_seqs.shape[0]}, "
                f"selected={len(selected_indices)}, "
                f"positive_mean_improvement={num_positive}"
            )
        else:
            raise ValueError(
                "Unsupported single-objective selection mode: "
                f"{self.offline_selection_mode!r}."
            )

        selected_meta = []
        for selection_rank, idx in enumerate(selected_indices, start=1):
            row = dict(candidate_meta[idx])
            row.update(
                {
                    "selection_mode": self.offline_selection_mode,
                    "selection_rank": selection_rank,
                    "pred_mean_max_space": float(candidate_means[idx]),
                    "pred_mean_improvement": float(improvements[idx]),
                }
            )
            selected_meta.append(row)

        return (
            candidate_seqs[np.asarray(selected_indices, dtype=int)],
            selected_meta,
        )

    def optimize(self, candidate_pool, pool_targets, all_seqs, all_targets, log_prefix=""):
        if int(self.bb_task.obj_dim) != 1:
            raise ValueError("SingleObjectiveEvoMOBO requires bb_task.obj_dim == 1.")

        project_root = self._resolve_project_root()
        round_out_dir = self._round_output_dir(project_root)
        input_dir = os.path.expanduser(self.bb_task.input_dir)
        if not os.path.isabs(input_dir):
            input_dir = os.path.join(project_root, input_dir)
        wild_type = self._load_wild_type(os.path.join(input_dir, "wt.fasta"))

        sign = float(self.bb_task.sign_vector[0])
        observed_seqs = np.asarray(all_seqs, dtype=object).copy()
        observed_targets = np.asarray(all_targets, dtype=float).reshape(-1, 1).copy()
        self.active_candidates = np.asarray(candidate_pool, dtype=object).copy()
        self.active_targets = np.asarray(pool_targets, dtype=float).reshape(-1, 1).copy()
        self.active_seqs = np.asarray(
            [candidate.mutant_residue_seq for candidate in self.active_candidates],
            dtype=object,
        )
        if len(self.active_candidates) == 0:
            raise ValueError("Single-objective task produced no active mutation parents.")

        batch_size = int(self.bb_task.batch_size)
        start_time = time.time()
        best_internal = float(observed_targets[:, 0].min())
        best_observed_real = best_internal * sign
        metrics = {
            "round_idx": 1,
            "best_observed": best_observed_real,
            "num_bb_evals": 0,
        }

        active_rows = []
        for seq, target in zip(self.active_seqs, self.active_targets[:, 0]):
            seq = str(seq)
            active_rows.append(
                {
                    "Name": self._mutation_name(seq, wild_type),
                    "mutant_sequence": seq,
                    "objective0": float(target * sign),
                }
            )
        self._write_csv(
            pd.DataFrame(active_rows),
            os.path.join(round_out_dir, "active_set.csv"),
            float_format="%.6f",
            quoting=csv.QUOTE_MINIMAL,
        )

        mean = observed_targets.mean(axis=0)
        std = observed_targets.std(axis=0)
        std = np.where(np.abs(std) < 1e-12, 1.0, std)

        def target_transform(values):
            return -(np.asarray(values, dtype=float).reshape(-1, 1) - mean) / std

        split = DataSplit(observed_seqs, observed_targets)
        self.train_split, self.val_split, self.test_split = update_splits(
            self.train_split,
            self.val_split,
            self.test_split,
            split,
            self.surrogate_model.holdout_ratio,
        )
        self._print_split("TRAIN", self.train_split, wild_type, sign)
        self._print_split("VAL", self.val_split, wild_type, sign)
        self._print_split("TEST", self.test_split, wild_type, sign)

        x_train = self.train_split.inputs
        y_train = target_transform(self.train_split.targets)
        x_val = self.val_split.inputs
        y_val = target_transform(self.val_split.targets)
        x_test = self.test_split.inputs
        y_test = target_transform(self.test_split.targets)
        with isolated_seed(self.surrogate_training_seed):
            records = self.surrogate_model.fit(
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                encoder_obj=self.encoder_obj,
                resampling_temp=None,
                csv_dir=round_out_dir,
            )
        last_entry = {str(key).split("/")[-1]: value for key, value in records[-1].items()}
        try:
            best_epoch = int(last_entry.get("best_epoch", 0))
        except Exception:
            best_epoch = 0
        best_epoch = max(0, min(best_epoch, len(records) - 1))
        best_entry = {
            str(key).split("/")[-1]: value
            for key, value in records[best_epoch].items()
        }
        self._log(self._format_df(pd.DataFrame([best_entry]), floatfmt=".4f"))
        metrics.update(
            {
                "test_rmse": best_entry.get("test_rmse", np.nan),
                "test_nll": best_entry.get("test_nll", np.nan),
                "test_s_rho": best_entry.get("test_s_rho", np.nan),
                "test_ece": best_entry.get("test_ece", np.nan),
                "test_post_var": best_entry.get("test_post_var", np.nan),
                "num_train": int(len(x_train)),
            }
        )

        transformed_observed_targets = target_transform(observed_targets)
        acq_fn = hydra.utils.instantiate(
            self.acquisition,
            known_targets=torch.as_tensor(
                transformed_observed_targets,
                dtype=torch.float32,
                device=self.surrogate_model.device,
            ),
            surrogate=self.surrogate_model,
        )
        self._log("\n---- optimizing single-objective candidates with q-EI ----")
        (
            base_cand_batches,
            new_seq_batches,
            new_seq_scores,
            batch_entropy,
        ) = self._optimize_candidate_batches(acq_fn, batch_size)

        order = np.argsort(new_seq_scores)
        for rank, batch_idx in enumerate(order, start=1):
            seqs = np.asarray(new_seq_batches[batch_idx], dtype=object)
            posterior = self.surrogate_model.posterior(seqs)
            pred_max_space = posterior.mean.detach().cpu().numpy().reshape(-1, 1)
            pred_sign_space = mean - pred_max_space * std
            pred_real = pred_sign_space * sign
            rank_data = {
                "Name": [self._mutation_name(str(seq), wild_type) for seq in seqs],
                "mutant_sequence": [str(seq) for seq in seqs],
                "pred_objective0": pred_real[:, 0],
            }
            self._write_csv(
                pd.DataFrame(rank_data),
                os.path.join(round_out_dir, f"rank{rank}.csv"),
                float_format="%.6f",
            )

        suggested_seqs, metadata = self._collect_single_objective_suggestions(
            base_cand_batches=base_cand_batches,
            new_seq_batches=new_seq_batches,
            new_seq_scores=new_seq_scores,
            batch_entropy=batch_entropy,
            all_seqs=observed_seqs,
            baseline_targets=transformed_observed_targets,
            batch_size=batch_size,
        )
        if suggested_seqs.size == 0:
            raise RuntimeError("No novel feasible single-objective suggestions were found.")

        posterior = self.surrogate_model.posterior(suggested_seqs)
        pred_max_space = posterior.mean.detach().cpu().numpy().reshape(-1, 1)
        pred_std_max_space = posterior.variance.sqrt().detach().cpu().numpy().reshape(-1, 1)
        pred_sign_space = mean - pred_max_space * std
        pred_real = pred_sign_space * sign
        pred_std_real = pred_std_max_space * std
        names = [self._mutation_name(str(seq), wild_type) for seq in suggested_seqs]
        df_out = pd.DataFrame(
            {
                "Name": names,
                "mutant_sequence": [str(seq) for seq in suggested_seqs],
                "pred_objective0": pred_real[:, 0],
                "pred_std_objective0": pred_std_real[:, 0],
            }
        )
        if metadata and len(metadata) == len(df_out):
            for key in metadata[0]:
                df_out[key] = [row.get(key) for row in metadata]
        self._write_csv(
            df_out,
            os.path.join(round_out_dir, self.offline_suggest_csv),
            float_format="%.6f",
        )
        self._write_csv(
            pd.DataFrame(
                {
                    "Name": names,
                    "mutant_sequence": [str(seq) for seq in suggested_seqs],
                    "objective0": np.nan,
                }
            ),
            os.path.join(round_out_dir, self.offline_exam_template_csv),
        )

        metrics["best_predicted"] = float(
            pred_real[:, 0].max()
            if self.bb_task.objective_direction == "max"
            else pred_real[:, 0].min()
        )
        metrics["acq_val"] = float(-new_seq_scores[order[0]])
        metrics["time_elapsed"] = float(time.time() - start_time)
        final_round_dir = self._commit_round_output(
            round_out_dir,
            best_epoch=best_epoch,
            extra_meta={
                "select_crit_key": "val_nll",
                "offline_mode": bool(self.offline_mode),
                "offline_selection_mode": self.offline_selection_mode,
                "single_objective": True,
            },
        )
        self._log(f"[offline] Suggestions saved to: {final_round_dir}")
        return metrics
