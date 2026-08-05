from __future__ import annotations

import datetime
import logging
import os
import shutil
from typing import Dict, Optional

import numpy as np
import pandas as pd

from evomobo.candidate import StringCandidate
from evomobo.tasks.base_task import BaseTask
from evomobo.tasks.opt_task.opt_task import (
    BASE_COLS,
    OBJECTIVE_COLUMN_PATTERN,
    _atomic_write_csv,
    _clean_df,
    _dedupe_keep_first,
    _ensure_expected_cols,
    _length_filter,
    _read_csv_if_exists,
    _require_objectives_present,
    _resolve_project_root,
    _select_seed_rows,
    resolve_sequence_lengths,
    sync_resolved_lengths,
)

logger = logging.getLogger(__name__)

OBJECTIVE_COLS = ("objective0",)
EXPECTED_COLS = BASE_COLS + OBJECTIVE_COLS
SEQUENCE_FILES = (
    "known_data.csv",
    "exam.csv",
    "seed_data.csv",
    "pool.csv",
)


def resolve_single_input_dir(project_root: str, input_dir: str) -> str:
    path = os.path.expanduser(str(input_dir))
    return path if os.path.isabs(path) else os.path.join(project_root, path)


def single_objective_sequence_paths(project_root: str, input_dir: str):
    base = resolve_single_input_dir(project_root, input_dir)
    return [os.path.join(base, name) for name in SEQUENCE_FILES]


def infer_single_objective_columns(path: str) -> tuple[str, ...]:
    """Validate the scalar-objective CSV header without multi-objective rules."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot infer the single objective; CSV not found: {path}"
        )

    columns = pd.read_csv(path, nrows=0).columns.tolist()
    indexed = []
    for col in columns:
        match = OBJECTIVE_COLUMN_PATTERN.fullmatch(str(col))
        if match is not None:
            indexed.append((int(match.group(1)), col))
    inferred = tuple(col for _, col in sorted(indexed))
    if inferred != OBJECTIVE_COLS:
        raise ValueError(
            "Single-objective optimization requires exactly objective0.\n"
            f"path={path}\nfound={list(inferred)}\n"
            f"expected={list(OBJECTIVE_COLS)}"
        )
    return inferred


class SingleObjectiveTask(BaseTask):
    """Semi-automatic scalar-objective protein optimization task."""

    def __init__(
        self,
        tokenizer,
        candidate_pool,
        obj_dim: int = 1,
        objective_direction: str = "min",
        input_dir: str = "evomobo/assets/input_learning",
        num_active_candidates: int = 16,
        num_start_examples: int = 1024,
        seq_max_len: Optional[int] = None,
        max_len: Optional[int] = None,
        seed_sampling_mode: str = "first",
        seed_sampling_seed: int = 0,
        update_known_from_exam: bool = True,
        update_pool_from_exam: bool = True,
        update_seed_from_exam: bool = False,
        clear_exam_after_merge: bool = False,
        unknown_score_policy: str = "error",
        transform=lambda x: x,
        **kwargs,
    ):
        obj_dim = int(obj_dim)
        if obj_dim != 1:
            raise ValueError(
                "SingleObjectiveTask requires obj_dim=1, "
                f"got obj_dim={obj_dim}."
            )

        super().__init__(
            tokenizer,
            candidate_pool,
            obj_dim=obj_dim,
            transform=transform,
            max_len=max_len,
            **kwargs,
        )

        direction = str(objective_direction).strip().lower()
        if direction not in {"min", "max"}:
            raise ValueError("objective_direction must be 'min' or 'max'.")
        self.objective_direction = direction
        self.objective_directions = [direction]
        self.sign_vector = np.array([1.0 if direction == "min" else -1.0])
        self.objective_cols = OBJECTIVE_COLS
        self.expected_cols = EXPECTED_COLS

        self.input_dir = str(input_dir)
        self.num_active_candidates = max(1, int(num_active_candidates))
        self.num_start_examples = max(0, int(num_start_examples))
        self.seq_max_len = None if seq_max_len is None else int(seq_max_len)
        self.seed_sampling_mode = str(seed_sampling_mode)
        self.seed_sampling_seed = int(seed_sampling_seed)
        self.update_known_from_exam = bool(update_known_from_exam)
        self.update_pool_from_exam = bool(update_pool_from_exam)
        self.update_seed_from_exam = bool(update_seed_from_exam)
        self.clear_exam_after_merge = bool(clear_exam_after_merge)

        self.unknown_score_policy = str(unknown_score_policy).strip().lower()
        if self.unknown_score_policy not in {"error", "zero", "nan"}:
            raise ValueError("unknown_score_policy must be error, zero, or nan.")
        self.seq_to_target_map: Dict[str, np.ndarray] = {}

    def _load_evaluated(self, path: str, context: str) -> pd.DataFrame:
        df = _read_csv_if_exists(path)
        if df is None:
            return pd.DataFrame(columns=list(self.expected_cols))
        df = _ensure_expected_cols(df, path, self.expected_cols)
        df = _clean_df(df, self.objective_cols)
        df = _dedupe_keep_first(df, subset="mutant_sequence")
        _require_objectives_present(
            df,
            path,
            context=context,
            obj_cols=self.objective_cols,
        )
        return df

    def task_setup(self, config=None, project_root: Optional[str] = None, *args, **kwargs):
        root = _resolve_project_root(project_root)
        input_dir = resolve_single_input_dir(root, self.input_dir)
        known_path = os.path.join(input_dir, "known_data.csv")
        exam_path = os.path.join(input_dir, "exam.csv")
        seed_path = os.path.join(input_dir, "seed_data.csv")
        pool_path = os.path.join(input_dir, "pool.csv")

        if not os.path.exists(known_path):
            raise FileNotFoundError(f"single-objective known_data.csv not found: {known_path}")

        self.seq_max_len, self.max_len = resolve_sequence_lengths(
            self.seq_max_len,
            self.max_len,
            single_objective_sequence_paths(root, self.input_dir),
        )
        sync_resolved_lengths(config, self.seq_max_len, self.max_len)

        df_known = self._load_evaluated(known_path, "known_data.csv")
        df_all = df_known.copy()

        if os.path.exists(exam_path):
            df_exam = self._load_evaluated(exam_path, "exam.csv")
            df_all = _dedupe_keep_first(
                pd.concat([df_all, df_exam], ignore_index=True),
                subset="mutant_sequence",
            )

            if self.update_pool_from_exam:
                df_pool = self._load_evaluated(pool_path, "pool.csv")
                df_pool = _dedupe_keep_first(
                    pd.concat([df_pool, df_exam], ignore_index=True),
                    subset="mutant_sequence",
                )
                _atomic_write_csv(df_pool, pool_path, self.expected_cols)
            if self.update_known_from_exam:
                _atomic_write_csv(df_all, known_path, self.expected_cols)
            if self.update_seed_from_exam:
                df_seed = self._load_evaluated(seed_path, "seed_data.csv")
                df_seed = _dedupe_keep_first(
                    pd.concat([df_seed, df_exam], ignore_index=True),
                    subset="mutant_sequence",
                )
                _atomic_write_csv(df_seed, seed_path, self.expected_cols)
            if self.clear_exam_after_merge:
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                archive = os.path.join(input_dir, f"exam_archived_{stamp}.csv")
                shutil.copyfile(exam_path, archive)
                os.remove(exam_path)

        df_all = _length_filter(df_all, self.seq_max_len)
        if len(df_all) == 0:
            raise ValueError("No evaluated sequences satisfy task.seq_max_len.")

        targets_sign = (
            df_all.loc[:, list(self.objective_cols)].to_numpy(dtype=float)
            * self.sign_vector.reshape(1, -1)
        )
        active_count = min(self.num_active_candidates, len(df_all))
        active_indices = np.argsort(targets_sign[:, 0], kind="stable")[:active_count]
        df_active = df_all.iloc[active_indices].reset_index(drop=True)
        active_targets = targets_sign[active_indices]
        active_candidates = np.asarray(
            [
                StringCandidate(
                    wild_seq=row["mutant_sequence"],
                    mutation_list=[],
                    tokenizer=self.tokenizer,
                    wild_name=row["Name"],
                )
                for _, row in df_active.iterrows()
            ],
            dtype=object,
        )

        all_seqs = df_all["mutant_sequence"].to_numpy(dtype=object)
        all_targets = targets_sign

        df_seed = self._load_evaluated(seed_path, "seed_data.csv")
        if len(df_seed) > 0 and self.num_start_examples > 0:
            df_seed = _select_seed_rows(
                df_seed,
                min(self.num_start_examples, len(df_seed)),
                self.seed_sampling_mode,
                self.seed_sampling_seed,
            )
            df_seed = _length_filter(df_seed, self.seq_max_len)
            seed_targets = (
                df_seed.loc[:, list(self.objective_cols)].to_numpy(dtype=float)
                * self.sign_vector.reshape(1, -1)
            )
            all_seqs = np.concatenate(
                [all_seqs, df_seed["mutant_sequence"].to_numpy(dtype=object)]
            )
            all_targets = np.concatenate([all_targets, seed_targets], axis=0)

        _, unique_indices = np.unique(all_seqs, return_index=True)
        unique_indices = np.sort(unique_indices)
        all_seqs = all_seqs[unique_indices]
        all_targets = all_targets[unique_indices]
        self.seq_to_target_map = {
            str(seq): np.asarray(target, dtype=float)
            for seq, target in zip(all_seqs, all_targets)
        }

        logger.info(
            "single-objective setup done. measured=%d, active_parents=%d, train_all=%d",
            len(df_all),
            len(active_candidates),
            len(all_seqs),
        )
        return active_candidates, active_targets, all_seqs, all_targets

    def score(self, candidates):
        values = []
        for candidate in candidates:
            seq = str(candidate.mutant_residue_seq)
            target = self.seq_to_target_map.get(seq)
            if target is not None:
                values.append(target)
            elif self.unknown_score_policy == "error":
                raise KeyError(f"Unknown sequence encountered in score(): {seq[:60]}...")
            elif self.unknown_score_policy == "zero":
                values.append(np.zeros(1, dtype=float))
            else:
                values.append(np.full(1, np.nan, dtype=float))
        return np.stack(values, axis=0)
