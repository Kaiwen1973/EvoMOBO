from __future__ import annotations

import csv
import datetime
import logging
import os
import re
import shutil
from typing import Dict, List, Optional

import hydra
import numpy as np
import pandas as pd
import torch
from botorch.utils.multi_objective import pareto
from omegaconf import DictConfig, open_dict, read_write

from evomobo.candidate import StringCandidate
from evomobo.tasks.base_task import BaseTask

logger = logging.getLogger(__name__)

BASE_COLS = ("Name", "mutant_sequence")
OBJ_COLS = ("objective0", "objective1")
EXPECTED_COLS = BASE_COLS + OBJ_COLS
OBJECTIVE_COLUMN_PATTERN = re.compile(r"^objective(\d+)$")
SEQUENCE_LENGTH_SOURCE_FILES = (
    "known_data.csv",
    "exam.csv",
    "seed_data.csv",
    "pool.csv",
    "par_cand_extra.csv",
)



def _resolve_project_root(project_root: Optional[str]) -> str:
    if project_root is not None:
        return project_root
    try:
        return hydra.utils.get_original_cwd()
    except Exception:
        return os.getcwd()


def _read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def objective_columns(obj_dim: int) -> tuple[str, ...]:
    obj_dim = int(obj_dim)
    if obj_dim < 2:
        raise ValueError(f"obj_dim must be at least 2, got {obj_dim}")
    return tuple(f"objective{idx}" for idx in range(obj_dim))


def infer_objective_columns(path: str) -> tuple[str, ...]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot infer objectives; CSV not found: {path}")

    columns = pd.read_csv(path, nrows=0).columns.tolist()
    indexed = sorted(
        (int(match.group(1)), col)
        for col in columns
        if (match := OBJECTIVE_COLUMN_PATTERN.fullmatch(str(col))) is not None
    )
    if not indexed:
        raise ValueError(
            f"No objective columns found in {path}; expected objective0, objective1, ..."
        )

    inferred = tuple(col for _, col in indexed)
    expected = tuple(f"objective{idx}" for idx in range(len(indexed)))
    if inferred != expected:
        raise ValueError(
            "Objective columns must be contiguous and start at objective0.\n"
            f"path={path}\nfound={list(inferred)}\nexpected={list(expected)}"
        )
    if len(inferred) < 2:
        raise ValueError(
            f"Multi-objective optimization requires at least 2 objectives, found {len(inferred)}"
        )
    return inferred


def input_learning_sequence_paths(project_root: Optional[str] = None) -> List[str]:
    root = _resolve_project_root(project_root)
    fp_dir = os.path.join(root, "evomobo", "assets", "input_learning")
    return [os.path.join(fp_dir, name) for name in SEQUENCE_LENGTH_SOURCE_FILES]


def _ensure_expected_cols(
    df: pd.DataFrame,
    path: str,
    expected_cols=EXPECTED_COLS,
) -> pd.DataFrame:
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing columns {missing}:\n"
            f"path={path}\n"
            f"expected={list(expected_cols)}\n"
            f"got={list(df.columns)}"
        )

    expected_objectives = tuple(
        col for col in expected_cols if OBJECTIVE_COLUMN_PATTERN.fullmatch(str(col))
    )
    actual_indexed = []
    for col in df.columns:
        match = OBJECTIVE_COLUMN_PATTERN.fullmatch(str(col))
        if match is not None:
            actual_indexed.append((int(match.group(1)), col))
    actual_objectives = tuple(col for _, col in sorted(actual_indexed))
    if actual_objectives != expected_objectives:
        raise ValueError(
            "CSV objective columns do not match the configured campaign.\n"
            f"path={path}\nexpected={list(expected_objectives)}\n"
            f"got={list(actual_objectives)}"
        )
    return df


def _clean_df(df: pd.DataFrame, obj_cols=OBJ_COLS) -> pd.DataFrame:
    df = df.copy()

    df["mutant_sequence"] = df["mutant_sequence"].fillna("").astype(str).str.strip()
    df["Name"] = df["Name"].fillna("").astype(str).str.strip()

    for col in obj_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["mutant_sequence"].str.len() > 0].reset_index(drop=True)
    return df


def _dedupe_keep_first(df: pd.DataFrame, subset: str = "mutant_sequence") -> pd.DataFrame:
    return df.drop_duplicates(subset=[subset], keep="first").reset_index(drop=True)


def _length_filter(df: pd.DataFrame, len_thr: Optional[int]) -> pd.DataFrame:
    if len_thr is None:
        return df
    return df[df["mutant_sequence"].str.len() <= int(len_thr)].reset_index(drop=True)


def infer_seq_max_len(paths: List[str]) -> int:
    max_len = 0
    found_sequences = False

    for path in paths:
        df = _read_csv_if_exists(path)
        if df is None or "mutant_sequence" not in df.columns:
            continue

        seqs = df["mutant_sequence"].dropna().astype(str).str.strip()
        seqs = seqs[seqs.str.len() > 0]
        if len(seqs) == 0:
            continue

        found_sequences = True
        max_len = max(max_len, int(seqs.str.len().max()))

    if not found_sequences:
        raise ValueError(
            "Cannot infer task.seq_max_len: no non-empty mutant_sequence values "
            "were found in the input_learning CSV files."
        )
    return max_len


def resolve_sequence_lengths(
    seq_max_len: Optional[int],
    max_len: Optional[int],
    sequence_paths: List[str],
) -> tuple[int, int]:
    resolved_seq_max_len = (
        infer_seq_max_len(sequence_paths)
        if seq_max_len is None
        else int(seq_max_len)
    )
    resolved_max_len = (
        resolved_seq_max_len + 2
        if max_len is None
        else int(max_len)
    )

    if resolved_seq_max_len <= 0:
        raise ValueError(
            f"task.seq_max_len must be positive, got {resolved_seq_max_len}"
        )
    if resolved_max_len < resolved_seq_max_len + 2:
        raise ValueError(
            "task.max_len must be at least task.seq_max_len + 2 to accommodate "
            f"<cls>/<eos>; got max_len={resolved_max_len}, "
            f"seq_max_len={resolved_seq_max_len}."
        )

    return resolved_seq_max_len, resolved_max_len


def sync_resolved_lengths(config, seq_max_len: int, max_len: int) -> None:
    if config is None:
        return

    task_cfg = (
        config.get("task")
        if isinstance(config, dict)
        else getattr(config, "task", None)
    )
    if task_cfg is not None:
        if isinstance(task_cfg, DictConfig):
            with read_write(task_cfg), open_dict(task_cfg):
                task_cfg.seq_max_len = int(seq_max_len)
                task_cfg.max_len = int(max_len)
        elif isinstance(task_cfg, dict):
            task_cfg["seq_max_len"] = int(seq_max_len)
            task_cfg["max_len"] = int(max_len)
        else:
            setattr(task_cfg, "seq_max_len", int(seq_max_len))
            setattr(task_cfg, "max_len", int(max_len))

    # Replace any earlier null interpolation.
    encoder_cfg = (
        config.get("encoder")
        if isinstance(config, dict)
        else getattr(config, "encoder", None)
    )
    if encoder_cfg is None:
        return

    model_cfg = (
        encoder_cfg.get("model")
        if isinstance(encoder_cfg, dict)
        else getattr(encoder_cfg, "model", None)
    )
    if model_cfg is None:
        return

    if isinstance(model_cfg, DictConfig):
        with read_write(model_cfg), open_dict(model_cfg):
            model_cfg.max_len = int(max_len)
    elif isinstance(model_cfg, dict):
        model_cfg["max_len"] = int(max_len)
    else:
        setattr(model_cfg, "max_len", int(max_len))


def sync_resolved_objectives(config, obj_dim: int) -> None:
    task_cfg = (
        config.get("task")
        if isinstance(config, dict)
        else getattr(config, "task", None)
    )
    if task_cfg is None:
        return

    directions = (
        task_cfg.get("objective_directions")
        if isinstance(task_cfg, dict)
        else getattr(task_cfg, "objective_directions", None)
    )
    if directions is None:
        directions = ["min"] * int(obj_dim)
    else:
        directions = list(directions)
    if len(directions) != int(obj_dim):
        raise ValueError(
            "task.objective_directions length must equal the number of objective columns; "
            f"got directions={len(directions)}, objectives={obj_dim}."
        )

    if isinstance(task_cfg, DictConfig):
        with read_write(task_cfg), open_dict(task_cfg):
            task_cfg.obj_dim = int(obj_dim)
            task_cfg.objective_directions = directions
    elif isinstance(task_cfg, dict):
        task_cfg["obj_dim"] = int(obj_dim)
        task_cfg["objective_directions"] = directions
    else:
        setattr(task_cfg, "obj_dim", int(obj_dim))
        setattr(task_cfg, "objective_directions", directions)


def _require_objectives_present(
    df: pd.DataFrame,
    path: str,
    context: str,
    obj_cols=OBJ_COLS,
) -> None:
    bad = df[list(obj_cols)].isna().any(axis=1)
    if bool(bad.any()):
        n_bad = int(bad.sum())
        preview = df.loc[bad, [*BASE_COLS, *obj_cols]].head(10)
        raise ValueError(
            f"Found {n_bad} rows with missing objectives in {context}.\n"
            f"path={path}\n"
            f"This will corrupt BO training data (unobserved points mixed into known set).\n"
            f"Fix: fill all objective columns with real experimental values, or remove these rows.\n"
            f"Preview (up to 10 rows):\n{preview.to_string(index=False)}"
        )


def _stack_targets(
    df: pd.DataFrame,
    sign_vec: np.ndarray,
    obj_cols=OBJ_COLS,
) -> np.ndarray:
    arr = df.loc[:, list(obj_cols)].to_numpy(dtype=float)
    return arr * sign_vec.reshape(1, -1)


def _compute_pareto_mask(targets_sign_space: np.ndarray) -> np.ndarray:
    if targets_sign_space.size == 0:
        return np.zeros((0,), dtype=bool)
    mask = pareto.is_non_dominated(-torch.tensor(targets_sign_space, dtype=torch.float32)).cpu().numpy()
    return mask.astype(bool)


def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _append_csv(df: pd.DataFrame, path: str, expected_cols=EXPECTED_COLS) -> None:
    _ensure_parent_dir(path)
    df = df.loc[:, list(expected_cols)]
    df.to_csv(
        path,
        mode="a",
        index=False,
        header=not os.path.exists(path),
        float_format="%.6f",
        quoting=csv.QUOTE_MINIMAL,
    )


def _atomic_write_csv(df: pd.DataFrame, path: str, expected_cols=EXPECTED_COLS) -> None:
    _ensure_parent_dir(path)
    df = df.loc[:, list(expected_cols)]
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, float_format="%.6f", quoting=csv.QUOTE_MINIMAL)
    os.replace(tmp, path)


def _legacy_rng_burn_seed_sample(df_seed_full: pd.DataFrame, n_seed: int) -> None:
    if n_seed <= 0 or len(df_seed_full) == 0:
        return
    _ = df_seed_full.sample(n=n_seed, replace=False)  # noqa: F841


def _select_seed_rows(
    df_seed_full: pd.DataFrame,
    n_seed: int,
    mode: str,
    fixed_seed: int,
) -> pd.DataFrame:
    mode = (mode or "first").strip().lower()
    n_seed = int(max(0, n_seed))
    if n_seed == 0 or len(df_seed_full) == 0:
        return df_seed_full.iloc[0:0].copy()

    if mode == "first":
        return df_seed_full.head(n_seed).copy()

    if mode == "sorted":
        return df_seed_full.sort_values(by=["mutant_sequence"], kind="mergesort").head(n_seed).copy()

    if mode == "fixed_random":
        rng = np.random.default_rng(int(fixed_seed))
        idx = rng.choice(len(df_seed_full), size=n_seed, replace=False)
        return df_seed_full.iloc[idx].copy()

    raise ValueError(f"Unsupported seed_sampling_mode: {mode}")



class OPTTask(BaseTask):
    def __init__(
        self,
        tokenizer,
        candidate_pool,
        obj_dim: int,
        transform=lambda x: x,
        num_start_examples: int = 1024,
        objective_directions: Optional[List[str]] = None,
        seq_max_len: Optional[int] = None,
        max_len: Optional[int] = None,
        seed_sampling_mode: str = "first",
        seed_sampling_seed: int = 0,
        update_known_from_exam: bool = True,
        update_pool_from_exam: bool = True,
        update_seed_from_exam: bool = False,
        clear_exam_after_merge: bool = False,
        unknown_score_policy: str = "error",
        **kwargs,
    ):
        self.objective_cols = objective_columns(obj_dim)
        self.expected_cols = BASE_COLS + self.objective_cols

        super().__init__(
            tokenizer,
            candidate_pool,
            obj_dim,
            transform,
            max_len=max_len,
            **kwargs,
        )

        self.num_start_examples = int(num_start_examples)

        if objective_directions is None:
            objective_directions = ["min"] * int(obj_dim)
        if len(objective_directions) != int(obj_dim):
            raise ValueError("objective_directions length must equal obj_dim")

        self.objective_directions = [str(d).lower() for d in objective_directions]
        for d in self.objective_directions:
            if d not in {"min", "max"}:
                raise ValueError(f"unsupported objective direction: {d}")

        self.sign_vector = np.array([1.0 if d == "min" else -1.0 for d in self.objective_directions], dtype=float)

        self.seq_max_len = None if seq_max_len is None else int(seq_max_len)

        self.seed_sampling_mode = str(seed_sampling_mode)
        self.seed_sampling_seed = int(seed_sampling_seed)

        self.update_known_from_exam = bool(update_known_from_exam)
        self.update_pool_from_exam = bool(update_pool_from_exam)
        self.update_seed_from_exam = bool(update_seed_from_exam)
        self.clear_exam_after_merge = bool(clear_exam_after_merge)

        self.unknown_score_policy = str(unknown_score_policy).strip().lower()
        if self.unknown_score_policy not in {"error", "zero", "nan"}:
            raise ValueError(
                f"unknown_score_policy must be one of error/zero/nan, got {self.unknown_score_policy}"
            )

        self.seq_to_target_map: Dict[str, np.ndarray] = {}

    def task_setup(self, config=None, project_root: Optional[str] = None, *args, **kwargs):
        root = _resolve_project_root(project_root)
        fp_dir = os.path.join(root, "evomobo", "assets", "input_learning")

        known_path = os.path.join(fp_dir, "known_data.csv")
        pool_path = os.path.join(fp_dir, "pool.csv")
        seed_path = os.path.join(fp_dir, "seed_data.csv")
        exam_path = os.path.join(fp_dir, "exam.csv")
        pareto_extra_path = os.path.join(fp_dir, "par_cand_extra.csv")

        seq_max_len_was_auto = self.seq_max_len is None
        max_len_was_auto = self.max_len is None
        self.seq_max_len, self.max_len = resolve_sequence_lengths(
            self.seq_max_len,
            self.max_len,
            input_learning_sequence_paths(root),
        )

        if seq_max_len_was_auto:
            logger.info(
                "Auto-inferred task.seq_max_len=%d from input_learning CSV sequences.",
                self.seq_max_len,
            )

        if max_len_was_auto:
            logger.info(
                "Auto-resolved task.max_len=%d as task.seq_max_len + 2.",
                self.max_len,
            )

        sync_resolved_lengths(config, self.seq_max_len, self.max_len)

        len_thr = self.seq_max_len

        # Load observations.
        df_known = _read_csv_if_exists(known_path)
        if df_known is None:
            raise FileNotFoundError(f"known_data.csv not found: {known_path}")
        df_known = _ensure_expected_cols(df_known, known_path, self.expected_cols)
        df_known = _clean_df(df_known, self.objective_cols)
        df_known = _dedupe_keep_first(df_known, subset="mutant_sequence")

        _require_objectives_present(
            df_known,
            known_path,
            context="known_data.csv",
            obj_cols=self.objective_cols,
        )

        df_known_valid = _length_filter(df_known, len_thr)

        # Cache the current Pareto set.
        if len(df_known_valid) > 0:
            known_targets_sign = _stack_targets(
                df_known_valid,
                self.sign_vector,
                self.objective_cols,
            )
            old_mask = _compute_pareto_mask(known_targets_sign)
            df_old_pareto = df_known_valid.loc[old_mask].reset_index(drop=True)
        else:
            df_old_pareto = df_known_valid

        # Merge completed evaluations.
        df_all = df_known.copy()

        df_exam = _read_csv_if_exists(exam_path)
        if df_exam is not None:
            df_exam = _ensure_expected_cols(df_exam, exam_path, self.expected_cols)
            df_exam = _clean_df(df_exam, self.objective_cols)
            df_exam = _dedupe_keep_first(df_exam, subset="mutant_sequence")

            _require_objectives_present(
                df_exam,
                exam_path,
                context="exam.csv",
                obj_cols=self.objective_cols,
            )

            df_all = _dedupe_keep_first(pd.concat([df_all, df_exam], ignore_index=True), subset="mutant_sequence")

            if self.update_pool_from_exam:
                df_pool = _read_csv_if_exists(pool_path)
                if df_pool is None:
                    df_pool = pd.DataFrame(columns=list(self.expected_cols))
                else:
                    df_pool = _ensure_expected_cols(
                        df_pool,
                        pool_path,
                        self.expected_cols,
                    )
                    df_pool = _clean_df(df_pool, self.objective_cols)
                    _require_objectives_present(
                        df_pool,
                        pool_path,
                        context="pool.csv",
                        obj_cols=self.objective_cols,
                    )

                df_pool = _dedupe_keep_first(pd.concat([df_pool, df_exam], ignore_index=True), subset="mutant_sequence")
                _atomic_write_csv(df_pool, pool_path, self.expected_cols)

            if self.update_known_from_exam:
                _atomic_write_csv(df_all, known_path, self.expected_cols)

            if self.update_seed_from_exam:
                df_seed_exist = _read_csv_if_exists(seed_path)
                if df_seed_exist is None:
                    df_seed_exist = pd.DataFrame(columns=list(self.expected_cols))
                else:
                    df_seed_exist = _ensure_expected_cols(
                        df_seed_exist,
                        seed_path,
                        self.expected_cols,
                    )
                    df_seed_exist = _clean_df(df_seed_exist, self.objective_cols)
                    _require_objectives_present(
                        df_seed_exist,
                        seed_path,
                        context="seed_data.csv",
                        obj_cols=self.objective_cols,
                    )

                df_seed_new = _dedupe_keep_first(
                    pd.concat([df_seed_exist, df_exam], ignore_index=True),
                    subset="mutant_sequence",
                )
                _atomic_write_csv(df_seed_new, seed_path, self.expected_cols)

            if self.clear_exam_after_merge:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                archive = os.path.join(fp_dir, f"exam_archived_{ts}.csv")
                shutil.copyfile(exam_path, archive)
                os.remove(exam_path)
                logger.info("exam.csv archived to %s and removed.", archive)

        # Recompute the feasible Pareto set.
        df_all_valid = _length_filter(df_all, len_thr)
        if len(df_all_valid) == 0:
            logger.warning(
                "No sequences satisfy length <= %s. Disable length filtering for this run.",
                str(len_thr),
            )
            df_all_valid = df_all.copy()
            len_thr = None

        all_targets_sign = _stack_targets(
            df_all_valid,
            self.sign_vector,
            self.objective_cols,
        )
        new_mask = _compute_pareto_mask(all_targets_sign)
        df_new_pareto = df_all_valid.loc[new_mask].reset_index(drop=True)

        # Archive displaced Pareto points.
        try:
            old_seqs = set(df_old_pareto["mutant_sequence"].tolist())
            new_seqs = set(df_new_pareto["mutant_sequence"].tolist())
            dropped = list(old_seqs - new_seqs)
            if dropped:
                df_extra = df_old_pareto[df_old_pareto["mutant_sequence"].isin(dropped)].copy()
                if len(df_extra) > 0:
                    _append_csv(
                        df_extra.loc[:, list(self.expected_cols)],
                        pareto_extra_path,
                        self.expected_cols,
                    )
        except Exception:
            logger.exception("Failed to append dropped pareto points. Continue without it.")

        # Build the active candidate set.
        base_candidates = np.asarray(
            [
                StringCandidate(
                    wild_seq=row["mutant_sequence"],
                    mutation_list=[],
                    tokenizer=self.tokenizer,
                    wild_name=row["Name"],
                )
                for _, row in df_new_pareto.iterrows()
            ],
            dtype=object,
        )
        base_targets = _stack_targets(
            df_new_pareto,
            self.sign_vector,
            self.objective_cols,
        )

        # Load optional seed data.
        seed_seqs = np.array([], dtype=object)
        seed_targets = np.zeros((0, self.obj_dim), dtype=float)

        df_seed_full = _read_csv_if_exists(seed_path)
        if df_seed_full is not None:
            df_seed_full = _ensure_expected_cols(
                df_seed_full,
                seed_path,
                self.expected_cols,
            )
            df_seed_full = _clean_df(df_seed_full, self.objective_cols)
            df_seed_full = _dedupe_keep_first(df_seed_full, subset="mutant_sequence")
            _require_objectives_present(
                df_seed_full,
                seed_path,
                context="seed_data.csv",
                obj_cols=self.objective_cols,
            )

            if len(df_seed_full) > 0 and self.num_start_examples > 0:
                n_seed = min(int(self.num_start_examples), int(len(df_seed_full)))

                _legacy_rng_burn_seed_sample(df_seed_full, n_seed)

                df_seed_pick = _select_seed_rows(
                    df_seed_full=df_seed_full,
                    n_seed=n_seed,
                    mode=self.seed_sampling_mode,
                    fixed_seed=self.seed_sampling_seed,
                )
                df_seed_pick = _length_filter(df_seed_pick, len_thr)

                if len(df_seed_pick) > 0:
                    seed_seqs = df_seed_pick["mutant_sequence"].values.astype(object)
                    seed_targets = _stack_targets(
                        df_seed_pick,
                        self.sign_vector,
                        self.objective_cols,
                    )

        # Merge and deduplicate training data.
        all_seqs = np.concatenate([df_all_valid["mutant_sequence"].values.astype(object), seed_seqs])
        all_targets = np.concatenate([all_targets_sign, seed_targets], axis=0)

        if len_thr is not None:
            m = np.array([len(str(s)) <= int(len_thr) for s in all_seqs], dtype=bool)
            all_seqs = all_seqs[m]
            all_targets = all_targets[m]

        uniq_seqs, uniq_idx = np.unique(all_seqs.astype(object), return_index=True)
        all_seqs = uniq_seqs.astype(object)
        all_targets = all_targets[uniq_idx]

        # Cache measured targets.
        self.seq_to_target_map.clear()
        for s, y in zip(all_seqs, all_targets):
            self.seq_to_target_map[str(s)] = np.asarray(y, dtype=float)

        logger.info(
            "setup done. known_all=%d, pareto=%d, train_all=%d, seq_max_len=%s, max_len=%s",
            int(len(df_all_valid)),
            int(len(df_new_pareto)),
            int(len(all_seqs)),
            str(len_thr),
            str(self.max_len),
        )

        return base_candidates, base_targets, all_seqs, all_targets

    def score(self, candidates):
        out = []
        for c in candidates:
            seq = str(c.mutant_residue_seq)
            y = self.seq_to_target_map.get(seq, None)
            if y is not None:
                out.append(y)
                continue

            if self.unknown_score_policy == "error":
                raise KeyError(f"Unknown sequence encountered in score(): {seq[:60]}...")
            if self.unknown_score_policy == "zero":
                out.append(np.zeros(self.obj_dim, dtype=float))
            else: 
                out.append(np.full(self.obj_dim, np.nan, dtype=float))

        return np.stack(out, axis=0)
