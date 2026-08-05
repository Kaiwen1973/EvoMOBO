from __future__ import annotations

import csv
import copy
import datetime
import logging
import os
import random
import shutil
import time
from typing import Any, Dict, List, Optional, Sequence

import hydra
import numpy as np
import pandas as pd
import torch
from botorch.utils.multi_objective import infer_reference_point
from omegaconf import DictConfig, open_dict, read_write
from pymoo.factory import get_performance_indicator
from torch.nn import functional as F

from evomobo.candidate import StringCandidate
from evomobo.models.lanmt import corrupt_tok_idxs
from evomobo.models.mlm import enumerate_windows, evaluate_windows, sample_tokens
from evomobo.models.shared_elements import check_early_stopping
from evomobo.optimizers.pymoo import Normalizer, pareto_frontier
from evomobo.rng_utils import isolated_seed
from evomobo.utils import (
    DataSplit,
    str_to_tokens,
    tokens_to_str,
    update_splits,
    weighted_resampling,
)
from evomobo.wandb_utils import wandb_log

logger = logging.getLogger(__name__)


class EvoMOBO:
    def __init__(
        self,
        bb_task,
        tokenizer,
        encoder,
        surrogate,
        acquisition,
        num_rounds: int,
        num_gens: int,
        lr: float,
        num_opt_steps: int,
        concentrate_pool: int,
        patience: int,
        mask_ratio: float,
        resampling_weight: float,
        encoder_obj: str,
        optimize_latent: bool,
        position_sampler: str,
        entropy_penalty: float,
        latent_trust_radius: float,
        window_size: int,
        latent_init,
        mutable_positions: Optional[str] = None,
        positions_index_base: int = 1,
        obj_weights: Optional[Sequence[float]] = None,
        offline_mode: bool = True,
        offline_selection_mode: str = "rank1_batch_only",
        offline_num_suggestions: Optional[int] = None,  # Defaults to q.
        offline_suggest_csv: str = "to_eval.csv",
        offline_exam_template_csv: str = "exam_template.csv",
        offline_backfill_from_ranked_batches: bool = True,
        use_pool_csv_points: bool = False,
        use_pareto_extra_csv_points: bool = False,
        ckpt_dir: Optional[str] = None,
        ckpt_name: str = "default",
        resume: bool = True,
        save_ckpt_every_round: bool = True,
        resume_allow_unsafe: bool = False,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.bb_task = bb_task

        self.num_rounds = int(num_rounds)
        self.num_gens = int(num_gens)
        self.concentrate_pool = int(concentrate_pool)

        self.lr = float(lr)
        self.num_opt_steps = int(num_opt_steps)
        self.patience = int(patience)
        self.mask_ratio = float(mask_ratio)

        self.resampling_weight = resampling_weight
        self.optimize_latent = bool(optimize_latent)
        self.position_sampler = str(position_sampler)
        self.entropy_penalty = float(entropy_penalty)
        self.latent_trust_radius = float(latent_trust_radius)
        if self.latent_trust_radius <= 0:
            raise ValueError("latent_trust_radius must be positive.")
        self.window_size = int(window_size)
        self.latent_init = latent_init

        self.positions_index_base = int(positions_index_base)
        if self.positions_index_base not in {0, 1}:
            raise ValueError("positions_index_base must be either 0 or 1.")
        self.mutable_positions = self._parse_positions(mutable_positions)
        if self.mutable_positions:
            invalid_positions = sorted(
                pos
                for pos in self.mutable_positions
                if pos < self.positions_index_base
            )
            if invalid_positions:
                raise ValueError(
                    "mutable_positions contains positions below "
                    f"positions_index_base={self.positions_index_base}: "
                    f"{invalid_positions}"
                )
            self.mutable_token_positions = {
                pos - self.positions_index_base + 1
                for pos in self.mutable_positions
            }
        else:
            self.mutable_token_positions = None

        self.offline_mode = bool(offline_mode)
        self.offline_selection_mode = str(offline_selection_mode).strip().lower()
        if self.offline_selection_mode not in {
            "ranked_batch",
            "greedy_mean_hvi",
            "rank1_batch_only",
        }:
            raise ValueError(
                "offline_selection_mode must be 'ranked_batch', "
                "'greedy_mean_hvi', or 'rank1_batch_only'; "
                f"got {offline_selection_mode!r}."
            )
        self.offline_num_suggestions = offline_num_suggestions
        self.offline_suggest_csv = str(offline_suggest_csv)
        self.offline_exam_template_csv = str(offline_exam_template_csv)
        self.offline_backfill_from_ranked_batches = bool(
            offline_backfill_from_ranked_batches
        )

        self.use_pool_csv_points = bool(use_pool_csv_points)
        self.use_pareto_extra_csv_points = bool(use_pareto_extra_csv_points)

        self.print_splits = bool(kwargs.pop("print_splits", True))
        self.print_splits_max_rows = int(kwargs.pop("print_splits_max_rows", 0))

        if obj_weights is None:
            obj_weights = [1.0] * int(self.bb_task.obj_dim)
        obj_weights = np.asarray(obj_weights, dtype=float)
        if obj_weights.shape != (int(self.bb_task.obj_dim),):
            raise ValueError(
                f"obj_weights must have length {self.bb_task.obj_dim}, got {obj_weights.shape}"
            )
        if np.any(obj_weights <= 0):
            raise ValueError(" obj_weights must be positive.")
        self.obj_weights = obj_weights

        resolved_max_len = getattr(self.bb_task, "max_len", None)
        if resolved_max_len is None:
            raise ValueError(
                "bb_task.max_len is unresolved before encoder instantiation. "
                "Run task_setup() and resolve task lengths first."
            )

        self.encoder_config = self._encoder_config_with_max_len(
            encoder,
            int(resolved_max_len),
        )
        self.encoder_obj = str(encoder_obj)
        self.encoder = hydra.utils.instantiate(
            self.encoder_config,
            tokenizer=tokenizer,
        )

        self.surrogate_config = surrogate
        self.surrogate_training_seed = int(
            self.surrogate_config.get("training_seed", 0) or 0
        )
        surrogate_instantiate_config = copy.deepcopy(self.surrogate_config)
        if isinstance(surrogate_instantiate_config, DictConfig):
            with read_write(surrogate_instantiate_config), open_dict(
                surrogate_instantiate_config
            ):
                surrogate_instantiate_config.pop("training_seed", None)
        else:
            surrogate_instantiate_config.pop("training_seed", None)

        with isolated_seed(self.surrogate_training_seed):
            self.surrogate_model = hydra.utils.instantiate(
                surrogate_instantiate_config,
                tokenizer=self.encoder.tokenizer,
                encoder=self.encoder,
            )
        self.acquisition = acquisition

        if self.surrogate_training_seed != 0:
            logger.info(
                "Surrogate RNG isolation enabled with training_seed=%d.",
                self.surrogate_training_seed,
            )

        self._hv_ref = None
        self._hv_space = None
        self._ref_point = np.array([1] * int(self.bb_task.obj_dim), dtype=float)

        self.active_candidates = None
        self.active_targets = None
        self.active_seqs = None

        self.train_split = DataSplit()
        self.val_split = DataSplit()
        self.test_split = DataSplit()

        self._init_checkpointing(ckpt_dir=ckpt_dir, ckpt_name=ckpt_name)
        self._resume_enabled = bool(resume)
        self._resume_allow_unsafe = bool(resume_allow_unsafe)
        self._save_ckpt_every_round = bool(save_ckpt_every_round)
        self._pending_round_dir = None
        self._pending_final_round_dir = None
        self._pending_round_number = None

        self._global_round_offset = 0
        if self._resume_enabled:
            self._maybe_load_checkpoint()

    # Logging
    def _log(self, msg: str, level: str = "info") -> None:
        try:
            root_has_handlers = bool(logging.getLogger().handlers)
            local_has_handlers = bool(logger.handlers)
            if root_has_handlers or local_has_handlers:
                getattr(logger, level)(msg)
            else:
                print(msg)
        except Exception:
            print(msg)

    def _wandb_log(self, data: Dict[str, Any]) -> None:
        wandb_log(data)

    def _format_df(self, df: pd.DataFrame, floatfmt: str = ".4f") -> str:
        try:
            return df.to_markdown(floatfmt=floatfmt, index=False)
        except Exception:
            # Fallback without tabulate.
            return df.to_string(index=False)

    def _ensure_dir_for_file(self, path: str) -> None:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    def _write_csv(
        self,
        df: pd.DataFrame,
        path: str,
        *,
        float_format: Optional[str] = None,
        quoting: int = csv.QUOTE_MINIMAL,
    ) -> None:
        self._ensure_dir_for_file(path)
        df.to_csv(path, index=False, float_format=float_format, quoting=quoting)

    # Checkpoints
    def _resolve_project_root(self) -> str:
        try:
            return hydra.utils.get_original_cwd()
        except Exception:
            return os.getcwd()

    def _init_checkpointing(self, ckpt_dir: Optional[str], ckpt_name: str) -> None:
        project_root = self._resolve_project_root()
        if ckpt_dir is None:
            ckpt_root = os.path.join(project_root, "evomobo_checkpoints")
        else:
            ckpt_root = ckpt_dir if os.path.isabs(ckpt_dir) else os.path.join(project_root, ckpt_dir)

        self._ckpt_root = ckpt_root
        self._ckpt_name = str(ckpt_name)
        self._ckpt_dir = os.path.join(self._ckpt_root, self._ckpt_name)
        os.makedirs(self._ckpt_dir, exist_ok=True)
        self._ckpt_latest_path = os.path.join(self._ckpt_dir, "latest.pt")

    def _state_dict_to_cpu(self, sd: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in sd.items():
            out[k] = v.detach().cpu() if torch.is_tensor(v) else v
        return out

    def _ensure_py(self, obj: Any) -> Any:
        if obj is None:
            return None
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): self._ensure_py(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._ensure_py(v) for v in obj]
        if isinstance(obj, (np.str_, bytes)):
            return str(obj)
        return obj

    def _split_to_dict(self, split: DataSplit) -> Dict[str, Any]:
        if split is None:
            return {"inputs": None, "targets": None}
        inputs = getattr(split, "inputs", None)
        targets = getattr(split, "targets", None)
        if inputs is None or targets is None:
            return {"inputs": None, "targets": None}
        inputs_list = [str(x) for x in list(np.asarray(inputs, dtype=object))]
        targets_list = np.asarray(targets, dtype=float).tolist()
        return {"inputs": inputs_list, "targets": targets_list}

    def _dict_to_split(self, d: Any) -> DataSplit:
        if not isinstance(d, dict):
            return DataSplit()
        inputs = d.get("inputs", None)
        targets = d.get("targets", None)
        if inputs is None or targets is None:
            return DataSplit()
        return DataSplit(np.asarray(list(inputs), dtype=object), np.asarray(targets, dtype=float))

    def _torch_load_ckpt(self, path: str) -> Dict[str, Any]:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")
        except Exception as e:
            if not self._resume_allow_unsafe:
                raise e

        self._log(
            f"Safe load failed; falling back to unsafe torch.load(weights_only=False) for {path}.",
            level="warning",
        )
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _np_allclose(
        self,
        a: Any,
        b: Any,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> bool:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=atol)

    def _hard_check_ckpt_compat(self, ckpt: Dict[str, Any]) -> None:
        meta = ckpt.get("meta", {}) or {}

        ckpt_surrogate = meta.get("surrogate_target", None)
        current_surrogate = self.surrogate_config.get("_target_", None)
        ckpt_implementation = (
            str(ckpt_surrogate).split(".", 1)[-1]
            if ckpt_surrogate is not None
            else None
        )
        current_implementation = (
            str(current_surrogate).split(".", 1)[-1]
            if current_surrogate is not None
            else None
        )
        if (
            ckpt_implementation is not None
            and current_implementation is not None
            and ckpt_implementation != current_implementation
        ):
            raise ValueError(
                "[CKPT CHECK] surrogate mismatch: "
                f"ckpt='{ckpt_surrogate}' vs current='{current_surrogate}'."
            )

        ckpt_obj_dim = meta.get("bb_obj_dim", None)
        cur_obj_dim = int(getattr(self.bb_task, "obj_dim", -1))
        if ckpt_obj_dim is not None and int(ckpt_obj_dim) != cur_obj_dim:
            raise ValueError(
                f"[CKPT CHECK] obj_dim mismatch: ckpt={ckpt_obj_dim} vs current={cur_obj_dim}."
            )

        ckpt_encoder_obj = meta.get("encoder_obj", None)
        if ckpt_encoder_obj is not None and str(ckpt_encoder_obj) != str(self.encoder_obj):
            raise ValueError(
                f"[CKPT CHECK] encoder_obj mismatch: ckpt='{ckpt_encoder_obj}' vs current='{self.encoder_obj}'."
            )

        ckpt_w = meta.get("obj_weights", None)
        if ckpt_w is not None:
            ckpt_w = np.asarray(ckpt_w, dtype=float)
            cur_w = np.asarray(self.obj_weights, dtype=float)
            if not self._np_allclose(ckpt_w, cur_w):
                raise ValueError(
                    "[CKPT CHECK] obj_weights mismatch:\n"
                    f" ckpt={ckpt_w.tolist()}\n"
                    f" current={cur_w.tolist()}\n"
                    "This changes objective geometry; stop."
                )

        ckpt_dirs = meta.get("objective_directions", None)
        cur_dirs = getattr(self.bb_task, "objective_directions", None)
        if ckpt_dirs is not None and cur_dirs is not None:
            ckpt_dirs = [str(d).lower() for d in ckpt_dirs]
            cur_dirs = [str(d).lower() for d in cur_dirs]
            if ckpt_dirs != cur_dirs:
                raise ValueError(
                    "[CKPT CHECK] objective_directions mismatch:\n"
                    f" ckpt={ckpt_dirs}\n"
                    f" current={cur_dirs}\n"
                )

        ckpt_sign = meta.get("sign_vector", None)
        cur_sign = getattr(self.bb_task, "sign_vector", None)
        if ckpt_sign is not None and cur_sign is not None:
            ckpt_sign = np.asarray(ckpt_sign, dtype=float)
            cur_sign = np.asarray(cur_sign, dtype=float)
            if not self._np_allclose(ckpt_sign, cur_sign):
                raise ValueError(
                    "[CKPT CHECK] sign_vector mismatch:\n"
                    f" ckpt={ckpt_sign.tolist()}\n"
                    f" current={cur_sign.tolist()}\n"
                )

        for meta_key, task_attr in (
            ("task_seq_max_len", "seq_max_len"),
            ("task_max_len", "max_len"),
        ):
            ckpt_len = meta.get(meta_key, None)
            cur_len = getattr(self.bb_task, task_attr, None)
            if ckpt_len is not None and cur_len is not None and int(ckpt_len) != int(cur_len):
                raise ValueError(
                    f"[CKPT CHECK] {task_attr} mismatch: "
                    f"ckpt={int(ckpt_len)} vs current={int(cur_len)}."
                )

        hv_space = ckpt.get("hv_space", None)
        if isinstance(hv_space, dict):
            loc = hv_space.get("loc", None)
            scale = hv_space.get("scale", None)
            refp = hv_space.get("ref_point", None)
            if loc is not None and np.asarray(loc, dtype=float).shape != (cur_obj_dim,):
                raise ValueError("[CKPT CHECK] hv_space.loc shape mismatch.")
            if scale is not None and np.asarray(scale, dtype=float).shape != (cur_obj_dim,):
                raise ValueError("[CKPT CHECK] hv_space.scale shape mismatch.")
            if refp is not None and np.asarray(refp, dtype=float).shape != (cur_obj_dim,):
                raise ValueError("[CKPT CHECK] hv_space.ref_point shape mismatch.")

    def _maybe_load_checkpoint(self) -> bool:
        if not os.path.exists(self._ckpt_latest_path):
            self._log(f"No checkpoint found at {self._ckpt_latest_path}. Start fresh.")
            return False

        try:
            ckpt = self._torch_load_ckpt(self._ckpt_latest_path)
        except Exception as e:
            self._log(
                " Failed to load checkpoint (safe mode).\n"
                f" path={self._ckpt_latest_path}\n"
                f" err={e}\n"
                "If this is an old ckpt with unsafe objects, convert it or set resume_allow_unsafe=True (not recommended).",
                level="warning",
            )
            return False

        self._hard_check_ckpt_compat(ckpt)

        self._global_round_offset = int(ckpt.get("global_round", 0))

        enc_state = ckpt.get("encoder_state", None)
        if isinstance(enc_state, dict):
            msg = self.encoder.load_state_dict(enc_state, strict=False)
            if getattr(msg, "missing_keys", None) or getattr(msg, "unexpected_keys", None):
                self._log(
                    f"Encoder load warnings (strict=False): "
                    f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}",
                    level="warning",
                )

        sur_state = ckpt.get("surrogate_state", None)
        if isinstance(sur_state, dict):
            msg = self.surrogate_model.load_state_dict(sur_state, strict=False)
            if getattr(msg, "missing_keys", None) or getattr(msg, "unexpected_keys", None):
                self._log(
                    f"Surrogate load warnings (strict=False): "
                    f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}",
                    level="warning",
                )

        splits = ckpt.get("splits", None)
        if isinstance(splits, dict):
            self.train_split = self._dict_to_split(splits.get("train", {}))
            self.val_split = self._dict_to_split(splits.get("val", {}))
            self.test_split = self._dict_to_split(splits.get("test", {}))

        hv_space = ckpt.get("hv_space", None)
        if isinstance(hv_space, dict) and ("loc" in hv_space) and ("scale" in hv_space):
            self._hv_space = hv_space
            if "ref_point" in hv_space:
                try:
                    self._ref_point = np.asarray(hv_space["ref_point"], dtype=float)
                except Exception:
                    pass
            self._log(" Loaded hv_space (hypercube loc/scale + ref_point) from checkpoint.")
        else:
            self._hv_space = None
            self._log(" No hv_space in checkpoint; will initialize HV space this run.")

        hv_ref = ckpt.get("hv_ref", None)
        if hv_ref is not None:
            try:
                self._hv_ref = float(hv_ref)
                self._log(f"Loaded hv_ref baseline from checkpoint: {self._hv_ref:.6f}")
            except Exception:
                pass

        if hasattr(self.surrogate_model, "clear_cache"):
            try:
                self.surrogate_model.clear_cache()
            except Exception:
                pass

        meta = ckpt.get("meta", {})
        be = ckpt.get("best_epoch", None)
        self._log(
            f"Loaded checkpoint: global_round={self._global_round_offset}, "
            f"best_epoch={be}, meta_keys={list(meta.keys())[:10]}"
        )
        return True

    def _save_checkpoint(
        self, global_round: int, best_epoch: Optional[int] = None, extra_meta: Optional[Dict[str, Any]] = None
    ) -> bool:
        os.makedirs(self._ckpt_dir, exist_ok=True)
        round_path = os.path.join(self._ckpt_dir, f"round_{int(global_round):04d}.pt")
        tmp_round_path = f"{round_path}.incomplete"
        tmp_latest_path = f"{self._ckpt_latest_path}.incomplete"

        objective_directions = getattr(self.bb_task, "objective_directions", None)
        sign_vector = getattr(self.bb_task, "sign_vector", None)
        if sign_vector is not None:
            try:
                sign_vector = [float(x) for x in list(np.asarray(sign_vector, dtype=float).reshape(-1))]
            except Exception:
                sign_vector = None

        ckpt = {
            "version": 3,
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "global_round": int(global_round),
            "best_epoch": None if best_epoch is None else int(best_epoch),
            "encoder_state": self._state_dict_to_cpu(self.encoder.state_dict()),
            "surrogate_state": self._state_dict_to_cpu(self.surrogate_model.state_dict()),
            "splits": {
                "train": self._split_to_dict(self.train_split),
                "val": self._split_to_dict(self.val_split),
                "test": self._split_to_dict(self.test_split),
            },
            "meta": {
                "encoder_obj": str(self.encoder_obj),
                "surrogate_target": str(
                    self.surrogate_config.get("_target_", "")
                ),
                "surrogate_training_seed": int(self.surrogate_training_seed),
                "obj_weights": [float(x) for x in list(self.obj_weights)],
                "bb_obj_dim": int(getattr(self.bb_task, "obj_dim", -1)),
                "objective_directions": objective_directions,
                "sign_vector": sign_vector,
                "task_seq_max_len": int(getattr(self.bb_task, "seq_max_len", 0)),
                "task_max_len": int(getattr(self.bb_task, "max_len", 0)),
            },
            "hv_space": None if self._hv_space is None else self._ensure_py(self._hv_space),
            "hv_ref": None if self._hv_ref is None else float(self._hv_ref),
        }

        if extra_meta:
            ckpt["meta"].update(self._ensure_py(extra_meta))

        ckpt = self._ensure_py(ckpt)

        round_committed = False
        try:
            for path in (tmp_round_path, tmp_latest_path):
                if os.path.lexists(path):
                    os.remove(path)
            if os.path.lexists(round_path):
                raise FileExistsError(f"Checkpoint already exists: {round_path}")

            torch.save(ckpt, tmp_round_path)
            try:
                _ = torch.load(
                    tmp_round_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                _ = torch.load(tmp_round_path, map_location="cpu")

            try:
                os.symlink(os.path.basename(round_path), tmp_latest_path)
            except OSError:
                shutil.copyfile(tmp_round_path, tmp_latest_path)

            os.replace(tmp_round_path, round_path)
            round_committed = True
            os.replace(tmp_latest_path, self._ckpt_latest_path)
            self._log(f"Checkpoint saved: {round_path}")
            return True
        except Exception as e:
            if round_committed and os.path.lexists(round_path):
                os.remove(round_path)
            for path in (tmp_round_path, tmp_latest_path):
                if os.path.lexists(path):
                    os.remove(path)
            self._log(
                f"Failed to save checkpoint: {round_path}\n  err={e}",
                level="warning",
            )
            return False

    def _begin_round_output(self, out_base_dir: str) -> str:
        os.makedirs(out_base_dir, exist_ok=True)
        round_number = int(self._global_round_offset) + 1
        while True:
            final_dir = os.path.join(out_base_dir, f"round{round_number}")
            checkpoint_path = os.path.join(
                self._ckpt_dir,
                f"round_{round_number:04d}.pt",
            )
            if not os.path.exists(final_dir) and not os.path.lexists(checkpoint_path):
                break
            round_number += 1

        work_dir = os.path.join(
            out_base_dir,
            f".round{round_number}.incomplete",
        )
        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir)
        elif os.path.lexists(work_dir):
            os.remove(work_dir)
        os.makedirs(work_dir)

        self._pending_round_dir = work_dir
        self._pending_final_round_dir = final_dir
        self._pending_round_number = round_number
        return work_dir

    def cleanup_incomplete_round(self) -> None:
        pending_dir = self._pending_round_dir
        if pending_dir and os.path.isdir(pending_dir):
            shutil.rmtree(pending_dir)
        elif pending_dir and os.path.lexists(pending_dir):
            os.remove(pending_dir)
        self._pending_round_dir = None
        self._pending_final_round_dir = None
        self._pending_round_number = None

    def _commit_round_output(
        self,
        work_dir: str,
        best_epoch: Optional[int],
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        if (
            work_dir != self._pending_round_dir
            or self._pending_final_round_dir is None
            or self._pending_round_number is None
        ):
            raise RuntimeError("Round output staging state is inconsistent.")

        final_dir = self._pending_final_round_dir
        round_number = int(self._pending_round_number)
        try:
            os.replace(work_dir, final_dir)
            self._pending_round_dir = final_dir

            if self._save_ckpt_every_round:
                checkpoint_meta = dict(extra_meta or {})
                checkpoint_meta["round_out_dir"] = final_dir
                if not self._save_checkpoint(
                    global_round=round_number,
                    best_epoch=best_epoch,
                    extra_meta=checkpoint_meta,
                ):
                    raise RuntimeError(
                        f"Checkpoint commit failed for round {round_number}."
                    )
                self._global_round_offset = round_number

            self._pending_round_dir = None
            self._pending_final_round_dir = None
            self._pending_round_number = None
            return final_dir
        except Exception:
            self.cleanup_incomplete_round()
            raise

    # Helpers
    def _encoder_config_with_max_len(self, encoder_cfg: Any, max_len: int):
        cfg = copy.deepcopy(encoder_cfg)
        model_cfg = (
            cfg.get("model")
            if isinstance(cfg, dict)
            else getattr(cfg, "model", None)
        )
        if model_cfg is None:
            raise ValueError(
                "Encoder config has no model section; cannot set model.max_len."
            )

        if isinstance(model_cfg, DictConfig):
            with read_write(model_cfg), open_dict(model_cfg):
                model_cfg.max_len = int(max_len)
        elif isinstance(model_cfg, dict):
            model_cfg["max_len"] = int(max_len)
        else:
            setattr(model_cfg, "max_len", int(max_len))

        return cfg

    def _parse_positions(self, cfg: Any) -> Optional[set]:
        if cfg is None:
            return None
        if isinstance(cfg, (list, tuple)):
            return {int(x) for x in cfg}
        if isinstance(cfg, str):
            parts = [p.strip() for p in cfg.split(",") if p.strip()]
            out = set()
            for p in parts:
                if "-" in p:
                    a, b = p.split("-", 1)
                    a, b = int(a), int(b)
                    lo, hi = (a, b) if a <= b else (b, a)
                    out.update(range(lo, hi + 1))
                else:
                    out.add(int(p))
            return out if out else None
        return None

    def _enumerate_mutation_windows(self, base_seqs):
        return enumerate_windows(
            base_seqs,
            self.encoder.tokenizer,
            self.window_size,
            allowed_token_positions=self.mutable_token_positions,
        )

    def _latent_mean_to_real_objectives(
        self,
        mean_latent_np,
        z_score_transform,
        sign_vec,
        weight_vec,
    ):
        z_hat = -mean_latent_np / weight_vec
        y_sign = z_score_transform.inv_transform(z_hat)
        y_real = y_sign * sign_vec
        return y_real

    def _collect_offline_suggestions(
        self,
        base_cand_batches,
        new_seq_batches,
        new_seq_scores,
        batch_entropy,
        all_seqs,
        batch_size: int,
        baseline_targets=None,
        ref_point=None,
    ):
        desired_n = (
            batch_size
            if self.offline_num_suggestions is None
            else int(self.offline_num_suggestions)
        )
        # Evaluation count may differ from acquisition q.
        desired_n = max(1, int(desired_n))
        if self.offline_selection_mode == "rank1_batch_only":
            # Rank 1 is limited to acquisition q.
            desired_n = int(batch_size)

        existing = set(str(x) for x in np.asarray(all_seqs, dtype=object).tolist())
        chosen_set = set()
        chosen: List[str] = []
        meta: List[Dict[str, Any]] = []

        order = np.argsort(new_seq_scores)
        rank_map = {int(b): int(i + 1) for i, b in enumerate(order)}

        def _try_add_from_batch(b: int) -> None:
            base_candidates_b = base_cand_batches[b]
            cand_seqs_b = new_seq_batches[b]
            try:
                new_cands_b = self.bb_task.make_new_candidates(base_candidates_b, cand_seqs_b)
            except Exception:
                return

            feas = self.bb_task.is_feasible(new_cands_b)
            if feas is None:
                feas = np.ones(len(cand_seqs_b), dtype=bool)
            feas = np.asarray(feas, dtype=bool)
            if feas.shape[0] != cand_seqs_b.shape[0]:
                feas = np.ones(len(cand_seqs_b), dtype=bool)

            for s in cand_seqs_b[feas]:
                ss = str(s)
                if (ss in existing) or (ss in chosen_set):
                    continue
                chosen.append(ss)
                chosen_set.add(ss)
                meta.append(
                    dict(
                        batch_rank=rank_map.get(int(b), None),
                        batch_idx=int(b) + 1,
                        batch_acq_val=float(np.mean(new_seq_scores[b])),
                        batch_entropy=float(batch_entropy[b]),
                    )
                )
                if len(chosen) >= desired_n:
                    return

        # Preserve the top acquisition-ranked batch.
        best_batch_idx = int(order[0])
        _try_add_from_batch(best_batch_idx)

        num_rank1 = len(chosen)
        if self.offline_selection_mode == "rank1_batch_only":
            self._log(
                "[offline] rank1_batch_only "
                f"requested_q={batch_size}, selected={num_rank1}, "
                "backfill=disabled"
            )
            return np.asarray(chosen, dtype=object), meta

        if (
            len(chosen) < desired_n
            and self.offline_backfill_from_ranked_batches
            and (baseline_targets is None or ref_point is None)
        ):
            for b in order[1:]:
                _try_add_from_batch(int(b))
                if len(chosen) >= desired_n:
                    break
            return np.asarray(chosen, dtype=object), meta

        if len(chosen) < desired_n and self.offline_backfill_from_ranked_batches:
            backfill_seqs: List[str] = []
            backfill_meta: List[Dict[str, Any]] = []
            backfill_seen = set()

            for b in order[1:]:
                b = int(b)
                base_candidates_b = base_cand_batches[b]
                cand_seqs_b = new_seq_batches[b]
                try:
                    new_cands_b = self.bb_task.make_new_candidates(
                        base_candidates_b,
                        cand_seqs_b,
                    )
                except Exception:
                    continue

                feas = self.bb_task.is_feasible(new_cands_b)
                if feas is None:
                    feas = np.ones(len(cand_seqs_b), dtype=bool)
                feas = np.asarray(feas, dtype=bool).reshape(-1)
                if feas.shape[0] != cand_seqs_b.shape[0]:
                    feas = np.ones(len(cand_seqs_b), dtype=bool)

                for s in cand_seqs_b[feas]:
                    ss = str(s)
                    if (
                        ss in existing
                        or ss in chosen_set
                        or ss in backfill_seen
                    ):
                        continue
                    backfill_seen.add(ss)
                    backfill_seqs.append(ss)
                    backfill_meta.append(
                        dict(
                            batch_rank=rank_map.get(b, None),
                            batch_idx=b + 1,
                            batch_acq_val=float(np.mean(new_seq_scores[b])),
                            batch_entropy=float(batch_entropy[b]),
                        )
                    )

            if backfill_seqs:
                prediction_seqs = np.asarray(
                    chosen + backfill_seqs,
                    dtype=object,
                )
                self.encoder.eval()
                self.surrogate_model.eval()
                with torch.inference_mode():
                    posterior = self.surrogate_model.predict_mvn(
                        prediction_seqs,
                        latent=True,
                    )
                prediction_means = posterior.mean.detach().cpu().numpy()
                if prediction_means.ndim == 1:
                    prediction_means = prediction_means.reshape(-1, 1)

                candidate_means = prediction_means[num_rank1:]
                baseline_targets = np.asarray(
                    baseline_targets,
                    dtype=float,
                ).reshape(-1, candidate_means.shape[-1])
                ref_point = np.asarray(ref_point, dtype=float).reshape(-1)
                if ref_point.shape[0] != candidate_means.shape[-1]:
                    raise ValueError(
                        "Reference-point dimension does not match GP objective dimension."
                    )

                hv_indicator = get_performance_indicator(
                    "hv",
                    ref_point=-ref_point,
                )
                working_points = baseline_targets.copy()
                if num_rank1:
                    working_points = np.concatenate(
                        [working_points, prediction_means[:num_rank1]],
                        axis=0,
                    )

                # Match the greedy HVI tie-breaker.
                mean_quality = candidate_means.mean(axis=1)
                remaining = list(range(len(backfill_seqs)))
                num_to_fill = desired_n - len(chosen)

                for _ in range(num_to_fill):
                    current_hv = self._maximization_hypervolume(
                        working_points,
                        ref_point,
                        hv_indicator,
                    )
                    best_idx = None
                    best_key = None

                    for idx in remaining:
                        trial_points = np.concatenate(
                            [working_points, candidate_means[idx : idx + 1]],
                            axis=0,
                        )
                        marginal_hvi = max(
                            0.0,
                            self._maximization_hypervolume(
                                trial_points,
                                ref_point,
                                hv_indicator,
                            )
                            - current_hv,
                        )
                        key = (
                            marginal_hvi,
                            mean_quality[idx],
                            -backfill_meta[idx]["batch_rank"],
                        )
                        if best_key is None or key > best_key:
                            best_idx = idx
                            best_key = key

                    if best_idx is None:
                        break

                    chosen.append(backfill_seqs[best_idx])
                    chosen_set.add(backfill_seqs[best_idx])
                    meta.append(backfill_meta[best_idx])
                    remaining.remove(best_idx)
                    working_points = np.concatenate(
                        [working_points, candidate_means[best_idx : best_idx + 1]],
                        axis=0,
                    )

        self._log(
            f"[offline] {self.offline_selection_mode} rank1="
            f"{num_rank1}, mean_hvi_backfill={len(chosen) - num_rank1}, "
            f"selected={len(chosen)}"
        )

        return np.asarray(chosen, dtype=object), meta

    def _collect_novel_feasible_candidate_pool(
        self,
        base_cand_batches,
        new_seq_batches,
        new_seq_scores,
        batch_entropy,
        all_seqs,
    ):
        existing = set(str(x) for x in np.asarray(all_seqs, dtype=object))
        seen = set()
        pool: List[str] = []
        meta: List[Dict[str, Any]] = []

        order = np.argsort(new_seq_scores)
        rank_map = {int(batch_idx): rank + 1 for rank, batch_idx in enumerate(order)}

        for batch_idx in order:
            batch_idx = int(batch_idx)
            base_candidates = base_cand_batches[batch_idx]
            candidate_seqs = new_seq_batches[batch_idx]
            try:
                new_candidates = self.bb_task.make_new_candidates(
                    base_candidates,
                    candidate_seqs,
                )
            except Exception:
                continue

            feasible = self.bb_task.is_feasible(new_candidates)
            if feasible is None:
                feasible = np.ones(candidate_seqs.shape[0], dtype=bool)
            feasible = np.asarray(feasible, dtype=bool).reshape(-1)
            if feasible.shape[0] != candidate_seqs.shape[0]:
                feasible = np.ones(candidate_seqs.shape[0], dtype=bool)

            for seq in candidate_seqs[feasible]:
                seq = str(seq)
                if seq in existing or seq in seen:
                    continue
                seen.add(seq)
                pool.append(seq)
                meta.append(
                    {
                        "source_batch_rank": int(rank_map[batch_idx]),
                        "source_batch_idx": batch_idx + 1,
                        "source_batch_score": float(new_seq_scores[batch_idx]),
                        "source_batch_entropy": float(batch_entropy[batch_idx]),
                    }
                )

        return np.asarray(pool, dtype=object), meta

    @staticmethod
    def _maximization_hypervolume(points, ref_point, indicator) -> float:
        points = np.asarray(points, dtype=float)
        ref_point = np.asarray(ref_point, dtype=float).reshape(-1)
        if points.size == 0:
            return 0.0
        points = points.reshape(-1, ref_point.shape[0])
        contributes = np.isfinite(points).all(axis=1) & np.all(
            points > ref_point.reshape(1, -1),
            axis=1,
        )
        if not bool(contributes.any()):
            return 0.0
        # pymoo expects minimization.
        return float(indicator.do(-points[contributes]))

    def _collect_greedy_mean_hvi_suggestions(
        self,
        base_cand_batches,
        new_seq_batches,
        new_seq_scores,
        batch_entropy,
        all_seqs,
        baseline_targets,
        ref_point,
        batch_size: int,
    ):
        candidate_seqs, candidate_meta = self._collect_novel_feasible_candidate_pool(
            base_cand_batches=base_cand_batches,
            new_seq_batches=new_seq_batches,
            new_seq_scores=new_seq_scores,
            batch_entropy=batch_entropy,
            all_seqs=all_seqs,
        )
        if candidate_seqs.size == 0:
            return candidate_seqs, []

        desired_n = (
            batch_size
            if self.offline_num_suggestions is None
            else int(self.offline_num_suggestions)
        )
        # Limit evaluation count to the generated pool.
        desired_n = max(1, min(desired_n, candidate_seqs.shape[0]))

        self.encoder.eval()
        self.surrogate_model.eval()
        with torch.inference_mode():
            posterior = self.surrogate_model.predict_mvn(
                candidate_seqs,
                latent=True,
            )
        candidate_means = posterior.mean.detach().cpu().numpy()
        if candidate_means.ndim == 1:
            candidate_means = candidate_means.reshape(-1, 1)

        baseline_targets = np.asarray(baseline_targets, dtype=float).reshape(
            -1,
            candidate_means.shape[-1],
        )
        ref_point = np.asarray(ref_point, dtype=float).reshape(-1)
        if ref_point.shape[0] != candidate_means.shape[-1]:
            raise ValueError(
                "Reference-point dimension does not match GP objective dimension."
            )

        hv_indicator = get_performance_indicator(
            "hv",
            ref_point=-ref_point,
        )
        baseline_hv = self._maximization_hypervolume(
            baseline_targets,
            ref_point,
            hv_indicator,
        )
        single_hvi = np.zeros(candidate_seqs.shape[0], dtype=float)
        for idx in range(candidate_seqs.shape[0]):
            trial_points = np.concatenate(
                [baseline_targets, candidate_means[idx : idx + 1]],
                axis=0,
            )
            single_hvi[idx] = max(
                0.0,
                self._maximization_hypervolume(
                    trial_points,
                    ref_point,
                    hv_indicator,
                )
                - baseline_hv,
            )

        # Break zero-HVI ties by normalized mean quality.
        mean_quality = candidate_means.mean(axis=1)

        selected_indices: List[int] = []
        selected_meta: List[Dict[str, Any]] = []
        remaining = list(range(candidate_seqs.shape[0]))
        working_points = baseline_targets.copy()

        for selection_rank in range(1, desired_n + 1):
            current_hv = self._maximization_hypervolume(
                working_points,
                ref_point,
                hv_indicator,
            )
            best_idx = None
            best_key = None
            best_marginal_hvi = 0.0

            for idx in remaining:
                trial_points = np.concatenate(
                    [working_points, candidate_means[idx : idx + 1]],
                    axis=0,
                )
                marginal_hvi = max(
                    0.0,
                    self._maximization_hypervolume(
                        trial_points,
                        ref_point,
                        hv_indicator,
                    )
                    - current_hv,
                )
                key = (
                    marginal_hvi,
                    mean_quality[idx],
                    -candidate_meta[idx]["source_batch_rank"],
                )
                if best_key is None or key > best_key:
                    best_idx = idx
                    best_key = key
                    best_marginal_hvi = marginal_hvi

            if best_idx is None:
                break

            selected_indices.append(best_idx)
            remaining.remove(best_idx)
            working_points = np.concatenate(
                [working_points, candidate_means[best_idx : best_idx + 1]],
                axis=0,
            )
            row_meta = dict(candidate_meta[best_idx])
            row_meta.update(
                {
                    "selection_mode": "greedy_mean_hvi",
                    "selection_rank": selection_rank,
                    "pred_single_hvi": float(single_hvi[best_idx]),
                    "pred_marginal_hvi": float(best_marginal_hvi),
                    "pred_mean_quality": float(mean_quality[best_idx]),
                }
            )
            selected_meta.append(row_meta)

        num_positive = sum(
            meta_row["pred_marginal_hvi"] > 0.0
            for meta_row in selected_meta
        )
        self._log(
            "[offline] greedy_mean_hvi candidate_pool="
            f"{candidate_seqs.shape[0]}, selected={len(selected_indices)}, "
            f"positive_marginal_hvi={num_positive}"
        )
        return candidate_seqs[np.asarray(selected_indices, dtype=int)], selected_meta

    def optimize(self, candidate_pool, pool_targets, all_seqs, all_targets, log_prefix: str = ""):
        project_root = self._resolve_project_root()
        assets_dir = os.path.join(project_root, "evomobo", "assets", "input_learning")

        # Convert objectives to minimization space.
        sign_vec = np.asarray(
            getattr(self.bb_task, "sign_vector", [1.0] * int(self.bb_task.obj_dim)), dtype=float
        )
        weight_vec = self.obj_weights
        objective_cols = tuple(
            getattr(
                self.bb_task,
                "objective_cols",
                tuple(f"objective{idx}" for idx in range(int(self.bb_task.obj_dim))),
            )
        )
        predicted_objective_cols = tuple(f"pred_{col}" for col in objective_cols)

        # Validate direction encoding.
        directions = getattr(self.bb_task, "objective_directions", ["min"] * int(self.bb_task.obj_dim))
        expected_sign = np.asarray([1.0 if str(d).lower() == "min" else -1.0 for d in directions], dtype=float)
        if expected_sign.shape != sign_vec.shape or not np.allclose(expected_sign, sign_vec, rtol=0, atol=0):
            raise ValueError(
                "[ASSERT FAIL] objective_directions and sign_vector are inconsistent.\n"
                f" objective_directions={list(directions)}\n"
                f" sign_vector={sign_vec.tolist()}\n"
                f" expected_sign(from directions)={expected_sign.tolist()}\n"
            )

        wt_fasta = os.path.join(assets_dir, "wt.fasta")

        def _load_wt_seq(fp: str) -> str:
            if not os.path.exists(fp):
                return ""
            seq_parts = []
            with open(fp, "r") as f:
                for line in f:
                    if line.startswith(">"):
                        continue
                    seq_parts.append(line.strip())
            return "".join(seq_parts)

        WT_SEQ = _load_wt_seq(wt_fasta)

        def _mutation_name(mut_seq: str, wt_seq: str) -> str:
            if not wt_seq or not mut_seq:
                return "NA"
            L = min(len(mut_seq), len(wt_seq))
            muts = []
            for i in range(L):
                a, b = wt_seq[i], mut_seq[i]
                if a != b:
                    muts.append(f"{a}{i+1}{b}")
            return "WT" if not muts else ",".join(muts)
            
        def _as_seq_str(x) -> str:
            return self.tokenizer.to_smiles(x) if hasattr(self.tokenizer, "to_smiles") else str(x)

        def _print_split_csv(tag: str, split: DataSplit, max_rows: int = 0):
            if not self.print_splits:
                return
            xs = np.asarray(getattr(split, "inputs", np.array([])), dtype=object)
            ys = np.asarray(getattr(split, "targets", np.array([])), dtype=float)
            n = int(xs.shape[0]) if xs is not None else 0

            self._log(f"\n[{tag}] N={n}")
            if n == 0:
                return

            if ys.ndim == 1:
                ys = ys.reshape(-1, 1)

            k = n if max_rows == 0 else min(n, int(max_rows))
            for i in range(k):
                seq_str = _as_seq_str(xs[i])
                nm = _mutation_name(seq_str, WT_SEQ)

                y_real = ys[i : i + 1, :] * sign_vec.reshape(1, -1)
                vals = [f"{float(v):.2f}" for v in y_real.flatten().tolist()]
                self._log(f"{nm},{seq_str}," + ",".join(vals))

            if max_rows != 0 and n > k:
                self._log(f"... (truncated) total={n}, printed={k}")

        out_base_dir = os.path.join(project_root, "evomobo", "assets", "active_learning")
        round_out_dir = self._begin_round_output(out_base_dir)

        batch_size = int(self.bb_task.batch_size)

        hv_reused = False
        if isinstance(self._hv_space, dict) and ("loc" in self._hv_space) and ("scale" in self._hv_space):
            loc = np.asarray(self._hv_space["loc"], dtype=float)
            scale = np.asarray(self._hv_space["scale"], dtype=float)
            if loc.shape != (int(self.bb_task.obj_dim),) or scale.shape != (int(self.bb_task.obj_dim),):
                raise ValueError(" hv_space loc/scale shape mismatch.")
            hypercube_transform = Normalizer(loc=loc, scale=scale)
            hv_reused = True
            self._log(" Reusing hypercube HV space (loc/scale) from checkpoint.")
        else:
            target_min = all_targets.min(axis=0).copy()
            target_range = all_targets.max(axis=0).copy() - target_min
            safe_range = np.where(np.abs(target_range) < 1e-12, 1.0, target_range)
            loc = target_min + 0.5 * safe_range
            scale = safe_range / 2.0
            hypercube_transform = Normalizer(loc=loc, scale=scale)
            self._log(" Initialized new hypercube HV space (loc/scale) from current all_targets.")

        observed_seqs = all_seqs.copy()
        observed_targets = all_targets.copy()
        observed_set = set(str(s) for s in np.asarray(observed_seqs, dtype=object).tolist())
        len_thr = getattr(self.bb_task, "seq_max_len", None)

        is_feasible = self.bb_task.is_feasible(candidate_pool)
        pool_candidates = candidate_pool[is_feasible]
        pool_targets = pool_targets[is_feasible]
        pool_seqs = np.array([p_cand.mutant_residue_seq for p_cand in pool_candidates])

        pool_csv = os.path.join(assets_dir, "pool.csv")
        global_points = []
        if os.path.exists(pool_csv) and self.use_pool_csv_points:
            df_pool = pd.read_csv(pool_csv)
            df_pool["mutant_sequence"] = df_pool["mutant_sequence"].astype(str).str.strip()

            missing = [s for s in df_pool["mutant_sequence"].tolist() if s not in observed_set]
            if missing:
                show = missing[:10]
                raise AssertionError(
                    "[ASSERT FAIL] pool.csv points are NOT all in observed_seqs.\n"
                    f" source: {pool_csv}\n"
                    f" missing_count: {len(missing)} (showing up to 10): {show}\n"
                    f" likely causes:\n"
                    f"   1) length filtering dropped them (task.seq_max_len={len_thr}).\n"
                    f"   2) data source mismatch: pool.csv not derived from current known_data/exam.\n"
                )

            global_points = [
                dict(
                    seq=r["mutant_sequence"],
                    name=r["Name"],
                    target=r.loc[list(objective_cols)].to_numpy(dtype=float) * sign_vec,
                )
                for _, r in df_pool.iterrows()
            ]

        for g in global_points:
            if g["seq"] in pool_seqs:
                continue
            cand = StringCandidate(
                wild_seq=g["seq"],
                mutation_list=[],
                tokenizer=self.tokenizer,
                wild_name=g["name"],
            )
            pool_candidates = np.concatenate((pool_candidates, np.array([cand], dtype=object)))
            pool_targets = np.concatenate((pool_targets, g["target"][None, :]))
            pool_seqs = np.concatenate((pool_seqs, np.array([g["seq"]], dtype=object)))

        self.active_candidates, self.active_targets = (pool_candidates, pool_targets)
        self.active_seqs = pool_seqs

        pareto_candidates, pareto_targets = pareto_frontier(self.active_candidates, self.active_targets)
        pareto_seqs = np.array([p_cand.mutant_residue_seq for p_cand in pareto_candidates])

        pareto_cand_history = pareto_candidates.copy()
        pareto_seq_history = pareto_seqs.copy()
        pareto_target_history = pareto_targets.copy()

        pareto_csv = os.path.join(assets_dir, "par_cand_extra.csv")
        history_points = []
        if os.path.exists(pareto_csv) and not self.use_pareto_extra_csv_points:
            self._log(f"NOTE: {pareto_csv} exists but use_pareto_extra_csv_points=False; skip loading it.")
        if os.path.exists(pareto_csv) and self.use_pareto_extra_csv_points:
            df_par = pd.read_csv(pareto_csv)
            df_par["mutant_sequence"] = df_par["mutant_sequence"].astype(str).str.strip()

            missing = [s for s in df_par["mutant_sequence"].tolist() if s not in observed_set]
            if missing:
                show = missing[:10]
                raise AssertionError(
                    "[ASSERT FAIL] par_cand_extra.csv points are NOT all in observed_seqs.\n"
                    f" source: {pareto_csv}\n"
                    f" missing_count: {len(missing)} (showing up to 10): {show}\n"
                    f" likely causes:\n"
                    f"   1) length filtering dropped them (task.seq_max_len={len_thr}).\n"
                    f"   2) data source mismatch: par_cand_extra.csv not derived from current known_data/exam.\n"
                )

            history_points = [
                dict(
                    seq=r["mutant_sequence"],
                    name=r["Name"],
                    target=r.loc[list(objective_cols)].to_numpy(dtype=float) * sign_vec,
                )
                for _, r in df_par.iterrows()
            ]

        for h in history_points:
            if h["seq"] in pareto_seq_history:
                continue
            cand = StringCandidate(
                wild_seq=h["seq"],
                mutation_list=[],
                tokenizer=self.tokenizer,
                wild_name=h["name"],
            )
            pareto_cand_history = np.concatenate((pareto_cand_history, np.array([cand], dtype=object)))
            pareto_target_history = np.concatenate((pareto_target_history, h["target"][None, :]))
            pareto_seq_history = np.concatenate((pareto_seq_history, np.array([h["seq"]], dtype=object)))

        norm_pareto_unweighted = hypercube_transform(pareto_targets)
        norm_pareto_targets = norm_pareto_unweighted * weight_vec

        ref_reused = False
        if isinstance(self._hv_space, dict) and ("ref_point" in self._hv_space):
            try:
                self._ref_point = np.asarray(self._hv_space["ref_point"], dtype=float)
                ref_reused = True
                self._log(" Reusing ref_point from checkpoint.")
            except Exception:
                ref_reused = False

        if not ref_reused:
            self._ref_point = -infer_reference_point(
                -torch.tensor(norm_pareto_targets, dtype=torch.float32)
            ).numpy()
            self._log(" Initialized new ref_point from current pareto (infer_reference_point).")

        eps = 1e-6
        need_worsen = np.maximum(0.0, (norm_pareto_targets.max(axis=0) + eps) - self._ref_point)
        if np.any(need_worsen > 0):
            old_ref = self._ref_point.copy()
            self._ref_point = np.maximum(self._ref_point, norm_pareto_targets.max(axis=0) + eps)
            self._log(" NOTE: ref_point was not worse than current pareto in minimize-space; auto-worsened.")
            self._log(f" old_ref: {old_ref}")
            self._log(f" new_ref: {self._ref_point}")

        if self._hv_space is None or not hv_reused:
            self._hv_space = {}
        self._hv_space["loc"] = np.asarray(loc, dtype=float).tolist()
        self._hv_space["scale"] = np.asarray(scale, dtype=float).tolist()
        self._hv_space["ref_point"] = np.asarray(self._ref_point, dtype=float).tolist()
        self._hv_space["obj_weights"] = np.asarray(weight_vec, dtype=float).tolist()
        self._hv_space["sign_vec"] = np.asarray(sign_vec, dtype=float).tolist()

        rescaled_ref_point = hypercube_transform.inv_transform(self._ref_point.copy() / weight_vec)
        ref_sign = rescaled_ref_point
        ref_real = rescaled_ref_point * sign_vec
        pareto_real = pareto_targets * sign_vec

        self._log("\n[reference point] sign-space (minimize-space):")
        self._log(str(ref_sign))
        self._log("[reference point] real-space (objective display):")
        self._log(str(ref_real))
        self._log("[pareto real-space range] per-objective min/max:")
        self._log(f" min: {pareto_real.min(axis=0)}")
        self._log(f" max: {pareto_real.max(axis=0)}")

        total_bb_evals = 0
        start_time = time.time()
        round_idx = 0
        checkpoint_best_epoch = None
        checkpoint_meta = None

        self._log_candidates(pareto_candidates, pareto_targets, round_idx, log_prefix)
        metrics = self._log_optimizer_metrics(
            norm_pareto_targets, round_idx, total_bb_evals, start_time, log_prefix
        )

        self._log("\n best candidates")
        disp = pareto_targets * sign_vec
        obj_vals = {}
        for i, d in enumerate(directions):
            vals = disp[:, i]
            obj_vals[f"obj_val_{i}"] = float(vals.max() if str(d).lower() == "max" else vals.min())
        self._log(self._format_df(pd.DataFrame([obj_vals]), floatfmt=".4f"))

        for round_idx in range(1, self.num_rounds + 1):
            metrics = {}

            if (
                self.concentrate_pool > 0 and round_idx % self.concentrate_pool == 0
            ) or self.latent_init == "perturb_pareto":
                self.active_candidates, self.active_targets = pareto_frontier(
                    self.active_candidates, self.active_targets
                )
                self.active_seqs = np.array([a.mutant_residue_seq for a in self.active_candidates])
                self._log(f"\nactive set contracted to {self.active_candidates.shape[0]} pareto points")

            if self.active_candidates.shape[0] < batch_size:
                num_samples = min(batch_size, pareto_cand_history.shape[0])
                num_backtrack = min(num_samples, batch_size - self.active_candidates.shape[0])
                _, weights, _ = weighted_resampling(pareto_target_history, k=self.resampling_weight)
                hist_idxs = np.random.choice(
                    np.arange(pareto_cand_history.shape[0]),
                    num_samples,
                    p=weights,
                    replace=False,
                )
                is_active = np.in1d(pareto_seq_history[hist_idxs], self.active_seqs)
                hist_idxs = hist_idxs[~is_active]
                if hist_idxs.size > 0:
                    hist_idxs = hist_idxs[:num_backtrack]
                    backtrack_candidates = pareto_cand_history[hist_idxs]
                    backtrack_targets = pareto_target_history[hist_idxs]
                    backtrack_seqs = pareto_seq_history[hist_idxs]
                    self.active_candidates = np.concatenate((self.active_candidates, backtrack_candidates))
                    self.active_targets = np.concatenate((self.active_targets, backtrack_targets))
                    self.active_seqs = np.concatenate((self.active_seqs, backtrack_seqs))
                    self._log(f"active set augmented with {backtrack_candidates.shape[0]} backtrack points")

            if self.active_candidates.shape[0] < batch_size:
                num_samples = min(batch_size, pool_candidates.shape[0])
                num_rand = min(num_samples, batch_size - self.active_candidates.shape[0])
                _, weights, _ = weighted_resampling(pool_targets, k=self.resampling_weight)
                rand_idxs = np.random.choice(
                    np.arange(pool_candidates.shape[0]),
                    num_samples,
                    p=weights,
                    replace=False,
                )
                is_active = np.in1d(pool_seqs[rand_idxs], self.active_seqs)
                rand_idxs = rand_idxs[~is_active][:num_rand]
                rand_candidates = pool_candidates[rand_idxs]
                rand_targets = pool_targets[rand_idxs]
                rand_seqs = pool_seqs[rand_idxs]
                self.active_candidates = np.concatenate((self.active_candidates, rand_candidates))
                self.active_targets = np.concatenate((self.active_targets, rand_targets))
                self.active_seqs = np.concatenate((self.active_seqs, rand_seqs))
                self._log(f"active set augmented with {rand_candidates.shape[0]} random points")

            self._log("\nreference point (real-space; display only)")
            self._log(str(ref_real))

            _active_rows = []
            self._log(
                "\nactive set (Name,sequence," + ",".join(objective_cols) + "):"
            )
            for seq, obj in zip(self.active_seqs, self.active_targets):
                seq_str = _as_seq_str(seq)
                nm = _mutation_name(seq_str, WT_SEQ)
                obj_disp = obj * sign_vec
                obj_text = ",".join(f"{float(value):.2f}" for value in obj_disp)
                self._log(f"{nm},{seq_str},{obj_text}")
                row = {"Name": nm, "mutant_sequence": seq_str}
                row.update(
                    {
                        col: float(obj_disp[idx])
                        for idx, col in enumerate(objective_cols)
                    }
                )
                _active_rows.append(row)

            self._write_csv(
                pd.DataFrame(_active_rows),
                os.path.join(round_out_dir, "active_set.csv"),
                float_format="%.2f",
                quoting=csv.QUOTE_MINIMAL,
            )

            self._log("\n---- fitting surrogate model ----")

            mu = observed_targets.mean(0)
            sd = observed_targets.std(0)
            sd = np.where(np.abs(sd) < 1e-12, 1.0, sd)
            z_score_transform = Normalizer(mu, sd)

            def tgt_transform(x):
                z = z_score_transform(x)
                z_weighted = z * weight_vec
                return -z_weighted  # Maximization labels.

            transformed_ref_point = tgt_transform(ref_sign)

            observed_split = DataSplit(observed_seqs, observed_targets)
            holdout_ratio = self.surrogate_model.holdout_ratio
            self.train_split, self.val_split, self.test_split = update_splits(
                self.train_split,
                self.val_split,
                self.test_split,
                observed_split,
                holdout_ratio,
            )

            _print_split_csv("TRAIN", self.train_split, max_rows=self.print_splits_max_rows)
            _print_split_csv("VAL", self.val_split, max_rows=self.print_splits_max_rows)
            _print_split_csv("TEST", self.test_split, max_rows=self.print_splits_max_rows)

            X_train, Y_train = (self.train_split.inputs, tgt_transform(self.train_split.targets))
            X_val, Y_val = (self.val_split.inputs, tgt_transform(self.val_split.targets))
            X_test, Y_test = (self.test_split.inputs, tgt_transform(self.test_split.targets))

            with isolated_seed(self.surrogate_training_seed):
                records = self.surrogate_model.fit(
                    X_train,
                    Y_train,
                    X_val,
                    Y_val,
                    X_test,
                    Y_test,
                    encoder_obj=self.encoder_obj,
                    resampling_temp=None,
                    csv_dir=round_out_dir,
                )

            # Handle missing best-epoch metadata.
            last_entry = {str(k).split("/")[-1]: v for k, v in records[-1].items()}
            best_idx_raw = last_entry.get("best_epoch", None)
            try:
                best_idx = int(best_idx_raw) if best_idx_raw is not None else 0
            except Exception:
                best_idx = 0
            best_idx = max(0, min(best_idx, len(records) - 1))
            best_entry = {str(k).split("/")[-1]: v for k, v in records[best_idx].items()}

            self._log(self._format_df(pd.DataFrame([best_entry]), floatfmt=".4f"))

            metrics.update(
                dict(
                    test_rmse=best_entry.get("test_rmse", np.nan),
                    test_nll=best_entry.get("test_nll", np.nan),
                    test_s_rho=best_entry.get("test_s_rho", np.nan),
                    test_ece=best_entry.get("test_ece", np.nan),
                    test_post_var=best_entry.get("test_post_var", np.nan),
                    test_perplexity=best_entry.get("test_perplexity", np.nan),
                    round_idx=round_idx,
                    num_bb_evals=total_bb_evals,
                    num_train=int(X_train.shape[0]),
                    time_elapsed=time.time() - start_time,
                )
            )

            prefixed = {"/".join((log_prefix, "opt_metrics", k)): v for k, v in metrics.items()}
            self._wandb_log(prefixed)

            transformed_observed_targets = tgt_transform(observed_targets)

            checkpoint_best_epoch = (
                int(best_idx) if best_idx is not None else None
            )
            checkpoint_meta = {
                "select_crit_key": "val_nll",
                "offline_mode": bool(self.offline_mode),
                "offline_selection_mode": self.offline_selection_mode,
                "hv_space_reused": bool(hv_reused),
                "ref_point_reused": bool(ref_reused),
            }

            baseline_candidates, baseline_targets = pareto_frontier(
                self.active_candidates,
                self.active_targets,
            )
            baseline_seqs = np.asarray(
                [candidate.mutant_residue_seq for candidate in baseline_candidates],
                dtype=object,
            )
            baseline_targets = tgt_transform(baseline_targets)

            acq_fn = hydra.utils.instantiate(
                self.acquisition,
                X_baseline=baseline_seqs,
                known_targets=torch.as_tensor(baseline_targets, dtype=torch.float32, device=self.surrogate_model.device),
                surrogate=self.surrogate_model,
                ref_point=torch.as_tensor(transformed_ref_point, dtype=torch.float32, device=self.surrogate_model.device),
                obj_dim=int(self.bb_task.obj_dim),
            )

            self._log("\n---- optimizing candidates ----")
            if self.resampling_weight is None:
                weights = np.ones(self.active_targets.shape[0]) / self.active_targets.shape[0]
            else:
                _, weights, _ = weighted_resampling(self.active_targets, k=self.resampling_weight)

            base_cand_batches = []
            new_seq_batches = []
            new_seq_scores = []
            batch_entropy = []

            for gen_idx in range(self.num_gens):
                base_idxs = np.random.choice(
                    np.arange(weights.shape[0]),
                    batch_size,
                    p=weights,
                    replace=True,
                )
                base_candidates = self.active_candidates[base_idxs]
                base_seqs = np.array([cand.mutant_residue_seq for cand in base_candidates])

                base_tok_idxs = str_to_tokens(base_seqs, self.encoder.tokenizer)
                base_mask = base_tok_idxs != self.encoder.tokenizer.padding_idx
                base_lens = base_mask.float().sum(-1).long()
                tgt_lens = None if self.bb_task.allow_len_change else base_lens

                window_mask_idxs = self._enumerate_mutation_windows(base_seqs)

                if self.position_sampler == "entropy_method":
                    with torch.no_grad():
                        window_mask_idxs, window_entropy_map = evaluate_windows(
                            base_seqs,
                            self.encoder,
                            self.window_size,
                            replacement=True,
                            encoder_obj=self.encoder_obj,
                            window_mask_idxs=window_mask_idxs,
                        )
                    mask_idxs = self.sample_mutation_window(window_mask_idxs, window_entropy_map)
                elif self.position_sampler == "uniform":
                    mask_idxs = np.concatenate(
                        [random.choice(w_idxs) for w_idxs in window_mask_idxs.values()]
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
                        mask_idxs_t = torch.from_numpy(mask_idxs).to(device=device, dtype=torch.long)

                        src_tok_idxs_masked = src_tok_idxs.clone()
                        src_tok_idxs_masked.scatter_(1, mask_idxs_t, self.encoder.tokenizer.masking_idx)

                        src_tok_features, src_mask = self.encoder.get_token_features(src_tok_idxs_masked)
                        feat_dim = src_tok_features.size(-1)
                        gather_idx = mask_idxs_t.unsqueeze(-1).expand(-1, -1, feat_dim)
                        opt_features = src_tok_features.gather(1, gather_idx)

                    else:
                        raise ValueError(f"Unknown encoder_obj: {self.encoder_obj}")

                    opt_params = opt_features.clone().detach()
                    opt_origin = opt_features.clone().detach()
                    opt_params.requires_grad_(self.optimize_latent)

                optimizer = torch.optim.Adam(
                    params=[opt_params], lr=self.lr, betas=(0.0, 1e-2)
                )
                lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=self.patience
                )
                best_score, best_step = None, 0
                best_seqs = base_seqs
                best_entropy = float("nan")

                for step_idx in range(self.num_opt_steps):
                    opt_params.grad = None

                    if self.encoder_obj == "lanmt":
                        lat_tok_features, pooled_features = self.encoder.pool_features(
                            opt_params,
                            src_mask,
                            src_tok_idxs,
                        )
                        tgt_tok_logits, tgt_mask = self.encoder.logits_from_features(
                            opt_params, src_mask, lat_tok_features, tgt_lens
                        )
                        tgt_tok_idxs, logit_entropy = self.encoder.sample_tgt_tok_idxs(
                            tgt_tok_logits,
                            tgt_mask,
                            temp=1.0,
                        )

                    elif self.encoder_obj == "mlm":
                        device = self.surrogate_model.device
                        mask_idxs_t = torch.from_numpy(mask_idxs).to(device=device, dtype=torch.long)

                        current_features = src_tok_features.clone()
                        feat_dim = current_features.size(-1)
                        scatter_idx = mask_idxs_t.unsqueeze(-1).expand(-1, -1, feat_dim)
                        current_features = current_features.scatter(1, scatter_idx, opt_params)

                        lat_tok_features, pooled_features = self.encoder.pool_features(
                            current_features,
                            src_mask,
                            src_tok_idxs_masked,
                        )
                        tgt_tok_logits, tgt_mask = self.encoder.logits_from_features(
                            current_features,
                            src_mask,
                            lat_tok_features,
                            tgt_lens,
                        )

                        new_tok_idxs_full, logit_entropy_full = sample_tokens(
                            base_tok_idxs.to(device),
                            tgt_tok_logits,
                            self.encoder.tokenizer,
                            replacement=False,
                        )

                        new_tok_idxs_mut = new_tok_idxs_full.gather(
                            1,
                            mask_idxs_t,
                        )
                        tgt_tok_idxs = src_tok_idxs.clone()
                        tgt_tok_idxs.scatter_(
                            1,
                            mask_idxs_t,
                            new_tok_idxs_mut,
                        )

                        logit_entropy_mut = logit_entropy_full.gather(
                            1,
                            mask_idxs_t,
                        )

                    else:
                        raise ValueError(f"Unknown encoder_obj: {self.encoder_obj}")

                    lat_acq_vals = acq_fn(pooled_features.unsqueeze(0))
                    if self.encoder_obj == "mlm":
                        entropy_term = logit_entropy_mut.mean()
                    else:
                        entropy_term = logit_entropy[tgt_mask.bool()].mean()
                    loss = -lat_acq_vals.mean() + self.entropy_penalty * entropy_term

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
                        lr_sched.step(loss)

                    tgt_seqs = tokens_to_str(
                        tgt_tok_idxs,
                        self.encoder.tokenizer,
                    )
                    with torch.no_grad():
                        act_acq_vals = acq_fn(tgt_seqs[None, :]).mean().item()

                    best_score, best_step, _, stop = check_early_stopping(
                        model=None,
                        best_score=best_score,
                        best_epoch=best_step,
                        best_weights=None,
                        curr_score=-act_acq_vals,
                        curr_epoch=step_idx + 1,
                        patience=self.patience,
                        save_weights=False,
                    )
                    if (step_idx + 1) == best_step:
                        best_seqs = tgt_seqs.copy()
                        best_entropy = entropy_term.item()
                    if stop:
                        self._log(f"Early stopping at step {step_idx + 1}")
                        break

                base_cand_batches.append(base_candidates.copy())
                new_seq_batches.append(best_seqs.copy())
                new_seq_scores.append(best_score)
                batch_entropy.append(best_entropy)

            new_seq_batches = np.stack(new_seq_batches)
            new_seq_scores = np.stack(new_seq_scores)
            best_batch_idx = int(new_seq_scores.argmin())

            _sorted_idx = np.argsort(new_seq_scores)
            _max_batches = len(new_seq_batches)

            self._log("\n=== All candidate batches this round (display only; offline mode will output suggestions) ===")
            for _gen in range(_max_batches):
                _acq_val = float(np.mean(new_seq_scores[_gen]))
                _entropy = float(batch_entropy[_gen])
                _rank = int(np.where(_sorted_idx == _gen)[0][0] + 1)

                _tbl = pd.DataFrame(
                    [
                        dict(
                            gen_idx=_gen + 1,
                            rank=_rank,
                            acq_val=_acq_val,
                            entropy=_entropy,
                            round_idx=round_idx,
                            num_bb_evals=total_bb_evals,
                            time_elapsed=time.time() - start_time,
                        )
                    ]
                )
                self._log(self._format_df(_tbl, floatfmt=".4f"))

                self.encoder.eval()
                self.surrogate_model.eval()
                with torch.inference_mode():
                    mvn_all = self.surrogate_model.predict_mvn(new_seq_batches[_gen], latent=True)
                mean_all_np = mvn_all.mean.cpu().numpy()
                mean_all_disp = self._latent_mean_to_real_objectives(
                    mean_latent_np=mean_all_np,
                    z_score_transform=z_score_transform,
                    sign_vec=sign_vec,
                    weight_vec=weight_vec,
                )

                for i, _seq in enumerate(new_seq_batches[_gen]):
                    _s = _as_seq_str(_seq)
                    _name = _mutation_name(str(_s), WT_SEQ)
                    pred_text = ",".join(
                        f"{float(value):.2f}" for value in mean_all_disp[i]
                    )
                    self._log(f"{_name},{_s},{pred_text}")
                self._log("\n" + "-" * 60)

                _seq_list = [_as_seq_str(x) for x in new_seq_batches[_gen]]
                _name_list = [_mutation_name(str(s), WT_SEQ) for s in _seq_list]
                batch_data = {
                    "Name": _name_list,
                    "mutant_sequence": [str(s) for s in _seq_list],
                }
                batch_data.update(
                    {
                        col: mean_all_disp[:, idx]
                        for idx, col in enumerate(predicted_objective_cols)
                    }
                )
                _df_batch = pd.DataFrame(batch_data)
                self._write_csv(
                    _df_batch,
                    os.path.join(round_out_dir, f"rank{_rank}.csv"),
                    float_format="%.6f",
                    quoting=csv.QUOTE_MINIMAL,
                )

            metrics = dict(
                acq_val=float(np.mean(new_seq_scores[best_batch_idx])),
                entropy=float(batch_entropy[best_batch_idx]),
                round_idx=round_idx,
                num_bb_evals=total_bb_evals,
                time_elapsed=time.time() - start_time,
            )
            self._log(self._format_df(pd.DataFrame([metrics]), floatfmt=".4f"))
            self._wandb_log({"/".join((log_prefix, "opt_metrics", k)): v for k, v in metrics.items()})

            if self.offline_mode:
                if self.offline_selection_mode == "greedy_mean_hvi":
                    suggested_seqs, meta = self._collect_greedy_mean_hvi_suggestions(
                        base_cand_batches=base_cand_batches,
                        new_seq_batches=new_seq_batches,
                        new_seq_scores=new_seq_scores,
                        batch_entropy=batch_entropy,
                        all_seqs=observed_seqs,
                        baseline_targets=transformed_observed_targets,
                        ref_point=transformed_ref_point,
                        batch_size=batch_size,
                    )
                else:
                    suggested_seqs, meta = self._collect_offline_suggestions(
                        base_cand_batches=base_cand_batches,
                        new_seq_batches=new_seq_batches,
                        new_seq_scores=new_seq_scores,
                        batch_entropy=batch_entropy,
                        all_seqs=observed_seqs,
                        baseline_targets=transformed_observed_targets,
                        ref_point=transformed_ref_point,
                        batch_size=batch_size,
                    )

                self._log(
                    f"[offline] Selection mode: {self.offline_selection_mode}"
                )

                if suggested_seqs.size == 0:
                    self._log("[offline] WARNING: no novel feasible suggestions found. Stop.", level="warning")
                    self.cleanup_incomplete_round()
                    return metrics

                self.encoder.eval()
                self.surrogate_model.eval()
                with torch.inference_mode():
                    mvn = self.surrogate_model.predict_mvn(suggested_seqs, latent=True)
                mean_np = mvn.mean.cpu().numpy()
                pred_real = self._latent_mean_to_real_objectives(
                    mean_latent_np=mean_np,
                    z_score_transform=z_score_transform,
                    sign_vec=sign_vec,
                    weight_vec=weight_vec,
                )

                name_list = [_mutation_name(str(s), WT_SEQ) for s in suggested_seqs]
                output_data = {
                    "Name": name_list,
                    "mutant_sequence": [str(s) for s in suggested_seqs],
                }
                output_data.update(
                    {
                        col: pred_real[:, idx]
                        for idx, col in enumerate(predicted_objective_cols)
                    }
                )
                df_out = pd.DataFrame(output_data)
                if meta is not None and len(meta) == len(df_out):
                    for k in meta[0].keys():
                        df_out[k] = [m.get(k, None) for m in meta]

                out_path = os.path.join(round_out_dir, self.offline_suggest_csv)
                self._write_csv(df_out, out_path, float_format="%.6f", quoting=csv.QUOTE_MINIMAL)

                exam_data = {
                    "Name": name_list,
                    "mutant_sequence": [str(s) for s in suggested_seqs],
                }
                exam_data.update(
                    {col: np.nan for col in objective_cols}
                )
                df_exam = pd.DataFrame(exam_data)
                exam_path = os.path.join(round_out_dir, self.offline_exam_template_csv)
                self._write_csv(df_exam, exam_path, quoting=csv.QUOTE_MINIMAL)

                final_round_dir = self._commit_round_output(
                    round_out_dir,
                    best_epoch=checkpoint_best_epoch,
                    extra_meta=checkpoint_meta,
                )
                final_out_path = os.path.join(
                    final_round_dir,
                    self.offline_suggest_csv,
                )
                final_exam_path = os.path.join(
                    final_round_dir,
                    self.offline_exam_template_csv,
                )
                self._log(
                    f"\n[offline] Suggestions saved to: {final_out_path}"
                )
                self._log(
                    f"[offline] Exam template saved to: {final_exam_path}"
                )
                self._log("[offline] Stop after suggest-only (no pseudo-labels were added).")
                return metrics

        if checkpoint_meta is None:
            self.cleanup_incomplete_round()
        else:
            self._commit_round_output(
                round_out_dir,
                best_epoch=checkpoint_best_epoch,
                extra_meta=checkpoint_meta,
            )
        return metrics

    def sample_mutation_window(self, window_mask_idxs, window_entropy, temp: float = 1.0):
        selected_mask_idxs = []
        for seq_idx, entropies in window_entropy.items():
            mask_idxs = window_mask_idxs[seq_idx]
            assert len(mask_idxs) == len(entropies)
            window_idxs = np.arange(len(mask_idxs)).astype(int)
            entropies_t = torch.tensor(entropies)
            weights = F.softmax(entropies_t / temp, dim=0).cpu().numpy()
            selected_window = np.random.choice(window_idxs, 1, p=weights).item()
            selected_mask_idxs.append(mask_idxs[selected_window])
        return np.concatenate(selected_mask_idxs)

    def _log_candidates(self, candidates, targets, round_idx: int, log_prefix: str):
        sign_vec = np.asarray(
            getattr(self.bb_task, "sign_vector", [1.0] * int(self.bb_task.obj_dim)), dtype=float
        )
        table_cols = ["round_idx", "cand_uuid", "cand_ancestor", "cand_seq"]
        table_cols.extend([f"obj_val_{idx}" for idx in range(int(self.bb_task.obj_dim))])

        for cand, obj in zip(candidates, targets):
            disp = obj * sign_vec
            new_row = [round_idx, cand.uuid, cand.wild_name, cand.mutant_residue_seq]
            new_row.extend([float(elem) for elem in disp])
            record = {
                "/".join((log_prefix, "candidates", key)): val
                for key, val in zip(table_cols, new_row)
            }
            self._wandb_log(record)

    def _log_optimizer_metrics(self, normed_targets, round_idx: int, num_bb_evals: int, start_time: float, log_prefix: str):
        hv_indicator = get_performance_indicator("hv", ref_point=self._ref_point)
        new_hypervol = hv_indicator.do(normed_targets)
        self._hv_ref = new_hypervol if self._hv_ref is None else self._hv_ref

        metrics = dict(
            round_idx=round_idx,
            hypervol_abs=float(new_hypervol),
            hypervol_rel=float(new_hypervol / max(1e-6, float(self._hv_ref))),
            num_bb_evals=int(num_bb_evals),
            time_elapsed=float(time.time() - start_time),
        )
        self._log(self._format_df(pd.DataFrame([metrics]), floatfmt=".4f"))
        self._wandb_log({"/".join((log_prefix, "opt_metrics", k)): v for k, v in metrics.items()})
        return metrics
