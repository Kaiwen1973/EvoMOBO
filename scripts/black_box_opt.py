import logging
import random
import sys
import warnings
from pathlib import Path

# Prefer this checkout over stale editable installs.
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
try:
    sys.path.remove(str(_SOURCE_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(_SOURCE_ROOT))

import hydra
import wandb
from gpytorch.settings import max_cholesky_size
from omegaconf import OmegaConf

from upcycle.logging.analysis import flatten_config
from upcycle.scripting import startup

from evomobo.tasks.opt_task.opt_task import (
    infer_objective_columns,
    input_learning_sequence_paths,
    resolve_sequence_lengths,
    sync_resolved_objectives,
    sync_resolved_lengths,
)
from evomobo.tasks.single_objective_task import infer_single_objective_columns
from evomobo.wandb_utils import configure_wandb_logging


LOGGER = logging.getLogger(__name__)


def _is_single_objective(config) -> bool:
    task_target = str(getattr(config.task, "_target_", ""))
    return (
        "SingleObjectiveTask" in task_target
        or getattr(config.task, "objective_direction", None) is not None
    )


def _flatten_cfg_for_wandb(cfg) -> dict:
    flat = flatten_config(OmegaConf.to_container(cfg, resolve=True), sep="/")
    return {"config/" + k: v for k, v in flat.items()}


def _resolve_wandb_settings(config) -> tuple[bool, str]:
    enabled = bool(getattr(config, "wandb_enabled", True))
    mode = str(getattr(config, "wandb_mode", "online")).strip().lower()
    valid_modes = {"online", "offline", "disabled"}
    if mode not in valid_modes:
        raise ValueError(
            f"wandb_mode must be one of {sorted(valid_modes)}, got {mode!r}."
        )
    return enabled and mode != "disabled", mode


def _resolve_config_sequence_lengths(config, project_root: Path) -> None:
    if _is_single_objective(config):
        sequence_paths = input_learning_sequence_paths(str(project_root))
    else:
        sequence_paths = input_learning_sequence_paths(str(project_root))
    seq_max_len, max_len = resolve_sequence_lengths(
        config.task.seq_max_len,
        config.task.max_len,
        sequence_paths,
    )
    sync_resolved_lengths(config, seq_max_len, max_len)


def _resolve_config_objectives(config, project_root: Path) -> None:
    if _is_single_objective(config):
        known_path = (
            project_root
            / "evomobo"
            / "assets"
            / "input_learning"
            / "known_data.csv"
        )
        infer_single_objective_columns(str(known_path))
        LOGGER.info(
            "[black_box_opt] Effective single objective: column=objective0"
        )
        return

    known_path = (
        project_root
        / "evomobo"
        / "assets"
        / "input_learning"
        / "known_data.csv"
    )
    inferred_cols = infer_objective_columns(str(known_path))
    inferred_dim = len(inferred_cols)
    configured_dim = getattr(config.task, "obj_dim", None)
    if configured_dim is not None and int(configured_dim) != inferred_dim:
        raise ValueError(
            "Configured task.obj_dim does not match known_data.csv. "
            f"configured={configured_dim}, inferred={inferred_dim}, "
            f"columns={list(inferred_cols)}. Use task=multi_objective_task "
            "to infer the objective count automatically."
        )
    sync_resolved_objectives(config, inferred_dim)
    LOGGER.info(
        "[black_box_opt] Effective objectives: obj_dim=%d, columns=%s",
        inferred_dim,
        list(inferred_cols),
    )


def _validate_effective_sequence_lengths(config, bb_task) -> None:
    seq_lengths = {
        "config.task.seq_max_len": int(config.task.seq_max_len),
        "bb_task.seq_max_len": int(bb_task.seq_max_len),
    }
    max_lengths = {
        "config.task.max_len": int(config.task.max_len),
        "config.encoder.model.max_len": int(config.encoder.model.max_len),
        "bb_task.max_len": int(bb_task.max_len),
    }
    if len(set(seq_lengths.values())) != 1 or len(set(max_lengths.values())) != 1:
        raise ValueError(
            "Sequence-length configuration is out of sync: "
            f"seq_lengths={seq_lengths}, max_lengths={max_lengths}"
        )

    seq_max_len = next(iter(seq_lengths.values()))
    max_len = next(iter(max_lengths.values()))
    if max_len < seq_max_len + 2:
        raise ValueError(
            "Effective max_len must be at least seq_max_len + 2; "
            f"got seq_max_len={seq_max_len}, max_len={max_len}."
        )

    LOGGER.info(
        "[black_box_opt] Effective sequence lengths: seq_max_len=%d, max_len=%d",
        seq_max_len,
        max_len,
    )


@hydra.main(config_path="../hydra_config", config_name="black_box_opt")
def main(config):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 

        random.seed(None)

        project_root = Path(hydra.utils.get_original_cwd())
        _resolve_config_objectives(config, project_root)
        _resolve_config_sequence_lengths(config, project_root)
        wandb_enabled, wandb_mode = _resolve_wandb_settings(config)
        configure_wandb_logging(False)

        config, _ = startup(config)

        run = None
        if wandb_enabled:
            try:
                log_config = _flatten_cfg_for_wandb(config)

                if wandb_mode == "online":
                    wandb.login(host=getattr(config, "wandb_host", None))
                run = wandb.init(
                    project=getattr(config, "project_name", "OPT_TASK"),
                    config=log_config,
                    mode=wandb_mode,
                    group=getattr(config, "exp_name", None),
                    reinit=True,
                )
                if run is None:
                    raise RuntimeError("wandb.init returned no active run.")
                configure_wandb_logging(True)
                LOGGER.info(
                    "[black_box_opt] WandB initialized with mode=%s.",
                    wandb_mode,
                )
            except Exception as e:
                LOGGER.warning(
                    "[black_box_opt] Failed to initialize wandb; continue without wandb. Error=%s",
                    str(e),
                )
                run = None
                configure_wandb_logging(False)
        else:
            LOGGER.info(
                "[black_box_opt] WandB disabled by configuration "
                "(wandb_enabled=%s, wandb_mode=%s).",
                getattr(config, "wandb_enabled", True),
                wandb_mode,
            )

        if run is not None:
            try:
                config.job_name = run.name
            except Exception:
                try:
                    config["job_name"] = run.name
                except Exception:
                    pass

        ret_val = float("NaN")
        optimizer = None
        try:
            tokenizer = hydra.utils.instantiate(config.tokenizer)
            bb_task = hydra.utils.instantiate(
                config.task, tokenizer=tokenizer, candidate_pool=[]
            )

            base_candidates, base_targets, all_seqs, all_targets = bb_task.task_setup(
                config, project_root=project_root
            )
            _validate_effective_sequence_lengths(config, bb_task)

            if run is not None:
                try:
                    run.config.update(
                        _flatten_cfg_for_wandb(config),
                        allow_val_change=True,
                    )
                except Exception:
                    pass

            max_chol_sz = int(getattr(config.surrogate, "max_cholesky_size", int(1e5)))
            with max_cholesky_size(max_chol_sz):
                optimizer = hydra.utils.instantiate(
                    config.optimizer,
                    bb_task=bb_task,
                    surrogate=config.surrogate,
                    acquisition=config.acquisition,
                    encoder=config.encoder,
                    tokenizer=tokenizer,
                )
                surrogate_cls = type(optimizer.surrogate_model)
                surrogate_module = sys.modules.get(surrogate_cls.__module__)
                LOGGER.info(
                    "[black_box_opt] Surrogate implementation: %s.%s (%s)",
                    surrogate_cls.__module__,
                    surrogate_cls.__name__,
                    getattr(surrogate_module, "__file__", "unknown"),
                )

                metrics = optimizer.optimize(
                    base_candidates,
                    base_targets,
                    all_seqs,
                    all_targets,
                    log_prefix=config.task.log_prefix,
                )

            metrics = {k.split("/")[-1]: v for k, v in metrics.items()}
            metric_key = (
                "best_observed"
                if _is_single_objective(config)
                else "hypervol_rel"
            )
            ret_val = float(metrics.get(metric_key, float("NaN")))

        except Exception as err:
            if optimizer is not None:
                try:
                    optimizer.cleanup_incomplete_round()
                except Exception as cleanup_err:
                    LOGGER.error(
                        "[black_box_opt] Failed to clean incomplete round: %s",
                        str(cleanup_err),
                    )
            LOGGER.exception("[black_box_opt] Fatal error: %s", str(err))
            ret_val = float("NaN")

        if run is not None:
            try:
                run.finish()
            except Exception:
                pass
        configure_wandb_logging(False)

        return ret_val


if __name__ == "__main__":
    main()
    sys.exit()
