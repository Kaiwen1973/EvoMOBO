from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf, open_dict


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else project_root / path


def _resolve_checkpoint(cfg, project_root: Path) -> Path:
    if cfg.ckpt_path:
        path = _resolve_path(project_root, cfg.ckpt_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    candidates = [
        project_root / "evomobo_checkpoints" / str(cfg.ckpt_name) / "latest.pt",
        project_root / "single_objective_checkpoints" / str(cfg.ckpt_name) / "latest.pt",
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected exactly one checkpoint for ckpt_name="
            f"{cfg.ckpt_name!r}; found {matches}. Set ckpt_path explicitly."
        )
    return matches[0]


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _read_sequences(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "mutant_sequence" not in df.columns:
        raise ValueError(f"Input CSV must contain mutant_sequence: {path}")
    if "Name" not in df.columns:
        df.insert(0, "Name", "")
    df = df.copy()
    df["mutant_sequence"] = df["mutant_sequence"].fillna("").astype(str).str.strip()
    df = df[df["mutant_sequence"].str.len() > 0].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No non-empty sequences found in {path}")
    return df


def _checkpoint_observations(ckpt):
    sequences = []
    targets = []
    splits = ckpt.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("Checkpoint does not contain train/val/test splits.")

    for name in ("train", "val", "test"):
        split = splits.get(name) or {}
        split_x = split.get("inputs")
        split_y = split.get("targets")
        if split_x is None or split_y is None or len(split_x) == 0:
            continue
        split_y = np.asarray(split_y, dtype=float)
        if split_y.ndim == 1:
            split_y = split_y[:, None]
        sequences.extend(str(seq) for seq in split_x)
        targets.append(split_y)

    if not sequences or not targets:
        raise ValueError("Checkpoint splits contain no observations.")
    target_array = np.concatenate(targets, axis=0)
    if len(sequences) != len(target_array):
        raise ValueError("Checkpoint sequence and target counts do not match.")
    return np.asarray(sequences, dtype=object), target_array


def _checkpoint_train_split(ckpt):
    split = (ckpt.get("splits") or {}).get("train") or {}
    split_sequences = split.get("inputs")
    split_targets = split.get("targets")
    sequences = np.asarray(
        [] if split_sequences is None else split_sequences,
        dtype=object,
    )
    targets = np.asarray(
        [] if split_targets is None else split_targets,
        dtype=float,
    )
    if sequences.size == 0 or targets.size == 0:
        raise ValueError("Checkpoint train split contains no observations.")
    if targets.ndim == 1:
        targets = targets[:, None]
    if len(sequences) != len(targets):
        raise ValueError("Checkpoint train split sequence and target counts do not match.")
    return sequences, targets


def _surrogate_config(
    project_root: Path,
    obj_dim: int,
    encoder_batch_size: int,
    surrogate_target: str | None = None,
):
    target_to_filename = {
        "models.ibnn_models.IBNNMultiTaskExactGP": (
            "multi_task_ibnn.yaml"
        ),
        "models.vbll_models.MultiTaskVBLL": "multi_task_vbll.yaml",
        "models.gaussian_mlp_models.MultiTaskGaussianMLP": (
            "multi_task_gaussian_mlp.yaml"
        ),
    }
    target_implementation = str(surrogate_target).split(".", 1)[-1]
    filename = target_to_filename.get(target_implementation)
    if filename is None:
        filename = (
            "single_task_exact_gp.yaml"
            if obj_dim == 1
            else "multi_task_exact_gp.yaml"
        )
    config = OmegaConf.load(project_root / "hydra_config" / "surrogate" / filename)
    with open_dict(config):
        config.bs = int(encoder_batch_size)
        if "out_dim" in config:
            config.out_dim = int(obj_dim)
        if obj_dim == 1 and "covar_module" in config:
            config.covar_module.ard_num_dims = int(config.feature_dim)
        elif "task_covar_prior" in config:
            config.task_covar_prior.n = int(obj_dim)
            config.likelihood.num_tasks = int(obj_dim)
            if (
                "data_covar_module" in config
                and "ard_num_dims" in config.data_covar_module
            ):
                config.data_covar_module.ard_num_dims = int(config.feature_dim)
    return config


def _restore_real_scale(mean, observed_targets, meta):
    obj_dim = observed_targets.shape[1]
    sign = np.asarray(meta.get("sign_vector") or [1.0] * obj_dim, dtype=float)
    weights = np.asarray(meta.get("obj_weights") or [1.0] * obj_dim, dtype=float)
    if sign.shape != (obj_dim,) or weights.shape != (obj_dim,):
        raise ValueError("Checkpoint objective metadata has the wrong dimension.")

    center = observed_targets.mean(axis=0)
    scale = observed_targets.std(axis=0)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    signed_values = (-mean / weights.reshape(1, -1)) * scale + center
    return signed_values * sign.reshape(1, -1)


@hydra.main(config_path="../hydra_config", config_name="inference")
def main(cfg) -> None:
    project_root = Path(hydra.utils.get_original_cwd())
    input_path = _resolve_path(project_root, cfg.input_csv)
    output_path = _resolve_path(project_root, cfg.output_csv)
    checkpoint_path = _resolve_checkpoint(cfg, project_root)

    df = _read_sequences(input_path)
    ckpt = _load_checkpoint(checkpoint_path)
    observed_sequences, observed_targets = _checkpoint_observations(ckpt)
    train_sequences, train_targets = _checkpoint_train_split(ckpt)
    obj_dim = int(observed_targets.shape[1])
    meta = ckpt.get("meta") or {}
    if int(meta.get("bb_obj_dim", obj_dim)) != obj_dim:
        raise ValueError("Checkpoint objective dimension is inconsistent with its splits.")

    seq_max_len = int(meta.get("task_seq_max_len") or max(map(len, observed_sequences)))
    max_len = int(meta.get("task_max_len") or (seq_max_len + 2))
    with open_dict(cfg.task):
        cfg.task.seq_max_len = seq_max_len
        cfg.task.max_len = max_len
        cfg.task.obj_dim = obj_dim
    with open_dict(cfg.encoder.model):
        cfg.encoder.model.max_len = max_len

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    encoder = hydra.utils.instantiate(cfg.encoder, tokenizer=tokenizer)
    surrogate_cfg = _surrogate_config(
        project_root,
        obj_dim=obj_dim,
        encoder_batch_size=int(cfg.encoder.batch_size),
        surrogate_target=meta.get("surrogate_target"),
    )
    with open_dict(surrogate_cfg):
        surrogate_cfg.pop("training_seed", None)
    surrogate = hydra.utils.instantiate(
        surrogate_cfg,
        tokenizer=encoder.tokenizer,
        encoder=encoder,
    )

    encoder.load_state_dict(ckpt["encoder_state"], strict=False)
    surrogate.load_state_dict(ckpt["surrogate_state"], strict=False)

    device = torch.device(
        "cuda" if str(cfg.device).lower() == "auto" and torch.cuda.is_available()
        else "cpu" if str(cfg.device).lower() == "auto"
        else str(cfg.device)
    )
    surrogate = surrogate.to(device)
    surrogate.device = device
    encoder.device = device
    surrogate.eval()
    surrogate.requires_grad_(False)

    center = observed_targets.mean(axis=0)
    scale = observed_targets.std(axis=0)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    weights = np.asarray(meta.get("obj_weights") or [1.0] * obj_dim, dtype=float)
    transformed_targets = -(
        (train_targets - center.reshape(1, -1))
        / scale.reshape(1, -1)
        * weights.reshape(1, -1)
    )
    surrogate.set_train_data(
        train_sequences,
        torch.as_tensor(transformed_targets, dtype=torch.float32, device=device),
        strict=False,
    )

    sequences = df["mutant_sequence"].to_numpy(dtype=object)
    chunk_size = max(1, int(cfg.predict_chunk_size))
    predicted_chunks = []
    with torch.inference_mode():
        for start in range(0, len(sequences), chunk_size):
            chunk = sequences[start : start + chunk_size]
            _, mean, _ = surrogate.predict(chunk, latent=True)
            predicted_chunks.append(mean.numpy())

    predicted_train_scale = np.concatenate(predicted_chunks, axis=0)
    predictions = _restore_real_scale(
        predicted_train_scale,
        observed_targets,
        meta,
    )
    for idx in range(obj_dim):
        df[f"pred_objective{idx}"] = predictions[:, idx]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"Predicted {len(df)} sequences with {obj_dim} objective(s).")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
