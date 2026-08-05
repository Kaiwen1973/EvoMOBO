from __future__ import annotations

import copy
import logging
import math
import os
import re
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

from botorch.models import KroneckerMultiTaskGP, SingleTaskGP

try:
    from gpytorch import lazify 
except Exception: 
    try:
        from linear_operator import to_linear_operator as lazify  
    except Exception:
        lazify = None  

try:
    from gpytorch.lazy import ConstantDiagLazyTensor 
except Exception: 
    try:
        from linear_operator.operators import DiagLinearOperator as ConstantDiagLazyTensor
    except Exception:
        ConstantDiagLazyTensor = None

from gpytorch.settings import cholesky_jitter
from gpytorch.variational import IndependentMultitaskVariationalStrategy

from evomobo import transforms as gfp_transforms
from evomobo.models.lanmt import lanmt_eval_epoch, lanmt_train_step
from evomobo.models.lm_elements import LanguageModel
from evomobo.models.mlm import mlm_eval_epoch, mlm_train_step
from evomobo.models.shared_elements import check_early_stopping
from evomobo.wandb_utils import wandb_log
from evomobo.utils import batched_call, draw_bootstrap, to_tensor, weighted_resampling

logger = logging.getLogger(__name__)



def _resolve_project_root() -> str:
    try:
        import hydra 
        return hydra.utils.get_original_cwd()
    except Exception:
        return os.getcwd()


def _parse_round_dirname(name: str) -> Optional[int]:
    m = re.fullmatch(r"round(\d+)", str(name).strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _latest_active_learning_round_dir(project_root: str) -> Tuple[str, bool]:
    env_dir = os.environ.get("EVOMOBO_ROUND_OUT_DIR", "").strip()
    if env_dir:
        env_dir = os.path.expanduser(env_dir)
        if not os.path.isabs(env_dir):
            env_dir = os.path.join(project_root, env_dir)
        if os.path.isdir(env_dir):
            return env_dir, False

    base = os.path.join(project_root, "evomobo", "assets", "active_learning")
    os.makedirs(base, exist_ok=True)

    max_n = None
    for entry in os.listdir(base):
        p = os.path.join(base, entry)
        if not os.path.isdir(p):
            continue
        n = _parse_round_dirname(entry)
        if n is None:
            continue
        max_n = n if (max_n is None or n > max_n) else max_n

    if max_n is None:
        out_dir = os.path.join(base, "round1")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir, True

    out_dir = os.path.join(base, f"round{max_n}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, False


def _resolve_gp_records_dir(csv_dir: Optional[str]) -> str:
    if csv_dir is not None:
        s = str(csv_dir).strip()
        if s not in {"", ".", "./"}:
            return s

    project_root = _resolve_project_root()
    round_dir, created = _latest_active_learning_round_dir(project_root)
    if created:
        logger.info("No existing round dir found; created: %s", round_dir)
    else:
        logger.info("Auto-resolved round dir for gp_training csv: %s", round_dir)
    return round_dir



def compute_mll_terms(mvn_dist, targets):
    mean, covar = mvn_dist.loc, mvn_dist.lazy_covariance_matrix
    diff = targets.view(*mean.shape) - mean

    if diff.shape[:-1] != covar.batch_shape:
        if len(diff.shape[:-1]) < len(covar.batch_shape):
            diff = diff.expand(covar.shape[:-1])
        else:
            padded_batch_shape = (
                *(1 for _ in range(diff.dim() + 1 - covar.dim())),
                *covar.batch_shape,
            )
            covar = covar.repeat(
                *(
                    diff_size // covar_size
                    for diff_size, covar_size in zip(diff.shape[:-1], padded_batch_shape)
                ),
                1,
                1,
            )

    covar = covar.evaluate_kernel()
    inv_quad, logdet = covar.inv_quad_logdet(inv_quad_rhs=diff.unsqueeze(-1), logdet=True)
    return inv_quad, logdet


def gp_train_step(surrogate, optimizer, inputs, targets, mll):
    surrogate.zero_grad(set_to_none=True)
    surrogate.clear_cache()

    features = surrogate.get_features(inputs.to(surrogate.device), surrogate.bs, transform=False)

    dtype = features.dtype if (features.dtype is not torch.bool) else torch.float32
    targets = surrogate.reshape_targets(targets).to(features.device, dtype=dtype)

    if isinstance(surrogate, (SingleTaskGP, KroneckerMultiTaskGP)):
        surrogate.set_train_data(features, targets, strict=False)

    output = surrogate.forward(features)
    loss = -mll(output, targets).mean()

    loss.backward()
    optimizer.step()
    return loss


def fit_encoder_only(surrogate, optimizer, mll, train_loader, num_epochs: int):
    assert hasattr(surrogate, "encoder")
    surrogate.requires_grad_(False)
    surrogate.encoder.requires_grad_(True)

    total = 0.0
    steps = 0
    for _ in range(int(num_epochs)):
        surrogate.train()
        for inputs, targets in train_loader:
            loss = gp_train_step(surrogate, optimizer, inputs, targets, mll)
            total += float(loss.detach().item())
            steps += 1

    return total / max(1, steps)


def fit_gp_surrogate(
    surrogate,
    mll,
    X_train,
    Y_train,
    X_val,
    Y_val,
    X_test,
    Y_test,
    eval_bs: Optional[int] = None,
    train_bs: Optional[int] = None,
    shuffle_train: bool = False,
    log_prefix: str = "",
    encoder_obj: Optional[str] = "mll",
    resampling_temp: Optional[float] = None,
    select_crit_key: str = "val_nll",
    save_csv: bool = True,
    csv_dir: Optional[str] = ".",  # "." resolves to the round directory.
    csv_name: Optional[str] = None,
):
    assert encoder_obj in ["mll", "mlm", "lanmt", None], "unsupported encoder objective"

    num_val = 0 if X_val is None else int(X_val.shape[0])
    logger.info(f"{X_train.shape[0]} train, {num_val} val, {X_test.shape[0]} test")

    if surrogate.bootstrap_ratio is not None:
        X_train, Y_train = draw_bootstrap(
            X_train,
            Y_train,
            bootstrap_ratio=surrogate.bootstrap_ratio,
            min_samples=surrogate.min_num_train,
        )

    if resampling_temp is not None:
        logger.info("\n---- resampling training/validation data ----")
        _, _, train_idxs = weighted_resampling(-Y_train, k=resampling_temp)
        X_train, Y_train = X_train[train_idxs], Y_train[train_idxs]

        if X_val is not None and Y_val is not None:
            _, _, val_idxs = weighted_resampling(-Y_val, k=resampling_temp)
            X_val, Y_val = X_val[val_idxs], Y_val[val_idxs]

    def collate_fn(x):
        return gfp_transforms.padding_collate_fn(x, surrogate.tokenizer.padding_idx)

    train_bs = int(X_train.shape[0]) if train_bs is None else int(train_bs)

    if num_val > 0:
        _, val_dataset = surrogate._get_datasets(X_train, X_val, Y_train, Y_val)
    else:
        val_dataset = None

    train_dataset, test_dataset = surrogate._get_datasets(X_train, X_test, Y_train, Y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=shuffle_train,
        collate_fn=collate_fn,
    )

    if eval_bs is None:
        if X_val is None:
            eval_bs = int(X_test.shape[0])
        else:
            eval_bs = int(max(X_val.shape[0], X_test.shape[0]))
    else:
        eval_bs = int(eval_bs)

    val_loader = (
        DataLoader(val_dataset, batch_size=eval_bs, shuffle=False, collate_fn=collate_fn)
        if val_dataset is not None
        else None
    )
    test_loader = DataLoader(test_dataset, batch_size=eval_bs, shuffle=False, collate_fn=collate_fn)

    Y_train_t = to_tensor(Y_train, device=surrogate.device)
    Y_train_t = surrogate.reshape_targets(Y_train_t)
    Y_train_t = Y_train_t.to(dtype=list(surrogate.parameters())[0].dtype)

    has_encoder = hasattr(surrogate, "encoder") and len(list(getattr(surrogate.encoder, "parameters", lambda: [])())) > 0
    is_lm_encoder = bool(has_encoder and isinstance(surrogate.encoder, LanguageModel))

    if (not is_lm_encoder) and encoder_obj in ("mlm", "lanmt"):
        logger.info(
            "encoder_obj='%s' requested, but encoder is not a LanguageModel; fallback to 'mll'.",
            encoder_obj,
        )
        effective_encoder_obj = "mll"
    else:
        effective_encoder_obj = encoder_obj

    logger.info("\n---- preparing checkpoint ----")
    surrogate.eval()
    surrogate.requires_grad_(False)
    surrogate.set_train_data(X_train, Y_train_t, strict=False)

    start_metrics = {}
    if val_loader is not None:
        start_metrics.update(surrogate.evaluate(val_loader, split="val"))
    start_metrics.update(surrogate.evaluate(test_loader, split="test"))
    start_metrics["epoch"] = 0

    if has_encoder and is_lm_encoder and effective_encoder_obj == "mlm":
        if val_loader is not None:
            start_metrics.update(mlm_eval_epoch(surrogate.encoder, val_loader, surrogate.encoder.mask_ratio, split="val"))
        start_metrics.update(mlm_eval_epoch(surrogate.encoder, test_loader, surrogate.encoder.mask_ratio, split="test"))

    if has_encoder and is_lm_encoder and effective_encoder_obj == "lanmt":
        if val_loader is not None:
            start_metrics.update(lanmt_eval_epoch(surrogate.encoder.model, val_loader, split="val"))
        start_metrics.update(lanmt_eval_epoch(surrogate.encoder.model, test_loader, split="test"))

    best_score = start_metrics.get(select_crit_key, None)
    best_score_epoch = 0

    surrogate.cpu()
    best_weights = copy.deepcopy(surrogate.state_dict())
    surrogate.to(surrogate.device)

    if best_score is not None:
        logger.info(f"starting {select_crit_key}: {best_score:.4f}")

    if has_encoder and any(isinstance(module, _BatchNorm) for module in surrogate.encoder.modules()):
        logger.info("\n---- initializing encoder normalization buffers ----")
        num_warmup_epochs = 8
        surrogate.train()
        surrogate.requires_grad_(False)
        for _ in range(num_warmup_epochs):
            for inputs, _ in train_loader:
                _ = surrogate.get_features(inputs.to(surrogate.device), surrogate.bs, transform=False)

    if hasattr(surrogate, "init_inducing_points") and getattr(surrogate, "num_inducing_points", 0) <= X_train.shape[0]:
        logger.info("\n---- initializing GP variational params ----")
        surrogate.eval()
        surrogate.requires_grad_(False)
        init_features = torch.cat(batched_call(surrogate.get_features, X_train, train_bs))
        try:
            surrogate.train()
            surrogate.init_inducing_points(init_features)
            initialize_var_dist_sgpr(
                surrogate,
                init_features,
                Y_train_t.to(init_features),
                noise_lb=1.0,
            )
            logger.info("variational initialization successful")
        except Exception as exp:
            logger.exception("variational initialization failed: %s", exp)
            logger.info("variational initialization failed")

    mll.to(surrogate.device)
    if hasattr(mll, "num_data"):
        mll.num_data = len(train_loader.dataset)

    stop_crit_key = "train_loss"
    best_loss, best_loss_epoch = None, 0
    stop = False

    gp_optimizer = torch.optim.Adam(surrogate.param_groups)
    gp_lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        gp_optimizer,
        patience=math.ceil(surrogate.patience / 2.0),
        threshold=1e-3,
    )

    records = [start_metrics]
    logger.info("\n---- fitting all params ----")
    for epoch_idx in range(int(surrogate.num_epochs)):
        metrics = {}

        if has_encoder and effective_encoder_obj == "mll":
            enc_sup_loss = fit_encoder_only(surrogate, gp_optimizer, mll, train_loader, num_epochs=1)
        else:
            enc_sup_loss = 0.0

        avg_train_loss = float(enc_sup_loss)

        surrogate.train()
        for inputs, targets in train_loader:
            if has_encoder and is_lm_encoder and effective_encoder_obj == "mlm":
                surrogate.encoder.requires_grad_(True)
                mlm_loss, _, _ = mlm_train_step(
                    surrogate.encoder,
                    gp_optimizer,
                    inputs,
                    surrogate.encoder.mask_ratio,
                    loss_scale=1.0,
                )
            elif has_encoder and is_lm_encoder and effective_encoder_obj == "lanmt":
                surrogate.encoder.requires_grad_(True)
                mlm_loss, _, _ = lanmt_train_step(
                    surrogate.encoder.model,
                    gp_optimizer,
                    inputs,
                    loss_scale=1.0,
                )
            else:
                mlm_loss = torch.zeros((), device=surrogate.device)

            surrogate.requires_grad_(True)
            gp_loss = gp_train_step(surrogate, gp_optimizer, inputs, targets, mll)

            avg_train_loss += float((mlm_loss.detach() + gp_loss.detach()).item()) / len(train_loader)

        gp_lr_sched.step(avg_train_loss)

        metrics.update({"epoch": epoch_idx + 1, "train_loss": float(avg_train_loss)})

        if (epoch_idx + 1) % int(surrogate.eval_period) == 0:
            surrogate.requires_grad_(False)
            surrogate.eval()
            surrogate.set_train_data(X_train, Y_train_t, strict=False)

            if val_loader is not None:
                metrics.update(surrogate.evaluate(val_loader, split="val"))
            metrics.update(surrogate.evaluate(test_loader, split="test"))

            if has_encoder and is_lm_encoder and effective_encoder_obj == "mlm":
                if val_loader is not None:
                    metrics.update(mlm_eval_epoch(surrogate.encoder, val_loader, surrogate.encoder.mask_ratio, split="val"))
                metrics.update(mlm_eval_epoch(surrogate.encoder, test_loader, surrogate.encoder.mask_ratio, split="test"))

            elif has_encoder and is_lm_encoder and effective_encoder_obj == "lanmt":
                if val_loader is not None:
                    metrics.update(lanmt_eval_epoch(surrogate.encoder.model, val_loader, split="val"))
                metrics.update(lanmt_eval_epoch(surrogate.encoder.model, test_loader, split="test"))

        select_crit = metrics.get(select_crit_key, None)
        if select_crit is not None:
            best_score, best_score_epoch, best_weights, _ = check_early_stopping(
                model=surrogate,
                best_score=best_score,
                best_epoch=best_score_epoch,
                best_weights=best_weights,
                curr_score=select_crit,
                curr_epoch=epoch_idx + 1,
                patience=int(surrogate.patience),
                save_weights=True,
            )

        metrics.update({"best_score": best_score, "best_epoch": int(best_score_epoch)})

        stop_crit = metrics.get(stop_crit_key, None)
        if stop_crit is not None:
            best_loss, best_loss_epoch, _, stop = check_early_stopping(
                model=surrogate,
                best_score=best_loss,
                best_epoch=best_loss_epoch,
                best_weights=None,
                curr_score=stop_crit,
                curr_epoch=epoch_idx + 1,
                patience=int(surrogate.patience),
                save_weights=False,
            )

        metrics.update({"best_loss": best_loss, "best_loss_epoch": int(best_loss_epoch)})
        records.append(metrics)

        if log_prefix:
            log_metrics = {"/".join((log_prefix, k)): v for k, v in metrics.items()}
        else:
            log_metrics = metrics
        wandb_log(log_metrics)

        if bool(getattr(surrogate, "early_stopping", True)) and stop:
            break

    # Restore the best validation checkpoint.
    if best_weights is not None:
        logger.info(f"\n---- loading best checkpoint: epoch={best_score_epoch}, {select_crit_key}={best_score} ----")
        surrogate.load_state_dict(best_weights, strict=False)

    surrogate.requires_grad_(False)
    surrogate.train()
    surrogate.clear_cache()
    surrogate.eval()
    surrogate.set_train_data(X_train, Y_train_t, strict=False)

    # Save records in the round directory.
    if save_csv:
        try:
            if csv_name is None:
                csv_name = f"gp_training_{log_prefix}.csv" if log_prefix else "gp_training.csv"

            resolved_dir = _resolve_gp_records_dir(csv_dir)
            os.makedirs(resolved_dir, exist_ok=True)
            csv_path = os.path.join(resolved_dir, csv_name)

            pd.DataFrame(records).to_csv(csv_path, index=False)
            logger.info(f"\ntraining records saved to: {csv_path}")
        except Exception as e:
            logger.exception("Failed to save training records CSV: %s", e)

    return records


def initialize_var_dist_sgpr(model, train_x, train_y, noise_lb: float):
    if lazify is None or ConstantDiagLazyTensor is None:
        raise RuntimeError(
            "initialize_var_dist_sgpr requires gpytorch.lazify and ConstantDiagLazyTensor "
            "(or compatible linear_operator equivalents) in your environment."
        )

    if isinstance(model.model.variational_strategy, IndependentMultitaskVariationalStrategy):
        ind_pts = model.model.variational_strategy.base_variational_strategy.inducing_points
        train_y = train_y.transpose(-1, -2).unsqueeze(-1)
        is_batch_model = True
    else:
        ind_pts = model.model.variational_strategy.inducing_points
        is_batch_model = False

    with cholesky_jitter(1e-4):
        kuu = model.model.covar_module(ind_pts).double()
        kuu_chol = kuu.cholesky()
        kuv = model.model.covar_module(ind_pts, train_x).double()

        if hasattr(model.likelihood, "noise"):
            noise = model.likelihood.noise
        elif hasattr(model.likelihood, "task_noises"):
            noise = model.likelihood.task_noises.view(-1, 1, 1)
        else:
            raise AttributeError("Likelihood has no noise attribute.")
        noise = noise.clamp(min=noise_lb).double()

        if len(train_y.shape) < len(kuv.shape):
            train_y = train_y.unsqueeze(-1)
        if len(noise.shape) < len(kuv.shape):
            noise = noise.unsqueeze(-1)

        data_term = kuv.matmul(train_y.double()) / noise

        if is_batch_model:
            noise_as_lt = ConstantDiagLazyTensor(noise.squeeze(-1), diag_shape=kuv.shape[-1])
            inner_prod = kuv.matmul(noise_as_lt).matmul(kuv.transpose(-1, -2))
            inner_term = inner_prod + kuu
        else:
            inner_term = kuv @ kuv.transpose(-1, -2) / noise + kuu

        s_mat = kuu_chol.transpose(-1, -2).matmul(inner_term.inv_matmul(kuu_chol.evaluate()))
        s_root = lazify(s_mat).cholesky().evaluate()

        mean_param = kuu_chol.transpose(-1, -2).matmul(inner_term.inv_matmul(data_term))

    mean_param = mean_param.to(train_y)
    s_root = s_root.to(train_y)

    if not is_batch_model:
        dist = model.model.variational_strategy._variational_distribution
        dist.variational_mean.data = mean_param.data.detach().squeeze()
        dist.chol_variational_covar.data = s_root.data.detach()
        model.model.variational_strategy.variational_params_initialized.fill_(1)
    else:
        base_vs = model.model.variational_strategy.base_variational_strategy
        dist = base_vs._variational_distribution
        dist.variational_mean.data = mean_param.data.detach().squeeze()
        dist.chol_variational_covar.data = s_root.data.detach()
        base_vs.variational_params_initialized.fill_(1)
