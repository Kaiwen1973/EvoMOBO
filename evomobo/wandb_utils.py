from typing import Any, Mapping

import wandb


_WANDB_LOGGING_ENABLED = False


def configure_wandb_logging(enabled: bool) -> None:
    global _WANDB_LOGGING_ENABLED
    _WANDB_LOGGING_ENABLED = bool(enabled)


def wandb_run_is_active() -> bool:
    return _WANDB_LOGGING_ENABLED and wandb.run is not None


def wandb_log(data: Mapping[str, Any]) -> None:
    if not wandb_run_is_active():
        return
    try:
        wandb.log(dict(data))
    except Exception:
        pass
