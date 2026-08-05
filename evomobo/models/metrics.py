import torch


def quantile_calibration(mean, std, targets):
    quantiles = torch.linspace(0.05, 0.95, 10, device=mean.device).view(10, 1, 1)

    z_dist = torch.distributions.Normal(
        torch.tensor((0.0,), device=mean.device),
        torch.tensor((1.0,), device=mean.device),
    )
    tail_probs = (1 - quantiles) / 2
    z_scores = z_dist.icdf(1 - tail_probs)

    pred_mean = mean.unsqueeze(0)
    pred_std = std.unsqueeze(0)
    lb = pred_mean - z_scores * pred_std
    ub = pred_mean + z_scores * pred_std

    targets = targets.unsqueeze(0)
    targets_in_region = torch.le(lb, targets) * torch.le(targets, ub)
    occupancy_rates = targets_in_region.float().mean(-1, keepdim=True)
    occupancy_rates = occupancy_rates.mean(-2, keepdim=True)
    ece = (occupancy_rates - quantiles).abs().mean().item()
    calibration_metrics = {
        f"{quantile.item():.2f}_quantile": occ_rate.item()
        for quantile, occ_rate in zip(quantiles, occupancy_rates)
    }
    calibration_metrics["ece"] = ece
    calibration_metrics["occ_diff"] = (quantiles - occupancy_rates).mean().item()
    return calibration_metrics
