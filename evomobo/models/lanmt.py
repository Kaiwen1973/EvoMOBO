import torch
from torch.nn import functional as F
import math
import torchvision

from evomobo.models.mlm import sample_mask
from evomobo import dataset as gfp_dataset, transforms as gfp_transforms
from evomobo.models.shared_elements import check_early_stopping
from evomobo.wandb_utils import wandb_log


def _model_device(model):
    if hasattr(model, "device"):
        try:
            return torch.device(model.device)
        except Exception:
            pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _call_dec_tok_features(model, src_tok_features, src_mask, tgt_lens):
    return model.dec_tok_features(src_tok_features, src_mask, None, tgt_lens)


def _align_targets_to_logits(tgt_tok_idxs, tgt_tok_logits, padding_idx):
    num_logits = int(tgt_tok_logits.size(1))
    num_targets = int(tgt_tok_idxs.size(1))

    if num_targets > num_logits:
        return tgt_tok_idxs[:, :num_logits]
    if num_targets < num_logits:
        pad = tgt_tok_idxs.new_full(
            (tgt_tok_idxs.size(0), num_logits - num_targets),
            int(padding_idx),
        )
        return torch.cat([tgt_tok_idxs, pad], dim=1)
    return tgt_tok_idxs


def _sequence_perplexity(tgt_tok_logits, tgt_tok_idxs, padding_idx):
    tgt_tok_idxs = _align_targets_to_logits(tgt_tok_idxs, tgt_tok_logits, padding_idx)
    tgt_tok_idxs = tgt_tok_idxs.to(tgt_tok_logits.device)
    tgt_mask = tgt_tok_idxs.ne(padding_idx).to(dtype=tgt_tok_logits.dtype)

    log_prob = F.log_softmax(tgt_tok_logits, dim=-1)
    log_prob = log_prob.gather(-1, tgt_tok_idxs.unsqueeze(-1)).squeeze(-1)
    log_prob = log_prob * tgt_mask

    denom = tgt_mask.sum().clamp_min(1.0)
    return torch.exp(-log_prob.sum() / denom)


def _insert_tokens(src_row, selected, new_tok_idxs):
    out = src_row
    for pos, tok in sorted(
        zip(selected.detach().cpu().tolist(), new_tok_idxs.detach().cpu().tolist()),
        reverse=True,
    ):
        pos = max(0, min(int(pos), out.size(0)))
        tok_t = out.new_tensor([int(tok)])
        out = torch.cat([out[:pos], tok_t, out[pos:]], dim=0)
    return out


def corrupt_tok_idxs(tgt_tok_idxs, tokenizer, max_len_delta, select_idxs=None):
    device = tgt_tok_idxs.device
    viable_idxs = torch.as_tensor(
        tokenizer.non_special_idxs + [tokenizer.masking_idx],
        dtype=torch.long,
        device=device,
    )

    if select_idxs is None:
        max_len_delta = int(max_len_delta or 0)
        if max_len_delta <= 0:
            return tgt_tok_idxs.clone()

        rand_idxs = sample_mask(tgt_tok_idxs, tokenizer, mask_size=max_len_delta)
        rand_idxs = rand_idxs.to(device=device, dtype=torch.long)
        len_deltas = torch.randint(
            0,
            max_len_delta + 1,
            (tgt_tok_idxs.size(0),),
            device=device,
        )
        rand_tok_idxs = viable_idxs[
            torch.randint(viable_idxs.numel(), rand_idxs.shape, device=device)
        ]
    else:
        select_idxs = torch.as_tensor(select_idxs, dtype=torch.long, device=device)
        if select_idxs.dim() == 1:
            select_idxs = select_idxs.unsqueeze(0)
        rand_tok_idxs = viable_idxs[
            torch.randint(viable_idxs.numel(), select_idxs.shape, device=device)
        ]

    src_tok_idxs = []
    for row_idx, tgt_row in enumerate(tgt_tok_idxs):
        src_row = tgt_row.clone()

        if select_idxs is None:
            delta = int(len_deltas[row_idx].item())
            if delta == 0:
                src_tok_idxs.append(src_row)
                continue
            selected = rand_idxs[row_idx, :delta]
            new_tok_idxs = rand_tok_idxs[row_idx, :delta]
        else:
            selected = select_idxs[row_idx]
            new_tok_idxs = rand_tok_idxs[row_idx]

        valid = (selected >= 0) & (selected < src_row.size(0))
        selected = selected[valid]
        new_tok_idxs = new_tok_idxs[valid]
        if selected.numel() == 0:
            src_tok_idxs.append(src_row)
            continue

        op = int(torch.randint(3, (), device=device).item())
        if op == 0:
            keep = torch.ones(src_row.size(0), dtype=torch.bool, device=device)
            keep[selected] = False
            src_tok_idxs.append(src_row[keep])
        elif op == 1:
            src_tok_idxs.append(_insert_tokens(src_row, selected, new_tok_idxs))
        else:
            src_row = src_row.clone()
            src_row[selected] = new_tok_idxs
            src_tok_idxs.append(src_row)

    max_len = max([src_row.size(0) for src_row in src_tok_idxs])
    for row_idx, src_row in enumerate(src_tok_idxs):
        if src_row.size(0) == max_len:
            continue
        padding = src_row.new_full(
            (max_len - src_row.size(0),),
            int(tokenizer.padding_idx),
        )
        src_tok_idxs[row_idx] = torch.cat([src_row, padding])

    src_tok_idxs = torch.stack(src_tok_idxs).to(tgt_tok_idxs)

    return src_tok_idxs


def lanmt_train_step(model, optimizer, tgt_tok_idxs, loss_scale=1.):
    optimizer.zero_grad(set_to_none=True)

    device = _model_device(model)
    tgt_tok_idxs = tgt_tok_idxs.to(device=device, dtype=torch.long)

    max_len_delta = int(getattr(model, "max_len_delta", 0) or 0)
    src_tok_idxs = corrupt_tok_idxs(tgt_tok_idxs, model.tokenizer, max_len_delta)
    src_tok_idxs = src_tok_idxs.to(device=device, dtype=torch.long)

    src_tok_features, src_mask = model.enc_tok_features(src_tok_idxs)
    src_lens = src_mask.float().sum(-1)
    tgt_lens = tgt_tok_idxs.ne(model.tokenizer.padding_idx).float().sum(-1)
    _, tgt_tok_features, _, len_delta_logits = _call_dec_tok_features(
        model, src_tok_features, src_mask, tgt_lens
    )
    tgt_tok_logits = model.tgt_tok_logits(tgt_tok_features)
    tgt_tok_idxs = _align_targets_to_logits(
        tgt_tok_idxs, tgt_tok_logits, model.tokenizer.padding_idx
    )

    tok_loss = F.cross_entropy(
        tgt_tok_logits.flatten(end_dim=-2), tgt_tok_idxs.flatten(), ignore_index=model.tokenizer.padding_idx
    )
    if len_delta_logits is None:
        len_loss = tgt_tok_logits.new_zeros(())
    else:
        len_deltas = (tgt_lens.to(src_lens.device) - src_lens).long()
        len_targets = len_deltas + max_len_delta
        len_loss = F.cross_entropy(len_delta_logits, len_targets.to(len_delta_logits.device))

    loss = loss_scale * (tok_loss + len_loss)
    loss.backward()
    optimizer.step()

    return loss, tgt_tok_logits, tgt_tok_idxs


def lanmt_train_epoch(model, optimizer, train_loader):
    metrics = dict(
        train_loss=0.,
        train_perplexity=0.,
    )
    model.train()
    for minibatch in train_loader:
        if isinstance(minibatch, tuple):
            tgt_tok_idxs = minibatch[0]
        else:
            assert torch.is_tensor(minibatch)
            tgt_tok_idxs = minibatch

        loss, tgt_tok_logits, tgt_tok_idxs = lanmt_train_step(model, optimizer, tgt_tok_idxs)

        perplexity = _sequence_perplexity(
            tgt_tok_logits, tgt_tok_idxs, model.tokenizer.padding_idx
        )
        metrics['train_perplexity'] += perplexity.item() / len(train_loader)
        metrics['train_loss'] += loss.item() / len(train_loader)
    return metrics


def lanmt_eval_epoch(model, eval_loader, split):
    metrics = dict(
        perplexity=0.,
    )
    model.eval()
    device = _model_device(model)
    with torch.no_grad():
        for minibatch in eval_loader:
            if isinstance(minibatch, tuple):
                tgt_tok_idxs = minibatch[0]
            else:
                assert torch.is_tensor(minibatch)
                tgt_tok_idxs = minibatch

            tgt_tok_idxs = tgt_tok_idxs.to(device=device, dtype=torch.long)

            max_len_delta = int(getattr(model, "max_len_delta", 0) or 0)
            src_tok_idxs = corrupt_tok_idxs(tgt_tok_idxs, model.tokenizer, max_len_delta)
            src_tok_idxs = src_tok_idxs.to(device=device, dtype=torch.long)

            src_tok_features, src_mask = model.enc_tok_features(src_tok_idxs)
            tgt_lens = tgt_tok_idxs.ne(model.tokenizer.padding_idx).float().sum(-1)
            _, tgt_tok_features, _, _ = _call_dec_tok_features(
                model, src_tok_features, src_mask, tgt_lens
            )
            tgt_tok_logits = model.tgt_tok_logits(tgt_tok_features)

            perplexity = _sequence_perplexity(
                tgt_tok_logits, tgt_tok_idxs, model.tokenizer.padding_idx
            )
            metrics['perplexity'] += perplexity.item() / len(eval_loader)

    metrics = {f'{split}_{key}': val for key, val in metrics.items()}

    return metrics


def fit_lanmt_model(model, train_seqs, num_epochs, batch_size, lr, patience, max_shift,
                              weights=None, log_prefix=''):

    train_transform = []
    if max_shift > 0:
        train_transform.append(gfp_transforms.SequenceTranslation(max_shift))
    train_transform.append(gfp_transforms.StringToLongTensor(model.tokenizer))
    train_transform = torchvision.transforms.Compose(train_transform)

    train_dataset = gfp_dataset.TransformTensorDataset([train_seqs], train_transform)

    if weights is None:
        loader_kwargs = dict(batch_size=batch_size, shuffle=True)
    else:
        sampler = torch.utils.data.WeightedRandomSampler(weights, batch_size, replacement=True)
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        loader_kwargs = dict(batch_sampler=batch_sampler)

    collate_fn = lambda x: gfp_transforms.padding_collate_fn(x, model.tokenizer.padding_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=collate_fn, **loader_kwargs
    )

    optimizer = torch.optim.Adam(model.param_groups(lr))
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=math.ceil(patience / 2)
    )

    records = []
    best_score, best_epoch, best_weights = None, 0, None
    model.requires_grad_(True)
    for epoch in range(num_epochs):
        metrics = {}
        metrics.update(
            lanmt_train_epoch(model, optimizer, train_loader)
        )
        lr_sched.step(metrics['train_loss'])
        best_score, best_epoch, best_weights, stop = check_early_stopping(
            model,
            best_score,
            best_epoch,
            best_weights,
            metrics['train_loss'],
            epoch + 1,
            patience,
            save_weights=True,
            )

        metrics.update(dict(best_score=best_score, best_epoch=best_epoch))
        if len(log_prefix) > 0:
            metrics = {'/'.join((log_prefix, key)): val for key, val in metrics.items()}
        wandb_log(metrics)
        records.append(metrics)

        if stop:
            break

    model.load_state_dict(best_weights)
    model.requires_grad_(False)

    return records
