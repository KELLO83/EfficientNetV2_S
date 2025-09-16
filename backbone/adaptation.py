import torch
import torch.nn as nn
import torch.nn.functional as F


def set_bn_momentum(model: nn.Module, momentum: float = 0.01):
    """Set BatchNorm momentum across the model (useful to stabilize stats under shift)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.momentum = momentum


def _set_bn_train_only(model: nn.Module):
    """Put the whole model in eval, but BN layers in train mode (for running stats update)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.train()


@torch.no_grad()
def adabn_update(model: nn.Module, data_loader, device: torch.device, max_batches: int | None = None):
    """
    Adaptive BatchNorm (AdaBN): re-estimate BN running stats on target domain B using unlabeled data.

    - Does not update weights; only updates BN running mean/var via forward pass.
    - Works even if the model was trained/fine-tuned on a different domain.
    """
    was_training = model.training
    _set_bn_train_only(model)

    count = 0
    for batch in data_loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device, non_blocking=True)
        _ = model(images)
        count += 1
        if max_batches is not None and count >= max_batches:
            break

    # restore mode
    model.train(was_training)


def tent_collect_bn_affine(model: nn.Module):
    """Collect BN affine parameters (gamma/beta) for TENT."""
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if m.weight is not None:
                params.append(m.weight)
            if m.bias is not None:
                params.append(m.bias)
    return params


def tent_configure_model(model: nn.Module):
    """
    Configure model for TENT:
    - Freeze all params except BN affine (gamma/beta)
    - Put model in train mode so BN updates running stats
    """
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if m.weight is not None:
                m.weight.requires_grad = True
            if m.bias is not None:
                m.bias.requires_grad = True
    model.train()


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    return torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=1))


def tent_adapt_batch(model: nn.Module, images: torch.Tensor, optimizer) -> torch.Tensor:
    """
    One TENT step on a batch: minimize prediction entropy by updating BN affine only.
    Returns the (pre-update) logits for convenience.
    """
    model.train()
    logits = model(images)
    loss = entropy_loss(logits)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return logits.detach()


@torch.no_grad()
def tta_predict_logits(model: nn.Module, images: torch.Tensor):
    """
    Simple TTA: average logits of original and horizontal flip.
    Assumes images are a float tensor in NCHW. Returns averaged logits.
    """
    model.eval()
    logits_list = []
    logits_list.append(model(images))
    logits_list.append(model(torch.flip(images, dims=[3])))  # horizontal flip
    logits = torch.stack(logits_list, dim=0).mean(dim=0)
    return logits

