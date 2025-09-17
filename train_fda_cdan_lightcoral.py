import argparse
import logging
import os
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from natsort import natsorted
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import wandb
from backbone.model import ConvNext_V2_Tiny, GradientReversalFunction
from utils.early_stop import EarlyStopping


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def setup_ddp(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def wandb_init(project: str, name: str = 'None'):
    wandb.init(project=project, name=name)


@dataclass
class Sample:
    path: pathlib.Path
    class_label: int
    domain_label: int
    force_fda: bool = False
    is_real_target: bool = False
    is_synthetic: bool = False


class CDANFDADataset(data.Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        train: bool,
        img_size: int,
        fda_images: Optional[List[pathlib.Path]] = None,
        fda_beta: float = 0.0,
        fda_prob: float = 0.0,
        synthetic_force_prob: float = 1.0,
    ):
        self.samples = list(samples)
        self.train = train
        self.img_size = img_size
        self.fda_images = fda_images or []
        self.fda_beta = max(0.0, fda_beta)
        self.fda_prob = max(0.0, min(1.0, fda_prob))
        self.synthetic_force_prob = max(0.0, min(1.0, synthetic_force_prob))

        import torchvision.transforms.v2 as v2

        self.train_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.08, hue=0.01)], p=0.25),
            v2.RandomGrayscale(p=0.05),
            v2.RandomApply([
                v2.RandomChoice([
                    v2.RandomAffine(degrees=6, translate=(0.015, 0.015), scale=(0.97, 1.03), shear=(-2, 2)),
                    v2.RandomPerspective(distortion_scale=0.08),
                ])
            ], p=0.6),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1),
            v2.Resize(size=(img_size, img_size)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.RandomErasing(p=0.05, scale=(0.02, 0.05), ratio=(0.3, 3.3), value=0),
        ])

        self.val_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(img_size, img_size)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = cv2.imread(str(sample.path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {sample.path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.train and self.fda_images and self.fda_beta > 0:
            force_fda = sample.force_fda and random.random() < self.synthetic_force_prob
            apply_random = (not sample.force_fda) and random.random() < self.fda_prob
            if force_fda or apply_random:
                image = self._apply_fda(image)

        if self.train:
            tensor_image = self.train_transform(image)
        else:
            tensor_image = self.val_transform(image)

        metadata = {
            'is_real_target': torch.tensor(sample.is_real_target, dtype=torch.bool),
            'is_synthetic': torch.tensor(sample.is_synthetic, dtype=torch.bool),
        }

        return (
            tensor_image,
            torch.tensor(sample.class_label, dtype=torch.long),
            torch.tensor(sample.domain_label, dtype=torch.long),
            metadata,
        )

    def _apply_fda(self, src_img: np.ndarray) -> np.ndarray:
        style_path = random.choice(self.fda_images)
        style_img = cv2.imread(str(style_path))
        if style_img is None:
            return src_img

        style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
        h, w = src_img.shape[:2]
        if h == 0 or w == 0:
            return src_img

        style_img = cv2.resize(style_img, (w, h), interpolation=cv2.INTER_LINEAR)

        return self._fda_source_to_target(src_img, style_img, self.fda_beta)

    @staticmethod
    def _fda_source_to_target(src_img: np.ndarray, tgt_img: np.ndarray, beta: float) -> np.ndarray:
        src_img = src_img.astype(np.float32)
        tgt_img = tgt_img.astype(np.float32)

        src_fft = np.fft.fft2(src_img, axes=(0, 1))
        tgt_fft = np.fft.fft2(tgt_img, axes=(0, 1))

        src_amp, src_phase = np.abs(src_fft), np.angle(src_fft)
        tgt_amp = np.abs(tgt_fft)

        src_amp_shift = np.fft.fftshift(src_amp, axes=(0, 1))
        tgt_amp_shift = np.fft.fftshift(tgt_amp, axes=(0, 1))

        h, w = src_img.shape[:2]
        if beta <= 0:
            return src_img

        radius = max(1, int(np.floor(min(h, w) * beta)))
        c_h, c_w = h // 2, w // 2
        h1 = max(0, c_h - radius)
        h2 = min(h, c_h + radius)
        w1 = max(0, c_w - radius)
        w2 = min(w, c_w + radius)

        if h1 < h2 and w1 < w2:
            src_amp_shift[h1:h2, w1:w2] = tgt_amp_shift[h1:h2, w1:w2]

        src_amp = np.fft.ifftshift(src_amp_shift, axes=(0, 1))
        mixed_fft = src_amp * np.exp(1j * src_phase)
        mixed_img = np.fft.ifft2(mixed_fft, axes=(0, 1))
        mixed_img = np.real(mixed_img)
        mixed_img = np.clip(mixed_img, 0, 255).astype(np.uint8)
        return mixed_img


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha is None or alpha <= 0.0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def deep_coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.ndim > 2:
        source = source.flatten(start_dim=1)
    if target.ndim > 2:
        target = target.flatten(start_dim=1)

    ns = source.size(0)
    nt = target.size(0)

    if ns <= 1 or nt <= 1:
        return torch.zeros((), device=source.device, dtype=source.dtype)

    source = source - source.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)

    cov_source = (source.T @ source) / (ns - 1)
    cov_target = (target.T @ target) / (nt - 1)

    loss = (cov_source - cov_target).pow(2).sum()
    d = source.size(1)
    return loss / (4.0 * d * d)


def class_conditional_coral(
    features: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    is_real_target: torch.Tensor,
    is_synthetic: torch.Tensor,
    real_target_classes: Sequence[int],
    use_synth_fallback: bool = False,
    synth_weight: float = 0.5,
) -> torch.Tensor:
    coral_terms = []
    device = features.device
    dtype = features.dtype

    if synth_weight < 0:
        synth_weight = 0.0

    is_real_target = is_real_target.bool()
    is_synthetic = is_synthetic.bool()
    target_class_set = set(int(c) for c in real_target_classes)

    for cls in labels.unique():
        cls_mask = labels == cls
        src_mask = cls_mask & (domains == 0)
        if src_mask.sum() <= 1:
            continue

        cls_int = int(cls.item())
        real_target_mask = cls_mask & (domains == 1) & is_real_target
        synthetic_mask = cls_mask & (domains == 1) & is_synthetic

        target_mask = None
        weight = 1.0

        if cls_int in target_class_set and real_target_mask.sum() > 1:
            target_mask = real_target_mask

        elif use_synth_fallback and synthetic_mask.sum() > 1:
            target_mask = synthetic_mask
            weight = synth_weight

        if target_mask is None:
            continue

        coral_terms.append(deep_coral_loss(features[src_mask], features[target_mask]) * weight)

    if coral_terms:
        return torch.stack(coral_terms).mean()
    return torch.zeros((), device=device, dtype=dtype)


def compute_coral_lambda(epoch: int, args) -> float:
    warmup_epochs = max(1, args.coral_warmup_epochs)
    ramp_epochs = max(warmup_epochs, args.coral_ramp_epochs)

    if epoch < warmup_epochs:
        return args.coral_lambda_start * (epoch + 1) / warmup_epochs
    if epoch < ramp_epochs:
        progress = (epoch - warmup_epochs + 1) / (ramp_epochs - warmup_epochs + 1)
        return args.coral_lambda_start + progress * (args.coral_lambda_target - args.coral_lambda_start)
    return args.coral_lambda_target


def compute_partial_domain_weights(
    labels: torch.Tensor,
    domains: torch.Tensor,
    is_real_target: torch.Tensor,
    is_synthetic: torch.Tensor,
    real_target_classes: Sequence[int],
    weight_real: float,
    weight_synth: float,
) -> torch.Tensor:
    device = labels.device
    weights = torch.zeros(labels.size(0), device=device, dtype=torch.float32)

    is_real_target = is_real_target.bool()
    is_synthetic = is_synthetic.bool()
    real_target_set = set(int(c) for c in real_target_classes)

    for cls in labels.unique():
        cls_mask = labels == cls
        cls_int = int(cls.item())
        has_real_target = cls_int in real_target_set

        source_mask = cls_mask & (domains == 0)
        real_target_mask = cls_mask & (domains == 1) & is_real_target
        synthetic_mask = cls_mask & (domains == 1) & is_synthetic

        if has_real_target:
            if weight_real > 0:
                weights[source_mask] = weight_real
                weights[real_target_mask] = weight_real
            if weight_synth > 0:
                weights[synthetic_mask] = weight_synth
        else:
            if weight_synth > 0:
                weights[source_mask] = weight_synth
                weights[synthetic_mask] = weight_synth

    return weights


def compute_groupdro_group_ids(
    labels: torch.Tensor,
    domains: torch.Tensor,
    is_synthetic: torch.Tensor,
    num_classes: int,
    split_synth: bool,
) -> torch.Tensor:
    effective_domains = domains.clone()
    if split_synth:
        synth_mask = is_synthetic.bool() & (domains == 1)
        effective_domains = effective_domains + synth_mask.to(effective_domains.dtype)
    return (effective_domains * num_classes + labels).long()


class GroupDROState:
    def __init__(self, num_groups: int, eta: float, device: torch.device):
        self.num_groups = max(1, num_groups)
        self.eta = eta
        self.device = device
        self.eps = 1e-12
        self.q = torch.ones(self.num_groups, device=device, dtype=torch.float32) / float(self.num_groups)

    def to(self, device: torch.device):
        self.device = device
        self.q = self.q.to(device)
        return self

    def step(self, losses: torch.Tensor, group_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if losses.numel() == 0:
            zero = torch.zeros((), device=self.device, dtype=torch.float32)
            group_losses = torch.zeros(self.num_groups, device=self.device, dtype=torch.float32)
            return zero, group_losses

        group_losses = torch.zeros(self.num_groups, device=losses.device, dtype=losses.dtype)
        unique_groups = group_ids.unique()
        for g in unique_groups:
            idx = group_ids == g
            if idx.any():
                group_losses[g] = losses[idx].mean()

        dro_loss = (self.q * group_losses).sum()

        with torch.no_grad():
            updated = self.q * torch.exp(self.eta * group_losses.detach())
            updated = torch.clamp(updated, min=self.eps)
            norm = updated.sum().clamp_min(self.eps)
            self.q = updated / norm

        return dro_loss, group_losses


class LightCoralProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, trainable: bool = False):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        self.proj.weight.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CDANClassifier(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int, domain_hidden: int = 1024, coral_dim: int = 128, coral_trainable: bool = False):
        super().__init__()
        self.backbone = ConvNext_V2_Tiny(num_classes=num_classes)
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.domain_classifier = nn.Sequential(
            nn.Linear(num_classes * feature_dim, domain_hidden),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(domain_hidden, 2)
        )
        self.grl = GradientReversalFunction.apply
        self.coral_projector = LightCoralProjector(feature_dim, coral_dim, trainable=coral_trainable)

    def forward(self, x: torch.Tensor, alpha: float = 1.0, return_features: bool = False):
        logits, features = self.backbone(x, return_features=True)
        if features.ndim > 2:
            pooled = features.mean(dim=list(range(2, features.ndim)))
        else:
            pooled = features

        softmax_output = torch.softmax(logits, dim=1)
        outer = torch.bmm(softmax_output.unsqueeze(2), pooled.unsqueeze(1))
        domain_input = outer.reshape(outer.size(0), -1)
        reversed_feat = self.grl(domain_input, alpha)
        domain_logits = self.domain_classifier(reversed_feat)

        coral_features = self.coral_projector(pooled)

        if return_features:
            return logits, domain_logits, coral_features, pooled
        return logits, domain_logits


def gather_paths(*dirs: str) -> List[pathlib.Path]:
    paths: List[pathlib.Path] = []
    for d in dirs:
        if not d:
            continue
        path = pathlib.Path(d)
        if not path.exists():
            logging.warning(f"Data directory not found: {d}")
            continue
        paths.extend(natsorted([p for p in path.rglob('*.jpg')]))
    return paths


def prepare_samples(args) -> Tuple[List[Sample], List[Sample], List[pathlib.Path], List[int]]:
    source_wear = gather_paths(args.source_wear_dir, args.source_wear_dir2)
    source_nowear = gather_paths(args.source_nowear_dir, args.source_nowear_dir2)
    target_nowear = gather_paths(args.target_nowear_dir, args.target_nowear_dir2)
    target_wear = gather_paths(args.target_wear_dir , args.target_wear_dir2)

    if not source_wear or not source_nowear:
        raise ValueError("Source wear/nowear data not found. Check directory paths.")

    if not target_nowear:
        logging.warning("No target nowear data found. Domain diversity may be limited.")

    fda_images = []
    if args.FDA_data:
        fda_images = gather_paths(args.FDA_data)
        if fda_images:
            logging.info(f"Loaded {len(fda_images)} FDA reference images from {args.FDA_data}")
        else:
            logging.warning("No FDA reference images found; synthetic target wear may be ineffective.")

    samples: List[Sample] = []
    real_target_classes = set()
    for path in source_wear:
        samples.append(Sample(path=path, class_label=1, domain_label=0, force_fda=False, is_real_target=False, is_synthetic=False))
    for path in source_nowear:
        samples.append(Sample(path=path, class_label=0, domain_label=0, force_fda=False, is_real_target=False, is_synthetic=False))
    for path in target_nowear:
        samples.append(Sample(path=path, class_label=0, domain_label=1, force_fda=False, is_real_target=True, is_synthetic=False))
        real_target_classes.add(0)

    if target_wear:
        for path in target_wear:
            samples.append(Sample(path=path, class_label=1, domain_label=1, force_fda=False, is_real_target=True, is_synthetic=False))
            real_target_classes.add(1)
    else:
        if not fda_images:
            logging.warning("No target wear data and no FDA reference images. Target wear domain will be missing.")
        else:
            multiplier = max(1, args.synthetic_target_wear_multiplier)
            synth_count = min(len(source_wear) * multiplier, args.max_synthetic_target_wear or len(source_wear) * multiplier)
            selected = random.sample(source_wear, k=min(len(source_wear), synth_count)) if synth_count < len(source_wear) else source_wear
            for path in selected:
                samples.append(Sample(path=path, class_label=1, domain_label=1, force_fda=True, is_real_target=False, is_synthetic=True))

    labels_for_split = [sample.class_label * 2 + sample.domain_label for sample in samples]
    from sklearn.model_selection import train_test_split

    train_samples, val_samples = train_test_split(
        samples,
        test_size=args.val_split,
        random_state=42,
        stratify=labels_for_split if len(set(labels_for_split)) > 1 else None,
    )

    return train_samples, val_samples, fda_images, sorted(real_target_classes)


def compute_alpha(current_step: int, current_epoch: int, total_epochs: int, steps_per_epoch: int) -> float:
    progress = float(current_step + current_epoch * steps_per_epoch) / float(total_epochs * steps_per_epoch)
    return 2. / (1. + np.exp(-10 * progress)) - 1


def unwrap_model(m: nn.Module) -> nn.Module:
    if isinstance(m, DDP):
        m = m.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    return m


def main_worker(rank: int, world_size: int, args):
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if world_size > 1 and torch.cuda.is_available():
        torch.cuda.set_device(rank)
        setup_ddp(rank, world_size)

    if rank == 0 and args.wandb:
        wandb_init(project=args.project, name=args.wandb_name)

    torch.backends.cudnn.benchmark = True
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        train_samples, val_samples, fda_images, real_target_classes = prepare_samples(args)
    else:
        train_samples, val_samples, fda_images, real_target_classes = None, None, None, None

    if world_size > 1:
        obj_list = [train_samples, val_samples, fda_images, real_target_classes]
        dist.broadcast_object_list(obj_list, src=0)
        train_samples, val_samples, fda_images, real_target_classes = obj_list

    real_target_classes = real_target_classes or []
    real_target_classes_seq = tuple(sorted({int(c) for c in real_target_classes}))

    if rank == 0:
        logging.info(f"Real target classes observed: {real_target_classes_seq if real_target_classes_seq else 'None'}")

    train_dataset = CDANFDADataset(
        samples=train_samples,
        train=True,
        img_size=args.img_size,
        fda_images=fda_images,
        fda_beta=args.fda_beta,
        fda_prob=args.fda_prob,
        synthetic_force_prob=args.synthetic_target_force_prob,
    )
    val_dataset = CDANFDADataset(
        samples=val_samples,
        train=False,
        img_size=args.img_size,
        fda_images=fda_images,
        fda_beta=args.fda_beta,
        fda_prob=0.0,
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=4, pin_memory=True, sampler=train_sampler)
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, sampler=val_sampler)

    criterion_cls = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction='none')
    criterion_domain = nn.CrossEntropyLoss(reduction='none')

    temp_backbone = ConvNext_V2_Tiny(num_classes=args.num_classes)
    feature_dim = temp_backbone.model.num_features if hasattr(temp_backbone, 'model') and hasattr(temp_backbone.model, 'num_features') else 1536
    del temp_backbone

    model = CDANClassifier(
        num_classes=args.num_classes,
        feature_dim=feature_dim,
        domain_hidden=args.domain_hidden_dim,
        coral_dim=args.light_coral_dim,
        coral_trainable=args.light_coral_trainable,
    ).to(device)

    if args.pretrained and os.path.isfile(args.weight_path):
        logging.info(f"Loading weights from {args.weight_path}")
        checkpoint = torch.load(args.weight_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        result = model.backbone.load_state_dict(state_dict, strict=False)
        logging.info(f"Backbone weight loading result: {result}")

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)

    early_stopping = EarlyStopping(patience=args.early_stop_patience, verbose=(rank == 0), delta=args.early_stop_delta)
    scaler = torch.amp.GradScaler(device_type=device.type, enabled=(device.type == 'cuda'))

    best_val_loss = float('inf')
    num_groupdro_domains = 2 + (1 if args.groupdro_split_synth else 0)
    num_groupdro_groups = args.num_classes * num_groupdro_domains
    groupdro_state = GroupDROState(num_groupdro_groups, args.groupdro_eta, device) if args.use_groupdro else None

    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        current_coral_lambda = compute_coral_lambda(epoch, args)
        model.train()

        train_cls_loss = 0.0
        train_domain_loss = 0.0
        train_coral_loss = 0.0
        correct_cls = 0
        total_samples = 0
        max_group_loss_epoch = 0.0

        steps_per_epoch = len(trainloader)

        pbar = tqdm(enumerate(trainloader), total=steps_per_epoch, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", disable=(rank != 0))
        for step, (images, labels, domains, metadata) in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            domains = domains.to(device, non_blocking=True)
            is_real_target = metadata['is_real_target'].to(device=device)
            is_synthetic = metadata['is_synthetic'].to(device=device)

            alpha = compute_alpha(step, epoch, args.epochs, steps_per_epoch)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits, domain_logits, coral_feats, pooled_feats = model(images, alpha=alpha, return_features=True)
                cls_loss_samples = criterion_cls(logits, labels).float()
                raw_domain_loss = criterion_domain(domain_logits, domains)
                domain_weights = compute_partial_domain_weights(
                    labels,
                    domains,
                    is_real_target,
                    is_synthetic,
                    real_target_classes_seq,
                    args.domain_weight_real,
                    args.domain_weight_synth,
                ).to(raw_domain_loss.dtype)
                weight_sum = domain_weights.sum()
                if weight_sum.item() > 0:
                    dom_loss = (raw_domain_loss * domain_weights).sum() / weight_sum
                else:
                    dom_loss = torch.zeros((), device=device, dtype=raw_domain_loss.dtype)
                if args.use_groupdro:
                    group_ids = compute_groupdro_group_ids(
                        labels,
                        domains,
                        is_synthetic,
                        args.num_classes,
                        args.groupdro_split_synth,
                    )
                    cls_loss, batch_group_losses = groupdro_state.step(cls_loss_samples, group_ids)
                else:
                    cls_loss = cls_loss_samples.mean()
                    batch_group_losses = None

                coral_loss = torch.zeros((), device=device, dtype=pooled_feats.dtype)
                if current_coral_lambda > 0:
                    coral_loss = class_conditional_coral(
                        coral_feats,
                        labels,
                        domains,
                        is_real_target,
                        is_synthetic,
                        real_target_classes_seq,
                        use_synth_fallback=args.coral_use_synth_fallback,
                        synth_weight=args.coral_synth_weight,
                    )
                loss = cls_loss + args.domain_lambda * dom_loss + current_coral_lambda * coral_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_cls_loss += cls_loss.item()
            train_domain_loss += dom_loss.item()
            train_coral_loss += coral_loss.item()
            if batch_group_losses is not None:
                max_group_loss_epoch = max(max_group_loss_epoch, batch_group_losses.max().item())

            preds = torch.argmax(logits, dim=1)
            total_samples += labels.size(0)
            correct_cls += (preds == labels).sum().item()

            if rank == 0:
                postfix = {
                    'L_cls': cls_loss.item(),
                    'L_dom': dom_loss.item(),
                    'L_coral': coral_loss.item(),
                    'λ_coral': current_coral_lambda,
                    'α': alpha,
                }
                if batch_group_losses is not None:
                    postfix['GdroMax'] = batch_group_losses.max().item()
                pbar.set_postfix(postfix)

        train_acc = 100.0 * correct_cls / max(1, total_samples)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            pbar_val = tqdm(valloader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", disable=(rank != 0))
            for images, labels, _, _ in pbar_val:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    logits, _ = model(images, alpha=0.0, return_features=False)
                    loss = criterion_cls(logits, labels).mean()

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                total_val += labels.size(0)
                correct_val += (preds == labels).sum().item()

        avg_train_cls = train_cls_loss / len(trainloader)
        avg_train_dom = train_domain_loss / len(trainloader)
        avg_train_coral = train_coral_loss / len(trainloader)
        avg_val_loss = val_loss / len(valloader)
        val_acc = 100.0 * correct_val / max(1, total_val)

        if rank == 0:
            if args.use_groupdro:
                logging.info(
                    "Epoch %d, TrainCls %.4f, TrainDom %.4f, TrainCoral %.4f, TrainAcc %.2f%%, ValLoss %.4f, ValAcc %.2f%%, Coral λ %.4f, MaxGroupLoss %.4f",
                    epoch + 1,
                    avg_train_cls,
                    avg_train_dom,
                    avg_train_coral,
                    train_acc,
                    avg_val_loss,
                    val_acc,
                    current_coral_lambda,
                    max_group_loss_epoch,
                )
            else:
                logging.info(
                    "Epoch %d, TrainCls %.4f, TrainDom %.4f, TrainCoral %.4f, TrainAcc %.2f%%, ValLoss %.4f, ValAcc %.2f%%, Coral λ %.4f",
                    epoch + 1,
                    avg_train_cls,
                    avg_train_dom,
                    avg_train_coral,
                    train_acc,
                    avg_val_loss,
                    val_acc,
                    current_coral_lambda,
                )

            if args.wandb:
                wandb.log({
                    'Epoch': epoch + 1,
                    'Train Cls Loss': avg_train_cls,
                    'Train Domain Loss': avg_train_dom,
                    'Train Coral Loss': avg_train_coral,
                    'Train Accuracy': train_acc,
                    'Val Loss': avg_val_loss,
                    'Val Accuracy': val_acc,
                    'Coral Lambda': current_coral_lambda,
                    'Learning Rate': scheduler.get_last_lr()[0],
                    'Max Group Loss': max_group_loss_epoch if args.use_groupdro else 0.0,
                })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if rank == 0:
                state = unwrap_model(model).state_dict()
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                torch.save({'model_state_dict': state, 'epoch': epoch}, os.path.join(args.checkpoint_dir, 'best_model.pth'))

        if rank == 0:
            early_stopping(avg_val_loss)

        if world_size > 1:
            stop_tensor = torch.tensor(float(early_stopping.early_stop), device=device)
            dist.broadcast(stop_tensor, src=0)
            stop_signal = stop_tensor.item() == 1.0
        else:
            stop_signal = early_stopping.early_stop

        if stop_signal:
            if rank == 0:
                logging.info("Early stopping triggered.")
            break

        scheduler.step()

    if world_size > 1:
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description='ConvNeXt CDAN with Light CORAL for FDA data')
    """

-fda_beta: 저주파 패치를 얼마나 넓게 교환할지 결정하는 반경 비율. 0.05면 이미지 한 변의 5% 수준으로 스타일을 섞습니다.
  - --fda_prob: 강제 FDA가 아닌 일반 소스 샘플에 대해 FDA를 적용할 확률.
  - --synthetic_target_wear_multiplier: 타깃 착용이 없을 때 소스 착용 샘플을 몇 배까지 합성할지 결정.
  - --max_synthetic_target_wear: 합성 타깃 착용 샘플의 상한(0이면 제한 없음).
  - --synthetic_target_force_prob: force_fda=True 샘플(=합성 타깃 후보)에 대해 FDA를 강제로 적용할 확률.
  - --FDA_data: FDA 스타일을 뽑아올 타깃 스타일 이미지 폴더.

  - --domain_lambda: CDAN을 켤지/껄지 결정하는 가중치. 0이면 현재 코드 흐름(FDA+CORAL+GroupDRO)만 사용, >0이면 CDAN 도메인 손실이 추가됨.
  - --domain_hidden_dim: CDAN 도메인 분류기의 은닉 차원.
  - --domain_weight_real: 실제 타깃 클래스가 존재할 때 도메인 손실에 주는 가중치.
  - --domain_weight_synth: 합성 타깃을 도메인 손실에 포함시킬 때의 가중치(실제보다 낮게 두어 음의 전이 억제).

  - --light_coral_dim: Light CORAL을 위해 특성 차원을 낮추는 투영 차원.
  - --light_coral_trainable: 투영 행렬을 학습 가능하도록 할지 (기본은 고정).
  - --coral_lambda_start/target: CORAL 손실의 초기·최대 가중치.
  - --coral_warmup_epochs, --coral_ramp_epochs: CORAL 가중치를 램프업하는 에포크 수.
  - --coral_use_synth_fallback: 실 타깃 클래스가 없는 경우 합성 타깃을 CORAL에 쓰도록 허용.
  - --coral_synth_weight: 합성 타깃이 CORAL에 기여할 때의 추가 스케일.

  - --use_groupdro: GroupDRO를 켜서 그룹별 worst-case를 안정화할지 여부.
  - --groupdro_eta: GroupDRO의 q 갱신 학습률(크면 빠르게 전환하지만 불안정할 수 있음).
  - --groupdro_split_synth: 합성 타깃을 별도 도메인으로 취급해 더욱 보수적으로 가중치를 조정할지.

    """
    
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for classification')

    parser.add_argument('--source_wear_dir', type=str, default='/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/마스크_원시데이터/TS1')
    parser.add_argument('--source_wear_dir2', type=str, default='/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/마스크_원시데이터/TS2')
    parser.add_argument('--source_nowear_dir', type=str, default='/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/neckslice/refining_yaw_yaw')
    parser.add_argument('--source_nowear_dir2', type=str, default='')
    parser.add_argument('--target_nowear_dir', type=str, default='/home/ubuntu/KOR_DATA/high_resolution_not_wear_hat')
    parser.add_argument('--target_nowear_dir2', type=str, default='')
    parser.add_argument('--target_wear_dir', type=str, default='')
    parser.add_argument('--target_wear_dir2', type=str, default='')

    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--mixup_alpha', type=float, default=0.0)

    parser.add_argument('--fda_beta', type=float, default=0.05)
    parser.add_argument('--fda_prob', type=float, default=0.25)
    parser.add_argument('--synthetic_target_wear_multiplier', type=int, default=1)
    parser.add_argument('--max_synthetic_target_wear', type=int, default=0, help='Cap on synthetic target wear samples (0=disabled)')
    parser.add_argument('--synthetic_target_force_prob', type=float, default=1.0, help='Probability to apply FDA when force_fda flag is set')
    parser.add_argument('--FDA_data', type=str, default='/home/ubuntu/KOR_DATA/high_resolution_train_data_640')

    parser.add_argument('--domain_lambda', type=float, default=0.05, help='Weight for domain adversarial loss when using domain lambda 1.0')
    parser.add_argument('--domain_hidden_dim', type=int, default=1024, help='Hidden dimension of CDAN domain classifier')
    parser.add_argument('--domain_weight_real', type=float, default=1.0, help='Weight for domain loss on classes observed in real target data')
    parser.add_argument('--domain_weight_synth', type=float, default=0.3, help='Weight for domain loss when relying on synthetic target samples')

    parser.add_argument('--light_coral_dim', type=int, default=128, help='Projection dimension for Light CORAL')
    parser.add_argument('--light_coral_trainable', action='store_true', help='Allow Light CORAL projector weights to be trainable')
    parser.add_argument('--coral_lambda_start', type=float, default=0.01)
    parser.add_argument('--coral_lambda_target', type=float, default=0.05)
    parser.add_argument('--coral_warmup_epochs', type=int, default=2)
    parser.add_argument('--coral_ramp_epochs', type=int, default=10)
    parser.add_argument('--coral_use_synth_fallback', action='store_true', help='Use synthetic target samples for CORAL when real target samples are absent for a class')
    parser.add_argument('--coral_synth_weight', type=float, default=0.5, help='Scaling factor for CORAL loss contributed by synthetic target samples')
    parser.add_argument('--use_groupdro', action='store_true', help='Enable GroupDRO on classification loss')
    parser.add_argument('--groupdro_eta', type=float, default=0.1, help='Learning rate for GroupDRO weight updates')
    parser.add_argument('--groupdro_split_synth', action='store_true', help='Treat synthetic target samples as a separate GroupDRO domain')

    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--early_stop_delta', type=float, default=1e-3)

    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--weight_path', type=str, default='checkpoints/convnext_v2_tiny_best.pth')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/cdan_light_coral')

    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--project', type=str, default='ConvNeXtV2_FDA_CDAN')
    parser.add_argument('--wandb_name', type=str, default='CDAN_LightCORAL')

    parser.add_argument('--no_confirm', action='store_true')

    parser.set_defaults(
        domain_lambda=0.0,
        use_groupdro=True,
        coral_use_synth_fallback=True,
    )

    # # 타켓도메인 착용 / 미착용 존재할떄
    # parser.set_defaults(
    #      domain_lambda=0.05,   # CDAN 가중치
    #      use_groupdro=True,
    #      coral_use_synth_fallback=False,  # 합성 fallback 필요 없음
    #  )

    args = parser.parse_args()

    if not args.no_confirm:
        print("--- Training Arguments ---")
        for var in vars(args):
            print(f"{var}: {getattr(args, var)}")
        print("--------------------------")
        if sys.stdin.isatty():
            input("Press Enter to proceed...")

    world_size = torch.cuda.device_count()

    if world_size > 1:
        logging.info(f"Using {world_size} GPUs via DDP")
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        logging.info("Using single GPU")
        main_worker(0, 1, args)


if __name__ == '__main__':
    main()
