import logging
import sys
import torch
from torch.utils import data
from tqdm import tqdm
import wandb
import os
import pathlib
from dataset import custom_dataset_FDA_CORAL
import numpy as np
import cv2
import logging

from backbone.model import (
    EfficientNetV2_S,
    EfficientNetV2_L,
    EfficientNetV2_S_improved,
    ConvNext_V2_Tiny,
    build_param_groups_lrd,
)


from sklearn.model_selection import train_test_split
from utils.early_stop import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from collections import OrderedDict
from natsort import natsorted
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import math
from typing import Optional

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def wandb_init(project , name='None'):
    name = name
    wandb.init(
        project=f'{project}',
        name=name,
    )

def data_check(loader: DataLoader, is_bgr: bool):
    """
    Visualizes a batch of images from the DataLoader to check augmentations.
    Uses inverse ImageNet normalization to display.
    """
    logging.info("Fetching a batch of data to visualize...")
    it = iter(loader)

    # Inverse of Normalize(mean, std) used in dataset.py
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    labels = []
    cv2.namedWindow('Data Check - Press any key to continue', cv2.WINDOW_NORMAL)
    for images, label in it:
        labels = []
        images_unnorm = images.clone()
        labels.extend(label.cpu().numpy().tolist())
        images_unnorm.mul_(std).add_(mean)
        grid = torchvision.utils.make_grid(images_unnorm, nrow=8)
        grid = torch.clamp(grid, 0, 1)

        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        grid_np = (grid_np * 255).astype(np.uint8)

        if not is_bgr:
            grid_np = cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('Data Check - Press any key to continue', grid_np)
        print("Labels in this batch:", labels)
        key = cv2.waitKey(0)
        if key == 27:
            logging.info("ESC key pressed. Exiting data check.")
            break

    cv2.destroyAllWindows()
    exit(0)


def find_max_batch_size(model, input_shape, device):
    if 'cuda' not in str(device):
        logging.info("CUDA not found. Skipping max batch size search.")
        return None

    model.to(device)
    model.eval()
    max_found_bs = 0
    mid = 64
    limit = 1024 * 5
    while max_found_bs <= limit:
        if mid == 64:
            pass
        else:
            mid = max_found_bs * 2
        try:
            dummy_input = torch.randn(mid, *input_shape, device=device)
            with torch.no_grad():
                _ = model(dummy_input)
            max_found_bs = mid

            print(f"✅ Batch size {mid} succeeded.")
            mid = mid * 2
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"❌ Batch size {mid} failed (OOM).")
                return max_found_bs
            else:
                print(f"An unexpected error occurred: {e}")
                raise e
        finally:
            torch.cuda.empty_cache()
    print(f"Max batch size search completed. Maximum found limit {limit}: {max_found_bs}")
    return max_found_bs

def get_model(model_name):
    if model_name == 'efficientnetv2_s':
        model = EfficientNetV2_S()
    elif model_name == 'efficientnetv2_l':
        model = EfficientNetV2_L()
    elif model_name == 'efficientnetv2_s_improved':
        model = EfficientNetV2_S_improved()
    elif model_name == 'convnext_v2_tiny':
        model = ConvNext_V2_Tiny()
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    return model

def unwrap_model(m: torch.nn.Module):
    if isinstance(m, DDP):
        m = m.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    return m

def get_inner_timm(m: torch.nn.Module):
    m = unwrap_model(m)
    return getattr(m, 'model', m)

def freeze_bn_running_stats(model: torch.nn.Module):
    """Freeze BatchNorm running stats; keep affine trainable."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()

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
    """Deep CORAL loss as defined in https://arxiv.org/abs/1607.01719."""
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

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone().cpu()
        self.backup = {}

    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow and p.requires_grad:
                new = p.detach().cpu()
                self.shadow[name].mul_(self.decay).add_(new, alpha=1.0 - self.decay)

    def apply_shadow(self, model: torch.nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow and p.requires_grad:
                self.backup[name] = p.data.detach().clone()
                p.data.copy_(self.shadow[name].to(p.device))

    def restore(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

def load_extra_data(path, fraction):
    p = pathlib.Path(path)
    if not p.exists():
        logging.warning(f"Extra data path not found: {path}")
        return []
    subfolders = natsorted([f for f in p.iterdir() if f.is_dir()])
    num_folders_to_use = int(len(subfolders) * fraction)
    selected_folders = subfolders[:num_folders_to_use]
    selected_images = []
    for folder in selected_folders:
        selected_images.extend(natsorted(list(folder.glob('*.jpg'))))
    if not selected_images:
        logging.warning(f"No image files found in the specified extra data directory: {path}")
    return selected_images


def collect_image_paths(path: str):
    """Collect image files (jpg/png/jpeg/bmp) recursively from a directory."""
    p = pathlib.Path(path)
    if not p.exists():
        logging.warning(f"FDA data path not found: {path}")
        return []

    patterns = ['**/*.jpg', '**/*.jpeg', '**/*.png', '**/*.bmp']
    images = set()
    for pattern in patterns:
        images.update(p.glob(pattern))
        images.update(p.glob(pattern.upper()))

    image_list = natsorted(list(images))
    if not image_list:
        logging.warning(f"No image files found in the specified FDA directory: {path}")
    return image_list

    

def main_worker(rank, world_size, args):
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Ensure each rank is pinned to a unique GPU
    if world_size > 1 and torch.cuda.is_available():
        torch.cuda.set_device(rank)
    
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    if rank == 0 and args.wandb:
        wandb_init(project=f'{args.project}' , name=f'{args.wandb_name}')

    torch.backends.cudnn.benchmark = True
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # --- Data preparation (only on rank 0) ---
    if rank == 0:
        wear_path = pathlib.Path(args.wear_dir)
        nowear_path = pathlib.Path(args.nowear_dir)

        wear_list = list(wear_path.glob('*.jpg')) if wear_path.exists() else []
        no_wear_list = list(nowear_path.glob('**/*.jpg')) if nowear_path.exists() else []

        fda_images = collect_image_paths(args.FDA_data) if args.FDA_data else []
        if fda_images:
            logging.info(f"Loaded {len(fda_images)} FDA reference images from {args.FDA_data}")

        if args.data_fraction > 0:
            logging.info(f"Loading {args.data_fraction * 100:.0f}% of extra data from {args.wear_dir2} and {args.nowear_dir2}")
            no_wear_list.extend(load_extra_data(args.nowear_dir2, args.data_fraction))

        wear2_path = pathlib.Path(args.wear_dir2)
        wear_list.extend(list(wear2_path.glob('**/*.jpg')) if wear2_path.exists() else [])

        if not wear_list or not no_wear_list:
            raise ValueError("Image lists are empty. Please check data paths.")

        logging.info(f"wear num : {len(wear_list)}")
        logging.info(f"no wear num : {len(no_wear_list)}")

        wear_labels = [1] * len(wear_list)
        no_wear_labels = [0] * len(no_wear_list)

        all_data = wear_list + no_wear_list
        all_labels = wear_labels + no_wear_labels

        train_list, val_list, train_labels, val_labels = train_test_split(
            all_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        # Share data lists with other ranks
        if world_size > 1:
            data_to_share = {
                'train_list': train_list,
                'val_list': val_list,
                'train_labels': train_labels,
                'val_labels': val_labels,
                'fda_images': fda_images
            }
            dist.broadcast_object_list([data_to_share], src=0)
    else:
        # Receive data lists from rank 0
        data_to_share = [None]
        dist.broadcast_object_list(data_to_share, src=0)
        data_to_share = data_to_share[0]
        train_list = data_to_share['train_list']
        val_list = data_to_share['val_list']
        train_labels = data_to_share['train_labels']
        val_labels = data_to_share['val_labels']
        fda_images = data_to_share.get('fda_images', [])

    if world_size == 1:
        # Ensure fda_images initialized on single GPU setup
        if "fda_images" not in locals():
            fda_images = collect_image_paths(args.FDA_data) if args.FDA_data else []


    train_wear = [path for path, label in zip(train_list, train_labels) if label == 1]
    train_no_wear = [path for path, label in zip(train_list, train_labels) if label == 0]

    val_wear = [path for path, label in zip(val_list, val_labels) if label == 1]
    val_no_wear = [path for path, label in zip(val_list, val_labels) if label == 0]

    train_dataset = custom_dataset_FDA_CORAL(
        wear_data=train_wear,
        no_wear_data=train_no_wear,
        img_size=args.img_size,
        train=True,
        fda_images=fda_images,
        fda_beta=args.fda_beta,
        fda_prob=args.fda_prob,
        return_domain=True,
    )
    val_dataset = custom_dataset_FDA_CORAL(
        wear_data=val_wear,
        no_wear_data=val_no_wear,
        train=False,
        img_size=args.img_size,
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    trainloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), pin_memory=True, num_workers=4, sampler=train_sampler)
    valloader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4, sampler=val_sampler)

    # Run data_check safely under DDP (only rank 0 shows window)

    if args.data_check and rank == 0:
        data_check(trainloader, is_bgr=False)
        if world_size > 1:
            dist.barrier()
        
    if len(train_no_wear) > 0 and len(train_wear) > 0:
        count_no_wear = len(train_no_wear)
        count_wear = len(train_wear)
        
        # Calculate weight for each class to handle imbalance.
        # The weight for a class is inversely proportional to its number of samples.
        # This gives higher penalty for misclassifying the minority class.
        weight_for_class_0 = (count_no_wear + count_wear) / (2.0 * count_no_wear)
        weight_for_class_1 = (count_no_wear + count_wear) / (2.0 * count_wear)
        
        # The order in the tensor must match the class indices (0, 1)
        class_weights = torch.tensor([weight_for_class_0, weight_for_class_1], device=device)
        
        if rank == 0:
            logging.info(f"Handling class imbalance. Samples: [No-Wear: {count_no_wear}, Wear: {count_wear}]")
            logging.info(f"Applying weights: [No-Wear: {weight_for_class_0:.2f}, Wear: {weight_for_class_1:.2f}]")
    else:
        class_weights = None

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    model = get_model(args.model).to(device)

    if args.pretrained and os.path.isfile(args.weight_path):
        logging.info(f"Loading weights from {args.weight_path}")
        checkpoint = torch.load(args.weight_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model_weight = checkpoint['model_state_dict']
        else:
            model_weight = checkpoint
            
        result = model.load_state_dict(model_weight, strict=False)
        logging.info(f"Weight loading result: {result}")

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # Optionally freeze BN running stats for stability across domains
    if args.bn_freeze_stats:
        target_model = model.module if isinstance(model, DDP) else model
        freeze_bn_running_stats(target_model)

    if rank == 0 and args.wandb:
        wandb.watch(model, log='all', log_freq=100)

    if rank == 0:
        backbone_params = sum(p.numel() for p in model.parameters())
        trainable_backbone_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info('==' * 30)
        logging.info(f"Model total params: {backbone_params:,}")
        logging.info(f"Model trainable params: {trainable_backbone_params:,}")
        logging.info(f"traineable params percentage: {100 * trainable_backbone_params / backbone_params:.2f}%")
        logging.info('==' * 30)

    # Build optimizer (LLRD optional)
    if args.lrd:
        inner = get_inner_timm(model)
        try:
            param_groups = build_param_groups_lrd(inner)
            optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        except Exception as e:
            if rank == 0:
                logging.warning(f"LLRD failed, falling back to single LR: {e}")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    # Enable GradScaler only on CUDA to avoid CPU AMP issues (new API)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda'))
    ema = EMA(unwrap_model(model), decay=args.ema_decay) if args.use_ema else None
    
    if rank == 0:
        checkpoint_dir = os.path.join('checkpoints', args.model)
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        current_coral_lambda = compute_coral_lambda(epoch, args)
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()
        if args.bn_freeze_stats:
            target_model = model.module if isinstance(model, DDP) else model
            freeze_bn_running_stats(target_model)
            
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        coral_loss_sum = 0.0
        
        pbar_train = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", disable=(rank != 0))
        for batch in pbar_train:
            if len(batch) == 3:
                data_input, label, domain_labels = batch
                domain_labels = domain_labels.to(device, non_blocking=True)
            else:
                data_input, label = batch
                domain_labels = None
            data_input = data_input.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # Use autocast only when CUDA is available (safer across PyTorch versions)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits_base, features = model(data_input, return_features=True)
                if features.ndim > 2:
                    features = features.mean(dim=list(range(2, features.ndim)))

                coral_val = torch.zeros((), device=device, dtype=features.dtype)
                if domain_labels is not None and current_coral_lambda > 0.0:
                    source_mask_all = domain_labels == 0
                    target_mask_all = domain_labels == 1
                    coral_terms = []
                    for cls in torch.unique(label):
                        cls_mask = label == cls
                        src_mask = source_mask_all & cls_mask
                        tgt_mask = target_mask_all & cls_mask
                        if src_mask.sum() > 1 and tgt_mask.sum() > 1:
                            coral_terms.append(deep_coral_loss(features[src_mask], features[tgt_mask]))
                    if coral_terms:
                        coral_val = torch.stack(coral_terms).mean()

                if args.mixup_alpha > 0.0:
                    inputs_mix, targets_a, targets_b, lam = mixup_data(data_input, label, args.mixup_alpha)
                    logit = model(inputs_mix)
                    cls_loss = lam * criterion(logit, targets_a) + (1 - lam) * criterion(logit, targets_b)
                else:
                    logit = logits_base
                    cls_loss = criterion(logit, label)

                loss = cls_loss + (current_coral_lambda * coral_val)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(unwrap_model(model))

            train_loss += loss.item()
            coral_loss_sum += coral_val.item()
            _, predicted = torch.max(logit.data, 1)
            total_train += label.size(0)
            if args.mixup_alpha > 0.0:
                correct_train += (lam * (predicted == targets_a).sum().item() + (1 - lam) * (predicted == targets_b).sum().item())
            else:
                correct_train += (predicted == label).sum().item()
            
            if rank == 0:
                pbar_train.set_postfix({'loss': loss.item(), 'coral': coral_val.item(), 'λ_coral': current_coral_lambda})

        train_accuracy = 100 * correct_train / total_train

        if ema is not None and args.ema_eval:
            ema.apply_shadow(unwrap_model(model))
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            pbar_val = tqdm(valloader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", disable=(rank != 0))
            for data_input, label in pbar_val:
                data_input = data_input.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    if args.tta_flip:
                        logit1 = model(data_input)
                        logit2 = model(torch.flip(data_input, dims=[3]))
                        logit = (logit1 + logit2) / 2.0
                    else:
                        logit = model(data_input)
                    loss = criterion(logit, label)

                val_loss += loss.item()
                _, predicted = torch.max(logit.data, 1)
                total_val += label.size(0)
                correct_val += (predicted == label).sum().item()
                if rank == 0:
                    pbar_val.set_postfix({'loss': loss.item()})

        val_accuracy = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(valloader)

        if ema is not None and args.ema_eval:
            ema.restore(unwrap_model(model))
        
        if rank == 0 and args.wandb:
            wandb.log({
               "Train Loss": train_loss / len(trainloader),
               "Train Accuracy": train_accuracy,
               "Train Coral Loss": coral_loss_sum / len(trainloader),
               "Coral Lambda": current_coral_lambda,
               "Val Loss": avg_val_loss,
               "Val Accuracy": val_accuracy,
               "Learning Rate": scheduler.get_last_lr()[0]
            })

        if rank == 0:
            logging.info(f"Epoch {epoch+1}, Train Loss: {train_loss/len(trainloader):.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Coral λ: {current_coral_lambda:.4f}")

            # Save periodic checkpoints
            if (epoch + 1) % 2 == 0 :
                state_to_save = model.module.state_dict() if world_size > 1 else model.state_dict()
                
                cleaned_state_dict = OrderedDict()
                for k, v in state_to_save.items():
                    name = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
                    cleaned_state_dict[name] = v
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': cleaned_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss
                }

                checkpoint_path = os.path.join(checkpoint_dir, f'{args.model}_epoch_{epoch+1}.pth')
                torch.save(checkpoint, checkpoint_path)
                logging.info(f"Saved checkpoint to {checkpoint_path}")


            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                state_to_save = model.module.state_dict() if world_size > 1 else model.state_dict()
                
                cleaned_state_dict = OrderedDict()
                for k, v in state_to_save.items():
                    name = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
                    cleaned_state_dict[name] = v
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': cleaned_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss
                }
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(checkpoint, best_model_path)
                logging.info(f"Saved best model to {best_model_path} with val loss: {best_val_loss:.4f}")

            early_stopping(avg_val_loss)

        if world_size > 1:
            stop_signal = torch.tensor(1.0 if rank == 0 and early_stopping.early_stop else 0.0, device=device)
            dist.broadcast(stop_signal, src=0)
            if stop_signal.item() == 1.0:
                if rank == 0:
                    logging.info("Early stopping triggered.")
                break
        else:
            if early_stopping.early_stop:
                logging.info("Early stopping triggered.")
                break
        
        scheduler.step()

    if world_size > 1:
        cleanup_ddp()




def main():
    parser = argparse.ArgumentParser(description='EfficientNetV2 Training')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='convnext_v2_tiny', choices=['efficientnetv2_s', 'efficientnetv2_l', 'efficientnetv2_s_improved' , 'convnext_v2_tiny'], help='Model type')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--weight_path', type=str, default='effcientnet_s_mask_384/best_model.pth', help='Path to model weights')

    # Data arguments
    parser.add_argument('--wear_dir', type=str, default='/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/마스크_원시데이터/TS1')
    parser.add_argument('--wear_dir2' , type=str , default='/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/마스크_원시데이터/TS2')
    parser.add_argument('--nowear_dir', type=str, default='/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/neckslice/refining_yaw_yaw', help='Directory for "no wear" images')
    parser.add_argument('--nowear_dir2', type=str, default='/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/glasses/refining_yaw_yaw', help='Directory for additional "no wear" images')
    parser.add_argument('--data_fraction', type=float, default=1, help='Fraction of extra data to use (0.0 to 1.0)')
    parser.add_argument('--img_size', type=int, default=384, help='Input image size (assumed square)')
    parser.add_argument('--FDA_data', type=str, default='/home/ubuntu/KOR_DATA/high_resolution_train_data_640', help='FDA data path')
    parser.add_argument('--fda_beta', type=float, default=0.05, help='Relative radius of low-frequency swap for FDA (0 disables)')
    parser.add_argument('--fda_prob', type=float, default=0.5, help='Probability of applying FDA to a training sample')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--find_batch_size', action='store_true', help='Find the maximum batch size before training')
    parser.add_argument('--no_confirm', action='store_true', help='Skip argument confirmation prompt')
    parser.add_argument('--data_check', action='store_true', help='Visualize a batch of data to check augmentations')

    parser.add_argument('--mixup_alpha', type=float, default=0, help='Mixup alpha (0 disables mixup)')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='CrossEntropy label smoothing')
    parser.add_argument('--coral_lambda_start', type=float, default=0.01, help='Initial CORAL weight after warmup ramp')
    parser.add_argument('--coral_lambda_target', type=float, default=0.05, help='Target CORAL weight after ramp-up')
    parser.add_argument('--coral_warmup_epochs', type=int, default=2, help='Epochs to reach coral_lambda_start from 0')
    parser.add_argument('--coral_ramp_epochs', type=int, default=10, help='Epoch to finish ramping towards coral_lambda_target')

    # Boolean flags with default True via paired options
    parser.add_argument('--bn_freeze_stats', dest='bn_freeze_stats', action='store_true', help='Freeze BN running stats during training')
    parser.add_argument('--no-bn_freeze_stats', dest='bn_freeze_stats', action='store_false', help='Do not freeze BN running stats during training')
    parser.add_argument('--lrd', dest='lrd', action='store_true', help='Use layer-wise lr decay param groups')
    parser.add_argument('--no-lrd', dest='lrd', action='store_false', help='Disable layer-wise lr decay')
    parser.add_argument('--lr', type=float, default=1e-4, help='Base learning rate when not using LLRD')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for AdamW')
    parser.add_argument('--use_ema', dest='use_ema', action='store_true', help='Maintain EMA of weights')
    parser.add_argument('--no-use_ema', dest='use_ema', action='store_false', help='Disable EMA of weights')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay factor')
    parser.add_argument('--ema_eval', action='store_true', help='Evaluate with EMA weights during validation')
    parser.add_argument('--tta_flip', action='store_true', help='Enable horizontal flip TTA at validation')

    # Set defaults for paired booleans
    parser.set_defaults(bn_freeze_stats=True, lrd=True, use_ema=True)

    # W&B arguments
    parser.add_argument('--wandb', action='store_true', help='Use wandb or not')
    parser.add_argument('--project', type=str, default='ConvNextV2_FDA', help='wandb project name')
    parser.add_argument('--wandb_name', type=str, default='FDA_CORAL', help='wandb experiment name')
    
    args = parser.parse_args()

    if args.find_batch_size:
        print("Finding max batch size...")
        temp_model = get_model(args.model)
        # Use the configured img_size for the batch size search
        max_bs = find_max_batch_size(temp_model, input_shape=(3, args.img_size, args.img_size), device='cuda:0')
        if max_bs:
            print(f"Maximum possible batch size is {max_bs}. Setting batch_size to this value.")
            args.batch_size = max_bs

    print("--- Training Arguments ---")
    for var in vars(args):
        print(f"{var}: {getattr(args, var)}")
    print("--------------------------")

    if not args.no_confirm:
        if sys.stdin.isatty():
            input("Press Enter to proceed...")
        else:
            print("Non-interactive environment detected; skipping confirmation prompt. Use --no_confirm to suppress this message explicitly.")
    
    world_size = torch.cuda.device_count()
    if world_size > 1:
        logging.info(f"Using {world_size} GPUs for DDP.")
        mp.spawn(main_worker,
                 args=(world_size, args),
                 nprocs=world_size,
                 join=True)
    else:
        logging.info("Using single GPU.")
        main_worker(0, 1, args)

if __name__ == '__main__':
    main()
def compute_coral_lambda(epoch: int, args) -> float:
    warmup_epochs = max(1, args.coral_warmup_epochs)
    ramp_epochs = max(warmup_epochs, args.coral_ramp_epochs)

    if epoch < warmup_epochs:
        return args.coral_lambda_start * (epoch + 1) / warmup_epochs
    if epoch < ramp_epochs:
        progress = (epoch - warmup_epochs + 1) / (ramp_epochs - warmup_epochs + 1)
        return args.coral_lambda_start + progress * (args.coral_lambda_target - args.coral_lambda_start)
    return args.coral_lambda_target