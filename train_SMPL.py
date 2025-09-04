import logging
import torch
from torch.utils import data
from tqdm import tqdm
import wandb
import os
import pathlib
import numpy as np
import cv2
import logging
import os
from backbone.model import EfficientNetV2_S , EfficientNetV2_L
import torchinfo
from sklearn.model_selection import train_test_split
from utils.early_stop import EarlyStopping
from utils.polynomialLRWarmup import PolynomialLRWarmup
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from natsort import natsorted
from torchvision.transforms import v2

# Import the new SPMLDataset and loss functions
from dataset import SPMLDataset
from utils.smpl_loss import GR_loss # Updated import for GR_loss



"""
Boosting Single Positive Multi-label Classification with Generalized Robust Loss
https://arxiv.org/abs/2405.03501

"""

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def wandb_init(name='None'):
    name = name
    wandb.init(
        project='EfficientNetV2_SPML',
        name=name,
    )

def get_model(model_name, num_classes):
    if model_name == 'efficientnetv2_s':
        model = EfficientNetV2_S(num_classes=num_classes)
    elif model_name == 'efficientnetv2_l':
        model = EfficientNetV2_L(num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    return model

def main_worker(rank, world_size, args):
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if world_size > 1:
        setup_ddp(rank, world_size)

    if rank == 0 and args.wandb:
        wandb_init(name=f'{args.wandb_name}')

    torch.backends.cudnn.benchmark = True
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # --- Data Loading for SPMLDataset ---
    num_classes = 4 # Fixed for this task

    all_file_paths = []
    all_labels = []

    # Load data for each class
    if args.normal_person_dir:
        normal_person_paths = natsorted(list(pathlib.Path(args.normal_person_dir).glob('**/*.jpg')))
        all_file_paths.extend(normal_person_paths)
        all_labels.extend([0] * len(normal_person_paths))
        if rank == 0: logging.info(f"Loaded {len(normal_person_paths)} images for 'Normal Person'.")

    if args.hat_person_dir:
        hat_person_paths = natsorted(list(pathlib.Path(args.hat_person_dir).glob('**/*.jpg')))
        all_file_paths.extend(hat_person_paths)
        all_labels.extend([1] * len(hat_person_paths))
        if rank == 0: logging.info(f"Loaded {len(hat_person_paths)} images for 'Hat Person'.")

    if args.sunglass_person_dir:
        sunglass_person_paths = natsorted(list(pathlib.Path(args.sunglass_person_dir).glob('**/*.jpg')))
        all_file_paths.extend(sunglass_person_paths)
        all_labels.extend([2] * len(sunglass_person_paths))
        if rank == 0: logging.info(f"Loaded {len(sunglass_person_paths)} images for 'Sunglass Person'.")

    if args.mask_person_dir:
        mask_person_paths = natsorted(list(pathlib.Path(args.mask_person_dir).glob('**/*.jpg')))
        all_file_paths.extend(mask_person_paths)
        all_labels.extend([3] * len(mask_person_paths))
        if rank == 0: logging.info(f"Loaded {len(mask_person_paths)} images for 'Mask Person'.")

    if not all_file_paths:
        raise ValueError("No image files found. Please check the provided data directories.")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_file_paths, all_labels, test_size=args.val_split, random_state=42, stratify=all_labels
    )

    if rank == 0:
        logging.info(f"Total samples: {len(all_file_paths)}")
        logging.info(f"Train samples: {len(train_paths)}")
        logging.info(f"Validation samples: {len(val_paths)}")

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((640, 640)), 
        v2.TrivialAugmentWide(), 
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.RandomErasing(p=0.3, scale=(0.02, 0.15) , value=0), 
     ])
    

    val_transform = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Resize((640, 640)),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SPMLDataset(file_paths=train_paths, labels=train_labels, transform=train_transform)
    val_dataset = SPMLDataset(file_paths=val_paths, labels=val_labels, transform=val_transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    trainloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), pin_memory=True, num_workers=os.cpu_count()//world_size, sampler=train_sampler)
    valloader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=os.cpu_count()//world_size, sampler=val_sampler)

    model = get_model(args.model, num_classes=num_classes).to(device)
    if args.compile:
        model = torch.compile(model)

    if args.pretrained and os.path.isfile(args.weight_path):
        logging.info(f"Loading weights from {args.weight_path}")
        checkpoint = torch.load(args.weight_path, map_location=device)
        model_weight = checkpoint.get('model_state_dict', checkpoint)
        result = model.load_state_dict(model_weight, strict=False)
        logging.info(f"Weight loading result: {result}")

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    if rank == 0 and args.wandb:
        wandb.watch(model, log='all', log_freq=100)

    if rank == 0:
        backbone_params = sum(p.numel() for p in model.parameters())
        trainable_backbone_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info('==' * 30)
        logging.info(f"Model total params: {backbone_params:,}")
        logging.info(f"Model trainable params: {trainable_backbone_params:,}")
        logging.info(f"traineable params percentage: {100 * trainable_backbone_params / backbone_params:.2f}%%")
        logging.info('==' * 30)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = PolynomialLRWarmup(optimizer, warmup_iters=10, total_iters=args.epochs, power=1.0, limit_lr=1e-7)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    best_val_loss = float('inf')
    scaler = torch.amp.GradScaler()

    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss_sum = 0.0
        pbar_train = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", disable=(rank != 0))
        for data_input, label_one_hot in pbar_train:
            data_input = data_input.to(device, non_blocking=True)
            label_one_hot = label_one_hot.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type):
                preds = torch.sigmoid(model(data_input)) # Apply sigmoid to get probabilities

                # --- GR_loss Calculation ---
                # K and V are dynamically calculated based on model predictions,
                # representing the core idea of "soft pseudo-labeling".
                # Using preds.detach() for K is a common practice to treat
                # the model's own predictions as soft targets for the next iteration.
                with torch.no_grad():
                    K = preds.detach()
                    V = torch.ones_like(preds) # V can be used for class balancing if needed

                Q = (args.q2, args.q3)
                loss = GR_loss(preds, label_one_hot, K, V, Q, epoch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            if rank == 0:
                pbar_train.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss_sum / len(trainloader)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            pbar_val = tqdm(valloader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", disable=(rank != 0))
            for data_input, label_one_hot in pbar_val:
                data_input = data_input.to(device, non_blocking=True)
                label_one_hot = label_one_hot.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type):
                    preds = torch.sigmoid(model(data_input))

                    K = preds.detach()
                    V = torch.ones_like(preds)
                    Q = (args.q2, args.q3)
                    loss = GR_loss(preds, label_one_hot, K, V, Q, epoch)

                val_loss_sum += loss.item()
                if rank == 0:
                    pbar_val.set_postfix({'loss': loss.item()})

        avg_val_loss = val_loss_sum / len(valloader)

        if rank == 0 and args.wandb:
            wandb.log({
               "Train Loss": avg_train_loss,
               "Val Loss": avg_val_loss,
               "Learning Rate": scheduler.get_last_lr()[0]
            })

        if rank == 0:
            logging.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs('checkpoints', exist_ok=True)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss
                }
                torch.save(checkpoint, os.path.join('checkpoints', 'best_spml_model.pth'))
                logging.info("Saved best SPML model and optimizer state.")

            early_stopping(avg_val_loss)

        if world_size > 1:
            stop_signal = torch.tensor(1.0 if rank == 0 and early_stopping.early_stop else 0.0, device=device)
            dist.broadcast(stop_signal, src=0)
            if stop_signal.item() == 1.0:
                if rank == 0: logging.info("Early stopping triggered.")
                break
        else:
            if early_stopping.early_stop:
                logging.info("Early stopping triggered.")
                break

        scheduler.step()

    if world_size > 1:
        cleanup_ddp()

def main():
    parser = argparse.ArgumentParser(description='EfficientNetV2 SPML Training with Generalized Robust Loss')

    # Model arguments
    parser.add_argument('--model', type=str, default='efficientnetv2_l', choices=['efficientnetv2_s', 'efficientnetv2_l'], help='Model type')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--weight_path', type=str, default='checkpoints/best_spml_model.pth', help='Path to model weights')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile for model optimization')

    # Data arguments
    parser.add_argument('--normal_person_dir', type=str, required=True, help='Directory for "Normal Person" images (Class 0)')
    parser.add_argument('--hat_person_dir', type=str, required=True, help='Directory for "Hat Person" images (Class 1)')
    parser.add_argument('--sunglass_person_dir', type=str, required=True, help='Directory for "Sunglass Person" images (Class 2)')
    parser.add_argument('--mask_person_dir', type=str, required=True, help='Directory for "Mask Person" images (Class 3)')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of data to use for validation')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--no_confirm', action='store_true', help='Skip argument confirmation prompt')

    # GR_loss arguments
    parser.add_argument('--q2', type=float, default=0.7, help='q2 parameter for GR_loss (controls false positives)') # 음성레이블 손실함수 강도
    parser.add_argument('--q3', type=float, default=1.0, help='q3 parameter for GR_loss (controls false negatives)') # 양성레이블 손실함수 강도 

    # W&B arguments
    parser.add_argument('--wandb', action='store_true', help='Use wandb or not')
    parser.add_argument('--wandb_name', type=str, default='efficientnetv2_gr_loss_experiment', help='wandb experiment name')

    args = parser.parse_args()

    print("--- Training Arguments ---")
    for var in vars(args):
        print(f"{var}: {getattr(args, var)}")
    print("--------------------------")

    if not args.no_confirm:
        input("Press Enter to proceed...")

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
