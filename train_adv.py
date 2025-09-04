import logging
import torch
from torch.utils import data
from tqdm import tqdm
import wandb
import os
import pathlib
from dataset import Sun_glasses_Dataset, Domain_Sun_Glasses_Dataset
import numpy as np
import cv2
import logging
import os
from backbone.model import EfficientNetV2_S, EfficientNetV2_L, EfficientNetV2_S_DANN
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
import torchvision.transforms.v2 as v2

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def wandb_init(name='None'):
    wandb.init(
        project='EfficientNetV2_DANN',
        name=name,
    )

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

def get_model(model_name, **kwargs):
    if model_name == 'efficientnetv2_s':
        return EfficientNetV2_S()
    elif model_name == 'efficientnetv2_l':
        return EfficientNetV2_L()
    elif model_name == 'efficientnetv2_s_dann':
        return EfficientNetV2_S_DANN(num_classes=kwargs.get('num_classes', 2), num_domains=kwargs.get('num_domains', 2))
    else:
        raise ValueError(f"Model {model_name} not recognized.")

def main_worker(rank, world_size, args):
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    if rank == 0 and args.wandb:
        wandb_init(name=f'{args.wandb_name}')

    torch.backends.cudnn.benchmark = True
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    train_transform  = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32 ,scale=True),
            v2.RandomHorizontalFlip(p=0.3),
            v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)], p=0.3),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
            v2.Resize(size=(320, 320)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    val_transform  = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32 ,scale=True),
            v2.Resize(size=(320, 320)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])



    # --- Data Loading for DANN ---
    # Domain A: Main dataset
    domain_A_wear = list(pathlib.Path(args.wear_dir).glob('**/*.jpg'))
    domain_A_nowear = list(pathlib.Path(args.nowear_dir).glob('**/*.jpg')) + \
                      list(pathlib.Path(args.nowear_plus_dir).glob('**/*.jpg'))

    # Domain B: Extra dataset
    domain_B_wear = load_extra_data(args.extra_wear_dir, args.extra_data_fraction)
    domain_B_nowear = load_extra_data(args.extra_nowear_dir, args.extra_data_fraction)

    # Create a combined list for train/val split, preserving domain info
    all_data = [(path, 1, 0) for path in domain_A_wear] + \
               [(path, 0, 0) for path in domain_A_nowear] + \
               [(path, 1, 1) for path in domain_B_wear] + \
               [(path, 0, 1) for path in domain_B_nowear]
    
    all_paths = [d[0] for d in all_data]
    all_class_labels = [d[1] for d in all_data]

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_data, test_size=0.2, random_state=42, stratify=all_class_labels
    )

    # Reconstruct the domain-specific lists for the training set
    train_A_wear = [p for p, d in zip(train_paths, train_labels) if d[1] == 1 and d[2] == 0]
    train_A_nowear = [p for p, d in zip(train_paths, train_labels) if d[1] == 0 and d[2] == 0]
    train_B_wear = [p for p, d in zip(train_paths, train_labels) if d[1] == 1 and d[2] == 1]
    train_B_nowear = [p for p, d in zip(train_paths, train_labels) if d[1] == 0 and d[2] == 1]

    # For validation, we don't need domain labels, but we'll use the standard dataset
    val_wear = [p for p, d in zip(val_paths, val_labels) if d[1] == 1]
    val_nowear = [p for p, d in zip(val_paths, val_labels) if d[1] == 0]

    train_dataset = Domain_Sun_Glasses_Dataset(
        domain_A_wear_data=train_A_wear, domain_A_nowear_data=train_A_nowear,
        domain_B_wear_data=train_B_wear, domain_B_nowear_data=train_B_nowear,
        transform=train_transform # Define transform later
    )

    val_dataset = Domain_Sun_Glasses_Dataset(
        wear_data=val_wear, no_wear_data=val_nowear, transform=val_transform
        )

    # Define transforms (You can customize the augmentations here)
    train_transform = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Resize((640, 640)), v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Resize((640, 640)),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset.transform = train_transform
    val_dataset.transform = val_transform

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    trainloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), pin_memory=True, num_workers=os.cpu_count()//world_size, sampler=train_sampler)
    valloader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=os.cpu_count()//world_size, sampler=val_sampler)

    # --- Loss Functions for DANN ---
    # 1. Class label loss (with weighting for class imbalance)
    count_no_wear = len(train_A_nowear) + len(train_B_nowear)
    count_wear = len(train_A_wear) + len(train_B_wear)
    if count_no_wear > 0 and count_wear > 0:
        weight_for_class_0 = (count_no_wear + count_wear) / (2.0 * count_no_wear)
        weight_for_class_1 = (count_no_wear + count_wear) / (2.0 * count_wear)
        class_weights = torch.tensor([weight_for_class_0, weight_for_class_1], device=device)
    else:
        class_weights = None
    criterion_label = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # 2. Domain classification loss
    criterion_domain = torch.nn.CrossEntropyLoss()

    model = get_model(args.model).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    scaler = torch.amp.GradScaler()
    best_val_acc = 0.0

    # --- Training Loop for DANN ---
    for epoch in range(args.epochs):
        if train_sampler: train_sampler.set_epoch(epoch)
        model.train()
        total_loss, label_loss_sum, domain_loss_sum = 0, 0, 0
        correct_train, total_train = 0, 0
        len_dataloader = len(trainloader)

        pbar_train = tqdm(enumerate(trainloader), total=len_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", disable=(rank != 0))
        for i, (images, class_labels, domain_labels) in pbar_train:
            p = float(i + epoch * len_dataloader) / (args.epochs * len_dataloader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            images = images.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)
            domain_labels = domain_labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                label_output, domain_output = model(images, alpha=alpha)
                loss_l = criterion_label(label_output, class_labels)
                loss_d = criterion_domain(domain_output, domain_labels)
                loss = loss_l + (args.domain_lambda * loss_d)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            label_loss_sum += loss_l.item()
            domain_loss_sum += loss_d.item()

            _, predicted = torch.max(label_output.data, 1)
            total_train += class_labels.size(0)
            correct_train += (predicted == class_labels).sum().item()
            
            if rank == 0:
                pbar_train.set_postfix({'L_total': loss.item(), 'L_label': loss_l.item(), 'L_domain': loss_d.item()})

        train_accuracy = 100 * correct_train / total_train

        # --- Validation Loop ---
        model.eval()
        val_loss, correct_val, total_val = 0, 0, 0
        with torch.no_grad():
            pbar_val = tqdm(valloader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", disable=(rank != 0))
            for images, class_labels , _  in pbar_val:
                images = images.to(device, non_blocking=True)
                class_labels = class_labels.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type):
                    # For validation, we only care about the label prediction
                    label_output, _ = model(images, alpha=0)
                    loss = criterion_label(label_output, class_labels)

                val_loss += loss.item()
                _, predicted = torch.max(label_output.data, 1)
                total_val += class_labels.size(0)
                correct_val += (predicted == class_labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        
        if rank == 0 and args.wandb:
            wandb.log({
               "Train Total Loss": total_loss / len_dataloader,
               "Train Label Loss": label_loss_sum / len_dataloader,
               "Train Domain Loss": domain_loss_sum / len_dataloader,
               "Train Accuracy": train_accuracy,
               "Val Loss": val_loss / len(valloader),
               "Val Accuracy": val_accuracy,
               "Learning Rate": scheduler.get_last_lr()[0]
            })

        if rank == 0:
            logging.info(f"Epoch {epoch+1}, Train Loss: {total_loss/len_dataloader:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss/len(valloader):.4f}, Val Acc: {val_accuracy:.2f}%")

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(model.state_dict(), os.path.join('checkpoints', f'{args.model}_best.pth'))
                logging.info("Saved best model state.")

        if early_stopping.early_stop: break
        scheduler.step()

    if world_size > 1: cleanup_ddp()

def main():
    parser = argparse.ArgumentParser(description='DANN Training for EfficientNetV2')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='efficientnetv2_s_dann', choices=['efficientnetv2_s_dann'], help='Model type')
    parser.add_argument('--domain_lambda', type=float, default=1.0, help='Weight for the domain loss in DANN')


    # Data arguments
    parser.add_argument('--wear_dir', type=str, default='/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/sunglass/refining_yaw_yaw', help='Directory for "wear" images')
    parser.add_argument('--nowear_dir', type=str, default='/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/earing/refining_yaw_yaw', help='Directory for "no wear" images')
    parser.add_argument('--nowear_plus_dir', type=str, default='/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/glasses/refining_yaw_yaw', help='Directory for additional "no wear" images')
    parser.add_argument('--extra_wear_dir', type=str, default='/home/ubuntu/KOR_DATA/sunglass_dataset/wear/wear_data2', help='Directory for extra "wear" images')
    parser.add_argument('--extra_nowear_dir', type=str, default='/home/ubuntu/KOR_DATA/sunglass_dataset/nowear/no_wear_data2', help='Directory for extra "no wear" images')
    parser.add_argument('--extra_data_fraction', type=float, default=0.25, help='Fraction of extra data to use')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    
    # W&B arguments
    parser.add_argument('--wandb', action='store_true', help='Use wandb or not')
    parser.add_argument('--wandb_name', type=str, default='dann_experiment', help='wandb experiment name')
    
    args = parser.parse_args()

    print("---" + "-" * 17 + " Training Arguments " + "-" * 17 + "---")
    for var in vars(args):
        print(f"{var}: {getattr(args, var)}")
    print("-" * 50)
    
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        main_worker(0, 1, args)

if __name__ == '__main__':
    main()