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
from utils.sampler import DomainBalancedSampler, DistributedDomainBalancedSampler
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
from collections import OrderedDict


"""
Self-Paced Collaborative and Adversarial Network for Unsupervised Domain Adaptation
https://arxiv.org/abs/2506.19267
"""

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

def wandb_init(name='None'):
    wandb.init(
        project='EfficientNetV2_SPCAN', # Project name updated for SPCAN
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

    # --- Data Loading for DANN/SPCAN ---
    domain_A_wear = list(pathlib.Path(args.wear_dir).glob('**/*.jpg'))
    domain_A_nowear = list(pathlib.Path(args.nowear_dir).glob('**/*.jpg')) + \
                      list(pathlib.Path(args.nowear_plus_dir).glob('**/*.jpg'))

    domain_B_wear = load_extra_data(args.extra_wear_dir, args.extra_data_fraction)
    domain_B_nowear = load_extra_data(args.extra_nowear_dir, args.extra_data_fraction)

    all_data = [(path, 1, 0) for path in domain_A_wear] + \
               [(path, 0, 0) for path in domain_A_nowear] + \
               [(path, 1, 1) for path in domain_B_wear] + \
               [(path, 0, 1) for path in domain_B_nowear]
    
    all_paths = [d[0] for d in all_data]
    all_class_labels = [d[1] for d in all_data]

    train_paths, val_paths, train_labels_with_domain, val_labels_with_domain = train_test_split(
        all_paths, all_data, test_size=0.2, random_state=42, stratify=all_class_labels
    )

    train_A_wear = [p for p, d in zip(train_paths, train_labels_with_domain) if d[1] == 1 and d[2] == 0]
    train_A_nowear = [p for p, d in zip(train_paths, train_labels_with_domain) if d[1] == 0 and d[2] == 0]
    train_B_wear = [p for p, d in zip(train_paths, train_labels_with_domain) if d[1] == 1 and d[2] == 1]
    train_B_nowear = [p for p, d in zip(train_paths, train_labels_with_domain) if d[1] == 0 and d[2] == 1]

    val_A_wear = [p for p, d in zip(val_paths, val_labels_with_domain) if d[1] == 1 and d[2] == 0]
    val_A_nowear = [p for p, d in zip(val_paths, val_labels_with_domain) if d[1] == 0 and d[2] == 0]
    val_B_wear = [p for p, d in zip(val_paths, val_labels_with_domain) if d[1] == 1 and d[2] == 1]
    val_B_nowear = [p for p, d in zip(val_paths, val_labels_with_domain) if d[1] == 0 and d[2] == 1]

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((640, 640)), 
        v2.TrivialAugmentWide(), 
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.RandomErasing(p=0.5, scale=(0.02, 0.05) , value=0), 
     ])

    val_transform = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Resize((640, 640)),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = Domain_Sun_Glasses_Dataset(
        domain_A_wear_data=train_A_wear, domain_A_nowear_data=train_A_nowear,
        domain_B_wear_data=train_B_wear, domain_B_nowear_data=train_B_nowear,
        transform=train_transform
    )
    val_dataset = Domain_Sun_Glasses_Dataset(
        domain_A_wear_data=val_A_wear, domain_A_nowear_data=val_A_nowear,
        domain_B_wear_data=val_B_wear, domain_B_nowear_data=val_B_nowear,
        transform=val_transform
    )

    # --- Sampler and DataLoader Setup ---
    if world_size > 1:
        train_sampler = DistributedDomainBalancedSampler(
            train_dataset, num_replicas=world_size, rank=rank,
            batch_size=args.batch_size, domain_B_fraction=args.domain_b_fraction
        )
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = DomainBalancedSampler(train_dataset, batch_size=args.batch_size, domain_B_fraction=args.domain_b_fraction)
        val_sampler = None

    trainloader = data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=os.cpu_count()//world_size, sampler=train_sampler, shuffle=False)
    valloader = data.DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=os.cpu_count()//world_size, sampler=val_sampler, shuffle=False)

    # --- Loss Functions for DANN/SPCAN ---
    count_no_wear = len(train_A_nowear) + len(train_B_nowear)
    count_wear = len(train_A_wear) + len(train_B_wear)
    if count_no_wear > 0 and count_wear > 0:
        weight_for_class_0 = (count_no_wear + count_wear) / (2.0 * count_no_wear)
        weight_for_class_1 = (count_no_wear + count_wear) / (2.0 * count_wear)
        class_weights = torch.tensor([weight_for_class_0, weight_for_class_1], device=device)
    else:
        class_weights = None
    
    criterion_label = torch.nn.CrossEntropyLoss(weight=class_weights)
    # For SPCAN, we need per-sample loss, so set reduction='none'
    criterion_domain = torch.nn.CrossEntropyLoss(reduction='none' if args.spcan else 'mean')

    model = get_model(args.model).to(device)
    model = torch.compile(model)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    if rank == 0 and args.wandb:
        wandb.watch(model, log='all', log_freq=10)

    if rank == 0:
        backbone_params = sum(p.numel() for p in model.parameters())
        trainable_backbone_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info('==' * 30)
        logging.info(f"Model total params: {backbone_params:,}")
        logging.info(f"Model trainable params: {trainable_backbone_params:,}")
        logging.info(f"traineable params percentage: {100 * trainable_backbone_params / backbone_params:.2f}%")
        logging.info('==' * 30)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = PolynomialLRWarmup(
        optimizer, warmup_iters=10, total_iters=args.epochs, power=1.0 , limit_lr = 1e-5
    )

    early_stopping = EarlyStopping(patience=10, verbose=True)
    scaler = torch.amp.GradScaler()
    best_val_acc = 0.0

    # --- Training Loop for DANN/SPCAN ---
    for epoch in range(args.epochs):
        if world_size > 1: train_sampler.set_epoch(epoch)
        model.train()
        total_loss, label_loss_sum, domain_loss_sum = 0, 0, 0
        correct_train, total_train = 0, 0
        len_dataloader = len(trainloader)
        
        # SPCAN: Calculate gamma for the current epoch
        gamma = 0.0
        if args.spcan:
            gamma = args.spcan_gamma_init + (args.spcan_gamma_end - args.spcan_gamma_init) * (epoch / (args.epochs -1))

        pbar_train = tqdm(enumerate(trainloader), total=len_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", disable=(rank != 0))
        for i, (images, class_labels, domain_labels) in pbar_train:
            p = float(i + epoch * len_dataloader) / (args.epochs * len_dataloader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            images, class_labels, domain_labels = images.to(device), class_labels.to(device), domain_labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                label_output, domain_output = model(images, alpha=alpha)
                loss_l = criterion_label(label_output, class_labels)

                # --- SPCAN/DANN Domain Loss Calculation ---
                if args.spcan:
                    # 1. Calculate per-sample domain loss
                    domain_loss_per_sample = criterion_domain(domain_output, domain_labels)
                    
                    # 2. Calculate self-paced weights
                    # .detach() is used to prevent gradients from flowing through this weight calculation
                    weights = torch.exp(-gamma * domain_loss_per_sample.detach())
                    
                    # 3. Apply weights to get the final domain loss
                    loss_d = torch.mean(weights * domain_loss_per_sample)
                else:
                    # Standard DANN loss
                    loss_d = criterion_domain(domain_output, domain_labels)
                # --- End of SPCAN/DANN Block ---

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
            for images, class_labels, _ in pbar_val:
                images, class_labels = images.to(device), class_labels.to(device)

                with torch.amp.autocast(device_type=device.type):
                    label_output, _ = model(images, alpha=0)
                    loss = criterion_label(label_output, class_labels)

                val_loss += loss.item()
                _, predicted = torch.max(label_output.data, 1)
                total_val += class_labels.size(0)
                correct_val += (predicted == class_labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        
        if rank == 0 and args.wandb:
            log_data = {
               "Train Total Loss": total_loss / len_dataloader,
               "Train Label Loss": label_loss_sum / len_dataloader,
               "Train Domain Loss": domain_loss_sum / len_dataloader,
               "Train Accuracy": train_accuracy,
               "Val Loss": val_loss / len(valloader),
               "Val Accuracy": val_accuracy,
               "Learning Rate": scheduler.get_last_lr()[0],
               "DANN Alpha": alpha
            }
            if args.spcan:
                log_data["SPCAN Gamma"] = gamma
                # We can also log the mean of weights to see how it changes
                # This requires re-calculating weights for one batch or storing them
                # For simplicity, we log gamma here.

            wandb.log(log_data)
            
        if rank == 0:
            logging.info(f"Epoch {epoch+1}, Train Loss: {total_loss/len_dataloader:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss/len(valloader):.4f}, Val Acc: {val_accuracy:.2f}%")

            if (epoch + 1) % 2 == 0 :
                state_to_save = model.module.state_dict() if world_size > 1 else model.state_dict()
                
                cleaned_state_dict = OrderedDict()
                for k, v in state_to_save.items():
                    if '_orig_mod.' in k:
                        name = k.replace('_orig_mod.', '')
                    else:
                        name = k
                    cleaned_state_dict[name] = v
                
                torch.save(cleaned_state_dict, os.path.join('checkpoints' , f'{args.model}_{epoch+1}.pth'))
                logging.info(f"Saved checkpoint at epoch {epoch+1}.")

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                state_to_save = model.module.state_dict() if world_size > 1 else model.state_dict()
                
                cleaned_state_dict = OrderedDict()
                for k, v in state_to_save.items():
                    if '_orig_mod' in k:
                        name = k.replace('_orig_mod.', '')
                    else:
                        name = k
                    cleaned_state_dict[name] = v
                
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(cleaned_state_dict, os.path.join('checkpoints', f'{args.model}_best.pth'))
                logging.info("Saved best model state.")

        if early_stopping.early_stop: break
        scheduler.step()

    if world_size > 1: cleanup_ddp()

def main():
    parser = argparse.ArgumentParser(description='SPCAN/DANN Training for EfficientNetV2')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='efficientnetv2_s_dann', choices=['efficientnetv2_s_dann'], help='Model type')
    parser.add_argument('--domain_lambda', type=float, default=1.0, help='Weight for the domain loss in DANN/SPCAN')
    parser.add_argument('--domain_b_fraction', type=float, default=0.3, help='Fraction of domain B data in each batch for balanced sampling.')

    # --- SPCAN Arguments ---
    parser.add_argument('--spcan', default=True, help='Use SPCAN instead of standard DANN')
    parser.add_argument('--spcan_gamma_init', type=float, default=0.5, help='Initial gamma for SPCAN scheduler')
    parser.add_argument('--spcan_gamma_end', type=float, default=5.0, help='Final gamma for SPCAN scheduler')

    # Data arguments
    parser.add_argument('--wear_dir', type=str, default='/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/sunglass/refining_yaw_yaw', help='Directory for "wear" images in Domain A')
    parser.add_argument('--nowear_dir', type=str, default='/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/earing/refining_yaw_yaw', help='Directory for "no wear" images in Domain A')
    parser.add_argument('--nowear_plus_dir', type=str, default='/media/ubuntu/76A01D5EA01D25E1/009.패션 액세서리 착용 데이터/01-1.정식개방데이터/Training/01.원천데이터/glasses/refining_yaw_yaw', help='Directory for additional "no wear" images in Domain A')
    parser.add_argument('--extra_wear_dir', type=str, default='/home/ubuntu/KOR_DATA/sunglass_dataset/wear/wear_data2', help='Directory for "wear" images in Domain B (extra)')
    parser.add_argument('--extra_nowear_dir', type=str, default='/home/ubuntu/KOR_DATA/sunglass_dataset/nowear/no_wear_data2', help='Directory for "no wear" images in Domain B (extra)')
    parser.add_argument('--extra_data_fraction', type=float, default=0.25, help='Fraction of extra data to use')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--no_confirm', action='store_true', help='Skip argument confirmation prompt')

    # W&B arguments
    parser.add_argument('--wandb', action='store_true', help='Use wandb or not')
    parser.add_argument('--wandb_name', type=str, default='spcan_experiment', help='wandb experiment name')
    
    args = parser.parse_args()

    print("---" + "-" * 17 + " Training Arguments " + "-" * 17 + " ---")
    for var in vars(args):
        print(f"{var}: {getattr(args, var)}")
    print("-" * 50)
    
    if not args.no_confirm:
        input("Press Enter to proceed...")

    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        main_worker(0, 1, args)

if __name__ == '__main__':
    main()
